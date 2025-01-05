import torch
import torch.nn as nn
from clip import clip
import copy
import torch.nn.functional as F
import random
from torch.nn import Linear, Softmax, ReLU
import spacy


nlp = spacy.load("en_core_web_sm") # load spacy

def load_clip_to_cpu():
    url = clip._MODELS['ViT-B/32']
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VisionEncoder(nn.Module):
    def __init__(self, opt, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.size = opt.size

        v_dim = clip_model.visual.ln_pre.weight.shape[0]
        ctx_vectors = torch.empty(len(self.transformer)-2, self.size, v_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.pos_ctx = nn.Parameter(ctx_vectors)
        
        # f_temp
        self.ctx_temp = nn.Sequential(
                Linear(v_dim, v_dim//4, dtype=self.dtype),
                ReLU(),
                Linear(v_dim//4, v_dim, dtype=self.dtype)
            )

    def weighted_conv(self, x_prompt):
        x_prompt = F.pad(x_prompt, (0, 0, self.size//2, self.size//2), mode='replicate').permute(1, 0, 2)
        x_prompt = x_prompt.unfold(0, x_prompt.size(0)-self.size+1, 1).permute(1, 0, 3, 2).squeeze(0)
        return x_prompt

    def forward(self, video, prompts, mode=False):
        if not isinstance(video, list):
            video = [video]

        # two branch during training
        # mode == False -> using video_prompts
        # mode == True  -> using verb_prompts
        if mode == False:
            video_prompts = prompts
            if not isinstance(video_prompts, list):
                video_prompts = [video_prompts]
        else:
            verb_prompts = torch.cat([prompts]*3, dim=2)
            
        x_feats, x_feat_means, x_attns = [], [], []
        
        for video_idx, frames in enumerate(video):
            x = frames.cuda().to(self.dtype)
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)
            if self.pos_ctx.dim() == 3:
                positional_prompt = self.pos_ctx.unsqueeze(0).expand(len(x), -1, -1, -1).permute(1, 2, 0, 3)
            x = x.permute(1, 0, 2)

            if mode == True:
                verb_prompt = verb_prompts[:, :, video_idx].unsqueeze(2).expand(-1, -1, x.shape[1], -1)
            else:
                video_prompt = video_prompts[video_idx].unsqueeze(0)

            x_attn = []

            for layer_idx, layer in enumerate(self.transformer):
                x, attn = layer(x)
                if layer_idx == 0:
                    n_patch = x.shape[0]
                    # concatenate with action-aware prompt
                    if mode == True:
                        x = torch.cat([x, verb_prompt[layer_idx]], dim=0)
                    else:
                        x = torch.cat([x, video_prompt], dim=0)

                elif 0 < layer_idx < len(self.transformer) - 1:
                    # obtain attention map corresponding with action-aware prompt
                    attn = attn.sum(dim=1)[:, n_patch, 1:n_patch]
                    x_attn.append(attn)
                    
                    if mode == True:
                        x = torch.cat([x[:-1], verb_prompt[layer_idx]], dim=0)
                    else:
                        if layer_idx > 1:
                            x = x[:-self.size]
                        # concatenate with temporal prompt
                        attn_idx = torch.topk(attn, 1, dim=-1).indices
                        sampled_patches = torch.stack([x[1+attn_idx[idx], idx] for idx in range(x.shape[1])], dim=1)
                        temp_prompt = self.weighted_conv(sampled_patches) + positional_prompt[layer_idx-1]
                        temp_prompt = temp_prompt + self.ctx_temp(temp_prompt)
                        x = torch.cat([x, temp_prompt], dim=0)

            x_attn = torch.stack(x_attn, dim=0)
            x_attns.append(x_attn)

            x_feat = self.ln_post(x[0]) @ self.proj
            x_feat = x_feat / x_feat.norm(dim=-1, keepdim=True)
            x_feats.append(x_feat)

            x_feat_mean = torch.mean(x_feat, dim=0)
            x_feat_means.append(x_feat_mean)
            
        x_feat_means = torch.stack(x_feat_means, dim=0)
        return dict(x_feats=x_feats, x_feat_means=x_feat_means, x_attns=x_attns)
    
    
class TextEncoder(nn.Module):
    def __init__(self, opt, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        # f_veb
        single_layer = nn.Linear(512, 768, dtype=self.dtype)
        self.ctx_veb = _get_clones(single_layer, len(self.transformer) - 1) # text => image projection

    def forward(self, text):
        x_feats, x_feat_means, x_prompts = [], [], []

        if not isinstance(text, list):
            text = [text]

        # obtain verbs
        docs = [nlp(t) for t in text]
        verbs = [[token.text for token in doc if token.pos_ == 'VERB'] for doc in docs]
        prompts = clip.tokenize(text, context_length=77).cuda() # [batch, 77]

        # obtain verb index
        verbs_idx = []
        for prompt, verb in zip(prompts, verbs):
            tokens_text = [prompt[0]] if not verb else clip.tokenize(verb)[:, 1]
            verb_idx = [torch.where(prompt == token)[0][0].item() for token in tokens_text if torch.where(prompt == token)[0].shape[0] != 0]
            verbs_idx.append(verb_idx if verb_idx else [0])
        
        x = self.token_embedding(prompts)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2).type(self.dtype) 
        
        for layer_idx, layer in enumerate(self.transformer):
            x, attn = layer(x)
            if layer_idx != len(self.transformer) - 1:
                attn = attn.sum(dim=1)
                attn = attn[torch.arange(x.shape[1]), prompts.argmax(dim=-1)]

                # extract verb with the highest score
                new_verbs_idx = []
                for s_idx, verb_list in enumerate(verbs_idx):
                    max_attn_idx = torch.argmax(torch.tensor([attn[s_idx][v_idx] for v_idx in verb_list]))
                    new_verbs_idx.append(verb_list[max_attn_idx])

                x_prompt = x.permute(1, 0, 2)[torch.arange(x.shape[1]), new_verbs_idx].unsqueeze(0)
                x_prompt = self.ctx_veb[layer_idx](x_prompt)
                x_prompts.append(x_prompt)
            
        x_prompts = torch.stack(x_prompts, dim=0)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        valid_lengths = (prompts != 0).sum(1).tolist()
        for j, valid_len in enumerate(valid_lengths):
            x_feat = (x[j, :valid_len] @ self.text_projection) / (x[j, :valid_len] @ self.text_projection).norm(dim=-1, keepdim=True)
            x_feats.append(x_feat)
            x_feat_means.append(x_feat[-1])

        x_feat_means = torch.stack(x_feat_means, dim=0)

        return dict(x_feat_means=x_feat_means, x_feats=x_feats, x_prompts=x_prompts)
    
    
class Video_Encoder(nn.Module):
    def __init__(self, opt, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        # f_vid
        self.ctx_vid = Linear(2304, 768, dtype=self.dtype)

    def forward(self, features):
        x_prompt= [self.ctx_vid(feat.cuda()) for feat in features]
        return dict(x_prompt=x_prompt)

class CLIPExtractor(nn.Module):
    def __init__(self, opt):
        super().__init__()

        clip_model = load_clip_to_cpu().cuda()
        self.device = torch.device("cuda")
        self.image_encoder = VisionEncoder(opt, clip_model)
        self.text_encoder = TextEncoder(opt, clip_model)
        self.video_encoder = Video_Encoder(opt, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model
        self.seed = opt.seed

    def compute_matmul_(self, a, b):
        logits_all = [
            torch.stack([
                self.logit_scale.exp() * torch.mean(feat @ b[idx].t(), dim=0)
                for feat in a[idx]
            ], dim=0)
            for idx in range(len(a))
        ]
        return torch.stack(logits_all, dim=0)

    def forward(self, video, image, txt, eval=False):
        feat_len = len(video[0])
        (video_pos, video_neg_intra) = video
        video_neg_inter = video_neg_intra[:]
        (image_pos, image_neg_intra) = image
        image_neg_inter = image_neg_intra[:]

        combined = list(zip(video_neg_inter, image_neg_inter))
        random.seed(self.seed)
        random.shuffle(combined)

        video_neg_inter, image_neg_inter = zip(*combined)

        # make 4-tuple for ranking
        video = video_pos + video_neg_intra + list(video_neg_inter)
        image = image_pos + image_neg_intra + list(image_neg_inter)

        query_features = self.text_encoder(txt)
        query_feats = query_features['x_feat_means']
        verb_prompts = query_features['x_prompts']
        video_prompts = self.video_encoder(video)['x_prompt']  
        image_features_video = self.image_encoder(image, video_prompts)

        if eval==False:
            image_features_verb = self.image_encoder(image, verb_prompts, mode=True)
            image_features_video_attn, image_features_verb_attn = image_features_video['x_attns'][:feat_len], image_features_verb['x_attns'][:feat_len]
            
            for attn in image_features_verb_attn:
                attn = attn.detach()
                
            # consistency loss
            loss_f = nn.MSELoss()
            loss_con = sum(loss_f(attn, txt_attn) for attn, txt_attn in zip(image_features_video_attn, image_features_verb_attn))
            loss_con = loss_con / len(image_features_video_attn)
        else:
            loss_con = None
            
        image_feats_video = image_features_video['x_feats']
        image_features_video, image_features_video_neg_intra, image_features_video_neg_inter = \
            image_feats_video[:feat_len], image_feats_video[feat_len:2*feat_len], image_feats_video[-feat_len:]

        image_features_video = torch.stack([torch.mean(feat, dim=0) for feat in image_features_video])
        # ce loss
        logits_ce = image_features_video @ query_feats.t()
        
        image_features_video_all = [[image_features_video[idx], image_features_video_neg_intra[idx], image_features_video_neg_inter[idx]] \
                                    for idx in range(len(image_features_video))]
        # tri-ranking loss
        logits_tri_video = self.compute_matmul_(image_features_video_all, query_feats)

        if eval==False:
            image_feats_verb = image_features_verb['x_feats']
            image_features_verb, image_features_verb_neg_intra, image_features_verb_neg_inter = \
                image_feats_verb[:feat_len], image_feats_verb[feat_len:2*feat_len], image_feats_verb[-feat_len:]
            
            image_features_verb_all = [[image_features_verb[idx], image_features_verb_neg_intra[idx], image_features_verb_neg_inter[idx]] \
                                        for idx in range(len(image_features_verb))]
            # tri-ranking loss
            logits_tri_verb = self.compute_matmul_(image_features_verb_all, query_feats)
        else:
            logits_tri_verb = None

        return logits_ce, logits_tri_video, logits_tri_verb, loss_con