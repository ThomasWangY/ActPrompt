# Version 1
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
from basic_utils import Preprocessing, RandomSequenceSampler, \
    load_jsonl, l2_normalize_np_array
from tqdm import tqdm
import os
import h5py
import json
from collections import defaultdict


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def extract(model, opt):
    dataset = VideoLoader(
        opt.video_path,
        framerate=1/opt.clip_len,
        size=224 if opt.model_version == "ViT-B/32" else 288,
        centercrop=True,
        overwrite=opt.overwrite,
        model_version=opt.model_version
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=opt.ex_bsz,
        shuffle=False,
        num_workers=opt.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )
    preprocess = Preprocessing()
    image_encoder = model.image_encoder
    video_encoder = model.video_encoder

    data_json = load_jsonl(opt.train_path)
    data_json += load_jsonl(opt.test_path)
    if opt.eval_path != opt.test_path:
        data_json += load_jsonl(opt.eval_path)

    data_list = defaultdict(list)
    for item in data_json:
        if item['vid'] not in data_list:
            data_list[item['vid']] = [(item['qid'], item['query'])]
        else:
            data_list[item['vid']].append((item['qid'], item['query']))

    totatl_num_frames = 0
    with torch.no_grad():
        sl_dir = os.path.join(opt.dset_path, "vid_slowfast")
        vid_list = [".".join(file.split('.')[:-1]) for file in os.listdir(sl_dir)]
        for k, data in enumerate(tqdm(loader)):
            vid = data['vid'][0]
            if opt.dset_name == 'TaCoS':
                vid = vid[:7]
            if vid not in vid_list: 
                print(vid, "No vid!")
                continue
            if len(data['video'].shape) > 4:
                file = f'{vid}.npz'
                video = data['video'].squeeze(0)
                video_feature = [torch.from_numpy(np.load(os.path.join(sl_dir, file))['features'])]
                
                video_feature = video_encoder(video_feature)
            
                if len(video.shape) == 4:
                    image = preprocess(video)
                    image_batch = image.cuda()
                    video_prompt = video_feature['x_prompt'][0]

                    if len(video_prompt) == len(image_batch) + 1:
                        video_prompt = video_prompt[:-1]
                    elif len(video_prompt) == len(image_batch) - 1:
                        image_batch = image_batch[:-1]
                    image_features = image_encoder(image_batch, video_prompt)
                    features = image_features['x_feats'][0].cpu().numpy()
                    if opt.half_precision:
                        features = features.astype('float16')
                    totatl_num_frames += features.shape[0]
                    output_file = os.path.join(opt.dset_path, "vid_clip_new", file)
                    dirname = os.path.dirname(output_file)
                    if not os.path.exists(dirname):
                        print(f"Output directory {dirname} does not exists, creating...")
                        os.makedirs(dirname)
                    np.savez(output_file, features=features)

            else:
                print(f'Failed at ffprobe.\n')
        print("Finish Extraction!!!")

    print(f"Total number of frames: {totatl_num_frames}")

    data_path = [opt.train_path, opt.test_path]
    if opt.dset_name == "QVHighlights":
        data_path = [opt.train_path, opt.eval_path, opt.test_path]

    def load_data():
        datalist = []
        for dset_path in data_path:
            dset_list = load_jsonl(dset_path)
            datalist += dset_list
        return datalist

    q_feat_type = 'txt_clip'
    v_feat_type_1 = 'vid_clip_new'
    v_feat_type_2 = 'vid_slowfast'

    h5py_dir = os.path.join(opt.dset_path, 'h5py')
    if not os.path.exists(h5py_dir):
        os.mkdir(h5py_dir)

    data = load_data()

    with torch.no_grad():
        f_q = h5py.File(os.path.join(h5py_dir, q_feat_type + '.hdf5'), 'w')
        for meta in tqdm(data):
            qid = meta['qid']
            q_feat_path = os.path.join(opt.dset_path, q_feat_type, f"{qid}.npz")
            if os.path.exists(q_feat_path):
                if str(qid) not in f_q:
                    q_feat = np.load(q_feat_path)["last_hidden_state"].astype(np.float32)
                    q_feat = l2_normalize_np_array(q_feat) 
                    f_q[str(qid)] = q_feat
                    del q_feat
            else:
                print("File does not exists!!!")
        f_q.close()
        del f_q

        f_v = h5py.File(os.path.join(h5py_dir, v_feat_type_1[:-4] + '.hdf5'), 'w')
        for meta in tqdm(data):
            vid = meta['vid']
            v1_feat_path = os.path.join(opt.dset_path, v_feat_type_1, f"{vid}.npz")
            if os.path.exists(v1_feat_path):
                if str(vid) not in f_v:
                    v_feat = np.load(v1_feat_path)["features"].astype(np.float32)
                    v_feat = l2_normalize_np_array(v_feat)
                    f_v[str(vid)] = v_feat
                    del v_feat
            else:
                print("File does not exists!!!")
        f_v.close()
        del f_v

        f_s = h5py.File(os.path.join(h5py_dir, v_feat_type_2 + '.hdf5'), 'w')
        for meta in tqdm(data):
            vid = meta['vid']
            v2_feat_path = os.path.join(opt.dset_path, v_feat_type_2, f"{vid}.npz")
            if os.path.exists(v2_feat_path):
                if str(vid) not in f_s:
                    v_feat = np.load(v2_feat_path)["features"].astype(np.float32)
                    v_feat = l2_normalize_np_array(v_feat)
                    f_s[str(vid)] = v_feat
                    del v_feat
            else:
                print("File does not exists!!!")
        f_s.close()
        del f_s
