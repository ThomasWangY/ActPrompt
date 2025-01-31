# Version 6.0
import torch as th
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import ffmpeg
import math
import torch
from collections import defaultdict
from basic_utils import load_jsonl
import random
from basic_utils import Preprocessing


def start_end_collate_mr(batch):
    batched_data = [[e[i] for e in batch] for i in range(5)]
    return batched_data

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))

def get_loader(dataset, opt, data_split=None):
    bsz = opt.bsz 

    print("dataset for epoch:", len(dataset))

    return DataLoader(
        dataset,
        collate_fn=start_end_collate_mr,
        batch_size=bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

class DatasetMR(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            opt,
            size=224,
            centercrop=True,
            overwrite=True,
            model_version="ViT-B/32",
            data_split=None
    ):
        """
        Args:
        """
        self.centercrop = centercrop
        self.size = size
        self.framerate = 1/opt.clip_len
        self.overwrite = overwrite
        self.model_version = model_version
        self.data_path = opt.video_path
        self.cache_path = opt.cache_path
        self.data_split = data_split
        self.max_len = opt.max_len
        self.min_len = opt.min_len
        self.opt = opt
        self.seed = opt.seed
        self.process = Preprocessing()
        self.data_ratio = opt.data_ratio

        self.json_path = opt.train_path

        (self.data, self.vid_dic) = self.load_data_multi()

    def load_data(self):
        datalist = load_jsonl(self.json_path)
        return datalist
    
    def load_data_multi(self):
        datalist = load_jsonl(self.json_path)
        vid_dic = defaultdict(list)
        clip_len = int(1/self.framerate)
        datalist_multi = []
        for data in datalist:
            data["duration"] = int(data["duration"]) - int(data["duration"]) % clip_len
            if "relevant_windows" not in data.keys():
                data["relevant_windows"] = [[0, data["duration"]]]
            start, end = int(data["relevant_windows"][0][0]), int(data["relevant_windows"][0][1])
            start = start - start % clip_len
            end = end - end % clip_len
            if start == end:
                if start > 0: start = start - clip_len
                else: end = end + clip_len

            end_anchor = min(end, start+self.max_len)
            window_anchor = [start, end_anchor]

            segment_len = end_anchor - start

            if start > (data["duration"] - end):
                start_neg, end_neg = max(start-segment_len, 0), start
            else:
                start_neg, end_neg = end, min(end+segment_len, data["duration"])
            window_neg = [start_neg, end_neg]
            data["relevant_windows"] = [window_anchor, window_neg]

            datalist_multi.append(data)
            for l in [window_anchor, window_neg]:
                if l not in vid_dic[data['vid']]:
                    vid_dic[data['vid']].append(l)
        
        random.seed(self.seed)
        random.shuffle(datalist_multi)

        if self.data_split is not None:
            n_data = len(datalist_multi)
            n_split = int(n_data // self.opt.n_epoch * self.data_ratio)
            datalist_multi = datalist_multi[self.data_split*n_split:(self.data_split+1)*n_split]

        print("datalist_multi:", len(datalist_multi))
        return datalist_multi, vid_dic

    def __len__(self):
        return len(self.data)
        
    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        meta = self.data[idx]
        txt = meta["query"]
        windows = meta["relevant_windows"]

        start_anchor, end_anchor = windows[0][0], windows[0][1]
        start_neg, end_neg = windows[1][0], windows[1][1]

        sl_path = os.path.join("/".join(self.json_path.split('/')[:-2]), "vid_slowfast", meta["vid"]+'.npz')
        if os.path.isfile(sl_path):
            video_slowfast = torch.from_numpy(np.load(sl_path)['features'])
            sl_anchor = video_slowfast[int(start_anchor*self.framerate):int(end_anchor*self.framerate)]
            sl_neg = video_slowfast[int(start_neg*self.framerate):int(end_neg*self.framerate)]
        else:
            print("No slowfast feature!")
        if self.cache_path is not None:
            ffm_path_anchor = os.path.join(self.cache_path, "_".join([meta["vid"], str(start_anchor), str(end_anchor)])+'.npz')
            ffm_path_neg = os.path.join(self.cache_path, "_".join([meta["vid"], str(start_neg), str(end_neg)])+'.npz')
            if os.path.isfile(ffm_path_anchor) and os.path.isfile(ffm_path_neg):
                cl_anchor = self.process(torch.from_numpy(np.load(ffm_path_anchor)['features']))
                cl_neg = self.process(torch.from_numpy(np.load(ffm_path_neg)['features']))
            else:
                if 'TaCoS' in self.data_path:
                    video_path = os.path.join(self.data_path, meta["vid"]+'-cam-002.avi')
                else:
                    video_path = os.path.join(self.data_path, meta["vid"]+'.mp4')
                load_flag = os.path.isfile(video_path)

                if load_flag:
                    info = self._get_video_info(video_path)
                    h, w = info["height"], info["width"]
                    height, width = self._get_output_dim(h, w)

                    duration = info["duration"]
                    fps = self.framerate
                    if duration > 0 and duration < 1/fps+0.1:
                        fps = 2/max(int(duration), 1)
                        print(duration, fps)
                    cmd = (
                        ffmpeg
                        .input(video_path)
                        .filter('fps', fps=fps)
                        .filter('scale', width, height)
                    )
                    if self.centercrop:
                        x = int((width - self.size) / 2.0)
                        y = int((height - self.size) / 2.0)
                        cmd = cmd.crop(x, y, self.size, self.size)
                    out, _ = (
                        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .run(capture_stdout=True, quiet=True)
                    )
                    if self.centercrop and isinstance(self.size, int):
                        height, width = self.size, self.size
                    video = np.frombuffer(out, np.uint8).reshape(
                        [-1, height, width, 3])
                    video = th.from_numpy(video.astype('float32'))
                    video = video.permute(0, 3, 1, 2)
                else:
                    video = th.zeros(1)
                cl_anchor = video[int(start_anchor*self.framerate):int(end_anchor*self.framerate)]
                cl_neg = video[int(start_neg*self.framerate):int(end_neg*self.framerate)]

                if not os.path.exists(self.cache_path):
                    print(f"Output directory {self.cache_path} does not exists, creating...")
                    os.makedirs(self.cache_path)
                for tup in self.vid_dic[meta["vid"]]:
                    video_ = video[int(tup[0]*self.framerate):int(tup[1]*self.framerate)].cpu().numpy()
                    np.savez(os.path.join(self.cache_path, "_".join([meta["vid"], str(tup[0]), str(tup[1])])+'.npz'), features=video_)
        
        if len(sl_anchor) != len(cl_anchor): 
            print("anchor", meta['vid'], len(sl_anchor), len(cl_anchor), start_anchor, end_anchor)
            l_min = min(len(sl_anchor), len(cl_anchor))
            sl_anchor, cl_anchor = sl_anchor[:l_min], cl_anchor[:l_min]
        if len(sl_neg) != len(cl_neg): 
            print("neg", meta['vid'], len(sl_neg), len(cl_neg), start_neg, end_neg)
            l_min = min(len(sl_neg), len(cl_neg))
            sl_neg, cl_neg = sl_neg[:l_min], cl_neg[:l_min]
        if len(sl_neg) == 0:
            sl_neg = sl_anchor
            cl_neg = cl_anchor

        return sl_anchor, sl_neg, cl_anchor, cl_neg, txt
