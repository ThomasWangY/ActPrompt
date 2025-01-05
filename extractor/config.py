import os
import time
import torch
import argparse
from basic_utils import remkdirp, load_json, save_json, make_zipfile, dict_to_markdown, WarmupStepLR
from model import CLIPExtractor
from lr_scheduler import build_lr_scheduler


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        # * Running configs
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--seed", type=int, default=2024, help="random seed")

        # * DDP
        parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # * Training configs
        parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=64, help="mini-batch size")
        parser.add_argument("--n_epoch", type=int, default=10, help="number of epochs to run")
        parser.add_argument("--max_len", type=int, default=10, help="number of epochs to run")
        parser.add_argument("--min_len", type=int, default=4, help="number of epochs to run")
        parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=10, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--lr_gamma", type=float, default=0.1, help="lr reduces the gamma times after the `drop' epoch")
        parser.add_argument("--lr_warmup", type=float, default=1, help="linear warmup scheme")
        parser.add_argument("--wd", type=float, default=5e-2, help="weight decay")
        parser.add_argument("--margin", type=float, default=0.1, help="margin")
        parser.add_argument("--data_ratio", type=float, default=1,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")

        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--start_epoch", type=int, default=0,
                            help="if None, will be set automatically when using --resume_all")
        
        parser.add_argument('--ex_bsz', type=int, default=1, help='batch size')
        parser.add_argument('--clip_len', type=float, default=1, help='decoding length of clip (in seconds)')
        parser.add_argument('--overwrite', action='store_true', help='allow overwrite output files')
        parser.add_argument('--half_precision', type=int, default=1, help='output half precision float')
        parser.add_argument('--num_decoding_thread', type=int, default=4, help='Num parallel thread for video decoding')
        parser.add_argument('--model_version', type=str, default="ViT-B/32", choices=["ViT-B/32", "RN50x4"],
                            help='Num parallel thread for video decoding')
        
        parser.add_argument('--dset_name', type=str, default=None)
        parser.add_argument('--video_path', type=str, default=None)
        parser.add_argument('--train_path', type=str, default=None)
        parser.add_argument('--eval_path', type=str, default=None)
        parser.add_argument('--test_path', type=str, default=None)
        parser.add_argument('--cache_path', type=str, default=None)
        parser.add_argument('--dset_path', type=str, default=None)
        
        parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler type')
        parser.add_argument('--warmup_type', type=str, default='constant', help='warmup type')
        parser.add_argument('--warmup_lr', type=float, default=1e-5, help='warmup lr')
        parser.add_argument('--size', type=int, default=3, help='coef')
        parser.add_argument('--coef_1', type=float, default=5, help='')
        parser.add_argument('--coef_2', type=float, default=100, help='')

        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        print(dict_to_markdown(vars(opt), max_str_len=120))

    def initialize_from_json(self, json_file):
        import json
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            for key, value in json_data.items():
                self.parser.add_argument(f'--{key}', default=value)

    def parse(self, args=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        
        if args is not None:
            args_dict = vars(args)
            opt_dict = vars(opt)
            for key, value in args_dict.items():
                opt_dict[key] = value
            opt = argparse.Namespace(**opt_dict)    
            opt.model_dir = os.path.dirname(opt.resume)
            torch.cuda.set_device(opt.gpu_id)
            

        if int(opt.local_rank) in [0, -1]:
            self.display_save(opt)

        if int(opt.local_rank) in [-1]:
            torch.cuda.set_device(opt.gpu_id)
        opt.pin_memory = not opt.no_pin_memory

        if opt.local_rank == -1:
            torch.cuda.set_device(opt.gpu_id)

        self.opt = opt
        return opt


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""

    model = CLIPExtractor(opt)
    
    for name, param in model.named_parameters():
        if "ctx" not in name:
            param.requires_grad_(False)

    model.to(opt.gpu_id)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = build_lr_scheduler(optimizer, opt)

    return model, optimizer, lr_scheduler