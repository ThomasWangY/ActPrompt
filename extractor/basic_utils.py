import os
import json
import torch
import random
import zipfile
import numpy as np
import pickle
from collections import OrderedDict, Counter
import pandas as pd
import shutil
from torch.utils.data.sampler import Sampler
import datetime


# def set_seed(seed, use_cuda=True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if use_cuda:
#         torch.cuda.manual_seed_all(seed)

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def remkdirp(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """ convert '00:01:12' to 72 seconds.
    :hms_time (str): time in comma separated string, e.g. '00:01:12'
    :return (int): time in seconds, e.g. 72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_counter_dist(counter_object, sort_type="none"):
    _sum = sum(counter_object.values())
    dist = {k: float(f"{100 * v / _sum:.2f}") for k, v in counter_object.items()}
    if sort_type == "value":
        dist = OrderedDict(sorted(dist.items(), reverse=True))
    return dist


def get_show_name(vid_name):
    """
    get tvshow name from vid_name
    :param vid_name: video clip name
    :return: tvshow name
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name


def get_abspaths_by_ext(dir_path, ext=(".jpg",)):
    """Get absolute paths to files in dir_path with extensions specified by ext.
    Note this function does work recursively.
    """
    if isinstance(ext, list):
        ext = tuple(ext)
    if isinstance(ext, str):
        ext = tuple([ext, ])
    filepaths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(dir_path)
                 for name in files
                 if name.endswith(tuple(ext))]
    return filepaths


def get_basename_no_ext(path):
    """ '/data/movienet/240p_keyframe_feats/tt7672188.npz' --> 'tt7672188' """
    return os.path.splitext(os.path.split(path)[1])[0]


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


class WarmupStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, step_size, gamma=self.gamma, last_epoch=last_epoch)
    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        # e.g. warmup_steps = 10, case: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21...
        if self.last_epoch == self.warmup_steps or(self.last_epoch % self.step_size != 0 and self.last_epoch > self.warmup_steps):
            return [group['lr'] for group in self.optimizer.param_groups]
        # e.g. warmup_steps = 10, case: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        elif self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * float(self.last_epoch + 1) / float(self.warmup_steps) for group in self.optimizer.param_groups]
        
        
        # e.g. warmup_steps = 10, case: 10, 20, 30, 40...
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * float(self.last_epoch) / (self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** ((self.last_epoch -  self.warmup_steps)// self.step_size) for base_lr in self.base_lrs]



class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):

    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor



class RandomSequenceSampler(Sampler):

    def __init__(self, n_sample, seq_len):
        self.n_sample = n_sample
        self.seq_len = seq_len

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        np.random.shuffle(idx)
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)
    

def show_info(num_training_examples, batch_idx, batch_time, loss, optimizer=None, epoch_i=None, opt=None):
    nb_remain = 0
    nb_remain += num_training_examples - batch_idx - 1
    if opt is not None: nb_remain += (opt.n_epoch - epoch_i - 1) * num_training_examples
    eta_seconds = batch_time.avg * nb_remain
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    info = []
    if opt is not None: 
        info += [f"epoch [{epoch_i + 1}/{opt.n_epoch} - TRAIN]"]
    else:
        info += [f"epoch [EVAL]"]
    info += [f"batch [{batch_idx + 1}/{num_training_examples}]"]
    info += [f"{loss}"]
    if optimizer is not None: info += [f"lr {optimizer.param_groups[0]['lr']:.1e}"]
    info += [f"eta {eta}"]
    print(" ".join(info))
