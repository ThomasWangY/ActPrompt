import time
import torch
import torch.nn.functional as F

from config import BaseOptions, setup_model
from dataset import DatasetMR, start_end_collate_mr, get_loader
from basic_utils import set_seed, count_parameters, show_info
from metrics import AverageMeter, compute_accuracy, MetricMeter
from extract_utils import extract


def train_epoch(model, train_loader, optimizer, opt, epoch_i):
    model.train()

    loss_train = MetricMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    num_training_examples = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):    
        video, image, text = (batch[0], batch[1]), (batch[2], batch[3]), batch[4]
        logits_ce, logits_tri_video, logits_tri_verb, loss_con = model(video, image, text)
        
        ce_label = torch.arange(len(video[0])).cuda()
        tri_label = torch.tensor([0]*len(video[0])).cuda()

        loss_ce = F.cross_entropy(logits_ce, ce_label)
        loss_tri = (F.cross_entropy(logits_tri_video, tri_label) + F.cross_entropy(logits_tri_verb, tri_label))*opt.coef_1
        loss_con = loss_con*opt.coef_2
        loss = loss_ce + loss_tri + loss_con

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_summary = {
            "acc_vt": compute_accuracy(logits_ce, ce_label)[0].item(),
            "acc_tri": compute_accuracy(logits_tri_video, tri_label)[0].item(),
            "loss": loss.item(),
            "loss_contrastive": loss_ce.item(),
            "loss_tri-ranking": loss_tri.item(),
            "loss_consistency": loss_con.item()
        }
        loss_train.update(loss_summary)
        batch_time.update(time.time() - end)

        show_info(num_training_examples, batch_idx, 
                  batch_time, loss_train, optimizer, epoch_i, opt)

        end = time.time()


def train(model, optimizer, lr_scheduler, opt):
    if opt.start_epoch is None:
        start_epoch = -1
    else:
        start_epoch = opt.start_epoch

    for epoch_i in range(start_epoch, opt.n_epoch):
        train_dataset = DatasetMR(opt, data_split=epoch_i)
        train_loader = get_loader(train_dataset, opt)
        s_time = time.time()
        train_epoch(model, train_loader, optimizer, opt, epoch_i)
        print(f"Train Epoch Time:{round(time.time()-s_time)}s")
        lr_scheduler.step()


def main():
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    model, optimizer, lr_scheduler = setup_model(opt)
    count_parameters(model)
    train(model, optimizer, lr_scheduler, opt)
    extract(model, opt)


if __name__ == '__main__':
    main()
