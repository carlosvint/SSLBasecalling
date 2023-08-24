import numpy as np
import torch
import torchvision
import argparse
import shutil, time, os, requests, random, copy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils.dataloader import BasecallDataset
from utils.modules import SimCLRProjectionHead, w2v2ProjectionHead, FeatureExtractor, PreModel
from utils.SimCLRloss import SimCLR_Loss_multi
from utils.optimizer import LARS
from utils.model import load_optimizer, save_model
from sync_batchnorm import convert_model

from torch.optim.optimizer import Optimizer, required
import re

parser = argparse.ArgumentParser()

parser.add_argument('-seed', default=2022)
parser.add_argument('-optimizer', default='LARS', type=str)
parser.add_argument('-weight_decay', default=1.0e-6, type=float)
parser.add_argument('-temperature', default=0.5, type=float)
parser.add_argument('-port', default='1235', type=str,
                    help='port for distributed training')
parser.add_argument('-model_path', help="Save model path", required=True)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-train_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_train_signal_wick_small')
parser.add_argument('-epochs', default=10, type=int)
args = parser.parse_args()


def train(args, train_loader, model, criterion, optimizer, gpu):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True).float()
        x_j = x_j.cuda(non_blocking=True).float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if gpu == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")


        loss_epoch += loss.item()
    return loss_epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node 
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))


def main_worker(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + args.port,
        world_size= ngpus_per_node,
        rank=gpu)

    cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda else 'cpu')
    args.nodes = ngpus_per_node
    torch.cuda.set_device(gpu)

    set_seed(args.seed)

    train_dataset = BasecallDataset(signal_dir=args.train_dir)

    if args.nodes > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=ngpus_per_node, rank=gpu, shuffle=True
            )
    else:
            train_sampler = None

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, drop_last=True)    

    model = PreModel(feature_extractor='transformer').to(args.device)

    optimizer, scheduler = load_optimizer(args, model)
    criterion = SimCLR_Loss_multi(args.batch_size, args.temperature, ngpus_per_node)

    if args.nodes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)


    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, gpu)

        if gpu == 0 and scheduler:
            scheduler.step()

        if gpu == 0:
            save_model(args, model, optimizer)



if __name__ == "__main__":
    main()

    
