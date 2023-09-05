import wandb
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from utils.wav2vec2_model import BasecallWav2Vec2

from utils.utils import save_model, get_logits, get_targets

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
parser.add_argument('-val_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_val_signal_wick_small')
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-quantizer_size', default=320, type=int)
args = parser.parse_args()


def train(args, train_loader, model, criterion, optimizer, gpu):
    loss_epoch = 0
    for step, (x) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.cuda(non_blocking=True).float()
        B, T, C = x.shape
        # positive pair, with encoding
        net_output = model(x)
        #print(net_output.shape)
        logits = get_logits(net_output).float()
        #print(logits.shape)
        targets = get_targets(net_output)
        #print(targets.shape)

        
        loss = criterion(logits, targets)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if gpu == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()/T}")
            wandb.log({"training loss": loss.item()/T})

        loss_epoch += loss.item()/T
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

    if gpu == 0:
        wandb.init(project="wav2vec2", entity="carlos5")
        wandb.run.name = args.model_path
        wandb.config.update(args)

    train_dataset = BasecallDataset(signal_dir=args.train_dir)

    if args.nodes > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=ngpus_per_node, rank=gpu, shuffle=True
            )
    else:
            train_sampler = None

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, drop_last=True)    

    model = BasecallWav2Vec2(quantizer=True, quantizer_size=args.quantizer_size).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(reduction="sum")

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

        if gpu == 0:
            save_model(args, model, optimizer)



if __name__ == "__main__":
    main()
