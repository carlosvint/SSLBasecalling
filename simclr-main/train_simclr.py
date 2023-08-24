import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils.dataloader import BasecallDataset
from utils.modules import SimCLRProjectionHead, w2v2ProjectionHead, FeatureExtractor, PreModel
from utils.SimCLRloss import SimCLR_Loss
from utils.optimizer import LARS

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from torch.optim.optimizer import Optimizer, required
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-save_model', help="Save model path", required=True)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-train_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_train_signal_wick_small')
parser.add_argument('-val_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_val_signal_wick_small')
parser.add_argument('-epochs', default=10, type=int)
opt = parser.parse_args()

def save_model(model, optimizer, scheduler, epoch):
    model_name = opt.save_model + '.chkpt'

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'epoch': epoch}, model_name)
    print('    - [Info] The checkpoint file has been updated.')

def train(model, opt, optimizer, warmupscheduler, mainscheduler, criterion):
    train_signal_dir = opt.train_dir
    val_signal_dir = opt.val_dir


    train_dataset = BasecallDataset(signal_dir=train_signal_dir)
    val_dataset = BasecallDataset(signal_dir=val_signal_dir)
    train_loader = Data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    nr = 0
    current_epoch = 0
    epochs = opt.epochs
    tr_loss = []
    val_loss = []

    for epoch in range(opt.epochs):
            
        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()

        model.train()
        tr_loss_epoch = 0
        
        for step, (x_i, x_j) in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = x_i.to('cuda:0').float()
            x_j = x_j.to('cuda:0').float()

            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()
            
            if nr == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

            tr_loss_epoch += loss.item()

        if nr == 0 and epoch < 10:
            warmupscheduler.step()
        if nr == 0 and epoch >= 10:
            mainscheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]

        if nr == 0 and (epoch+1) % 50 == 0:
            save_model(model, optimizer, mainscheduler, current_epoch,"SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")

        model.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            for step, (x_i, x_j) in enumerate(valid_loader):
            
                x_i = x_i.to('cuda:0').float()
                x_j = x_j.to('cuda:0').float()

                # positive pair, with encoding
                z_i = model(x_i)
                z_j = model(x_j)

                loss = criterion(z_i, z_j)

                if nr == 0 and step % 50 == 0:
                    print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(),5)}")

                val_loss_epoch += loss.item()

        if nr == 0:
            tr_loss.append(tr_loss_epoch / len(train_loader))
            val_loss.append(val_loss_epoch / len(valid_loader))
            print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
            print(f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(valid_loader)}\t lr: {round(lr, 5)}")
            current_epoch += 1


        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")

    save_model(model, optimizer, mainscheduler, epoch)
 

def main():

    model = PreModel(feature_extractor='residual').to('cuda:0')

    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )
    # "decay the learning rate with the cosine decay schedule without restarts"
    #SCHEDULER OR LINEAR EWARMUP
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

    #SCHEDULER FOR COSINE DECAY
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

    #LOSS FUNCTION
    criterion = SimCLR_Loss(batch_size = opt.batch_size, temperature = 0.5)

    train(model, opt, optimizer, warmupscheduler, mainscheduler, criterion)


if __name__ == "__main__":
    main()
