import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil, time, os, requests, random, copy
import re

import torch.optim as optim
import torch.utils.data as Data
import argparse

from utils.dataloader import BasecallDataset
from utils.mae_model import BasecallMAE, MAETConv, MAEnn
from torch.optim.optimizer import Optimizer, required
from utils.model import save_model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-seed', default=2022)
parser.add_argument('-optimizer', default='LARS', type=str)
parser.add_argument('-weight_decay', default=1.0e-6, type=float)
parser.add_argument('-temperature', default=0.5, type=float)
parser.add_argument('-model_path', help="Save model path", required=True)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-train_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_train_signal_wick_small')
parser.add_argument('-val_dir', default='/nas/home/carlos/SACall/SACall-basecaller-master/generate_dataset/2k_val_signal_wick_small')
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-mode', default='normal')
args = parser.parse_args()



def train(args, train_loader, model, criterion, optimizer, epoch, model_name):
    epoch_loss = 0
    for step, (original, unmasked, mask, masked_signal, cheat) in enumerate(train_loader):
        optimizer.zero_grad()
        original = original.cuda().float()
        unmasked = unmasked.cuda().float()
        masked_signal = masked_signal.cuda().float()
        mask = mask.cuda()

        pred, masked_signal_output = model(masked_signal, mask)
        pred_masked = pred.squeeze() * mask
        original_masked = original.squeeze() * mask
        #import pdb; pdb.set_trace()
        vis_loss = criterion(masked_signal_output, masked_signal.squeeze())
        masked_loss = criterion(pred_masked, original_masked)

        loss = (vis_loss)+ masked_loss
        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if  step % 500 == 0:
            plt.plot(pred[0][:100].cpu().detach().numpy(), 'b', label='Prediction')
            plt.plot(masked_signal[0][:100].cpu().detach().numpy(), 'm', label='Masked')
            plt.plot(original[0][:100].cpu().detach().numpy(), 'g', label='Original')
            plt.legend()
            plt.show()
            plt.savefig(f'graphs/maev10/OGvsPred{epoch}_{step}_{model_name}.png')
            plt.clf()


        epoch_loss += loss.item()
    train_loss = epoch_loss/len(train_loader)
    return train_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else 'cpu')
    
    set_seed(args.seed)
  
    model = BasecallMAE().to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    train_dataset = BasecallDataset(signal_dir=args.train_dir)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.MSELoss(reduction='mean')

    for epoch in range(args.epochs):
        loss_epoch = train(args, train_loader, model, criterion, optimizer, epoch, args.model_path)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        chkpoint = {'model_state_dict': model.state_dict()}
        model_name = args.model_path + '.chkpt'
        torch.save(chkpoint, model_name)






if __name__ == "__main__":
    main()
