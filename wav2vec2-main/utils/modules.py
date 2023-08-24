import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()

        self.padding = 1
        self.kernel_size = 3
        self.stride = 2
        self.dilation = 1

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels= out_channels//2, kernel_size=self.kernel_size, 
                        stride=self.stride, padding=self.padding, dilation=self.dilation, bias=False),
            nn.BatchNorm1d(num_features=out_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True))

    def forward(self, signal):
        x = signal.transpose(-1, -2)
        h = self.feature_extractor(x)

        return h.transpose(-1,-2)


