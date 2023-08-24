import os
import torch
import torch.utils.data as Data
import numpy as np
import random
from tqdm import tqdm
from .data_augmentation import time_mask, noise_augment, gaussian_noise



class BasecallDataset(Data.Dataset):
    def __init__(self, signal_dir):
        file_list = os.listdir(signal_dir)

        self.file_count = len(file_list)
        print('file number:', self.file_count)
        self.file_list = file_list

        self.signal_path = signal_dir

        self.idx = 0
        self.signal_len = self.get_signal_length()

        #print('signal_length:', self.signal_len)
        #print('label_length:', self.label_len)
        self.signals_pool,  self.len_signals =self.getpool() 
        print('read_numer:', self.len_signals)

    def __len__(self):
        return self.len_signals

    def get_signal_length(self):
        signal = np.load(self.signal_path+'/'+self.file_list[0])
        signal_length = signal.shape[1]
        return signal_length
  
    def getpool(self):
        signals = []
        counter = 0
        for f in self.file_list:
            signal = np.load(self.signal_path+'/'+f)
            signals.append(signal)

        signal_array = torch.from_numpy(np.concatenate(signals))
        
        len_signals = signal_array.shape[0]

        return signal_array.unsqueeze(2), len_signals

    def __getitem__(self, item):
        signal = self.signals_pool[item]
        
        #print(signal.shape)
        return signal
  
