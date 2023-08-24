import os
import torch
import torch.utils.data as Data
import numpy as np
import random
from tqdm import tqdm
from .data_augmentation import time_mask, noise_augment, gaussian_noise




class TrainBatchBasecallDatasetMulti(Data.Dataset):
    def __init__(self, signal_dir, label_dir, augment=False):
        file_list = os.listdir(signal_dir)
        label_list = os.listdir(label_dir)
        self.file_count = len(file_list)
        print('file number:', self.file_count)
        self.file_list = file_list
        self.label_list = label_list
        self.signal_path = signal_dir
        self.label_path = label_dir
        self.idx = 0
        self.signal_len = self.get_signal_length()
        self.label_len = self.get_label_length()
        self.augment = augment
        #print('signal_length:', self.signal_len)
        #print('label_length:', self.label_len)
        self.signals_pool, self.labels_pool, self.len_signals =self.getpool() 
        print('read_numer:', self.len_signals)

    def __len__(self):
        return self.len_signals

    def get_signal_length(self):
        signal = np.load(self.signal_path+'/'+self.file_list[0])
        signal_length = signal.shape[1]
        return signal_length

    def get_label_length(self):
        label = np.load(self.label_path+'/'+self.label_list[0])
        label_length = label.shape[1]
        return label_length
  
    def getpool(self):
        signals = []
        labels = []
        counter = 0
        for f in self.file_list:
            signal = np.load(self.signal_path+'/'+f)
            signals.append(signal)

        for l in self.label_list:
            label = np.load(self.label_path+'/'+l)
            labels.append(label)

        signal_array = torch.from_numpy(np.concatenate(signals))
        label_array = torch.from_numpy(np.concatenate(labels))
        
        len_signals = signal_array.shape[0]
        #print(signal_array.shape)
        #print(label_array.shape)

        return signal_array.unsqueeze(2), label_array, len_signals

    def __getitem__(self, item):
        signal = self.signals_pool[item]
        if self.augment == True:
            a1 = time_mask(signal, num_masks=512, T=40, Pt=0.8)
            a2 = noise_augment(a1)
            signal = gaussian_noise(a2)
        else:
            signals = signal

        # label file has the same file name as signal file, but in different directory.
        label = self.labels_pool[item]
        #print(signal.shape)
        return signal, label
  
