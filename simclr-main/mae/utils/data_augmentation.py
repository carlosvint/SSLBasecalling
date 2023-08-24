import torch
import numpy as np
import random

def real_random_masking(signal, p = 0.70):
  clone = signal.squeeze()
  
  mask = torch.rand(clone.shape[0])
  ones = torch.ones(clone.shape[0])
  clone = clone * (mask>p)
  loss_mask = ones * (mask<p)
  
  return clone.unsqueeze(1), loss_mask, clone.unsqueeze(1)

def random_masking(signal, p=0.7, max_mask_length=20):
  clone = np.copy(signal)
  L = clone.shape[0]
  num_masks = int(np.ceil(L * p/10))
  mask = np.zeros(L)
  mask_ones = np.ones(L)
  indices = np.arange(0, L-max_mask_length, 1)
  total_mask_indices = np.array([])

  for i in range(num_masks):

    random_index = np.random.choice(indices, 1)
    random_index = int(random_index[0])
    mask_length =  int(abs(np.ceil(np.random.normal(10,8))))
    if mask_length > max_mask_length:
      mask_length = max_mask_length
    mask_indices_forward = np.arange(random_index,(random_index+mask_length), 1)

    mask_indices_backward = np.arange((random_index-8+1), (random_index), 1)

    mask[random_index:random_index+mask_length] = 1
    mask_ones[random_index:random_index+mask_length] = 0
    indices = np.setdiff1d(indices, mask_indices_forward)
    indices = np.setdiff1d(indices, mask_indices_backward)
    total_mask_indices = np.concatenate((total_mask_indices, mask_indices_forward), axis=0)

  clone = np.delete(clone, total_mask_indices.astype(int))

  mask_signal = signal.squeeze() * mask_ones

  return (mask_signal==0).sum(), torch.from_numpy(mask), mask_signal.unsqueeze(1)


def basecall_mask(x, w=3, epsilon=0.25, pmasking=0.70, padding_length=2048):
  x = x.squeeze(1).numpy()

  kernel = np.array([-1,0,1])
  clone = np.copy(x)
  ma = np.convolve(x, np.ones(w), 'same') / w
  ca = np.convolve(x, kernel, 'same') / kernel.shape
  
  ma = ca + ma

  index = []
  lens = []
  for i in range(ma.shape[0]-1):
    if np.absolute(ma[i])-np.absolute(ma[i+1])>epsilon:
      index.append(i)
  
  counter = 0
  for item in index:
    counter+=1


  start = 0    
  for i in index:
      end = i
      length = end - start
      start = i
      lens.append(length)

  index_array = np.array(index[:-1])
  len_array = np.array(lens[1:])

  mask_events = np.ceil(index_array.shape[0] * pmasking)

  #while True:
  mask_index = np.random.choice(index_array, mask_events.astype(int), replace=False)

  start = 0
  mask = np.zeros(x.shape[0])    

  for i in mask_index:
    j = np.where(index_array==i)[0]     
    start = index_array[j]
    length = len_array[j]

    mask[start.item():start.item() + length.item()] = 1
    clone[start.item():start.item() + length.item()] = 0


  mask_out = np.delete(clone, np.where(mask==1)[0].astype(int))

  #  if (560) < mask_out.shape[0] < (640):
  #    break
  
  enc_input = np.pad(mask_out, (0, padding_length-mask_out.shape[0]), mode='constant', constant_values=-10)

  return  torch.from_numpy(enc_input).unsqueeze(1), torch.from_numpy(mask), torch.from_numpy(clone).unsqueeze(1)


def cheat_masking(x, cheats, pmasking=0.15, padding_length=2048):
  x = x.squeeze(1).numpy()
  cheats = cheats.numpy()
  
  clone = np.copy(x)
  index = cheats[cheats != -10]
  
  lens = [] 
  counter = 0
  for item in index:
    counter+=1


  start = 0    
  for i in index:
      end = i
      length = end - start
      start = i
      lens.append(length)

  index_array = np.array(index[:-1])
  len_array = np.array(lens[:-1])

  mask_events = np.ceil(index_array.shape[0] * pmasking)

  #while True:
  mask_index = np.random.choice(index_array, mask_events.astype(int), replace=False)

  start = 0
  mask = np.zeros(x.shape[0])    

  for i in mask_index:
    j = np.where(index_array==i)[0]     
    end = index_array[j]
    length = len_array[j]
    
    mask[end.item() - length.item():end.item()] = 1
    clone[end.item() - length.item():end.item()] = 0
    #clone[end.item() - length.item():end.item()] = clone[end.item() - length.item():end.item()].mean()

  mask_out = np.delete(clone, np.where(mask==1)[0].astype(int))
  
  enc_input = np.pad(mask_out, (0, padding_length-mask_out.shape[0]), mode='constant', constant_values=-10)
  #import pdb; pdb.set_trace()
  return  torch.from_numpy(enc_input).unsqueeze(1), torch.from_numpy(mask), torch.from_numpy(clone).unsqueeze(1)
