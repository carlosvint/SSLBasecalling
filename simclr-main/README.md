# simclr

This directory contains the files necessary to pre train a basecaller model using SimCLR as part of my master thesis. 

To pretrain a model using SimCLR and two GPUs run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -model_path model -epochs 100
```
