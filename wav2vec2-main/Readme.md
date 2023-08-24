# wav2vec2

This directory contains all the files necessary to pretrain a basecaller model using the wav2vec2 framework as part of my master thesis.

To pre-train using wav2vec2 using four GPUs run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -model_path model -epochs 100
```

The ctc and transformer directories contain the modules for SACall, the basecaller model, and the utils module contains the wav2vec2.0 model as well as the helper functions. 
