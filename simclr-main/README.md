# simclr

This directory contains the files necessary to pre train a basecaller model using SimCLR as part of my master thesis. 

To pretrain a model using SimCLR and two GPUs run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_simclr.py -model_path model -epochs 100
```
We provide our pre-trained models in the pretrained directory.


To fine-tune a model use the following command:

```
python finetune_sacall_encoder.py -as training_signals_directory -al training_labels_directory -es validation_signals_directory -el training_signals_directory -model_dir model_name -pretrained pretrained_model

```

The fine-tuned model can be used to basecall by:

```
CUDA_VISIBLE_DEVICES=0 python call_ctc_encoder.py -model finetuned_model -records_dir signal_to_be_basecalled -output new_basecall_directory
