# SSLBasecalling
Code Implementation for the paper Self-Supervised Representation Learning for Basecalling.

This implementation is built over [SACall](https://github.com/huangnengCSU/SACall-basecaller). Hence, for supervised training and finetuning it is necessary to install [ctc-decode](https://github.com/parlance/ctcdecode).

To perform Self-Supervised pre-training, move to the corresponding directory and run the `train_simclr.py` or `train_wav2vec2.py`. 

We also provide the pre-trained models if you wish to fine-tune them with your own data. 

We are working on improving the data loader for it to be more memory efficent.
