# -*- coding: utf-8 -*-
import wandb
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from ctc.ctc_encoder import Encoder
import generate_dataset.constants as constants
from ctc.opts import add_decoder_args
from ctc.ctc_decoder import BeamCTCDecoder, GreedyDecoder
from ctc.ScheduledOptim import ScheduledOptim
from generate_dataset.train_dataloader_array import TrainBatchBasecallDatasetMulti
from generate_dataset.train_dataloader import TrainBatchBasecallDataset, TrainBatchProvider
from utils.modules import FeatureExtractor, PreModel
import time
import torch.distributed as dist
from tqdm import tqdm
import random


parser = argparse.ArgumentParser()
parser.add_argument('-save_model', help="Save model path", required=True)
parser.add_argument(
    '-from_model', help="load from exist model", default=None)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-train_signal_path', '-as', required=True)
parser.add_argument('-train_label_path', '-al', required=True)
parser.add_argument('-test_signal_path', '-es', required=True)
parser.add_argument('-test_label_path', '-el', required=True)
parser.add_argument('-learning_rate', '-lr', default=1e-4, type=float)
parser.add_argument('-weight_decay', '-wd', default=0.01, type=float)
parser.add_argument('-warmup_steps', default=500, type=int)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-d_model', type=int, default=256)
parser.add_argument('-d_ff', type=int, default=1024)
parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-label_vocab_size', type=int,
                    default=6)  # {0,1,2,3,4,5}
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-show_steps', type=int, default=500)
parser.add_argument('-cuda', default=True)
parser.add_argument('-port', default='1234', type=str,
                    help='port for distributed training')
parser.add_argument('-pretrained', help="Pretrained model", required=True)
parser.add_argument('-seed', default=2, type=int)
opt = parser.parse_args()


def train(model, optimizer, device, opt, gpu, ngpus_per_node):
    #print(model)
    train_dataset = TrainBatchBasecallDatasetMulti(
        signal_dir=opt.train_signal_path, label_dir=opt.train_label_path, augment=False)
    if gpu ==0:
        valid_dataset = TrainBatchBasecallDataset(
            signal_dir=opt.test_signal_path, label_dir=opt.test_label_path)
    list_charcter_error = []
    start = time.time()
    show_shape = True
    per_device_batch_size = opt.batch_size // ngpus_per_node
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=False, num_workers=4, sampler=train_sampler)
    for id in range(opt.epoch):
        train_sampler.set_epoch(id)
        log_file = open('log_file_' + opt.save_model + '.txt', "a+")
        model.train()
        total_loss = []
        batch_step = 0
        target_decoder = GreedyDecoder(
            '-ATCG ', blank_index=0)  # P表示padding，-表示blank
        decoder = BeamCTCDecoder(
            '-ATCG ', cutoff_top_n=6, beam_width=3, blank_index=0)
        for i, (signal, label) in enumerate(train_dataloader):
            if signal is not None and label is not None:
                batch_step += 1
                if show_shape:
                    print('signal shape:', signal.size())
                    print('label shape:', label.size())
                    show_shape = False
                signal = signal.type(torch.FloatTensor).cuda(dist.get_rank())
                label = label.type(torch.LongTensor).cuda(dist.get_rank())
                # forward
                optimizer.zero_grad()
                signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                enc_output, enc_output_lengths = model(
                    signal, signal_lengths)  # (N,L,C), [32, 256, 6]
                # print(enc_output[0])
                # print(enc_output_lengths)

                log_probs = enc_output.transpose(1, 0).log_softmax(
                    dim=-1)  # (L,N,C), [256,32,6]
                assert signal.size(2) == 1
                # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                target_lengths = label.ne(constants.PAD).sum(1)

                concat_label = torch.flatten(label)
                concat_label = concat_label[concat_label.lt(constants.PAD)]

                loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths,
                                  target_lengths, blank=0, reduction='sum').cuda(gpu)
                loss.backward()
                optimizer.step_and_update_lr()
                rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())
                total_loss.append(rd_train_loss.item() / signal.size(0))
                if batch_step % opt.show_steps == 0 and gpu == 0:
                    log_file.write('training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}\n'.format(
                        epoch=id,
                        step=batch_step,
                        loss=np.mean(total_loss),
                        t=(time.time() - start) / 60))
                    print('training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                        epoch=id,
                        step=batch_step,
                        loss=np.mean(total_loss),
                        t=(time.time() - start) / 60))
                    start = time.time()
                    wandb.log({"training loss": np.mean(total_loss)})
            else:
                log_file.write('training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}\n'.format(
                        epoch=id,
                        step=batch_step,
                        loss=np.mean(total_loss),
                        t=(time.time() - start) / 60))
                print('a training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                    epoch=id,
                    step=batch_step,
                    loss=np.mean(total_loss),
                    t=(time.time() - start) / 60))

        if gpu == 0:
            valid_provider = TrainBatchProvider(valid_dataset, per_device_batch_size, shuffle=False)
            start = time.time()
            model.eval()
            total_loss = []
            with torch.no_grad():
                total_wer, total_cer, num_tokens, num_chars = 0, 0, 0, 1
                while True:
                    batch = valid_provider.next()
                    signal, label = batch
                    if signal is not None and label is not None:
                        signal = signal.type(torch.FloatTensor).cuda(dist.get_rank())
                        label = label.type(torch.LongTensor).cuda(dist.get_rank())

                        signal_lengths = signal.squeeze(
                            2).ne(constants.SIG_PAD).sum(1)
                        enc_output, enc_output_lengths = model(
                            signal, signal_lengths)

                        log_probs = enc_output.transpose(
                            1, 0).log_softmax(2)  # (L,N,C)

                        assert signal.size(2) == 1
                        # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                        target_lengths = label.ne(constants.PAD).sum(1)
                        concat_label = torch.flatten(label)
                        concat_label = concat_label[concat_label.lt(constants.PAD)]

                        loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths, target_lengths, blank=0,
                                        reduction='sum').cuda(dist.get_rank())
                        total_loss.append(loss.item() / signal.size(0))
                        log_probs = log_probs.transpose(1, 0)  # (N,L,C)
                        # print(log_probs.size())
                        # print(input_lengths)
                        target_strings = target_decoder.convert_to_strings(
                            label, target_lengths)
                        decoded_output, _ = decoder.decode(
                            log_probs, enc_output_lengths)
                        # decoded_output, _ = target_decoder.decode(
                        #     log_probs, enc_output_lengths)
                        for x in range(len(label)):
                            transcript, reference = decoded_output[x][0], target_strings[x][0]
                            cer_inst = decoder.cer(transcript, reference)
                            total_cer += cer_inst
                            num_chars += len(reference)
                    else:
                        break
                cer = float(total_cer) / num_chars
                list_charcter_error.append(cer)
                log_file.write('validate: epoch {epoch:d}, loss {loss:.6f}, charcter error {cer:.3f} time: {time:.3f}\n'.format(
                        epoch=id,
                        loss=np.mean(total_loss),
                        cer=cer * 100,
                        time=(time.time() - start) / 60))
                print(
                    'validate: epoch {epoch:d}, loss {loss:.6f}, charcter error {cer:.3f} time: {time:.3f}'.format(
                        epoch=id,
                        loss=np.mean(total_loss),
                        cer=cer * 100,
                        time=(time.time() - start) / 60))
                start = time.time()
                wandb.log({"val loss": np.mean(total_loss), "character error": cer * 100})

                if cer <= min(list_charcter_error):
                    model_state_dic = model.module.state_dict()
                    model_name = opt.save_model + '.chkpt'
                    checkpoint = {'model': model_state_dic,
                                'settings': opt,
                                }
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node 
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))


def main_worker(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + opt.port,
        world_size= ngpus_per_node,
        rank=gpu)

    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else 'cpu')
    torch.cuda.set_device(gpu)
    set_seed(opt.seed)
    #print(device)

    if gpu == 0:
        wandb.init(project="SACall-hyperparameters-finetuning", entity="carlos5")
        wandb.config.update(opt)

    pretrained_model = torch.load(opt.pretrained, map_location=f'cuda:{gpu}')

    class Model(nn.Module):
        def __init__(self, checkpoint, label_vocab_size):
            super(Model, self).__init__()
            self.checkpoint = checkpoint
            self.feature_extractor = PreModel(feature_extractor = 'transformer', projection_head='basecall')
            self.feature_extractor.load_state_dict(checkpoint['model'])
            
            for p in self.feature_extractor.parameters():
                p.requires_grad = True
                
            for p in self.feature_extractor.projector.parameters():
                p.requires_grad = False

            self.final_proj = nn.Linear(256, label_vocab_size)

        def forward(self, signal, signal_lengths):
            """
            :param signal: a tensor shape of [batch, length, 1]
            :param signal_lengths:  a tensor shape of [batch,]
            :return:
            """
            enc_output, enc_output_lengths = self.feature_extractor.pretrained(signal)
            out = self.final_proj(enc_output)  # (N,L,C), [32, 256, 6]
            return out, enc_output_lengths

    if opt.from_model is None:
        model = Model(checkpoint=pretrained_model,
                      label_vocab_size=opt.label_vocab_size).to(device)

        for p in model.final_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)
    else:
        checkpoint = torch.load(opt.from_model)
        model_opt = checkpoint['settings']
        # use trained model setting cover current setting
        # opt = model_opt
        model = Model(d_model=model_opt.d_model,
                      d_ff=model_opt.d_ff,
                      n_head=model_opt.n_head,
                      n_layers=model_opt.n_layers,
                      label_vocab_size=model_opt.label_vocab_size,
                      dropout=model_opt.dropout).to(device)
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)

    train(model=model,
          optimizer=optim_schedule,
          device=device, opt=opt, gpu=gpu, ngpus_per_node=ngpus_per_node)


if __name__ == "__main__":
    main()
