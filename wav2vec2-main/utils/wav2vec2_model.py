import sys
sys.path.append('..')

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import compute_mask_indices, index_put, buffered_arange
from utils.modules import FeatureExtractor
from ctc.ctc_encoder import Encoder
import generate_dataset.constants as constants
from utils.gumbel_vector_quantizer import GumbelVectorQuantizer


class BasecallWav2Vec2(nn.Module):
    def __init__(self, quantizer=False):
        super(BasecallWav2Vec2, self).__init__()

        self.final_dim = 256
        self.latent_dim = 0
        self.embed = 256

        self.feature_extractor = FeatureExtractor(1, self.embed)
        #self.post_extract_proj = nn.Linear()

        self.mask_prob = 0.65
        self.mask_selection = 'static'
        self.mask_other = 0
        self.mask_length = 10
        self.no_mask_overlap = False
        self.mask_min_space = 1
        self.require_same_masks = True
        self.mask_dropout = 0
        self.mask_emb = nn.Parameter(torch.FloatTensor(self.embed).uniform_())

        self.dropout_input = nn.Dropout(0.1)
        self.dropout_features = nn.Dropout(0.1)

        self.quantizer = None
        self.input_quantizer = None
        self.quantize_targets = quantizer
        self.latent_vars = 320
        self.latent_temp = (2, 0.5, 0.999995) #"can be tuple of 3 values (start, end, decay)"
        self.latent_groups = 2
        self.quantizer_depth = 1
        self.quantizer_factor = 3


        self.n_negatives = 50
        self.cross_sample_negatives = 0
        self.codebook_negatives = 0
        self.negatives_from_everywhere = False

        self.logit_temp = 0.1

        final_dim = self.final_dim if self.final_dim > 0 else opt.model_dim

        if self.quantize_targets:
            print('Training with quantizer :)')
            vq_dim = self.latent_dim if self.latent_dim> 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=self.latent_vars,
                temp=self.latent_temp,
                groups=self.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=self.quantizer_depth,
                weight_proj_factor=self.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        self.d_model = 256
        self.d_ff = 1024
        self.n_head = 8
        self.n_layers = 6
        self.dropout = 0.1

        self.encoder = Encoder(d_model=self.embed,
                                   d_ff=self.d_ff,
                                   n_head=self.n_head,
                                   num_encoder_layers=self.n_layers,
                                   dropout=self.dropout)

        #self.layer_norm = nn.LayerNorm(self.embed)

        self.final_proj = nn.Linear(self.embed, final_dim)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.require_same_masks,
                    mask_dropout=self.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs
    
    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
        #print(targets.shape)
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)
        #print(logits.shape)

        if neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits


    def forward(self, 
        signal, 
        padding_mask=None, 
        mask=True, 
        features_only=False, 
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,):

        features = self.feature_extractor(signal)

        features_pen = features.transpose(1,2)
        #features = self.layer_norm(features)
        unmasked_features = features.clone()

        features = self.dropout_input(features)
        #print(features.shape)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices = mask_indices,
                mask_channel_indices = mask_channel_indices
            )
            if mask_indices is not None:
                
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1))
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
        x, layer_results = self.encoder(signal, signal_lengths, x)
        #print(x.shape)

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y) 
            #print(x.shape)               
            #print(y.shape)
            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        x = self.final_proj(x)
        #print(x.shape)
        x = self.compute_preds(x, y, negs)

        return x



        


