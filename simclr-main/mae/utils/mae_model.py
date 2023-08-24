import torch
import torch.nn as nn
from ctc.ctc_encoder import EncoderCNN, Encoder, MAEDecoder
import generate_dataset.constants as constants

class BasecallMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self):
        super(BasecallMAE, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.embed = 256
        self.mask_token = nn.Parameter(torch.rand(1, 1, 64))
        self.d_model = 256
        self.d_ff = 1024
        self.n_head = 8
        self.n_layers = 6
        self.dropout = 0.1

        self.encoder = EncoderCNN(d_model=self.embed,
                                   d_ff=self.d_ff,
                                   n_head=self.n_head,
                                   num_encoder_layers=self.n_layers,
                                   dropout=self.dropout)
 
        # --------------------------------------------------------------------------
        
        #self.pooling = nn.AvgPool1d(256)
        
        #self.extrapolator = nn.Linear(in_features=512, out_features=2048)
        #self.extrapolator_features = nn.Linear(in_features=256, out_features=1)
        # --------------------------------------------------------------------------

        self.decoder = MAEDecoder(d_model=64,
                                    d_ff=self.d_ff,
                                    n_head=self.n_head,
                                    num_encoder_layers=int(self.n_layers//2),
                                    dropout=self.dropout)

        
        self.upsample = nn.Sequential(nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, dilation=1, padding=1), 
                         nn.BatchNorm1d(128), nn.ReLU(),
                         nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, dilation=1, padding=1))

        self.decoder_pred = nn.Linear(64, 1, bias=True) 

        self.masked_signal_pred = nn.Linear(64, 1, bias=True)

        # --------------------------------------------------------------------------

    def forward(self, signal, mask):
        B = signal.shape[0]
        signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
        
        enc_output, enc_output_lengths = self.encoder(
            signal, signal_lengths)

        extrapolate_output = self.upsample(enc_output.transpose(-2, -1)).transpose(-2, -1)
        #import pdb; pdb.set_trace()
        masked_signal_pred = self.masked_signal_pred(extrapolate_output)

        placeholder = torch.zeros(signal.shape[0], 2048, 64).cuda()
        masks = (mask == 0)
        mask_tokens = placeholder.where((mask == 0).unsqueeze(2).repeat(1,1,64), self.mask_token)
        dec_input = mask_tokens.masked_scatter(masks.unsqueeze(2).repeat(1,1,64), extrapolate_output)

        dec_output, _ = self.decoder(dec_input)
        output = self.decoder_pred(dec_output)
        return output.squeeze(), masked_signal_pred.squeeze()



class MAETConv(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self):
        super(MAETConv, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.embed = 256
        self.mask_token = mask_token = nn.Parameter(torch.zeros(1, 1, self.embed))
        self.d_model = 256
        self.d_ff = 1024
        self.n_head = 8
        self.n_layers = 6
        self.dropout = 0.1

        self.encoder = EncoderCNN(d_model=self.embed,
                                   d_ff=self.d_ff,
                                   n_head=self.n_head,
                                   num_encoder_layers=self.n_layers,
                                   dropout=self.dropout)

        # --------------------------------------------------------------------------
        
        #self.pooling = nn.AvgPool1d(256)
        
        #self.extrapolator = nn.Linear(in_features=512, out_features=2048)
        #self.extrapolator_features = nn.Linear(in_features=256, out_features=1)
        # --------------------------------------------------------------------------

        #self.decoder = MAEDecoder(d_model=self.embed,
        #                           d_ff=self.d_ff,
        #                           n_head=self.n_head,
        #                           num_encoder_layers=self.n_layers,
        #                           dropout=self.dropout)

        self.upsample = nn.Sequential(nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, dilation=1, padding=1), 
                        nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=4, stride=2, dilation=1, padding=1))

        # --------------------------------------------------------------------------


    def forward(self, signal, mask):
        signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
        enc_output, enc_output_lengths = self.encoder(
            signal, signal_lengths)
        
        dec_pred = self.upsample(enc_output.transpose(-1,-2))
        return dec_pred.transpose(-1,-2).squeeze()


class MAEnn(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self):
        super(MAEnn, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.embed = 256
        self.mask_token = mask_token = nn.Parameter(torch.zeros(1, 1, self.embed))
        self.d_model = 256
        self.d_ff = 1024
        self.n_head = 8
        self.n_layers = 6
        self.dropout = 0.1

        self.encoder = EncoderCNN(d_model=self.embed,
                                   d_ff=self.d_ff,
                                   n_head=self.n_head,
                                   num_encoder_layers=self.n_layers,
                                   dropout=self.dropout)

        # --------------------------------------------------------------------------
        
        self.pooling = nn.AvgPool1d(256)
        
        # --------------------------------------------------------------------------


    def forward(self, signal, mask):
        signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
        enc_output, enc_output_lengths = self.encoder(
            signal, signal_lengths)
        
        pool_out = self.pooling(enc_output)
        #import pdb; pdb.set_trace()
        dec_pred = torch.nn.functional.interpolate(pool_out.transpose(-1,-2), 2048)
        
        return dec_pred.transpose(-1,-2).squeeze()
