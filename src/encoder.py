"""
encoder.py 
 - Desc: Contains Code for Timbre Encoder model 
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.random
from transformers import WavLMModel

from src.conv import DilatedConvEncoder

class MLP(nn.Module):
    def __init__(self, input_channel, hidden_dim, forward_expansion=1, n_midlayers=1, dropout=0):
        super().__init__()
        layers = [
            nn.Linear(input_channel, hidden_dim*forward_expansion),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_midlayers):
            layers.extend([
                nn.Linear(hidden_dim*forward_expansion, hidden_dim*forward_expansion),
                nn.ReLU()
            ])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim*forward_expansion, hidden_dim),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        #### input ####
        x   : in shape [*, input_channel]
        
        #### output ####
        out : in shape [*, hidden_dim]
        
        """
        out = self.layers(x)
        return out

def freeze_module(module):
    for p in module.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
    module.eval()

class TimbreEncoder(nn.Module):
    def __init__(self, config):
        """
        Produces an Embedding for the Timbre of an Instrument
         - Desc: Uses audio from first 1s of audio to output timbre embedding      
        """
        super().__init__()
        self.in_channels = config['input_channels']
        self.out_channels = config['out_channels']
        self.timbre_emb_dim = config['timbre_emb_dim']
        self.dilations = config['dilations']

        # Create DilatedConvStack
        self.convencoder = DilatedConvEncoder(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=3, 
            stride=1, 
            dilations=self.dilations, 
            nstacks=1
        )

        # Aggregate and Projection Layer
        self.proj = nn.Linear(self.out_channels * 2, self.timbre_emb_dim)

    def forward(self, x):
        """
        #### input ####
        x           : in shape [B, buffer_size * prev_ratio], sampled @48kHz
        NOTE: Assumed to be 1s Long always
        
        #### output ####
        timbre_emb    : in shape [B, timbre_emb_dim]
        
        """
        x = self.convencoder(x)

        # Then Simple Mean/Std Pooling over Time
        mean = x.mean(dim=-1)
        std  = x.std(dim=-1)
        x = torch.cat([mean, std], dim=-1)
        return self.proj(x)
    

class LinearFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, timbre_emb_dim):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.film = nn.Linear(timbre_emb_dim, out_channels*2)
                
    def forward(self, x, timbre_emb):
        # x in shape [B, t, in_channels], timbre_emb in shape [B, timbre_emb_dim]

        x = self.linear(x) # [B, t, out_channels]
        condition = self.film(timbre_emb).unsqueeze(1) # [B, 1, out_channels*2] 
        out = condition[:, :, :self.out_channels] * x + condition[:, :, self.out_channels:]
        return out
        
class MLPFiLM(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, timbre_emb_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearFiLM(in_channels=in_channels, out_channels=hidden_dim, timbre_emb_dim=timbre_emb_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        ])
    
    def forward(self, x, timbre_emb):
        # x in shape [B, t, in_channels], timbre_emb in shape [B, timbre_emb_dim]
        x = x.swapaxes(1, 2) # Switch to Appropriate shape
        out = x
        for layer in self.layers:
            if not isinstance(layer, LinearFiLM):
                out = layer(out)
            else:
                out = layer(out, timbre_emb) # [B, t, out_channels]
        out = out.swapaxes(1, 2)
        return out # [B, out_channels, t]
    

if __name__ == '__main__': 
    import yaml 
    import yaml 
    import time
    import sys 
    sys.path.append('./')

    from src.spectral_ops import STFT

    # Get Config
    yaml_name = 'midi_vocos_1st.yaml'
    with open('./yamls/' + yaml_name, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    device = config['device']  
    vocos_config = config['vocos_config'] 
    timbre_config = config['timbre_config']  

    # Create Preprocessing 
    stft_transform = STFT(
        n_fft=vocos_config['head']['n_fft'],
        hop_length=vocos_config['head']['hop_length'],
        win_length=vocos_config['head']['n_fft']
    )

    # Test out the Timbre Encoder 
    x = torch.randn((1, 48000))

    tmbr_encoder = TimbreEncoder(timbre_config)

    x_spec = stft_transform(x)
    start = time.time() 
    timbre_emb = tmbr_encoder(x_spec)
    end = time.time()

    print(timbre_emb.shape)
    print(f'Computed Timbre Emb (CPU) in {(end-start) * 1000} ms')