"""
main.py 
 - Desc: Wrapper file for training the RT Neural Midi model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import transformers
import transformers
import yaml 
import os 
import random 

# Local Imports
from src.discriminator import MultiScaleDiscriminator, HiFiGANMultiPeriodDiscriminator
from src.heads import FourierHead
from src.spectral_ops import STFT
from src.encoder import TimbreEncoder
from src.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MultiScaleSpectralLoss
from src.models import Vocos
from src.modules import safe_log
from src.dataset import Midi_Seg, train_collate_fn
from src.train import trainer
from src.utils import calc_nparam

from dotenv import load_dotenv

load_dotenv()

music_path = os.getenv("music_path")
sf_path = os.getenv('sf_path')
sample_path = os.getenv('sample_path')

if __name__ == '__main__': 
    # Load in Config 
    yaml_name = 'midi_vocos_1st.yaml'
    with open('./yamls/' + yaml_name, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    device = config['device']      

    vocos_config = config['vocos_config']

    # set random seeds
    seed = config.get('seed', 324)
    print(f'seed: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 

    sample_rate = config['sample_rate']
    buffer_size = config['buffer_size']
    prev_ratio = config.get('prev_ratio', 2.0)

    # Create Dataset 
    train_dataset = Midi_Seg(sf_path = sample_path, 
        sr = sample_rate, 
        buffer_size = buffer_size, 
        prev_ratio = prev_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_collate_fn, num_workers=8, prefetch_factor=4, pin_memory=True, persistent_workers=True)

    # Create Preprocessing 
    stft_transform = STFT(
        n_fft=vocos_config['head']['n_fft'],
        hop_length=vocos_config['head']['hop_length'],
        win_length=vocos_config['head']['n_fft']
    ).to(device)


    # Create Vocos Synth
    model = Vocos(vocos_config).to(device)

    # Create Timbre Encoder
    timbre_config = config['timbre_config']  
    tmbr_encoder = TimbreEncoder(timbre_config).to(device)

    print(f'Synth nparams: {calc_nparam(model)}, timbre encoder nparams: {calc_nparam(tmbr_encoder)}')

    # Create Discriminators
    print('Loading discriminators...')
    multiperioddisc = HiFiGANMultiPeriodDiscriminator(periods=config.get('mpd_periods', [2, 3, 5, 7, 11]))
    multiperioddisc = multiperioddisc.to(device)
    print(f'MultiPeriodDiscriminator nparams: {calc_nparam(multiperioddisc)}')
   
    multiresdisc = MultiScaleDiscriminator(nscales=len(config['scales']), 
                    features=config['features'], 
                    use_additional_conv=config.get('use_additional_conv', False))
    multiresdisc = multiresdisc.to(device)
    print(f'MultiResDiscriminator nparams: {calc_nparam(multiresdisc)}')

    # Define Losses
    disc_loss = DiscriminatorLoss()
    gen_loss = GeneratorLoss()
    feat_matching_loss = FeatureMatchingLoss()
    MSSLoss = MultiScaleSpectralLoss(nffts=config.get('nffts', [512, 256, 128, 64])).to(device)
    MSELoss = nn.MSELoss()

    # Instantiate Optimizers and Schedulers
    disc_params = list(multiperioddisc.parameters()) + list(multiresdisc.parameters())
    gen_params = list(model.parameters()) + list(tmbr_encoder.parameters())
    

    opt_disc = torch.optim.AdamW(disc_params, lr=config['lr_disc'], betas=config['betas'])
    opt_gen = torch.optim.AdamW(gen_params, lr=config['lr_gen'], betas=config['betas'])

    lr_scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=config['milestones'], gamma=config['gamma'])
    lr_scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=config['milestones'], gamma=config['gamma'])
    
    # Run Joint Training
    trainer(model, tmbr_encoder, stft_transform,
                        multiperioddisc, multiresdisc, opt_gen, opt_disc,
                        lr_scheduler_gen, lr_scheduler_disc, MSSLoss, MSELoss, 
                        train_dataloader, config['total_epochs'], config['comment'], 
                        device, config)