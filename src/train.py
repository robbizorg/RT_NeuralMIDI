"""
train.py 
 - Desc: Contains training code for training the Neural Midi Synth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torchaudio.transforms import Spectrogram, MelSpectrogram
from torch.utils.tensorboard.writer import SummaryWriter
from itertools import combinations 

# Local Imports
from src.discriminator import discriminator_loss, generator_loss
from src.utils import save_model

def trainer(model, tmbr_encoder, stft_transform,
                        multiperioddisc, multiresddisc, optimizer_gen, optimizer_disc,
                        lr_scheduler_gen, lr_scheduler_disc, MSSLoss, MSELoss, 
                        train_dataloader, total_epochs, comment, 
                        device, config): 
    print(f'Starting training on {device}...')

    # Establish ckpt location
    ckpt_path = config.get('ckpt_path', './ckpt')
    tgt_path = os.path.join(ckpt_path, comment)
    if not os.path.exists(tgt_path): 
        os.mkdir(tgt_path)

    writer = SummaryWriter(os.path.join(tgt_path, "logs"+comment))
    interval = 100 # print training loss every `interval` samples

    val_step = 1
    val_interval = (len(train_dataloader) * 5) // 10

    flush_interval = 200
    scales = config['scales'] # Disc. Scales
    fft_factor = config.get('fft_factor', 1)
    overlap = config.get('overlap', 0.75)

    mrd_loss_coeff = config.get('mrd_loss_coeff', 1.0)
    gen_loss_coeff = config.get('gen_loss_coeff', 5.0)
    fm_loss_coeff = config.get('fm_loss_coeff', 1.0)
    mel_loss_coeff = config.get('mel_loss_coeff', 1.0)

    save_epochs = [0, 1, 2, 5, 10, 15, 30, 50, 75, 99]
    for epoch in range(total_epochs):
        model.train()
        tmbr_encoder.train()
        running_loss_recon = 0.
        running_loss_disc = 0.
        running_loss_gen = 0.

        i = -1 # Count for Saving and Training
        for idx, batch in enumerate(train_dataloader):
            xs, prev_xs, timbre_x, pitch = batch 
            timbre_x = timbre_x.to(device).float()

            num_samples = len(xs)

            timbre_spec = stft_transform(timbre_x)

            for jdx in range(num_samples): 
                i += 1
                if i % interval == 0:
                    start = time.time()

                x = xs[jdx].to(device).float()
                prev_x = prev_xs[jdx].to(device).float()

                prev_spec = stft_transform(prev_x)
                if len(pitch.shape) != 3:
                    pitch = pitch[:, None, None].repeat(1, 1, prev_spec.shape[-1]).float().to(device)

                in_feats = torch.cat([pitch, prev_spec], dim = 1)

                timbre_emb = tmbr_encoder(timbre_spec)

                out = model(in_feats, timbre_emb=timbre_emb)

                # Match Time
                x_hat = out[:, :x.shape[-1]]
                

                # generate spectrogram for x and x_hat
                spec_reals = []
                spec_fakes = []
                for scale in scales:
                    win_length = scale
                    nfft = fft_factor * win_length
                    hopsize = int((1-overlap) * win_length)
                    spec = Spectrogram(n_fft=nfft, win_length=win_length, hop_length=hopsize, power=1).to(device)
                    spec_reals.append(spec(x).unsqueeze(1))
                    spec_fakes.append(spec(x_hat).unsqueeze(1))
        
                
                # train discriminator
                disc_reals = multiresddisc(spec_reals)
                loss_disc_real = 0.
                for disc_real in disc_reals:
                    loss_disc_real += MSELoss(disc_real, torch.ones_like(disc_real, device=device)) / len(disc_reals)
                        
                disc_fakes = multiresddisc(spec_fakes, detach=True)
                loss_disc_fake = 0.
                for disc_fake in disc_fakes:
                    loss_disc_fake += MSELoss(disc_fake, torch.zeros_like(disc_fake, device=device)) / len(disc_fakes)
                
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                
                # Calculate MultiPeriodDisc Loss
                x_df_hat_r, x_df_hat_g = multiperioddisc(x.unsqueeze(1), x_hat.unsqueeze(1).detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(x_df_hat_r, x_df_hat_g)
                loss_disc += mrd_loss_coeff * loss_disc_f

                optimizer_disc.zero_grad()
                loss_disc.backward()
                
                # check gradient norm of the discriminator
                disc_grad_norm = 0.0
                for p in multiresddisc.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        disc_grad_norm += param_norm.item() ** 2
                disc_grad_norm = disc_grad_norm ** 0.5

                mpdisc_grad_norm = 0.0
                for p in multiperioddisc.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        mpdisc_grad_norm += param_norm.item() ** 2
                mpdisc_grad_norm = mpdisc_grad_norm ** 0.5

                # gradient clipping 
                torch.nn.utils.clip_grad_norm_(multiresddisc.parameters(), max_norm=0.70)
                torch.nn.utils.clip_grad_norm_(multiperioddisc.parameters(), max_norm=0.70)

                optimizer_disc.step() 
            
                
                # train generator
                loss_recon = MSSLoss(x_hat, x)
                # Res Loss
                disc_fakes = multiresddisc(spec_fakes)
                loss_gen = 0.
                for disc_fake in disc_fakes:
                    loss_gen += MSELoss(disc_fake, torch.ones_like(disc_fake, device=device)) / len(disc_fakes)        
            
                # Period Loss
                x_df_hat_r, x_df_hat_g = multiperioddisc(x.unsqueeze(1), x_hat.unsqueeze(1))
                loss_gen_s, lossses_gen_s = generator_loss(x_df_hat_g)
                loss_gen += mrd_loss_coeff * loss_gen_s

                # Calculate generator multiperiod loss

                loss = mel_loss_coeff * loss_recon + gen_loss_coeff * loss_gen 


                # backprop and update
                optimizer_gen.zero_grad()
                loss.backward()
                optimizer_gen.step()

                # stats and writer
                running_loss_recon += loss_recon.item()
                running_loss_disc += loss_disc.item()
                running_loss_gen += loss_gen.item()
                # running_disc_grad_norm += disc_grad_norm

                if (i+1) % interval == 0:
                    
                    end = time.time()
                    elapsed_time = end - start
                    writer.add_scalar('MSSLoss_training', running_loss_recon / interval, epoch*5*len(train_dataloader)+i)
                    writer.add_scalar('Disc_loss_training', running_loss_disc / interval, epoch*5*len(train_dataloader)+i)
                    writer.add_scalar('Gen_loss_training', running_loss_gen / interval, epoch*5*len(train_dataloader)+i)

                    print(f'#################RUNTIME: {elapsed_time:.{4}f}#################')
                    print(f'Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(train_dataloader)*5}], MSSLoss_training: {(running_loss_recon / interval):.{6}f}, Gen_loss_training: {(running_loss_gen / interval):.{6}f}, Disc_loss_training: {(running_loss_disc / interval):.{6}f}')
                    running_loss_recon = 0.
                    running_loss_disc = 0.
                    running_loss_gen = 0.
                    # running_disc_grad_norm = 0.
                
                if (i+1) % flush_interval == 0:
                    writer.flush()
                    torch.cuda.empty_cache()
                
                # one val interval ends, save model
                if (i+1) % val_interval == 0:
                    val_step += 1
                    save_model(ckpt_path, comment, model, tmbr_encoder)

        # End of Epoch
        if epoch in save_epochs: 
            print(f'Saving Epoch {epoch}')
            save_model(ckpt_path, comment, model, tmbr_encoder, epoch_num=epoch)

        # scheduler step and save model
        lr_scheduler_gen.step()
        lr_scheduler_disc.step()        