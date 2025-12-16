from typing import List, Tuple

import torch
import torchaudio
from torch import nn

from src.modules import safe_log
from torchaudio.transforms import Spectrogram
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    def __init__(self, nfft, alpha=1.0, overlap=0.75):
        super().__init__()
        self.nfft = nfft
        self.alpha = alpha
        self.overlap = overlap
        self.hopsize = int(nfft*(1-overlap))
        self.spec = Spectrogram(n_fft=nfft, win_length=nfft, hop_length=self.hopsize, power=1) # power=1, magnitude

    def forward(self, x_hat, x):
        """
        #### input ####
        x_hat   : the synthesized waveform, in shape [B, t*framesize]
        x       : the ground truth segments, in shape [B, t*framesize]
        
        #### output ####
        loss    : the spectral loss between `x_hat` and `x`
        
        """
        x_hat_spec = self.spec(x_hat)
        x_spec = self.spec(x)
        loss = F.l1_loss(x_hat_spec, x_spec) + self.alpha * F.l1_loss(torch.log(x_hat_spec + 1e-7), torch.log(x_spec + 1e-7))
        return loss
        
class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, nffts=[2048, 1024, 512, 256, 128, 64], alpha=1.0, overlap=0.75):
        super().__init__()
        self.losses = nn.ModuleList([SpectralLoss(nfft, alpha, overlap) for nfft in nffts])
        
    def forward(self, x_hat, x):
        out = 0.
        for loss in self.losses:
            out += loss(x_hat, x)
        return out

class GeneratorLoss(nn.Module):
    """
    Generator Loss module. Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            disc_outputs (List[Tensor]): List of discriminator outputs.

        Returns:
            Tuple[Tensor, List[Tensor]]: Tuple containing the total loss and a list of loss values from
                                         the sub-discriminators
        """
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l

        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
            disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
                                                       the sub-discriminators for real outputs, and a list of
                                                       loss values for generated outputs.
        """
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            fmap_r (List[List[Tensor]]): List of feature maps from real samples.
            fmap_g (List[List[Tensor]]): List of feature maps from generated samples.

        Returns:
            Tensor: The calculated feature matching loss.
        """
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss