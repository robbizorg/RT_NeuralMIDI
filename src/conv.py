"""
conv.py 
 - Desc: A collection of convnets to use for the Timbre Encoder
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchaudio.transforms import Spectrogram
import torchcrepe
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
from scipy.fft import rfft
from IPython.display import Audio
from scipy.signal import butter, filtfilt, firwin
from transformers import Wav2Vec2Model, WavLMModel, Wav2Vec2Processor

def clpc(wav, order, windowsize, hopsize=80, return_spec_env=True, nfft=128, return_fft=False):
    """calculate chunked lpc
    #### input ####
    wav                 : in shape [L, ]
    order               : int, the lpc order, returns `order+1` coefficients, where the first coeff is always `1`
    windowsize          : int, the window size of the hann window, default 1024
    hopsize             : int, aka framesize, default 80
    return_spec_env     : bool, if True, return the spectral envelope instead of lpc
    nfft                : int, the number of points in fft, returns nfft // 2 + 1 points for rfft
    
    #### output ####
    out                 : in shape [nfft // 2 + 1, ceil(L / hopsize)] or [order, ceil(L / hopsize)]
    
    """
    L = len(wav)
    out = []
    fft_out = []
    hann = np.hanning(windowsize)
    wav_pad = np.pad(wav, (windowsize//2, windowsize//2), mode='reflect') # [L+windowsize, ]
    for i in range(0, L, hopsize):
        chunk = hann * wav_pad[i:i+windowsize]
        coeffs = librosa.lpc(chunk, order=order)
        if return_spec_env:
            r = rfft(coeffs, n=nfft)
            out.append(np.log10(1 / (np.abs(r) + 1e-5)))
        else:
            out.append(coeffs[1:])
            
        if return_fft:
            fft_out.append(np.log10(np.abs(rfft(chunk))))
        
    out = np.stack(out).T
    fft_out = np.stack(fft_out)
    return {
        'out': out,
        'fft_out': fft_out
    }

def calc_nparam(model):
    nparam = 0
    for p in model.parameters():
        if p.requires_grad:
            nparam += p.numel()
    return nparam

class ResBlock(nn.Module):
    '''
    Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
    '''
    def __init__(self, num_ins, num_outs, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(num_outs)
        # If stride != 1, we're downsampling
        if stride != 1: 
            self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=1, dilation=dilation)
        else:
            self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size, padding=(kernel_size-1)//2*dilation, stride=stride, dilation=dilation)

        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)
    
    
class ConvNet(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=3, num_blocks = 2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, d_model, kernel_size, padding=(kernel_size-1)//2),
            *[ResBlock(d_model, d_model, kernel_size, padding=(kernel_size-1)//2) for _ in range(num_blocks-1)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)
    
class HiFiGANResidualBlock(torch.nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize HiFiGANResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            self.convs1 += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x
    
class ConvEncoder(torch.nn.Module):
    def __init__(self, in_channels=14, out_channels=256, channels=256, kernel_size=7,
        resblock_kernel_sizes=(3, 7, 11), resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)], nloop=1,
        use_additional_convs=True, bias=True, nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1}, use_weight_norm=True
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.nloop = nloop

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        
        # define modules
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.blocks = torch.nn.ModuleList()
        for _ in range(nloop):
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    HiFiGANResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels,
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]

        self.output_conv = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        c = self.input_conv(c)
        for i in range(self.nloop):
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks  
        out = self.output_conv(c)
        return out

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.parametrizations.weight_norm(m)

        self.apply(_apply_weight_norm)

class DilatedConvStack(nn.Module):
    def __init__(self, hidden_dim, kernel_size, stride, dilations, use_transposed_conv, up_kernel_size):
        super().__init__()
        self.stack = []
        for dilation in dilations:
            self.stack.append(ResBlock(num_ins=hidden_dim, num_outs=hidden_dim, kernel_size=kernel_size, stride=stride, dilation=dilation))
        if use_transposed_conv:
            self.stack.extend([
                nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=up_kernel_size, stride=up_kernel_size//2, padding=up_kernel_size//4),
                nn.ReLU()
            ])
        self.stack = nn.Sequential(*self.stack)
        
    def forward(self, x):
        return self.stack(x)

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilations, nstacks, use_transposed_conv=False, up_kernel_size=None):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.stacks = []
        for _ in range(nstacks):
            self.stacks.append(DilatedConvStack(hidden_dim=out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, use_transposed_conv=use_transposed_conv, up_kernel_size=up_kernel_size))
        self.stacks = nn.Sequential(*self.stacks)
        self.out_conv = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.stacks(x)
        out = self.out_conv(x)
        return out