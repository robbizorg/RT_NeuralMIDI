import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram
from scipy.signal import firwin
import numpy as np
import copy

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_instance_norm=False, use_leaky_relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        ]
        
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if use_leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64, 128], use_additional_conv=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(in_channels=in_channels, out_channels=features[0], stride=2, use_instance_norm=False, use_leaky_relu=True))

        in_channels = features[0]
        for feature in features[1:]:
            self.layers.append(
                ConvBlock(in_channels, feature, stride=2, use_instance_norm=False, use_leaky_relu=True)
            )
            if use_additional_conv:
                self.layers.append(
                    ConvBlock(feature, feature, kernel_size=3, stride=1, padding=1, use_instance_norm=False, use_leaky_relu=True)
                )
            in_channels = feature
        
        self.layers.append(ConvBlock(features[-1], 1, stride=2, use_instance_norm=False, use_leaky_relu=False))

        self.initialize_param()
        self.apply_weight_norm()
        
    def initialize_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def apply_weight_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = nn.utils.parametrizations.weight_norm(m)
    
    def forward(self, x, save_feat=False):
        feat = x
        if save_feat:
            features = []
            for layer in self.layers:
                feat = layer(feat)
                features.append(feat)
            return features
        else:
            for layer in self.layers:
                feat = layer(feat)
            return feat
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, nscales=3, in_channels=1, features=[32, 64, 128], use_additional_conv=False):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for _ in range(nscales):
            self.discriminators += [Discriminator(in_channels, features, use_additional_conv)]
            
    def forward(self, xs, detach=False, save_feat=False):
        # xs in the format [spec_1024, spec_512, spec_256, spec_128...]
        outs = []
        for i, disc in enumerate(self.discriminators):
            if detach:
                outs += [disc(xs[i].detach(), save_feat)]
            else:
                outs += [disc(xs[i], save_feat)]
        return outs

class MultiWindowMultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=[1024, 512, 256, 128], in_channels=1, features=[32, 64, 128], window_fns=[torch.hann_window, torch.hamming_window, torch.kaiser_window, torch.bartlett_window], use_additional_conv=False, device='cuda'):
        super().__init__()
        overlap = 0.75
        hopsizes = [int((1-overlap)*scale) for scale in scales]
        self.transforms = []
        for window in window_fns:
            self.transforms.append([Spectrogram(n_fft=scale, win_length=scale, hop_length=hopsize, power=1, window_fn=window).to(device) for (scale, hopsize) in zip(scales, hopsizes)])
        
        self.discriminators = nn.ModuleList()
        for _ in range(len(window_fns)):
            self.discriminators += [MultiScaleDiscriminator(nscales=len(scales), in_channels=in_channels, features=features, use_additional_conv=use_additional_conv)]
            
    def forward(self, x, detach=False):
        x = x.unsqueeze(1) # [B, 1, t]
        specs = []
        for transforms in self.transforms:
            specs.append([transform(x) for transform in transforms])
            
        outs = []
        for i, disc in enumerate(self.discriminators):
            outs += disc(specs[i], detach)
        return outs
    
# xs = [torch.randn(4, 1, 1025, 32), torch.randn(4, 1, 513, 63), torch.randn(4, 1, 257, 126), torch.randn(4, 1, 129, 251), torch.randn(4, 1, 65, 501), torch.randn(4, 1, 33, 1001)]
class ResolutionDiscriminator(nn.Module):
    def __init__(self, stride_dict, in_channels=1, hidden_dim=128, windowsize=512):
        super().__init__()
        # stride_dict = {
        #     '2048': [(2,1), (2,1), (2,1), (2,1), (2,1)],
        #     '1024': [(2,2), (2,1), (2,1), (2,1), (1,1)],
        #     '512':  [(2,2), (2,2), (2,1), (1,1), (1,1)],
        #     '256':  [(2,2), (2,2), (1,2), (1,1), (1,1)],
        #     '128':  [(2,2), (1,2), (1,2), (1,2), (1,1)],
        #     '64':   [(1,2), (1,2), (1,2), (1,2), (1,2)]
        # }
        # kernel_size_dict = {
        #     (1,1): (3,3),
        #     (1,2): (3,4),
        #     (2,1): (4,3),
        #     (2,2): (4,4)
        # }
        kernel_size_dict = {
            1:3,
            2:4,
            4:8
        }
        padding_dict = {
            1:1,
            2:1,
            4:2
        }
        
        strides = stride_dict[str(windowsize)]
        
        self.input_conv = ConvBlock(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1, use_instance_norm=False, use_leaky_relu=True)
        
        layers = []
        for stride in strides:
            layers.append(
                ConvBlock(hidden_dim, hidden_dim, kernel_size=(kernel_size_dict[stride[0]], kernel_size_dict[stride[1]]), stride=stride, padding=(padding_dict[stride[0]], padding_dict[stride[1]]), use_instance_norm=False, use_leaky_relu=True)
            )
        self.layers = nn.Sequential(*layers)
        
        self.out_conv = ConvBlock(hidden_dim, 1, kernel_size=3, stride=1, padding=1, use_instance_norm=False, use_leaky_relu=False)

        self.initialize_param()
        self.apply_weight_norm()
        
    def initialize_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def apply_weight_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = nn.utils.parametrizations.weight_norm(m)
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.layers(x)
        x = self.out_conv(x)
        return x
    
class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, stride_dict, in_channels=1, hidden_dim=128, windowsizes=[2048, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for windowsize in windowsizes:
            self.discriminators += [ResolutionDiscriminator(stride_dict, in_channels, hidden_dim, windowsize)]
            
    def forward(self, xs, detach=False):
        # xs in the format [spec_2048, spec_1024, spec_512, spec_256, spec_128...]
        outs = []
        for i, disc in enumerate(self.discriminators):
            if detach:
                outs += [disc(xs[i].detach())]
            else:
                outs += [disc(xs[i])]
        return outs

class SubBandDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layers = nn.Sequential(
        #     ConvBlock(in_channels=1, out_channels=32, kernel_size=(3,4), stride=(1,2), padding=(1,1)),
        #     ConvBlock(in_channels=32, out_channels=64, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
        #     ConvBlock(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        #     ConvBlock(in_channels=128, out_channels=256, kernel_size=(1,3), stride=(1,1), padding=(0,1)),
        #     ConvBlock(in_channels=256, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(1,1), use_leaky_relu=False)
        # )
        
        self.layers = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=(3,4), stride=(1,2), padding=(1,1)),
            ConvBlock(in_channels=256, out_channels=1, kernel_size=(3,4), stride=(1,2), padding=(1,1), use_leaky_relu=False)
        )

        self.initialize_param()
        self.apply_weight_norm()
        
    def initialize_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def apply_weight_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = nn.utils.parametrizations.weight_norm(m)
    
    def forward(self, x):
        x = self.layers(x)
        return x

# xs = [torch.randn(4, 1, 1025, 32), torch.randn(4, 1, 513, 63), torch.randn(4, 1, 257, 126), torch.randn(4, 1, 129, 251), torch.randn(4, 1, 65, 501), torch.randn(4, 1, 33, 1001)]
class MultiBandDiscriminator(nn.Module):
    def __init__(self, nsubbands, overlap):
        super().__init__()
        self.nsubbands = nsubbands
        self.overlap = overlap
        self.discriminators = nn.ModuleList()
        for _ in range(nsubbands):
            self.discriminators += [SubBandDiscriminator()]

    def forward(self, x):
        # x in shape [B, 1, H, W]
        bandwidth = x.shape[2] // self.nsubbands
        outs = []
        for i in range(self.nsubbands):
            start = i*bandwidth - self.overlap if i != 0 else 0
            end = (i+1)*bandwidth + self.overlap
            subband = x[:, :, start:end, :]
            outs += [self.discriminators[i](subband)]
        return outs
        
class MultiBandMultiResolutionDiscriminator(nn.Module):
    def __init__(self, windowsizes, nsubbands_dict, overlap_dict):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for windowsize in windowsizes:
            self.discriminators += [MultiBandDiscriminator(nsubbands=nsubbands_dict[str(windowsize)], overlap=overlap_dict[str(windowsize)])]
            
    def forward(self, xs, detach=False):
        # xs in the format [spec_2048, spec_1024, spec_512, spec_256, spec_128...]
        outs = []
        for i, disc in enumerate(self.discriminators):
            if detach:
                outs += disc(xs[i].detach())
            else:
                outs += disc(xs[i])
        return outs

class MultiScaleDiscriminator_V2(nn.Module):
    def __init__(self, scales, overlap, features, device):
        super().__init__()
        hopsizes = [int((1-overlap)*scale) for scale in scales]
        self.multiscale_transforms = [Spectrogram(n_fft=scale, win_length=scale, hop_length=hopsize, power=1).to(device) for (scale, hopsize) in zip(scales, hopsizes)]
        
        self.discriminators = nn.ModuleList()
        for _ in range(len(scales)):
            self.discriminators += [Discriminator(in_channels=1, features=features)]
    
    def get_multiscale_spectrograms(self, x):
        # x in shape [B, t]
        x = x.unsqueeze(1) # [B, 1, t]
        outs = []
        for spec in self.multiscale_transforms:
            outs.append(spec(x))
        # each spec in outs is in shape [B, 1, Hi, Wi]
        return outs     
            
    def forward(self, x, detach=False):
        # x in shape [B, t]
        specs = self.get_multiscale_spectrograms(x)
        outs = []
        for i, disc in enumerate(self.discriminators):
            if detach:
                outs += [disc(specs[i].detach())]
            else:
                outs += [disc(specs[i])]
        return outs

class MultiRateDiscriminator(nn.Module):
    def __init__(self, nlevels, ntaps, scales, overlap, features, device):
        super().__init__()
        self.lpf = torch.fliplr(torch.from_numpy(firwin(numtaps=ntaps, cutoff=0.5, pass_zero=True)).unsqueeze(0)).unsqueeze(0).to(device).float() # [1, 1, ntaps]
        self.hpf = torch.fliplr(torch.from_numpy(firwin(numtaps=ntaps, cutoff=0.5, pass_zero=False)).unsqueeze(0)).unsqueeze(0).to(device).float() # [1, 1, ntaps]
        self.nlevels = nlevels
        self.ntaps = ntaps
        self.scales = scales
        hopsizes = [int((1-overlap)*scale) for scale in scales]
        self.multiscale_transforms = [Spectrogram(n_fft=scale, win_length=scale, hop_length=hopsize, power=1).to(device) for (scale, hopsize) in zip(scales, hopsizes)]
        
        # there are 2**nlevels sub-components in total! each sub-components requires a MultiScaleDiscriminator
        self.multiscale_discriminators = nn.ModuleList()
        for _ in range(2**nlevels):
            self.multiscale_discriminators += [MultiScaleDiscriminator(nscales=len(scales), features=features)]
    
    def multi_rate_decomposition(self, x):
        # x in shape [B, t]
        x = x.unsqueeze(1) # [B, 1, t]
        subcomps = [x]
        for _ in range(self.nlevels):
            newcomps = []
            for subcomp in subcomps:
                lpfed = F.conv1d(subcomp, self.lpf, padding=(self.ntaps-1)//2)
                newcomps.append(lpfed[:,:,::2])
                hpfed = F.conv1d(subcomp, self.hpf, padding=(self.ntaps-1)//2)
                newcomps.append(hpfed[:,:,::2])
            subcomps = newcomps
        # each component in subcomps is in shape [B, 1, t']
        return subcomps 
    
    def get_multiscale_spectrograms(self, x):
        # x in shape [B, 1, t']
        outs = []
        for spec in self.multiscale_transforms:
            outs.append(spec(x))
        # each spec in outs is in shape [B, 1, Hi, Wi]
        return outs    
        
    def forward(self, x, detach=False):
        # x in shape [B, t]
        subcomps = self.multi_rate_decomposition(x)
        outs = []
        for subcomp, multiscale_disc in zip(subcomps, self.multiscale_discriminators):
            multiscale_specs = self.get_multiscale_spectrograms(subcomp)
            outs.extend(multiscale_disc(multiscale_specs, detach))
        return outs

class MultiDownsampleDiscriminator(nn.Module):
    def __init__(self, cutoffs, ntaps, scales, overlap, features, device):
        super().__init__()
        self.lpfs = [torch.fliplr(torch.from_numpy(firwin(numtaps=ntaps, cutoff=cutoff, pass_zero=True)).unsqueeze(0)).unsqueeze(0).to(device).float() for cutoff in cutoffs] # [1, 1, ntaps]
        self.ntaps = ntaps
        self.scales = scales
        self.cutoffs = cutoffs
        hopsizes = [int((1-overlap)*scale) for scale in scales]
        self.multiscale_transforms = [Spectrogram(n_fft=scale, win_length=scale, hop_length=hopsize, power=1).to(device) for (scale, hopsize) in zip(scales, hopsizes)]
        
        # there are len(cutoffs) sub-components in total! each sub-components requires a MultiScaleDiscriminator
        self.multiscale_discriminators = nn.ModuleList()
        for _ in range(len(cutoffs)):
            self.multiscale_discriminators += [MultiScaleDiscriminator(nscales=len(scales), features=features)]
    
    def multi_downsampling(self, x):
        # x in shape [B, t]
        x = x.unsqueeze(1) # [B, 1, t]
        subcomps = []
        for cutoff, lpf in zip(self.cutoffs, self.lpfs):
            lpfed = F.conv1d(x, lpf, padding=(self.ntaps-1)//2)
            subcomps.append(lpfed[:,:,::int(1/cutoff)])
        # each component in subcomps is in shape [B, 1, t']
        return subcomps 
    
    def get_multiscale_spectrograms(self, x):
        # x in shape [B, 1, t']
        outs = []
        for spec in self.multiscale_transforms:
            outs.append(spec(x))
        # each spec in outs is in shape [B, 1, Hi, Wi]
        return outs    
        
    def forward(self, x, detach=False):
        # x in shape [B, t]
        subcomps = self.multi_downsampling(x)
        outs = []
        for subcomp, multiscale_disc in zip(subcomps, self.multiscale_discriminators):
            multiscale_specs = self.get_multiscale_spectrograms(subcomp)
            outs.extend(multiscale_disc(multiscale_specs, detach))
        return outs

class MultiScaleMultiDownsampleDiscriminator(nn.Module):
    def __init__(self, cutoffs, ntaps, scales_multiscale, scales_multidownsample, overlap, features, device):
        super().__init__()
        self.multiscale_disc = MultiScaleDiscriminator_V2(scales_multiscale, overlap, features, device)
        self.multidownsample_disc = MultiDownsampleDiscriminator(cutoffs, ntaps, scales_multidownsample, overlap, features, device)
        
    def forward(self, x, detach=False):
        outs_multiscale = self.multiscale_disc(x, detach)
        outs_multidownsample = self.multidownsample_disc(x, detach)
        outs = outs_multiscale + outs_multidownsample
        return outs

class MultiScaleMultiRateDiscriminator_V2(nn.Module):
    def __init__(self, nlevels, ntaps, scales_multiscale, scales_multirate, overlap, features, device):
        super().__init__()
        self.multiscale_disc = MultiScaleDiscriminator_V2(scales_multiscale, overlap, features, device)
        self.multirate_disc = MultiRateDiscriminator(nlevels, ntaps, scales_multirate, overlap, features, device)
        
    def forward(self, x, detach=False):
        outs_multiscale = self.multiscale_disc(x, detach)
        outs_multirate = self.multirate_disc(x, detach)
        outs = outs_multiscale + outs_multirate
        return outs

class MultiScaleMultiRateDiscriminator(nn.Module):
    def __init__(self, nlevels, ntaps, scales, overlap, features, device):
        super().__init__()
        self.lpf = torch.fliplr(torch.from_numpy(firwin(numtaps=ntaps, cutoff=0.5, pass_zero=True)).unsqueeze(0)).unsqueeze(0).to(device).float() # [1, 1, ntaps]
        self.hpf = torch.fliplr(torch.from_numpy(firwin(numtaps=ntaps, cutoff=0.5, pass_zero=False)).unsqueeze(0)).unsqueeze(0).to(device).float() # [1, 1, ntaps]
        self.nlevels = nlevels
        self.ntaps = ntaps
        self.scales = scales
        hopsizes = [int((1-overlap)*scale) for scale in scales]
        self.multiscale_transforms = [Spectrogram(n_fft=scale, win_length=scale, hop_length=hopsize, power=1).to(device) for (scale, hopsize) in zip(scales, hopsizes)]
        
        # there are 2**nlevels sub-components in total! each sub-components requires a MultiScaleDiscriminator
        self.multiscale_discriminators = nn.ModuleList()
        for _ in range(2**nlevels):
            self.multiscale_discriminators += [MultiScaleDiscriminator(nscales=len(scales), features=features)]
    
    def multi_rate_decomposition(self, x):
        # x in shape [B, t]
        x = x.unsqueeze(1) # [B, 1, t]
        subcomps = [x]
        for _ in range(self.nlevels):
            newcomps = []
            for subcomp in subcomps:
                lpfed = F.conv1d(subcomp, self.lpf, padding=(self.ntaps-1)//2)
                newcomps.append(lpfed[:,:,::2])
                hpfed = F.conv1d(subcomp, self.hpf, padding=(self.ntaps-1)//2)
                newcomps.append(hpfed[:,:,::2])
            subcomps = newcomps
        # each component in subcomps is in shape [B, 1, t']
        return subcomps 
    
    def get_multiscale_spectrograms(self, x):
        # x in shape [B, 1, t']
        outs = []
        for spec in self.multiscale_transforms:
            outs.append(spec(x))
        # each spec in outs is in shape [B, 1, Hi, Wi]
        return outs    
        
    def forward(self, x, detach=False):
        # x in shape [B, t]
        subcomps = self.multi_rate_decomposition(x)
        outs = []
        for subcomp, multiscale_disc in zip(subcomps, self.multiscale_discriminators):
            multiscale_specs = self.get_multiscale_spectrograms(subcomp)
            outs.extend(multiscale_disc(multiscale_specs, detach))
        return outs
    
class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 2, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            list: List of each layer's tensors.

        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.parametrizations.weight_norm(m)
                
        self.apply(_apply_weight_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 512,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    # Old forward func
    # def forward(self, x):
    #     """Calculate forward propagation.

    #     Args:
    #         x (Tensor): Input noise signal (B, 1, T).

    #     Returns:
    #         List: List of list of each discriminator outputs, which consists of each layer output tensors.

    #     """
    #     # outs = []
    #     # for f in self.discriminators:
    #     #     outs += [f(x)]

    #     # return outs

    # Based off of the hifigan implementation
    # y: true signal
    # y_hat: pred signal
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r = d(y)[-1]
            y_d_g = d(y_hat)[-1]
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)

        return y_d_rs, y_d_gs



if __name__ == '__main__':
    # x = torch.randn(4, 1, 257, 125)
    # discriminator = Discriminator(features=[32, 64, 128, 256])
    # out = discriminator(x, save_feat=True)
    # print(len(out))
    # for feat in out:
    #     print(feat.shape)
    # print(discriminator)
    
    xs = [torch.randn(4, 1, 1025, 32), torch.randn(4, 1, 513, 63), torch.randn(4, 1, 257, 126), torch.randn(4, 1, 129, 251), torch.randn(4, 1, 65, 501), torch.randn(4, 1, 33, 1001)]
    # xs = [torch.randn(4, 1, 513, 63), torch.randn(4, 1, 257, 126), torch.randn(4, 1, 129, 251), torch.randn(4, 1, 65, 501)]
    multidisc = MultiScaleDiscriminator(nscales=6, features=[32, 64, 128, 256])
    # print(multidisc)
    outs = multidisc(xs, save_feat=True)
    # print(outs[0].shape, outs[1].shape, outs[2].shape, outs[3].shape, outs[4].shape, outs[5].shape)
    print(len(outs))
    print(len(outs[0]))
    print(outs[0][0].shape)
    # print(outs[0].shape, outs[1].shape, outs[2].shape, outs[3].shape)
    
    # device = 'cuda:0'
    # x = torch.randn(4, 16000).to(device)
    # MSMRD = MultiScaleMultiRateDiscriminator(nlevels=2, ntaps=513, scales=[1024, 512, 256, 128], overlap=0.90, features=[32, 64, 128, 256], device=device).to(device)
    # outs = MSMRD(x)
    # print(MSMRD)
    # print(f'total number of params: {calc_nparam(MSMRD)}')
    # print(len(outs))
    # for out in outs:
    #     print(out.shape)
    
    # multibanddisc = MultiBandDiscriminator(nsubbands=8, overlap=8)
    # x = torch.randn(4, 1, 257, 126)
    # outs = multibanddisc(x)
    # for out in outs:
    #     print(out.shape)
    
    
# HifiGan helper function for computing discriminator loss
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses