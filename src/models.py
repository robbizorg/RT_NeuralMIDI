from typing import Optional, Any

import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm
from src.heads import ISTFTHead
from src.encoder import MLPFiLM

class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        timbre_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        if timbre_emb_dim is not None: 
            self.timbre_film = MLPFiLM(in_channels=dim, hidden_dim=intermediate_dim, out_channels=dim, timbre_emb_dim=timbre_emb_dim)

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        timbre_emb = kwargs.get('timbre_emb', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)

        if timbre_emb is not None: 
            x = self.timbre_film(x, timbre_emb)

        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x

class Vocos(nn.Module):
    """
    Modified Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. 
    
    Consists of two main components: 
        a backbone, and a head.
    """

    def __init__(
        self, config
    ):
        super().__init__()
        backbone_config = config['backbone']
        head_config = config['head']

        self.backbone = VocosBackbone(
            input_channels = backbone_config['input_channels'],
            dim = backbone_config['dim'],
            intermediate_dim = backbone_config['intermediate_dim'],
            num_layers = backbone_config['num_layers'],
            layer_scale_init_value = backbone_config.get('layer_scale_init_value', None),
            adanorm_num_embeddings = backbone_config.get('adanorm_num_embeddings', None), 
            timbre_emb_dim=backbone_config.get('timbre_emb_dim', None)
        )

        self.head = ISTFTHead(
            dim = head_config['dim'], 
            n_fft = head_config['n_fft'], 
            hop_length = head_config['hop_length'], 
            padding = head_config.get('padding', "same")
        )

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a synthesis from audio features. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

     
        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        audio_output = self.decode(features, **kwargs)
        return audio_output

    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

if __name__ == '__main__':
    import yaml 
    import time
    import sys 
    sys.path.append('./')

    # Get Config
    yaml_name = 'midi_vocos_1st.yaml'
    with open('./yamls/' + yaml_name, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    device = config['device']   
    vocos_config = config['vocos_config']   

    source_model = Vocos(vocos_config).to(device)

    features = torch.rand((32, 1, 18)).float().to(device) # Generate Fake Features
    start = time.time()
    x_hat = source_model(features)
    end = time.time()

    print(x_hat.shape)
    print(f'Computed Audio (GPU) in {(end-start) * 1000} ms')

    source_model = source_model.to('cpu')

    features = torch.rand((32, 1, 18)).float().to('cpu') # Generate Fake Features
    start = time.time()
    x_hat = source_model(features)
    end = time.time()

    print(x_hat.shape)
    print(f'Computed Audio (CPU) in {(end-start) * 1000} ms')