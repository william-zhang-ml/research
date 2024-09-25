"""ResNet50-based adversarial autoencoder. """
import torch
from torch import nn
from torch.nn import Module
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class Prior:
    """Base class for latent prior distributions. """
    def __init__(self) -> None:
        pass


class Encoder(Module):
    """ResNet50 and Gaussian noise encoder. """
    def __init__(self, latent_dim: int = 10, noise_dim: int = 10) -> None:
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self._backbone = create_feature_extractor(resnet, ['avgpool'])
        self._last_layer = nn.Linear(
            resnet.fc.in_features + noise_dim,
            latent_dim
        )

    def forward(self, inp: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Encode images into latent codes.

        Args:
            inp (torch.Tensor): images
            noise (torch.Tensor): generative noise (refer to GAN literature)

        Returns:
            torch.Tensor: latent codes
        """
        features = self._backbone(inp)['avgpool'].squeeze(-1).squeeze(-1)
        features_and_noise = torch.cat([features, noise], dim=1)
        return self._last_layer(features_and_noise)
