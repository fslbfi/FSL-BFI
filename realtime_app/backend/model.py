import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from realtime_app.config import (
    N_CHANNELS, EMBED_DIM, TARGET_SUBCARRIERS, WINDOW_SIZE, CHECKPOINT_PATH
)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        def conv_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, ceil_mode=True),
            )
        self.layer1 = conv_block(in_channels, embed_dim)
        self.layer2 = conv_block(embed_dim, embed_dim)
        self.layer3 = conv_block(embed_dim, embed_dim)
        self.layer4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.flatten(start_dim=1)


def load_encoder(checkpoint_path=None, device=None):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = CNNEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle both wrapped format {'encoder_state_dict': ...} and flat state_dict
    state = ckpt.get('encoder_state_dict', ckpt)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def windows_to_tensor(win_np, device):
    return torch.from_numpy(win_np).float().permute(0, 3, 1, 2).to(device)
