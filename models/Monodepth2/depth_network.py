import numpy as np
import torch
import torch.nn as nn

from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder

class DepthNetwork(nn.Module):
    def __init__(self):
        super(DepthNetwork, self).__init__()
        self.num_layers = 18 # choices=[18, 34, 50, 101, 152]
        self.weights_init = "pretrained" # choices=["pretrained", "scratch"]
        self.scales = [0, 1, 2, 3]
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
        self.decoder = DepthDecoder(self.num_ch_enc, self.scales)

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out