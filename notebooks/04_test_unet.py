import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from models.unet import UNet


# Creem una imatge falsa amb la mateixa forma que tindrà el nostre dataset
# batch_size = 4
# canals = 1
# height = 240
# width = 240
x = torch.randn((4, 1, 240, 240))

model = UNet(in_channels=1, out_channels=1)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape, "La sortida hauria de tenir la mateixa mida que l'entrada."

print("La U-Net funciona correctament.")
