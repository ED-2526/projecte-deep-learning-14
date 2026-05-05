# Importem sys i os per poder afegir la carpeta arrel del projecte al path.
import sys
import os

# Calculem la carpeta principal del projecte.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Afegim aquesta carpeta al path de Python.
# Així podem importar models.unet des d'una notebook/script dins de notebooks/.
sys.path.append(project_root)


# Importem PyTorch.
import torch

# Importem la nostra arquitectura U-Net.
from models.unet import UNet


# ============================================================
# Crear una entrada artificial
# ============================================================

# Creem una imatge falsa amb la mateixa forma que tindrà el nostre dataset.
# No fem servir dades reals perquè només volem comprovar si el model funciona.
#
# La forma és:
#   4   = batch_size, és a dir, 4 imatges alhora
#   1   = nombre de canals, perquè només utilitzem FLAIR
#   240 = altura de la imatge
#   240 = amplada de la imatge
x = torch.randn((4, 1, 240, 240))


# ============================================================
# Crear el model
# ============================================================

# Creem una U-Net.
# in_channels=1 perquè l'entrada només té un canal: FLAIR.
# out_channels=1 perquè volem una única màscara binària de sortida.
model = UNet(in_channels=1, out_channels=1)


# ============================================================
# Fer un forward pass
# ============================================================

# Passem el batch artificial pel model.
# Això comprova que la U-Net pot processar una entrada amb aquesta forma.
y = model(x)


# Mostrem la forma de l'entrada.
print("Input shape:", x.shape)

# Mostrem la forma de la sortida.
print("Output shape:", y.shape)


# Comprovem automàticament que la sortida té la mateixa mida que l'entrada.
# En segmentació això és necessari perquè volem una predicció per cada píxel.
# Si aquesta condició no es compleix, el programa s'atura amb un error.
assert y.shape == x.shape, "La sortida hauria de tenir la mateixa mida que l'entrada."


# Si arribem aquí, vol dir que no hi ha errors de dimensions.
print("La U-Net funciona correctament.")
