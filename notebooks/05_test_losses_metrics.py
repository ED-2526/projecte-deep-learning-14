# Importem sys i os per poder importar mòduls del projecte des de la carpeta notebooks.
import sys
import os

# Calculem la ruta de la carpeta arrel del projecte.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Afegim la carpeta arrel al path de Python.
sys.path.append(project_root)


# Importem PyTorch.
import torch

# Importem les funcions de loss que hem definit a utils/losses.py.
# DiceLoss mesura el desacord entre predicció i màscara real segons el Dice.
# BCEDiceLoss combina BCEWithLogitsLoss i DiceLoss.
from utils.losses import DiceLoss, BCEDiceLoss

# Importem les mètriques de segmentació que hem definit a utils/metrics.py.
from utils.metrics import dice_score, iou_score, precision_score, recall_score


# ============================================================
# Crear dades artificials per provar
# ============================================================

# Simulem una sortida del model.
# Aquesta sortida representa logits, no probabilitats.
# Els logits poden ser valors negatius o positius.
#
# Forma:
#   4 = batch size
#   1 = canal de sortida
#   240x240 = mida de la màscara
logits = torch.randn((4, 1, 240, 240))


# Simulem màscares reals binàries.
# torch.randint(0, 2, ...) genera valors 0 o 1.
# Convertim a float perquè les losses treballen amb tensors decimals.
targets = torch.randint(0, 2, (4, 1, 240, 240)).float()


# ============================================================
# Crear les funcions de loss
# ============================================================

# Creem una Dice Loss.
# Aquesta loss és útil en segmentació perquè mesura el solapament entre màscares.
dice_loss_fn = DiceLoss()

# Creem una loss combinada BCE + Dice.
# BCE ajuda a classificar cada píxel com tumor/no tumor.
# Dice ajuda especialment quan hi ha desequilibri entre fons i tumor.
bce_dice_loss_fn = BCEDiceLoss()


# ============================================================
# Calcular les losses
# ============================================================

# Calculem la Dice Loss entre els logits simulats i les màscares reals simulades.
dice_loss = dice_loss_fn(logits, targets)

# Calculem la loss combinada BCE + Dice.
bce_dice_loss = bce_dice_loss_fn(logits, targets)


# ============================================================
# Calcular les mètriques
# ============================================================

# Calculem el Dice Score.
# El Dice Score mesura el solapament entre la predicció i la màscara real.
# 1 significa segmentació perfecta i 0 significa cap solapament.
dice = dice_score(logits, targets)

# Calculem la IoU.
# IoU vol dir Intersection over Union.
# També mesura solapament, però és una mètrica més estricta que Dice.
iou = iou_score(logits, targets)

# Calculem la precision.
# Precision respon:
# "De tots els píxels que el model diu que són tumor, quants ho són realment?"
precision = precision_score(logits, targets)

# Calculem el recall.
# Recall respon:
# "De tots els píxels que realment són tumor, quants ha detectat el model?"
recall = recall_score(logits, targets)


# ============================================================
# Mostrar resultats
# ============================================================

# Mostrem el valor de la Dice Loss.
# Com les dades són aleatòries, no esperem un valor bo; només comprovem que funciona.
print("Dice Loss:", dice_loss.item())

# Mostrem el valor de BCE + Dice Loss.
print("BCE + Dice Loss:", bce_dice_loss.item())

# Mostrem el Dice Score.
print("Dice Score:", dice)

# Mostrem la IoU.
print("IoU:", iou)

# Mostrem la precision.
print("Precision:", precision)

# Mostrem el recall.
print("Recall:", recall)
