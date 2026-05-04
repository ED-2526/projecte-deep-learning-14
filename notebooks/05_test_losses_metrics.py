import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from utils.losses import DiceLoss, BCEDiceLoss
from utils.metrics import dice_score, iou_score, precision_score, recall_score


# Simulem una sortida del model
logits = torch.randn((4, 1, 240, 240))

# Simulem màscares reals binàries
targets = torch.randint(0, 2, (4, 1, 240, 240)).float()

dice_loss_fn = DiceLoss()
bce_dice_loss_fn = BCEDiceLoss()

dice_loss = dice_loss_fn(logits, targets)
bce_dice_loss = bce_dice_loss_fn(logits, targets)

dice = dice_score(logits, targets)
iou = iou_score(logits, targets)
precision = precision_score(logits, targets)
recall = recall_score(logits, targets)

print("Dice Loss:", dice_loss.item())
print("BCE + Dice Loss:", bce_dice_loss.item())

print("Dice Score:", dice)
print("IoU:", iou)
print("Precision:", precision)
print("Recall:", recall)
