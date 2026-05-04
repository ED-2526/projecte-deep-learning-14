import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss per segmentació binària.
    Com més alt és el Dice Score, millor.
    Com que una loss s'ha de minimitzar, fem:
        Dice Loss = 1 - Dice Score
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convertim logits a probabilitats
        probs = torch.sigmoid(logits)

        # Aplanem tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combinació de BCEWithLogitsLoss + DiceLoss.
    BCE ajuda píxel a píxel.
    Dice ajuda quan hi ha desbalanceig entre fons i tumor.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
