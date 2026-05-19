import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss per segmentació binària."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + DiceLoss per segmentació binària."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """Tversky Loss per segmentació binària."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index


class BCETverskyLoss(nn.Module):
    """BCEWithLogitsLoss + TverskyLoss per segmentació binària."""

    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.5, tversky_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.tversky_weight * self.tversky(logits, targets)
        )


class MulticlassDiceLoss(nn.Module):
    """
    Dice Loss per segmentació multiclasse.

    logits:  [B, C, H, W]
    targets: [B, H, W] amb valors enters 0..C-1

    Per defecte ignorem la classe 0 en la mitjana perquè és el fons i pot dominar
    la mètrica. Així optimitzem sobretot les classes tumorals.
    """

    def __init__(self, num_classes=4, smooth=1e-6, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]

        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        start_class = 0 if self.include_background else 1
        dice_losses = []

        for class_idx in range(start_class, self.num_classes):
            pred_c = probs[:, class_idx]
            target_c = targets_one_hot[:, class_idx]

            intersection = (pred_c * target_c).sum()
            denominator = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (
                denominator + self.smooth
            )
            dice_losses.append(1.0 - dice)

        return torch.stack(dice_losses).mean()


class CEDiceLoss(nn.Module):
    """
    CrossEntropyLoss + MulticlassDiceLoss.

    Aquesta és la loss recomanada per a la sortida multiclasse:
        - CrossEntropy aprèn la classe correcta píxel a píxel.
        - Dice ajuda a millorar el solapament de cada subregió tumoral.
    """

    def __init__(
        self,
        num_classes=4,
        ce_weight=0.5,
        dice_weight=0.5,
        include_background=False,
        class_weights=None,
    ):
        super().__init__()
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            include_background=include_background,
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]

        # Si hem definit pesos de classe, els movem al mateix device que logits.
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight.data = self.ce.weight.data.to(logits.device)

        ce_loss = self.ce(logits, targets.long())
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
