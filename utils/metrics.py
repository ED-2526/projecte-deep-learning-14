import torch


def binarize_predictions(logits, threshold=0.5):
    """
    Converteix els logits del model en una màscara binària.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    return preds


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula el Dice Score.
    Valor entre 0 i 1.
    1 = segmentació perfecta.
    """
    preds = binarize_predictions(logits, threshold)

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )

    return dice.item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula IoU / Jaccard.
    Valor entre 0 i 1.
    1 = segmentació perfecta.
    """
    preds = binarize_predictions(logits, threshold)

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Precision = TP / (TP + FP)
    Indica quina part del que el model diu que és tumor realment és tumor.
    """
    preds = binarize_predictions(logits, threshold)

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)

    return precision.item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Recall = TP / (TP + FN)
    Indica quina part del tumor real ha estat detectada pel model.
    """
    preds = binarize_predictions(logits, threshold)

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()

    recall = (tp + smooth) / (tp + fn + smooth)

    return recall.item()
