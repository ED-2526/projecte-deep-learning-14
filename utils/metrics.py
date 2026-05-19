import torch


def binarize_predictions(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def multiclass_predictions(logits):
    return torch.argmax(logits, dim=1)


def _prepare_multiclass_targets(targets):
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    return targets.long()


def multiclass_dice_score(
    logits,
    targets,
    num_classes=None,
    include_background=False,
    smooth=1e-6,
):
    """
    Dice mitjà per segmentació multiclasse.

    Calcula un Dice per classe i en fa la mitjana. Per defecte no inclou la
    classe 0 perquè és el fons. Les classes absents tant a predicció com a target
    s'ignoren per no inflar artificialment la mitjana.
    """
    if num_classes is None:
        num_classes = logits.shape[1]

    preds = multiclass_predictions(logits)
    targets = _prepare_multiclass_targets(targets)

    start_class = 0 if include_background else 1
    scores = []

    for class_idx in range(start_class, num_classes):
        pred_c = (preds == class_idx).float()
        target_c = (targets == class_idx).float()

        pred_sum = pred_c.sum()
        target_sum = target_c.sum()

        if pred_sum == 0 and target_sum == 0:
            continue

        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        scores.append(dice)

    if len(scores) == 0:
        return 1.0

    return torch.stack(scores).mean().item()


def multiclass_iou_score(
    logits,
    targets,
    num_classes=None,
    include_background=False,
    smooth=1e-6,
):
    """IoU mitjana per segmentació multiclasse."""
    if num_classes is None:
        num_classes = logits.shape[1]

    preds = multiclass_predictions(logits)
    targets = _prepare_multiclass_targets(targets)

    start_class = 0 if include_background else 1
    scores = []

    for class_idx in range(start_class, num_classes):
        pred_c = (preds == class_idx).float()
        target_c = (targets == class_idx).float()

        pred_sum = pred_c.sum()
        target_sum = target_c.sum()

        if pred_sum == 0 and target_sum == 0:
            continue

        intersection = (pred_c * target_c).sum()
        union = pred_sum + target_sum - intersection
        iou = (intersection + smooth) / (union + smooth)
        scores.append(iou)

    if len(scores) == 0:
        return 1.0

    return torch.stack(scores).mean().item()


def per_class_dice_iou(logits, targets, num_classes=None, smooth=1e-6):
    """
    Retorna mètriques per classe en un diccionari.

    Útil per veure si el model segmenta millor edema, necrosi o tumor realçat.
    """
    if num_classes is None:
        num_classes = logits.shape[1]

    preds = multiclass_predictions(logits)
    targets = _prepare_multiclass_targets(targets)

    results = {}
    for class_idx in range(num_classes):
        pred_c = (preds == class_idx).float()
        target_c = (targets == class_idx).float()

        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        union = pred_sum + target_sum - intersection

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)

        results[class_idx] = {
            "dice": dice.item(),
            "iou": iou.item(),
            "target_pixels": int(target_sum.item()),
            "pred_pixels": int(pred_sum.item()),
        }

    return results


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Dice Score automàtic.

    Si logits té 1 canal, calcula Dice binari.
    Si logits té més d'1 canal, calcula Dice multiclasse mitjà sense fons.
    """
    if logits.shape[1] > 1:
        return multiclass_dice_score(logits, targets, smooth=smooth)

    preds = binarize_predictions(logits, threshold)
    preds = preds.view(-1)
    targets = targets.view(-1).float()

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )
    return dice.item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    IoU automàtica.

    Si logits té 1 canal, calcula IoU binària.
    Si logits té més d'1 canal, calcula IoU multiclasse mitjana sense fons.
    """
    if logits.shape[1] > 1:
        return multiclass_iou_score(logits, targets, smooth=smooth)

    preds = binarize_predictions(logits, threshold)
    preds = preds.view(-1)
    targets = targets.view(-1).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Precision binària. Per multiclasse, usar per_class_dice_iou o afegir mètriques macro."""
    preds = binarize_predictions(logits, threshold)
    preds = preds.view(-1)
    targets = targets.view(-1).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Recall binari. Per multiclasse, usar per_class_dice_iou o afegir mètriques macro."""
    preds = binarize_predictions(logits, threshold)
    preds = preds.view(-1)
    targets = targets.view(-1).float()

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()
