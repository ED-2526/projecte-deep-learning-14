# Importem PyTorch perquè les prediccions i les màscares són tensors.
import torch


def binarize_predictions(logits, threshold=0.5):
    """
    Converteix els logits del model en una màscara binària.

    Args:
        logits: sortida bruta del model, abans de sigmoid.
        threshold: llindar per decidir si un píxel és tumor o no.

    Returns:
        preds: màscara binària amb valors 0 o 1.
    """

    # La U-Net retorna logits, no probabilitats.
    # Apliquem sigmoid per convertir cada logit en una probabilitat entre 0 i 1.
    probs = torch.sigmoid(logits)

    # Convertim les probabilitats en valors binaris.
    # Si la probabilitat és superior a 0.5, considerem que és tumor.
    # Si és inferior o igual a 0.5, considerem que és fons.
    preds = (probs > threshold).float()

    # Retornem la màscara binària.
    return preds


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula el Dice Score.

    Dice mesura el solapament entre la màscara predita i la màscara real.
    Valor entre 0 i 1:
        1 = segmentació perfecta
        0 = cap solapament
    """

    # Convertim els logits en una màscara binària.
    preds = binarize_predictions(logits, threshold)

    # Aplanem la predicció.
    # Passem de [batch, 1, H, W] a un vector llarg.
    preds = preds.view(-1)

    # Aplanem també la màscara real.
    targets = targets.view(-1)

    # Calculem la intersecció.
    # Com preds i targets són 0 o 1, el producte només és 1 quan tots dos són tumor.
    intersection = (preds * targets).sum()

    # Fórmula del Dice:
    # Dice = (2 * intersecció) / (píxels predits + píxels reals)
    # Afegim smooth per evitar divisions per zero.
    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )

    # Retornem el valor com a número Python.
    return dice.item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula la IoU, també anomenada Jaccard Index.

    IoU = intersecció / unió

    Valor entre 0 i 1:
        1 = segmentació perfecta
        0 = cap solapament
    """

    # Convertim logits a màscara binària.
    preds = binarize_predictions(logits, threshold)

    # Aplanem predicció i màscara real.
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Intersecció: píxels que són tumor tant a la predicció com a la màscara real.
    intersection = (preds * targets).sum()

    # Unió:
    # píxels predits com tumor + píxels reals de tumor - intersecció.
    # Restem la intersecció perquè si no la comptaríem dues vegades.
    union = preds.sum() + targets.sum() - intersection

    # Fórmula de la IoU.
    iou = (intersection + smooth) / (union + smooth)

    # Retornem el valor com a número Python.
    return iou.item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula la Precision.

    Precision respon:
        De tots els píxels que el model ha predit com a tumor,
        quants realment eren tumor?

    Precision = TP / (TP + FP)
    """

    # Convertim logits a màscara binària.
    preds = binarize_predictions(logits, threshold)

    # Aplanem tensors.
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True positives:
    # píxels que el model prediu com tumor i que realment són tumor.
    tp = (preds * targets).sum()

    # False positives:
    # píxels que el model prediu com tumor però que realment són fons.
    fp = (preds * (1 - targets)).sum()

    # Fórmula de la precision.
    precision = (tp + smooth) / (tp + fp + smooth)

    # Retornem el valor.
    return precision.item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Calcula el Recall.

    Recall respon:
        De tots els píxels que realment són tumor,
        quants ha detectat el model?

    Recall = TP / (TP + FN)
    """

    # Convertim logits a màscara binària.
    preds = binarize_predictions(logits, threshold)

    # Aplanem tensors.
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True positives:
    # píxels correctament detectats com a tumor.
    tp = (preds * targets).sum()

    # False negatives:
    # píxels que realment són tumor però el model ha predit com a fons.
    fn = ((1 - preds) * targets).sum()

    # Fórmula del recall.
    recall = (tp + smooth) / (tp + fn + smooth)

    # Retornem el valor.
    return recall.item()
