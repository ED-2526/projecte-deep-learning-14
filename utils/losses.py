# Importem PyTorch.
import torch

# Importem torch.nn perquè les losses personalitzades hereten de nn.Module.
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss per segmentació binària.

    El Dice Score és una mètrica que volem maximitzar.
    Però una loss s'ha de minimitzar.

    Per això:
        Dice Loss = 1 - Dice Score
    """

    def __init__(self, smooth=1e-6):
        """
        Constructor de la Dice Loss.

        Args:
            smooth: valor petit per evitar divisions per zero.
        """

        # Inicialitzem correctament la classe base nn.Module.
        super(DiceLoss, self).__init__()

        # Guardem smooth com a atribut de la classe.
        self.smooth = smooth


    def forward(self, logits, targets):
        """
        Calcula la Dice Loss.

        Args:
            logits: sortida bruta del model, amb forma [batch, 1, H, W].
            targets: màscara real binària, amb forma [batch, 1, H, W].

        Returns:
            loss: valor de Dice Loss.
        """

        # Convertim els logits en probabilitats entre 0 i 1.
        # La U-Net retorna logits, no probabilitats.
        probs = torch.sigmoid(logits)

        # Aplanem les probabilitats.
        # Passem de [batch, 1, H, W] a un vector llarg.
        probs = probs.view(-1)

        # Aplanem també les màscares reals.
        targets = targets.view(-1)

        # Calculem la intersecció suau.
        # Com probs són probabilitats, aquesta operació és diferenciable.
        # Això és necessari perquè la loss pugui fer backpropagation.
        intersection = (probs * targets).sum()

        # Calculem el Dice Score amb probabilitats.
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        # Retornem Dice Loss.
        # Si Dice és alt, la loss és baixa.
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combina BCEWithLogitsLoss i DiceLoss.

    BCE ajuda a penalitzar errors píxel a píxel.
    Dice ajuda a optimitzar el solapament global de la màscara.

    Aquesta combinació és útil en segmentació mèdica,
    especialment quan hi ha molt desequilibri entre fons i tumor.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Constructor de la loss combinada.

        Args:
            bce_weight: pes de la BCE dins la loss final.
            dice_weight: pes de la Dice Loss dins la loss final.
        """

        # Inicialitzem correctament nn.Module.
        super(BCEDiceLoss, self).__init__()

        # BCEWithLogitsLoss combina internament sigmoid + binary cross entropy.
        # Per això li passem logits directament.
        self.bce = nn.BCEWithLogitsLoss()

        # Creem una instància de la nostra Dice Loss.
        self.dice = DiceLoss()

        # Guardem el pes de la BCE.
        self.bce_weight = bce_weight

        # Guardem el pes de la Dice Loss.
        self.dice_weight = dice_weight


    def forward(self, logits, targets):
        """
        Calcula la loss final BCE + Dice.

        Args:
            logits: sortida bruta del model.
            targets: màscara real binària.

        Returns:
            loss combinada.
        """

        # Calculem la BCE.
        # Aquesta loss compara cada píxel individualment.
        bce_loss = self.bce(logits, targets)

        # Calculem la Dice Loss.
        # Aquesta loss mesura el solapament global entre màscares.
        dice_loss = self.dice(logits, targets)

        # Retornem la combinació ponderada.
        # Per defecte:
        #   0.5 * BCE + 0.5 * DiceLoss
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class TverskyLoss(nn.Module):
    """
    Tversky Loss per segmentació binària.

    És semblant a Dice Loss, però permet donar diferent pes a:
        - falsos positius
        - falsos negatius

    En el nostre cas ens interessen especialment els falsos negatius,
    perquè el model falla sobretot quan hi ha tumor petit però prediu 0 píxels.

    alpha controla el pes dels falsos positius.
    beta controla el pes dels falsos negatius.

    Si beta > alpha, penalitzem més deixar escapar tumor.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: sortida crua del model, abans de sigmoid
        targets: màscara real binària, amb valors 0 o 1
        """

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        true_positive = (probs * targets).sum()
        false_positive = (probs * (1 - targets)).sum()
        false_negative = ((1 - probs) * targets).sum()

        tversky_index = (
            (true_positive + self.smooth)
            / (
                true_positive
                + self.alpha * false_positive
                + self.beta * false_negative
                + self.smooth
            )
        )

        return 1 - tversky_index


class BCETverskyLoss(nn.Module):
    """
    Combina BCEWithLogitsLoss + TverskyLoss.

    BCE ajuda a aprendre píxel a píxel.
    Tversky ajuda a millorar el solapament i penalitza més els falsos negatius.

    Aquesta combinació és útil quan el model infra-segmenta o deixa escapar tumors petits.
    """

    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.5, tversky_weight=0.5):
        super(BCETverskyLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        tversky_loss = self.tversky(logits, targets)

        loss = (
            self.bce_weight * bce_loss
            + self.tversky_weight * tversky_loss
        )

        return loss
