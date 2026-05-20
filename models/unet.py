# Importem PyTorch.
# Ens farà falta per treballar amb tensors i per concatenar skip connections.
import torch

# Importem torch.nn, que conté totes les capes necessàries per construir xarxes neuronals.
# Per exemple: Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d, etc.
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Bloc bàsic de la U-Net.

    Aquest bloc aplica dues convolucions seguides.
    Cada convolució va seguida de BatchNorm i ReLU.

    Estructura:
        Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor del bloc DoubleConv.

        Args:
            in_channels: nombre de canals d'entrada.
            out_channels: nombre de canals de sortida.
        """

        # Inicialitzem correctament la classe base nn.Module.
        super(DoubleConv, self).__init__()

        # Definim el bloc de dues convolucions.
        # nn.Sequential permet agrupar diverses capes i executar-les una darrere l'altra.
        self.double_conv = nn.Sequential(

            # Primera convolució 2D.
            # kernel_size=3 vol dir que utilitza filtres de 3x3.
            # padding=1 manté la mida espacial de la imatge.
            # És a dir, si entra [H, W], surt també [H, W].
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            # BatchNorm normalitza les activacions i ajuda a estabilitzar l'entrenament.
            nn.BatchNorm2d(out_channels),

            # ReLU introdueix no-linealitat.
            # inplace=True estalvia memòria modificant el tensor directament.
            nn.ReLU(inplace=True),

            # Segona convolució 2D.
            # Ara l'entrada ja té out_channels canals i la sortida manté out_channels.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),

            # Segona normalització.
            nn.BatchNorm2d(out_channels),

            # Segona activació ReLU.
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Defineix com passa la informació pel bloc DoubleConv.

        Args:
            x: tensor d'entrada.

        Returns:
            sortida després de dues convolucions.
        """

        # Passem l'entrada per totes les capes definides a self.double_conv.
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net 2D per segmentació binària.

    Entrada:
        [batch_size, in_channels, height, width]

    Sortida:
        [batch_size, out_channels, height, width]

    En el nostre projecte:
        in_channels = 1  perquè utilitzem FLAIR.
        out_channels = 1 perquè fem segmentació binària tumor/no tumor.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        """
        Constructor de la U-Net.

        Args:
            in_channels: canals d'entrada. En el baseline és 1.
            out_channels: canals de sortida. En segmentació binària és 1.
            features: nombre de canals utilitzats a cada nivell de l'encoder.
        """

        # Inicialitzem correctament nn.Module.
        super(UNet, self).__init__()

        # ModuleList permet guardar una llista de capes de PyTorch.
        # A diferència d'una llista normal, PyTorch sí que registra els paràmetres
        # de les capes guardades dins ModuleList.
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # ============================================================
        # Encoder
        # ============================================================

        # L'encoder redueix la resolució espacial i augmenta els canals.
        # Recorrem la llista features: [32, 64, 128, 256].
        for feature in features:

            # Afegim un bloc DoubleConv.
            # Al primer bloc passem de 1 canal a 32.
            # Després de 32 a 64, de 64 a 128, etc.
            self.downs.append(DoubleConv(in_channels, feature))

            # Actualitzem in_channels perquè el següent bloc rebi
            # el nombre de canals que acaba de produir el bloc actual.
            in_channels = feature

        # MaxPool redueix la mida espacial a la meitat.
        # Per exemple:
        #   240x240 -> 120x120
        #   120x120 -> 60x60
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ============================================================
        # Bottleneck
        # ============================================================

        # El bottleneck és la part més profunda de la xarxa.
        # Rep features[-1] canals, és a dir, 256.
        # Retorna features[-1] * 2 canals, és a dir, 512.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ============================================================
        # Decoder
        # ============================================================

        # El decoder recupera la resolució espacial.
        # Recorrem features en ordre invers: [256, 128, 64, 32].
        for feature in reversed(features):

            # Primer afegim una convolució transposada.
            # Aquesta capa fa upsampling, és a dir, duplica la mida espacial.
            #
            # Exemple:
            #   [512, 15, 15] -> [256, 30, 30]
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )

            # Després de l'upsampling concatenarem amb la skip connection.
            # Això fa que el nombre de canals sigui feature * 2.
            # Per això el DoubleConv rep feature * 2 canals i torna a feature canals.
            self.ups.append(DoubleConv(feature * 2, feature))

        # ============================================================
        # Capa final
        # ============================================================

        # Aquesta convolució 1x1 transforma els canals interns finals en el nombre
        # de canals de sortida.
        #
        # En el nostre cas:
        #   [32, 240, 240] -> [1, 240, 240]
        #
        # No apliquem sigmoid aquí perquè la loss BCEWithLogitsLoss espera logits.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defineix el forward pass de la U-Net.

        Args:
            x: imatge d'entrada amb forma [batch, canals, H, W].

        Returns:
            logits de sortida amb forma [batch, out_channels, H, W].
        """

        # Aquesta llista guardarà les sortides de l'encoder.
        # Després les utilitzarem al decoder com skip connections.
        skip_connections = []

        # ============================================================
        # Encoder
        # ============================================================

        # Recorrem tots els blocs de baixada.
        for down in self.downs:

            # Apliquem el bloc DoubleConv.
            x = down(x)

            # Guardem aquesta sortida abans del pooling.
            # Aquesta informació conserva detalls espacials importants.
            skip_connections.append(x)

            # Reduïm la mida espacial amb MaxPool.
            x = self.pool(x)

        # ============================================================
        # Bottleneck
        # ============================================================

        # Apliquem el bloc més profund de la xarxa.
        x = self.bottleneck(x)

        # Invertim l'ordre de les skip connections.
        # El decoder ha d'utilitzar primer la skip connection més profunda.
        skip_connections = skip_connections[::-1]

        # ============================================================
        # Decoder
        # ============================================================

        # self.ups conté parelles:
        #   ConvTranspose2d
        #   DoubleConv
        #
        # Per això avancem de 2 en 2.
        for idx in range(0, len(self.ups), 2):

            # Primer fem upsampling amb ConvTranspose2d.
            x = self.ups[idx](x)

            # Recuperem la skip connection corresponent.
            skip_connection = skip_connections[idx // 2]

            # Pot passar que per petits desajustos de mida, x i skip_connection
            # no tinguin exactament la mateixa altura i amplada.
            # Si passa, ajustem x perquè coincideixi amb la skip connection.
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True
                )

            # Concatenem la informació del decoder amb la skip connection.
            # dim=1 és la dimensió dels canals.
            # Això combina informació semàntica profunda amb informació espacial fina.
            x = torch.cat((skip_connection, x), dim=1)

            # Apliquem el bloc DoubleConv després de concatenar.
            x = self.ups[idx + 1](x)

        # Apliquem la convolució final 1x1.
        # Retorna els logits de la màscara.
        return self.final_conv(x)
        
# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):
    """
    Bloc residual tipus ResNet.

    En una convolució normal, la sortida és:
        output = F(x)

    En un bloc residual, la sortida és:
        output = F(x) + shortcut(x)

    Això ajuda a entrenar xarxes més profundes perquè la informació
    pot passar pel camí residual sense perdre's.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Si els canals d'entrada i sortida són diferents,
        # fem una projecció 1x1 perquè es puguin sumar.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out = out + residual
        out = self.relu(out)
        return out


# ============================================================
# ResUNet
# ============================================================

class ResUNet(nn.Module):
    """
    ResUNet 2D per segmentació.

    És igual que una U-Net, però substitueix els DoubleConv
    per blocs residuals.

    Serveix tant per:
        - segmentació binària: out_channels = 1
        - segmentació multiclasse: out_channels = 4

    En el nostre cas multiclasse:
        input  = [B, 4, H, W]
        output = [B, 4, H, W]

    Les 4 classes són:
        0 = background
        1 = necrosis / NCR-NET
        2 = edema
        3 = enhancing tumor
    """

    def __init__(self, in_channels=4, out_channels=4, features=[32, 64, 128, 256]):
        super(ResUNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # -----------------------------
        # Encoder residual
        # -----------------------------
        for feature in features:
            self.downs.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=feature
                )
            )
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -----------------------------
        # Bottleneck residual
        # -----------------------------
        self.bottleneck = ResidualBlock(
            in_channels=features[-1],
            out_channels=features[-1] * 2
        )

        # -----------------------------
        # Decoder residual
        # -----------------------------
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )

            # Després de concatenar skip + upsample tenim feature * 2 canals.
            self.ups.append(
                ResidualBlock(
                    in_channels=feature * 2,
                    out_channels=feature
                )
            )

        # -----------------------------
        # Capa final
        # -----------------------------
        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
