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
