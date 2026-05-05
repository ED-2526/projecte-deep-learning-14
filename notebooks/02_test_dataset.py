# Importem matplotlib per poder visualitzar una imatge i la seva màscara.
import matplotlib.pyplot as plt

# Importem el nostre Dataset personalitzat.
# Aquesta classe està definida a utils/dataset.py.
# S'encarrega de llegir BraTS2020 i retornar parelles:
#   imatge MRI 2D + màscara binària 2D
from utils.dataset import BraTSSegmentationDataset


# Ruta principal del dataset Training de BraTS2020.
# Aquesta carpeta ha de contenir subcarpetes del tipus:
# BraTS20_Training_001, BraTS20_Training_002, etc.
root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"


# Creem una instància del nostre Dataset.
dataset = BraTSSegmentationDataset(

    # root_dir indica on es troben les carpetes dels pacients.
    root_dir=root_dir,

    # modality="flair" indica que només utilitzarem la modalitat FLAIR com a entrada.
    # Això vol dir que cada imatge tindrà 1 canal.
    modality="flair",

    # only_tumor_slices=True fa que el Dataset només inclogui slices on hi ha tumor.
    # Això és útil per al primer baseline, perquè evita entrenar amb massa slices buides.
    only_tumor_slices=True
)


# Mostrem el nombre total de mostres del Dataset.
# Cada mostra és una slice 2D, no un pacient complet.
print("Nombre total de mostres:", len(dataset))


# Agafem la primera mostra del Dataset.
# El Dataset retorna dues coses:
#   image = slice FLAIR normalitzada
#   mask = màscara binària corresponent
image, mask = dataset[0]


# Mostrem la forma de la imatge.
# Esperem torch.Size([1, 240, 240]):
#   1 = canal FLAIR
#   240 = altura
#   240 = amplada
print("Shape image:", image.shape)

# Mostrem la forma de la màscara.
# Ha de tenir la mateixa forma que la imatge:
# torch.Size([1, 240, 240])
print("Shape mask:", mask.shape)

# Mostrem el valor mínim de la imatge.
# Com el Dataset normalitza la imatge, hauria d'estar aproximadament entre 0 i 1.
print("Valor mínim image:", image.min().item())

# Mostrem el valor màxim de la imatge.
# Normalment esperem un valor màxim proper a 1.
print("Valor màxim image:", image.max().item())

# Mostrem els valors únics de la màscara.
# Esperem tensor([0., 1.]) perquè hem convertit la segmentació a binària.
print("Valors únics mask:", mask.unique())


# Creem una figura amb dues columnes:
# una per la imatge i una per la màscara.
plt.figure(figsize=(10, 4))

# Mostrem la imatge FLAIR.
# image té forma [1, 240, 240].
# Per visualitzar-la amb matplotlib traiem el canal amb image[0].
plt.subplot(1, 2, 1)
plt.imshow(image[0], cmap="gray")
plt.title("Imatge FLAIR")
plt.axis("off")

# Mostrem la màscara binària.
# mask[0] treu també la dimensió del canal.
plt.subplot(1, 2, 2)
plt.imshow(mask[0], cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

# Ajustem els espais de la figura.
plt.tight_layout()

# Mostrem la visualització.
plt.show()
