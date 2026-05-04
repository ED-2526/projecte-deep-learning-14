import matplotlib.pyplot as plt
from utils.dataset import BraTSSegmentationDataset


root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

dataset = BraTSSegmentationDataset(
    root_dir=root_dir,
    modality="flair",
    only_tumor_slices=True
)

print("Nombre total de mostres:", len(dataset))

image, mask = dataset[0]

print("Shape image:", image.shape)
print("Shape mask:", mask.shape)
print("Valor mínim image:", image.min().item())
print("Valor màxim image:", image.max().item())
print("Valors únics mask:", mask.unique())


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[0], cmap="gray")
plt.title("Imatge FLAIR")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask[0], cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

plt.tight_layout()
plt.show()
