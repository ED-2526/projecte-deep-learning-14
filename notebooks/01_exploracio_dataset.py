import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Part 1

# Canvia aquesta ruta per la ruta real del teu ordinador
case_path = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_091"

case_id = "BraTS20_Training_091"

flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
t1_path = os.path.join(case_path, f"{case_id}_t1.nii")
t1ce_path = os.path.join(case_path, f"{case_id}_t1ce.nii")
t2_path = os.path.join(case_path, f"{case_id}_t2.nii")
seg_path = os.path.join(case_path, f"{case_id}_seg.nii")


# Carregar els volums
flair = nib.load(flair_path).get_fdata()
t1 = nib.load(t1_path).get_fdata()
t1ce = nib.load(t1ce_path).get_fdata()
t2 = nib.load(t2_path).get_fdata()
seg = nib.load(seg_path).get_fdata()


# Mostrar informació bàsica
print("Shape FLAIR:", flair.shape)
print("Shape T1:", t1.shape)
print("Shape T1CE:", t1ce.shape)
print("Shape T2:", t2.shape)
print("Shape SEG:", seg.shape)

print("Valors únics a la màscara SEG:", np.unique(seg))

print("Valor mínim FLAIR:", np.min(flair))
print("Valor màxim FLAIR:", np.max(flair))


# Part 2

# Triem una slice central
slice_idx = flair.shape[2] // 2

flair_slice = flair[:, :, slice_idx]
seg_slice = seg[:, :, slice_idx]

# Convertim la màscara original a binària
binary_mask = (seg_slice > 0).astype(np.float32)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title(f"FLAIR - slice {slice_idx}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(seg_slice, cmap="gray")
plt.title("Màscara original")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

plt.tight_layout()
plt.show()


# Part 3

# Busquem quines slices tenen tumor
tumor_slices = []

for i in range(seg.shape[2]):
    if np.sum(seg[:, :, i] > 0) > 0:
        tumor_slices.append(i)

print("Nombre de slices amb tumor:", len(tumor_slices))
print("Primeres slices amb tumor:", tumor_slices[:10])

# Agafem una slice del mig de les que tenen tumor
slice_idx = tumor_slices[len(tumor_slices) // 2]

flair_slice = flair[:, :, slice_idx]
seg_slice = seg[:, :, slice_idx]
binary_mask = (seg_slice > 0).astype(np.float32)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title(f"FLAIR - slice {slice_idx}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(seg_slice, cmap="gray")
plt.title("Màscara original")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

plt.tight_layout()
plt.show()


# Part 4

slice_idx = tumor_slices[len(tumor_slices) // 2]

plt.figure(figsize=(18, 6))

plt.subplot(1, 5, 1)
plt.imshow(flair[:, :, slice_idx], cmap="gray")
plt.title("FLAIR")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(t1[:, :, slice_idx], cmap="gray")
plt.title("T1")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(t1ce[:, :, slice_idx], cmap="gray")
plt.title("T1CE")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(t2[:, :, slice_idx], cmap="gray")
plt.title("T2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

plt.tight_layout()
plt.show()
