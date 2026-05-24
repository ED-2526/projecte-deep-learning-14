import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓ
# ============================================================

# Canvia aquesta ruta si cal.
# Ha d'apuntar a la carpeta que conté els pacients BraTS20_Training_xxx
DATA_ROOT = "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData"

# Pacient que vols visualitzar.
# Pots canviar-lo per qualsevol pacient existent.
CASE_ID = "BraTS20_Training_091"

# Carpeta on es guardaran les figures
OUT_DIR = "results/figures_dataset"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# FUNCIONS AUXILIARS
# ============================================================

def load_nifti(path):
    """
    Carrega un fitxer .nii o .nii.gz i retorna el volum com a array NumPy.
    """
    return nib.load(path).get_fdata()


def normalize_slice(slice_2d):
    """
    Normalitza una slice entre 0 i 1 per visualitzar-la millor.
    Això no modifica la informació de la màscara, només millora la visualització.
    """
    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val == 0:
        return slice_2d

    return (slice_2d - min_val) / (max_val - min_val)


def find_best_tumor_slice(seg):
    """
    Busca la slice amb més píxels de tumor.
    Això és útil perquè així la figura mostra clarament el tumor.
    """
    tumor_pixels_per_slice = []

    for i in range(seg.shape[2]):
        tumor_pixels = np.sum(seg[:, :, i] > 0)
        tumor_pixels_per_slice.append(tumor_pixels)

    best_slice = int(np.argmax(tumor_pixels_per_slice))
    return best_slice


# ============================================================
# CÀRREGA DEL PACIENT
# ============================================================

case_path = os.path.join(DATA_ROOT, CASE_ID)

flair_path = os.path.join(case_path, f"{CASE_ID}_flair.nii")
t1_path = os.path.join(case_path, f"{CASE_ID}_t1.nii")
t1ce_path = os.path.join(case_path, f"{CASE_ID}_t1ce.nii")
t2_path = os.path.join(case_path, f"{CASE_ID}_t2.nii")
seg_path = os.path.join(case_path, f"{CASE_ID}_seg.nii")

flair = load_nifti(flair_path)
t1 = load_nifti(t1_path)
t1ce = load_nifti(t1ce_path)
t2 = load_nifti(t2_path)
seg = load_nifti(seg_path)

print("Shapes:")
print("FLAIR:", flair.shape)
print("T1:", t1.shape)
print("T1CE:", t1ce.shape)
print("T2:", t2.shape)
print("SEG:", seg.shape)
print("Valors únics SEG:", np.unique(seg))


# ============================================================
# SELECCIÓ DE LA SLICE AMB MÉS TUMOR
# ============================================================

slice_idx = find_best_tumor_slice(seg)
print(f"Slice seleccionada amb més tumor: {slice_idx}")

flair_slice = normalize_slice(flair[:, :, slice_idx])
t1_slice = normalize_slice(t1[:, :, slice_idx])
t1ce_slice = normalize_slice(t1ce[:, :, slice_idx])
t2_slice = normalize_slice(t2[:, :, slice_idx])

seg_slice = seg[:, :, slice_idx]

# Màscara binària: qualsevol valor > 0 és tumor
binary_mask = (seg_slice > 0).astype(np.float32)


# ============================================================
# FIGURA 1:
# FLAIR | T1 | T1CE | T2 | MÀSCARA BINÀRIA
# ============================================================

plt.figure(figsize=(18, 5))

plt.subplot(1, 5, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title("FLAIR", fontsize=14)
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(t1_slice, cmap="gray")
plt.title("T1", fontsize=14)
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(t1ce_slice, cmap="gray")
plt.title("T1CE", fontsize=14)
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(t2_slice, cmap="gray")
plt.title("T2", fontsize=14)
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària", fontsize=14)
plt.axis("off")

plt.suptitle(
    f"Modalitats MRI i màscara binària - {CASE_ID} - Slice {slice_idx}",
    fontsize=16
)

plt.tight_layout()

output_path_1 = os.path.join(OUT_DIR, "modalitats_mri_i_mascara_binaria.png")
plt.savefig(output_path_1, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figura guardada a: {output_path_1}")


# ============================================================
# FIGURA 2:
# MÀSCARA ORIGINAL MULTICLASSE → MÀSCARA BINÀRIA
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(seg_slice, cmap="viridis")
plt.title("Màscara original multiclasse\n0, 1, 2, 4", fontsize=14)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària\n0 = fons, 1 = tumor", fontsize=14)
plt.axis("off")

plt.suptitle(
    f"Conversió de màscara multiclasse a binària - Slice {slice_idx}",
    fontsize=16
)

plt.tight_layout()

output_path_2 = os.path.join(OUT_DIR, "conversio_mascara_multiclasse_a_binaria.png")
plt.savefig(output_path_2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figura guardada a: {output_path_2}")


# ============================================================
# FIGURA 3 OPCIONAL:
# FLAIR + MÀSCARA SOBREPOSADA
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(flair_slice, cmap="gray")
plt.imshow(binary_mask, cmap="Reds", alpha=0.35)
plt.title("FLAIR amb màscara tumoral sobreposada", fontsize=14)
plt.axis("off")

output_path_3 = os.path.join(OUT_DIR, "flair_amb_overlay_mascara.png")
plt.savefig(output_path_3, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figura guardada a: {output_path_3}")
