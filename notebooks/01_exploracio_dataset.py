# Importem os per poder construir rutes de fitxers de manera segura.
# Això evita escriure manualment totes les rutes i fa que el codi sigui més ordenat.
import os

# Importem NumPy perquè els volums mèdics carregats amb nibabel es manipulen com arrays.
# Ens servirà per calcular mínims, màxims, valors únics i per treballar amb màscares.
import numpy as np

# Importem nibabel, que és la llibreria que permet llegir fitxers mèdics en format NIfTI (.nii).
# Les imatges del dataset BraTS2020 estan en aquest format, no en format .jpg o .png.
import nibabel as nib

# Importem matplotlib per visualitzar slices 2D del volum MRI i les seves màscares.
import matplotlib.pyplot as plt


# ============================================================
# PART 1: Carregar un pacient concret del dataset
# ============================================================

# Aquesta ruta apunta a la carpeta d'un pacient concret del dataset BraTS2020.
# Dins d'aquesta carpeta hi ha els fitxers FLAIR, T1, T1CE, T2 i SEG d'aquest pacient.
# En aquest cas triem el pacient BraTS20_Training_091 només per explorar les dades.
case_path = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_091"

# Guardem l'identificador del pacient.
# Aquest identificador s'utilitza perquè els fitxers segueixen un patró de nom:
# BraTS20_Training_091_flair.nii, BraTS20_Training_091_t1.nii, etc.
case_id = "BraTS20_Training_091"

# Construïm la ruta completa al fitxer FLAIR.
# FLAIR és la modalitat que utilitzarem com a entrada principal del baseline.
flair_path = os.path.join(case_path, f"{case_id}_flair.nii")

# Construïm la ruta completa al fitxer T1.
# T1 és una altra modalitat MRI disponible al dataset.
t1_path = os.path.join(case_path, f"{case_id}_t1.nii")

# Construïm la ruta completa al fitxer T1CE.
# T1CE és T1 amb contrast i pot ressaltar algunes parts del tumor.
t1ce_path = os.path.join(case_path, f"{case_id}_t1ce.nii")

# Construïm la ruta completa al fitxer T2.
# T2 és una altra modalitat MRI que també pot contenir informació útil.
t2_path = os.path.join(case_path, f"{case_id}_t2.nii")

# Construïm la ruta completa al fitxer SEG.
# SEG és la màscara real del tumor, és a dir, la resposta correcta que el model haurà d'aprendre.
seg_path = os.path.join(case_path, f"{case_id}_seg.nii")


# ============================================================
# PART 2: Llegir els volums .nii amb nibabel
# ============================================================

# Carreguem el volum FLAIR.
# nib.load(...) obre el fitxer .nii.
# get_fdata() converteix el contingut en un array de NumPy.
flair = nib.load(flair_path).get_fdata()

# Carreguem el volum T1.
t1 = nib.load(t1_path).get_fdata()

# Carreguem el volum T1CE.
t1ce = nib.load(t1ce_path).get_fdata()

# Carreguem el volum T2.
t2 = nib.load(t2_path).get_fdata()

# Carreguem la màscara de segmentació real.
seg = nib.load(seg_path).get_fdata()


# ============================================================
# PART 3: Comprovar formes i valors bàsics
# ============================================================

# Mostrem la forma del volum FLAIR.
# Normalment en BraTS2020 esperem una mida semblant a (240, 240, 155).
# Això vol dir: 240 píxels d'altura, 240 d'amplada i 155 slices.
print("Shape FLAIR:", flair.shape)

# Mostrem la forma del volum T1.
# Ha de coincidir amb FLAIR perquè totes les modalitats han d'estar alineades.
print("Shape T1:", t1.shape)

# Mostrem la forma del volum T1CE.
print("Shape T1CE:", t1ce.shape)

# Mostrem la forma del volum T2.
print("Shape T2:", t2.shape)

# Mostrem la forma de la màscara SEG.
# És molt important que la màscara tingui la mateixa forma que les imatges.
# Així podem comparar cada píxel de la imatge amb el píxel corresponent de la màscara.
print("Shape SEG:", seg.shape)

# Mostrem els valors únics de la màscara.
# En BraTS2020 normalment apareixen els valors 0, 1, 2 i 4.
# 0 significa no tumor.
# 1, 2 i 4 representen diferents parts del tumor.
print("Valors únics a la màscara SEG:", np.unique(seg))

# Mostrem el valor mínim de la imatge FLAIR.
# Això ens ajuda a entendre el rang d'intensitats de la imatge.
print("Valor mínim FLAIR:", np.min(flair))

# Mostrem el valor màxim de la imatge FLAIR.
# Les MRI no tenen necessàriament valors entre 0 i 255, per això després normalitzarem.
print("Valor màxim FLAIR:", np.max(flair))


# ============================================================
# PART 4: Visualitzar una slice central
# ============================================================

# Triem una slice central del volum.
# flair.shape[2] és el nombre total de slices.
# Amb // 2 agafem aproximadament la slice del mig.
slice_idx = flair.shape[2] // 2

# Extraiem la slice 2D de la modalitat FLAIR.
# El volum és 3D: [altura, amplada, slice].
# Amb flair[:, :, slice_idx] agafem tota l'altura, tota l'amplada i només una slice.
flair_slice = flair[:, :, slice_idx]

# Extraiem la slice corresponent de la màscara.
# Ha de ser la mateixa slice perquè imatge i màscara estiguin alineades.
seg_slice = seg[:, :, slice_idx]

# Convertim la màscara original a una màscara binària.
# La màscara original pot tenir valors 0, 1, 2 i 4.
# Nosaltres fem segmentació binària:
#   0 = no tumor
#   1 = tumor
# Per això qualsevol valor superior a 0 passa a ser 1.
binary_mask = (seg_slice > 0).astype(np.float32)


# Creem una figura ampla per posar tres imatges una al costat de l'altra.
plt.figure(figsize=(15, 5))

# Primer subplot: imatge FLAIR.
plt.subplot(1, 3, 1)
plt.imshow(flair_slice, cmap="gray")
plt.title(f"FLAIR - slice {slice_idx}")
plt.axis("off")

# Segon subplot: màscara original amb valors 0, 1, 2 i 4.
plt.subplot(1, 3, 2)
plt.imshow(seg_slice, cmap="gray")
plt.title("Màscara original")
plt.axis("off")

# Tercer subplot: màscara binària, on només distingim tumor i no tumor.
plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

# Ajustem l'espai entre subplots perquè no se solapin.
plt.tight_layout()

# Mostrem la figura.
plt.show()


# ============================================================
# PART 5: Buscar automàticament slices que tenen tumor
# ============================================================

# Creem una llista buida per guardar els índexs de les slices on hi ha tumor.
tumor_slices = []

# Recorrem totes les slices del volum.
for i in range(seg.shape[2]):

    # Comprovem si en aquesta slice hi ha algun píxel amb valor superior a 0.
    # Si seg[:, :, i] > 0 és True en algun píxel, vol dir que hi ha tumor.
    # np.sum(...) compta quants píxels de tumor hi ha.
    if np.sum(seg[:, :, i] > 0) > 0:

        # Si hi ha almenys un píxel de tumor, guardem l'índex de la slice.
        tumor_slices.append(i)

# Mostrem quantes slices tenen tumor.
# Això és útil perquè no totes les slices del cervell contenen tumor.
print("Nombre de slices amb tumor:", len(tumor_slices))

# Mostrem les primeres slices on apareix tumor.
print("Primeres slices amb tumor:", tumor_slices[:10])

# Agafem una slice del mig de les slices que tenen tumor.
# Això és millor que agafar la slice central del volum, perquè garanteix que hi ha tumor visible.
slice_idx = tumor_slices[len(tumor_slices) // 2]

# Extraiem la nova slice FLAIR amb tumor.
flair_slice = flair[:, :, slice_idx]

# Extraiem la màscara corresponent.
seg_slice = seg[:, :, slice_idx]

# Tornem a convertir la màscara a binària.
binary_mask = (seg_slice > 0).astype(np.float32)


# Visualitzem de nou la imatge, la màscara original i la màscara binària.
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


# ============================================================
# PART 6: Visualitzar les quatre modalitats MRI
# ============================================================

# Tornem a seleccionar una slice amb tumor.
# Així podem comparar les modalitats en una zona on el tumor és present.
slice_idx = tumor_slices[len(tumor_slices) // 2]

# Creem una figura amb 5 columnes:
# FLAIR, T1, T1CE, T2 i màscara binària.
plt.figure(figsize=(18, 6))

# Mostrem la modalitat FLAIR.
plt.subplot(1, 5, 1)
plt.imshow(flair[:, :, slice_idx], cmap="gray")
plt.title("FLAIR")
plt.axis("off")

# Mostrem la modalitat T1.
plt.subplot(1, 5, 2)
plt.imshow(t1[:, :, slice_idx], cmap="gray")
plt.title("T1")
plt.axis("off")

# Mostrem la modalitat T1CE.
plt.subplot(1, 5, 3)
plt.imshow(t1ce[:, :, slice_idx], cmap="gray")
plt.title("T1CE")
plt.axis("off")

# Mostrem la modalitat T2.
plt.subplot(1, 5, 4)
plt.imshow(t2[:, :, slice_idx], cmap="gray")
plt.title("T2")
plt.axis("off")

# Mostrem la màscara binària del tumor.
plt.subplot(1, 5, 5)
plt.imshow(binary_mask, cmap="gray")
plt.title("Màscara binària")
plt.axis("off")

plt.tight_layout()
plt.show()
