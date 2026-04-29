"""
Exploració inicial del dataset BraTS2020.

Aquest script carrega un cas concret del dataset BraTS2020, visualitza les
modalitats MRI disponibles i transforma la màscara original de segmentació
en una màscara binària tumor/no tumor.

Objectiu del fitxer:
    1. Comprovar que podem llegir fitxers mèdics en format .nii.
    2. Inspeccionar les dimensions dels volums 3D.
    3. Analitzar els valors de la màscara de segmentació.
    4. Visualitzar slices 2D del volum.
    5. Convertir la màscara multiclasse a una màscara binària.
    6. Comparar visualment les diferents modalitats MRI.
"""

# Llibreria del sistema operatiu.
# Ens permet construir rutes de fitxers de manera segura amb os.path.join().
import os

# Llibreria principal per treballar amb matrius i càlcul numèric.
# En aquest projecte la fem servir per manipular els volums MRI com a arrays.
import numpy as np

# Llibreria per llegir fitxers mèdics en format NIfTI (.nii o .nii.gz).
# BraTS2020 proporciona les imatges en aquest format.
import nibabel as nib

# Llibreria per generar gràfics i visualitzacions.
# La fem servir per mostrar slices 2D de les imatges i de les màscares.
import matplotlib.pyplot as plt






# ======================================================================
# PART 1: DEFINICIÓ DE RUTES I CÀRREGA DELS VOLUMS 3D
# ======================================================================

# Ruta a la carpeta d'un pacient/cas concret del dataset BraTS2020.
# En aquest cas estem explorant el pacient BraTS20_Training_091.
# IMPORTANT: aquesta ruta és local i s'ha d'adaptar a l'ordinador de cada membre del grup.
case_path = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_091"

# Identificador del cas. Aquest nom coincideix amb el prefix dels fitxers dins la carpeta.
# Per exemple: BraTS20_Training_091_flair.nii, BraTS20_Training_091_seg.nii, etc.
case_id = "BraTS20_Training_091"

# Construïm la ruta completa del fitxer FLAIR.
# FLAIR és una modalitat MRI útil per detectar edema i zones alterades del cervell.
flair_path = os.path.join(case_path, f"{case_id}_flair.nii")

# Construïm la ruta completa del fitxer T1.
# T1 aporta informació anatòmica del cervell.
t1_path = os.path.join(case_path, f"{case_id}_t1.nii")

# Construïm la ruta completa del fitxer T1CE.
# T1CE és T1 amb contrast i ajuda a veure zones tumorals que capten contrast.
t1ce_path = os.path.join(case_path, f"{case_id}_t1ce.nii")

# Construïm la ruta completa del fitxer T2.
# T2 ajuda a visualitzar teixits, líquids i alteracions relacionades amb el tumor.
t2_path = os.path.join(case_path, f"{case_id}_t2.nii")

# Construïm la ruta completa del fitxer SEG.
# SEG és la màscara real de segmentació del tumor, és a dir, el target del model.
seg_path = os.path.join(case_path, f"{case_id}_seg.nii")


# Carreguem els fitxers .nii amb nibabel.
# nib.load(path) obre el fitxer NIfTI.
# get_fdata() converteix el volum mèdic en una matriu NumPy 3D.
flair = nib.load(flair_path).get_fdata()
t1 = nib.load(t1_path).get_fdata()
t1ce = nib.load(t1ce_path).get_fdata()
t2 = nib.load(t2_path).get_fdata()
seg = nib.load(seg_path).get_fdata()


# Mostrem les dimensions de cada volum.
# En BraTS2020 és habitual obtenir una forma aproximada de (240, 240, 155):
#   - 240 píxels d'alçada
#   - 240 píxels d'amplada
#   - 155 slices o talls 2D dins del volum 3D
print("Shape FLAIR:", flair.shape)
print("Shape T1:", t1.shape)
print("Shape T1CE:", t1ce.shape)
print("Shape T2:", t2.shape)
print("Shape SEG:", seg.shape)

# Mostrem els valors únics que apareixen a la màscara de segmentació.
# En BraTS, els valors esperats són:
#   0 = fons / no tumor
#   1 = nucli necròtic o part no realçada del tumor
#   2 = edema
#   4 = tumor realçat
print("Valors únics a la màscara SEG:", np.unique(seg))

# Mostrem el valor mínim i màxim de la modalitat FLAIR.
# Això ens ajuda a saber el rang d'intensitats abans de normalitzar les imatges.
print("Valor mínim FLAIR:", np.min(flair))
print("Valor màxim FLAIR:", np.max(flair))


# ======================================================================
# PART 2: VISUALITZACIÓ D'UNA SLICE CENTRAL
# ======================================================================

# Seleccionem una slice central del volum.
# flair.shape[2] és el nombre total de slices del volum.
# Dividint per 2 obtenim aproximadament el tall central.
slice_idx = flair.shape[2] // 2

# Extraiem la mateixa slice de la imatge FLAIR i de la màscara SEG.
# És important agafar el mateix índex perquè imatge i màscara estiguin alineades.
flair_slice = flair[:, :, slice_idx]
seg_slice = seg[:, :, slice_idx]

# Convertim la màscara original multiclasse a una màscara binària.
# Qualsevol valor superior a 0 es considera tumor.
# Això transforma:
#   0       -> 0  (no tumor)
#   1,2,4   -> 1  (tumor)
binary_mask = (seg_slice > 0).astype(np.float32)


# Creem una figura amb tres visualitzacions:
#   1. La slice FLAIR original.
#   2. La màscara original amb les classes 0, 1, 2 i 4.
#   3. La màscara binària tumor/no tumor.
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


# ======================================================================
# PART 3: CERCA DE SLICES QUE CONTENEN TUMOR
# ======================================================================

# Creem una llista buida on guardarem els índexs de les slices que tenen tumor.
tumor_slices = []

# Recorrem totes les slices del volum de segmentació.
# Per cada slice, comprovem si hi ha algun píxel amb valor superior a 0.
# Si n'hi ha, aquella slice conté alguna regió tumoral.
for i in range(seg.shape[2]):
    if np.sum(seg[:, :, i] > 0) > 0:
        tumor_slices.append(i)

# Mostrem quantes slices del pacient tenen tumor i els primers índexs trobats.
# Això ens ajuda a entendre on apareix el tumor dins del volum.
print("Nombre de slices amb tumor:", len(tumor_slices))
print("Primeres slices amb tumor:", tumor_slices[:10])

# Seleccionem una slice situada aproximadament al mig de totes les slices amb tumor.
# Aquesta selecció sol mostrar millor la regió tumoral que una slice central genèrica.
slice_idx = tumor_slices[len(tumor_slices) // 2]

# Tornem a extreure la slice FLAIR i la màscara corresponent.
flair_slice = flair[:, :, slice_idx]
seg_slice = seg[:, :, slice_idx]

# Tornem a convertir la màscara d'aquesta slice a format binari.
binary_mask = (seg_slice > 0).astype(np.float32)

# Visualitzem la slice amb tumor, la seva màscara original i la màscara binària.
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


# ======================================================================
# PART 4: COMPARACIÓ VISUAL DE LES QUATRE MODALITATS MRI
# ======================================================================

# Tornem a utilitzar una slice amb tumor per comparar les modalitats MRI.
slice_idx = tumor_slices[len(tumor_slices)  // 2]

# Creem una figura amb cinc columnes:
#   1. FLAIR
#   2. T1
#   3. T1CE
#   4. T2
#   5. Màscara binària
# Aquesta comparació ajuda a veure que cada modalitat aporta informació diferent.
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
