# Importem os per treballar amb rutes de fitxers i carpetes.
# Ens permet unir rutes amb os.path.join i llistar carpetes amb os.listdir.
import os

# Importem NumPy perquè les imatges i màscares carregades amb nibabel
# es manipulen inicialment com arrays de NumPy.
import numpy as np

# Importem nibabel perquè el dataset BraTS2020 està en format NIfTI (.nii).
# Aquesta llibreria permet llegir els volums mèdics 3D.
import nibabel as nib

# Importem PyTorch perquè al final volem retornar tensors.
import torch

# Importem Dataset, que és la classe base de PyTorch per crear datasets personalitzats.
from torch.utils.data import Dataset

from scipy.ndimage import rotate, shift

# Definim el nostre Dataset personalitzat per segmentació binària de tumors.
# Hereta de torch.utils.data.Dataset perquè PyTorch el pugui utilitzar amb DataLoader.
class BraTSSegmentationDataset(Dataset):
    def __init__(self,root_dir,case_ids=None,modalities=None,only_tumor_slices=True,augment=False,rotation_degrees=10,shift_pixels=5,intensity_jitter=0.10):
        """
        Dataset per a segmentació binària de tumors amb BraTS2020.

        Args:
            root_dir: ruta a la carpeta Training, que conté les carpetes dels pacients.
            case_ids: llista opcional de pacients. Si és None, agafa tots els casos.
            modality: modalitat MRI a utilitzar. Per ara, fem servir totes.
            only_tumor_slices: si True, només usa slices on hi ha tumor.
        """

        # Guardem la ruta principal del dataset.
        self.root_dir = root_dir

        # Guardem les modalitats MRI que volem utilitzar com a canals d'entrada.
        # Si no se n'indica cap, fem servir les quatre modalitats de BraTS.
        if modalities is None:
            modalities = ["flair", "t1", "t1ce", "t2"]
        
        self.modalities = modalities

        # Guardem si volem només slices amb tumor o totes les slices.
        self.only_tumor_slices = only_tumor_slices

        # Si no ens passen una llista concreta de pacients,
        # agafem automàticament tots els pacients de root_dir.
        if case_ids is None:

            # os.listdir(root_dir) retorna tots els elements de la carpeta.
            # Ens quedem només amb els que comencen per "BraTS20_Training_".
            # sorted(...) assegura que l'ordre sigui estable i reproduïble.
            self.case_ids = sorted([
                folder for folder in os.listdir(root_dir)
                if folder.startswith("BraTS20_Training_")
            ])

        # Si ens passen una llista de pacients, utilitzem només aquests.
        # Això és clau per poder fer split per pacient.
        else:
            self.case_ids = case_ids

        # Aquesta llista guardarà totes les mostres disponibles.
        # Cada mostra serà una parella: (case_id, slice_idx).
        self.samples = []

        # Construïm l'índex de mostres.
        self._build_index()

        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.shift_pixels = shift_pixels
        self.intensity_jitter = intensity_jitter

    def _apply_augmentation(self, image, mask):
        """
        Aplica data augmentation a una imatge multicanal i la seva màscara.
    
        image té forma [C, H, W]
        mask té forma [H, W]
    
        Les transformacions geomètriques s'apliquen igual a tots els canals i a la màscara.
        Les transformacions d'intensitat només s'apliquen als canals de la imatge.
        """
    
        # -----------------------------
        # Flip horitzontal aleatori
        # -----------------------------
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=1)
    
        # -----------------------------
        # Flip vertical aleatori
        # -----------------------------
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=0)
    
        # -----------------------------
        # Rotació petita aleatòria
        # -----------------------------
        angle = np.random.uniform(
            -self.rotation_degrees,
            self.rotation_degrees
        )
    
        rotated_channels = []
    
        for c in range(image.shape[0]):
            rotated_channel = rotate(
                image[c],
                angle,
                reshape=False,
                order=1,
                mode="nearest"
            )
            rotated_channels.append(rotated_channel)
    
        image = np.stack(rotated_channels, axis=0)
    
        mask = rotate(
            mask,
            angle,
            reshape=False,
            order=0,
            mode="nearest"
        )
    
        # -----------------------------
        # Petit desplaçament aleatori
        # -----------------------------
        shift_y = np.random.uniform(-self.shift_pixels, self.shift_pixels)
        shift_x = np.random.uniform(-self.shift_pixels, self.shift_pixels)
    
        shifted_channels = []
    
        for c in range(image.shape[0]):
            shifted_channel = shift(
                image[c],
                shift=(shift_y, shift_x),
                order=1,
                mode="nearest"
            )
            shifted_channels.append(shifted_channel)
    
        image = np.stack(shifted_channels, axis=0)
    
        mask = shift(
            mask,
            shift=(shift_y, shift_x),
            order=0,
            mode="nearest"
        )
    
        # -----------------------------
        # Canvi suau d'intensitat
        # Només afecta la imatge, no la màscara
        # -----------------------------
        factor = np.random.uniform(
            1.0 - self.intensity_jitter,
            1.0 + self.intensity_jitter
        )
    
        image = image * factor
        image = np.clip(image, 0.0, 1.0)
    
        # Ens assegurem que la màscara continua sent binària
        mask = (mask > 0.5).astype(np.float32)
    
        return image.copy(), mask.copy()

    def _build_index(self):
        """
        Crea una llista amb tots els exemples disponibles.
        Cada exemple és una parella:
            (case_id, slice_idx)
        """

        # Recorrem tots els pacients que formen part d'aquest Dataset.
        for case_id in self.case_ids:

            # Construïm la ruta a la carpeta del pacient.
            case_path = os.path.join(self.root_dir, case_id)

            # Construïm la ruta al fitxer de segmentació real.
            # Aquest fitxer conté la màscara del tumor.
            seg_path = os.path.join(case_path, f"{case_id}_seg.nii")

            # Si no existeix el fitxer de segmentació, saltem aquest pacient.
            # Això és important perquè la validació oficial de BraTS no té seg.nii.
            if not os.path.exists(seg_path):
                continue

            # Carreguem la màscara 3D del pacient.
            # get_fdata() la converteix en un array de NumPy.
            seg = nib.load(seg_path).get_fdata()

            # Recorrem totes les slices del volum.
            # seg.shape[2] és el nombre de slices del volum 3D.
            for slice_idx in range(seg.shape[2]):

                # Extraiem la màscara 2D corresponent a aquesta slice.
                mask_slice = seg[:, :, slice_idx]

                # Si només volem slices amb tumor, comprovem si hi ha algun píxel tumoral.
                if self.only_tumor_slices:

                    # mask_slice > 0 detecta qualsevol classe tumoral.
                    # np.sum(...) compta quants píxels tumorals hi ha.
                    # Si és més gran que 0, afegim aquesta slice.
                    if np.sum(mask_slice > 0) > 0:
                        self.samples.append((case_id, slice_idx))

                # Si no filtrem per tumor, afegim totes les slices.
                else:
                    self.samples.append((case_id, slice_idx))

        # Mostrem quantes slices s'han afegit al Dataset.
        print(f"Dataset creat amb {len(self.samples)} slices.")


    def __len__(self):
        """
        Retorna el nombre total de mostres del Dataset.
        PyTorch ho utilitza per saber la mida del Dataset.
        """

        return len(self.samples)


    def __getitem__(self, idx):
        """
        Retorna una mostra concreta del Dataset.

        Args:
            idx: índex de la mostra.

        Returns:
            image: tensor de la imatge MRI amb forma [1, H, W].
            mask: tensor de la màscara binària amb forma [1, H, W].
        """

        # Recuperem quin pacient i quina slice corresponen a aquest índex.
        case_id, slice_idx = self.samples[idx]

        # Construïm la ruta a la carpeta del pacient.
        case_path = os.path.join(self.root_dir, case_id)

        # Construïm la ruta a la màscara real de segmentació.
        seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
        
        # Carreguem el volum complet de la màscara.
        seg_volume = nib.load(seg_path).get_fdata()
        
        # Carreguem totes les modalitats MRI i les apilem com a canals.
        image_channels = []
        
        for modality in self.modalities:
            image_path = os.path.join(case_path, f"{case_id}_{modality}.nii")
        
            image_volume = nib.load(image_path).get_fdata()
        
            # Extraiem la slice 2D d'aquesta modalitat.
            image_slice = image_volume[:, :, slice_idx]
        
            # Convertim a float32.
            image_slice = image_slice.astype(np.float32)
        
            # Normalitzem cada modalitat per separat entre 0 i 1.
            if image_slice.max() > image_slice.min():
                image_slice = (image_slice - image_slice.min()) / (
                    image_slice.max() - image_slice.min()
                )
        
            image_channels.append(image_slice)
        
        # Convertim la llista de modalitats en un array multicanal.
        # Forma final: [C, H, W], on C = nombre de modalitats.
        image = np.stack(image_channels, axis=0)

        # Extraiem la slice 2D corresponent de la màscara.
        mask = seg_volume[:, :, slice_idx]

        # Convertim la màscara original a binària.
        # BraTS té valors 0, 1, 2 i 4.
        # Nosaltres fem:
        #   0 -> 0
        #   1, 2, 4 -> 1
        mask = (mask > 0).astype(np.float32)

        # Apliquem data augmentation només si aquest Dataset ho té activat.
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # La imatge ja té forma [C, H, W], per tant no cal afegir canal.
        # La màscara sí que necessita canal: [H, W] -> [1, H, W]
        mask = np.expand_dims(mask, axis=0)

        # Convertim la imatge de NumPy a tensor de PyTorch.
        image = torch.tensor(image, dtype=torch.float32)

        # Convertim la màscara de NumPy a tensor de PyTorch.
        mask = torch.tensor(mask, dtype=torch.float32)

        # Retornem la imatge i la màscara.
        return image, mask
