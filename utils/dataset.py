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


# Definim el nostre Dataset personalitzat per segmentació binària de tumors.
# Hereta de torch.utils.data.Dataset perquè PyTorch el pugui utilitzar amb DataLoader.
class BraTSSegmentationDataset(Dataset):

    def __init__(self, root_dir, case_ids=None, modality="flair", only_tumor_slices=True):
        """
        Dataset per a segmentació binària de tumors amb BraTS2020.

        Args:
            root_dir: ruta a la carpeta Training, que conté les carpetes dels pacients.
            case_ids: llista opcional de pacients. Si és None, agafa tots els casos.
            modality: modalitat MRI a utilitzar. Per ara, fem servir "flair".
            only_tumor_slices: si True, només usa slices on hi ha tumor.
        """

        # Guardem la ruta principal del dataset.
        self.root_dir = root_dir

        # Guardem la modalitat que volem utilitzar com a entrada.
        # En el baseline fem servir FLAIR.
        self.modality = modality

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

        # Construïm la ruta a la imatge MRI d'entrada.
        # Com self.modality és "flair", carregarà el fitxer *_flair.nii.
        image_path = os.path.join(case_path, f"{case_id}_{self.modality}.nii")

        # Construïm la ruta a la màscara real de segmentació.
        seg_path = os.path.join(case_path, f"{case_id}_seg.nii")

        # Carreguem el volum complet de la imatge MRI.
        image_volume = nib.load(image_path).get_fdata()

        # Carreguem el volum complet de la màscara.
        seg_volume = nib.load(seg_path).get_fdata()

        # Extraiem la slice 2D de la imatge.
        image = image_volume[:, :, slice_idx]

        # Extraiem la slice 2D corresponent de la màscara.
        mask = seg_volume[:, :, slice_idx]

        # Convertim la màscara original a binària.
        # BraTS té valors 0, 1, 2 i 4.
        # Nosaltres fem:
        #   0 -> 0
        #   1, 2, 4 -> 1
        mask = (mask > 0).astype(np.float32)

        # Convertim la imatge a float32.
        # Això és necessari perquè PyTorch treballa amb tensors decimals.
        image = image.astype(np.float32)

        # Normalitzem la imatge entre 0 i 1.
        # Afegim la condició per evitar dividir per zero si la imatge fos constant.
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # Afegim la dimensió de canal.
        # Abans:
        #   image -> [H, W]
        #   mask  -> [H, W]
        #
        # Després:
        #   image -> [1, H, W]
        #   mask  -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convertim la imatge de NumPy a tensor de PyTorch.
        image = torch.tensor(image, dtype=torch.float32)

        # Convertim la màscara de NumPy a tensor de PyTorch.
        mask = torch.tensor(mask, dtype=torch.float32)

        # Retornem la imatge i la màscara.
        return image, mask
