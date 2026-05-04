import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BraTSSegmentationDataset(Dataset):
    def __init__(self, root_dir, case_ids=None, modality="flair", only_tumor_slices=True):
        """
        Dataset per a segmentació binària de tumors amb BraTS2020.

        Args:
            root_dir: ruta a la carpeta Training, que conté les carpetes dels pacients.
            case_ids: llista opcional de pacients. Si és None, agafa tots els casos.
            modality: modalitat MRI a utilitzar. Per ara: "flair".
            only_tumor_slices: si True, només usa slices on hi ha tumor.
        """
        self.root_dir = root_dir
        self.modality = modality
        self.only_tumor_slices = only_tumor_slices

        if case_ids is None:
            self.case_ids = sorted([
                folder for folder in os.listdir(root_dir)
                if folder.startswith("BraTS20_Training_")
            ])
        else:
            self.case_ids = case_ids

        self.samples = []
        self._build_index()

    def _build_index(self):
        """
        Crea una llista amb tots els exemples disponibles.
        Cada exemple és: (case_id, slice_idx)
        """
        for case_id in self.case_ids:
            case_path = os.path.join(self.root_dir, case_id)
            seg_path = os.path.join(case_path, f"{case_id}_seg.nii")

            if not os.path.exists(seg_path):
                continue

            seg = nib.load(seg_path).get_fdata()

            for slice_idx in range(seg.shape[2]):
                mask_slice = seg[:, :, slice_idx]

                if self.only_tumor_slices:
                    if np.sum(mask_slice > 0) > 0:
                        self.samples.append((case_id, slice_idx))
                else:
                    self.samples.append((case_id, slice_idx))

        print(f"Dataset creat amb {len(self.samples)} slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, slice_idx = self.samples[idx]
        case_path = os.path.join(self.root_dir, case_id)

        image_path = os.path.join(case_path, f"{case_id}_{self.modality}.nii")
        seg_path = os.path.join(case_path, f"{case_id}_seg.nii")

        image_volume = nib.load(image_path).get_fdata()
        seg_volume = nib.load(seg_path).get_fdata()

        image = image_volume[:, :, slice_idx]
        mask = seg_volume[:, :, slice_idx]

        # Convertir màscara a binària
        mask = (mask > 0).astype(np.float32)

        # Normalitzar imatge
        image = image.astype(np.float32)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # Afegir dimensió de canal: [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
