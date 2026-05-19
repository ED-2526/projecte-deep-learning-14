import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate, shift


class BraTSSegmentationDataset(Dataset):
    """
    Dataset PyTorch per BraTS2020.

    Pot treballar en dos modes:
        - binary:     màscara [1, H, W] amb 0=fons i 1=tumor.
        - multiclass: màscara [H, W] amb classes:
              0 = fons
              1 = nucli necròtic / no realçat
              2 = edema
              3 = tumor realçat originalment etiquetat com 4 a BraTS

    Important:
    En multiclasse no retornem la màscara en one-hot. Retornem una màscara
    d'enters perquè nn.CrossEntropyLoss espera targets amb forma [B, H, W].
    """

    BRATS_TO_MULTICLASS = {
        0: 0,
        1: 1,
        2: 2,
        4: 3,
    }

    def __init__(
        self,
        root_dir,
        case_ids=None,
        modalities=None,
        only_tumor_slices=True,
        augment=False,
        rotation_degrees=10,
        shift_pixels=5,
        intensity_jitter=0.10,
        segmentation_type="binary",
    ):
        self.root_dir = root_dir

        if modalities is None:
            modalities = ["flair", "t1", "t1ce", "t2"]
        self.modalities = modalities

        if segmentation_type not in {"binary", "multiclass"}:
            raise ValueError("segmentation_type ha de ser 'binary' o 'multiclass'.")
        self.segmentation_type = segmentation_type

        self.only_tumor_slices = only_tumor_slices
        self.augment = augment
        self.rotation_degrees = rotation_degrees
        self.shift_pixels = shift_pixels
        self.intensity_jitter = intensity_jitter

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
        Crea l'índex de mostres. Cada mostra és (case_id, slice_idx).
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

    def _normalize_slice(self, image_slice):
        image_slice = image_slice.astype(np.float32)
        if image_slice.max() > image_slice.min():
            image_slice = (image_slice - image_slice.min()) / (
                image_slice.max() - image_slice.min()
            )
        return image_slice

    def _map_mask(self, mask):
        """
        Converteix la màscara original de BraTS al format desitjat.
        """
        if self.segmentation_type == "binary":
            return (mask > 0).astype(np.float32)

        mapped = np.zeros_like(mask, dtype=np.int64)
        for original_label, new_label in self.BRATS_TO_MULTICLASS.items():
            mapped[mask == original_label] = new_label
        return mapped

    def _apply_augmentation(self, image, mask):
        """
        Aplica la mateixa transformació geomètrica a imatge i màscara.

        image: [C, H, W]
        mask:
            - binary: [H, W] float
            - multiclass: [H, W] int
        """
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=1)

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=0)

        angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)

        image = np.stack([
            rotate(channel, angle, reshape=False, order=1, mode="nearest")
            for channel in image
        ], axis=0)

        # order=0 és clau: nearest neighbor. Així no inventem classes intermèdies.
        mask = rotate(mask, angle, reshape=False, order=0, mode="nearest")

        shift_y = np.random.uniform(-self.shift_pixels, self.shift_pixels)
        shift_x = np.random.uniform(-self.shift_pixels, self.shift_pixels)

        image = np.stack([
            shift(channel, shift=(shift_y, shift_x), order=1, mode="nearest")
            for channel in image
        ], axis=0)

        mask = shift(mask, shift=(shift_y, shift_x), order=0, mode="nearest")

        factor = np.random.uniform(
            1.0 - self.intensity_jitter,
            1.0 + self.intensity_jitter,
        )
        image = np.clip(image * factor, 0.0, 1.0)

        if self.segmentation_type == "binary":
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.rint(mask).astype(np.int64)
            mask = np.clip(mask, 0, 3)

        return image.copy(), mask.copy()

    def __getitem__(self, idx):
        case_id, slice_idx = self.samples[idx]
        case_path = os.path.join(self.root_dir, case_id)

        image_channels = []
        for modality in self.modalities:
            image_path = os.path.join(case_path, f"{case_id}_{modality}.nii")
            image_volume = nib.load(image_path).get_fdata()
            image_slice = image_volume[:, :, slice_idx]
            image_channels.append(self._normalize_slice(image_slice))

        image = np.stack(image_channels, axis=0)  # [C, H, W]

        seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
        seg_volume = nib.load(seg_path).get_fdata()
        mask = self._map_mask(seg_volume[:, :, slice_idx])

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        image = torch.tensor(image, dtype=torch.float32)

        if self.segmentation_type == "binary":
            mask = np.expand_dims(mask, axis=0)  # [1, H, W]
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            # CrossEntropyLoss espera LongTensor amb forma [H, W].
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
        
