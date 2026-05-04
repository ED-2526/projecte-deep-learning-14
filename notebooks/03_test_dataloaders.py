import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from torch.utils.data import random_split, DataLoader
from utils.dataset import BraTSSegmentationDataset


root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

batch_size = 4

dataset = BraTSSegmentationDataset(
    root_dir=root_dir,
    modality="flair",
    only_tumor_slices=True
)

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("Total mostres:", total_size)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

images, masks = next(iter(train_loader))

print("Batch images:", images.shape)
print("Batch masks:", masks.shape)
print("Valor mínim images:", images.min().item())
print("Valor màxim images:", images.max().item())
print("Valors únics masks:", masks.unique())
