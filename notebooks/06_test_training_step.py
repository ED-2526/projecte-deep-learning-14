import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader, random_split, Subset

from utils.dataset import BraTSSegmentationDataset
from utils.losses import BCEDiceLoss
from models.unet import UNet
from train import train


root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"


device = torch.device("cudxza" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset = BraTSSegmentationDataset(
    root_dir=root_dir,
    modality="flair",
    only_tumor_slices=True
)

# Agafem només una part petita per provar ràpid
small_dataset = Subset(dataset, range(32))

train_size = int(0.8 * len(small_dataset))
val_size = len(small_dataset) - train_size

train_dataset, val_dataset = random_split(
    small_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model = UNet(in_channels=1, out_channels=1).to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

config = {
    "epochs": 1
}

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    config=config,
    device=device,
    save_path="test_best_model.pth"
)

print("Training de prova completat.")
print(history)
