import os
import random
import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader, random_split

from utils.dataset import BraTSSegmentationDataset
from utils.losses import BCEDiceLoss
from models.unet import UNet
from train import train


# -----------------------------
# Reproductibilitat
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Crear datasets i dataloaders
# -----------------------------
def create_dataloaders(config):
    dataset = BraTSSegmentationDataset(
        root_dir=config["root_dir"],
        modality=config["modality"],
        only_tumor_slices=config["only_tumor_slices"]
    )

    total_size = len(dataset)

    train_size = int(config["train_split"] * total_size)
    val_size = int(config["val_split"] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    print("Total mostres:", total_size)
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    return train_loader, val_loader, test_loader


# -----------------------------
# Pipeline principal
# -----------------------------
def model_pipeline(config):
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(config["models_dir"], exist_ok=True)

    # Iniciar wandb en mode offline o online segons configuració
    os.environ["WANDB_MODE"] = config["wandb_mode"]

    with wandb.init(
        project=config["wandb_project"],
        config=config
    ):
        train_loader, val_loader, test_loader = create_dataloaders(config)

        model = UNet(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"]
        ).to(device)

        criterion = BCEDiceLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"]
        )

        save_path = os.path.join(config["models_dir"], config["model_name"])

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=device,
            save_path=save_path
        )

    return history


# -----------------------------
# Configuració del projecte
# -----------------------------
if __name__ == "__main__":

    config = {
        # Dataset
        "root_dir": "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        "modality": "flair",
        "only_tumor_slices": True,

        # Splits
        "train_split": 0.8,
        "val_split": 0.1,

        # Model
        "in_channels": 1,
        "out_channels": 1,

        # Training
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_workers": 2,

        # Reproductibilitat
        "seed": 42,

        # Guardar models
        "models_dir": "results/models",
        "model_name": "unet_flair_baseline.pth",

        # Wandb
        "wandb_project": "brats2020-tumor-segmentation",
        "wandb_mode": "offline"
    }

    history = model_pipeline(config)

    print("Entrenament finalitzat.")
    print(history)
