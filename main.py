import os
import random
import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader

from utils.dataset import BraTSSegmentationDataset
from utils.losses import BCEDiceLoss
from models.unet import UNet
from train import train, validate_one_epoch

# -----------------------------
# Reproductibilitat
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Obtenir tots els pacients
# -----------------------------
def get_case_ids(root_dir):
    """
    Retorna la llista ordenada de pacients disponibles al dataset BraTS.

    Cada pacient és una carpeta del tipus:
    BraTS20_Training_001, BraTS20_Training_002, etc.
    """
    case_ids = sorted([
        folder for folder in os.listdir(root_dir)
        if folder.startswith("BraTS20_Training_")
    ])

    return case_ids


# -----------------------------
# Dividir pacients en train/val/test
# -----------------------------
def split_case_ids(case_ids, train_split, val_split, seed):
    """
    Divideix els pacients en train, validation i test.

    Important:
    Aquesta divisió es fa a nivell de pacient, no a nivell de slice.
    Això evita que slices del mateix pacient apareguin en més d'un conjunt.
    """
    case_ids = list(case_ids)

    rng = random.Random(seed)
    rng.shuffle(case_ids)

    total_cases = len(case_ids)

    train_size = int(train_split * total_cases)
    val_size = int(val_split * total_cases)

    train_case_ids = case_ids[:train_size]
    val_case_ids = case_ids[train_size:train_size + val_size]
    test_case_ids = case_ids[train_size + val_size:]

    # Comprovació de seguretat: cap pacient pot aparèixer en més d'un split
    assert set(train_case_ids).isdisjoint(set(val_case_ids))
    assert set(train_case_ids).isdisjoint(set(test_case_ids))
    assert set(val_case_ids).isdisjoint(set(test_case_ids))

    return train_case_ids, val_case_ids, test_case_ids


# -----------------------------
# Crear datasets i dataloaders
# -----------------------------
def create_dataloaders(config):
    """
    Crea els DataLoaders fent primer una divisió per pacients.

    Abans:
        Es dividien slices individuals amb random_split.
        Això podia provocar data leakage.

    Ara:
        1. Obtenim tots els pacients.
        2. Dividim pacients en train/val/test.
        3. Creem un dataset separat per a cada grup de pacients.
    """
    case_ids = get_case_ids(config["root_dir"])

    train_case_ids, val_case_ids, test_case_ids = split_case_ids(
        case_ids=case_ids,
        train_split=config["train_split"],
        val_split=config["val_split"],
        seed=config["seed"]
    )

    train_dataset = BraTSSegmentationDataset(
        root_dir=config["root_dir"],
        case_ids=train_case_ids,
        modality=config["modality"],
        only_tumor_slices=config["only_tumor_slices"]
    )

    val_dataset = BraTSSegmentationDataset(
        root_dir=config["root_dir"],
        case_ids=val_case_ids,
        modality=config["modality"],
        only_tumor_slices=config["only_tumor_slices"]
    )

    test_dataset = BraTSSegmentationDataset(
        root_dir=config["root_dir"],
        case_ids=test_case_ids,
        modality=config["modality"],
        only_tumor_slices=config["only_tumor_slices"]
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

    print("\nSplit per pacients:")
    print("Total pacients:", len(case_ids))
    print("Pacients train:", len(train_case_ids))
    print("Pacients validation:", len(val_case_ids))
    print("Pacients test:", len(test_case_ids))

    print(
        f"Percentatges pacients: "
        f"train={len(train_case_ids) / len(case_ids):.2%}, "
        f"val={len(val_case_ids) / len(case_ids):.2%}, "
        f"test={len(test_case_ids) / len(case_ids):.2%}"
    )

    print("\nSplit per slices generades:")
    print("Slices train:", len(train_dataset))
    print("Slices validation:", len(val_dataset))
    print("Slices test:", len(test_dataset))

    print("\nExemples de pacients:")
    print("Train:", train_case_ids[:5])
    print("Validation:", val_case_ids[:5])
    print("Test:", test_case_ids[:5])

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

        # -----------------------------
        # Avaluació final en test
        # -----------------------------
        model.load_state_dict(torch.load(save_path, map_location=device))

        test_loss, test_dice, test_iou = validate_one_epoch(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device
        )

        print("\nResultats finals en test:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Dice: {test_dice:.4f}")
        print(f"Test IoU:  {test_iou:.4f}")

        history["test_loss"] = test_loss
        history["test_dice"] = test_dice
        history["test_iou"] = test_iou

        try:
            wandb.log({
                "test_loss": test_loss,
                "test_dice": test_dice,
                "test_iou": test_iou
            })
        except Exception:
            pass

    return history


# -----------------------------
# Configuració del projecte
# -----------------------------
if __name__ == "__main__":

    config = {
        # Dataset
        "root_dir": "/home/edxnG14/laia/data/data",
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
        "model_name": "unet_flair_patient_split.pth",

        # Wandb
        "wandb_project": "brats2020-tumor-segmentation",
        "wandb_mode": "offline"
    }

    history = model_pipeline(config)

    print("Entrenament finalitzat.")
    print(history)
