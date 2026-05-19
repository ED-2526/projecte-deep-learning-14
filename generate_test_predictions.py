import os
import random
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataset import BraTSSegmentationDataset


# ============================================================
# CONFIGURACIÓ
# ============================================================

CONFIG = {
    "root_dir": "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData",

    "model_path": "results/models/unet_multimodal_patient_split_20epochs_all_slices.pth",

    "output_dir": "results/predictions_multimodal_all_slices",

    "modalities": ["flair", "t1", "t1ce", "t2"],
    "in_channels": 4,
    "out_channels": 1,

    "segmentation_type": "binary",
    "only_tumor_slices": False,

    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,

    "batch_size": 1,
    "num_workers": 2,

    "num_predictions": 12,
    "threshold": 0.5,
}


# ============================================================
# SPLIT PER PACIENT
# ============================================================

def get_case_ids(root_dir):
    """
    Retorna tots els pacients del dataset BraTS.
    """
    case_ids = sorted([
        folder for folder in os.listdir(root_dir)
        if folder.startswith("BraTS20_Training_")
    ])

    if len(case_ids) == 0:
        raise RuntimeError(f"No s'han trobat pacients a: {root_dir}")

    return case_ids


def split_case_ids(case_ids, train_split, val_split, seed):
    """
    Divideix els pacients en train, validation i test.
    Ha de ser igual que al main.py per recuperar el mateix test set.
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

    assert set(train_case_ids).isdisjoint(set(val_case_ids))
    assert set(train_case_ids).isdisjoint(set(test_case_ids))
    assert set(val_case_ids).isdisjoint(set(test_case_ids))

    return train_case_ids, val_case_ids, test_case_ids


# ============================================================
# VISUALITZACIÓ
# ============================================================

def save_prediction_figure(image, mask_true, mask_pred, save_path, title=""):
    """
    Guarda una figura amb:
    - MRI
    - màscara real
    - màscara predita
    - overlay de predicció sobre MRI
    """

    # image shape: [4, H, W]
    # Per visualitzar, fem servir el canal FLAIR, que és el primer.
    flair = image[0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title("MRI FLAIR")
    axes[0].axis("off")

    axes[1].imshow(mask_true, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(flair, cmap="gray")
    axes[3].imshow(mask_pred, alpha=0.4, cmap="Reds")
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty-slices",
        action="store_true",
        help="Si s'activa, també guarda prediccions de slices sense tumor real."
    )
    args = parser.parse_args()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------------------------------------------
    # Reconstruir el mateix split de main.py
    # ------------------------------------------------------------
    case_ids = get_case_ids(CONFIG["root_dir"])

    train_case_ids, val_case_ids, test_case_ids = split_case_ids(
        case_ids=case_ids,
        train_split=CONFIG["train_split"],
        val_split=CONFIG["val_split"],
        seed=CONFIG["seed"],
    )

    print("\nSplit per pacients:")
    print("Total pacients:", len(case_ids))
    print("Pacients train:", len(train_case_ids))
    print("Pacients validation:", len(val_case_ids))
    print("Pacients test:", len(test_case_ids))
    print("Exemples test:", test_case_ids[:5])

    # ------------------------------------------------------------
    # Crear dataset de test
    # ------------------------------------------------------------
    test_dataset = BraTSSegmentationDataset(
        root_dir=CONFIG["root_dir"],
        case_ids=test_case_ids,
        modalities=CONFIG["modalities"],
        modality=None,
        segmentation_type=CONFIG["segmentation_type"],
        only_tumor_slices=CONFIG["only_tumor_slices"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    print("\nSlices test:", len(test_dataset))

    # ------------------------------------------------------------
    # Carregar model
    # ------------------------------------------------------------
    model = UNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
    ).to(device)

    if not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError(f"No s'ha trobat el model: {CONFIG['model_path']}")

    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    print("\nModel carregat:")
    print(CONFIG["model_path"])

    # ------------------------------------------------------------
    # Generar prediccions
    # ------------------------------------------------------------
    saved = 0

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, masks = batch

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > CONFIG["threshold"]).float()

            # Passem a CPU i traiem batch dimension
            image_np = images[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

            # En binari, la màscara pot venir com [1, H, W]
            if mask_np.ndim == 3:
                mask_np = mask_np[0]

            if pred_np.ndim == 3:
                pred_np = pred_np[0]

            # Per defecte saltem slices buides, perquè visualment no aporten gaire.
            if not args.allow_empty_slices:
                if np.sum(mask_np > 0) == 0:
                    continue

            save_path = os.path.join(
                CONFIG["output_dir"],
                f"test_prediction_{saved + 1:02d}.png"
            )

            save_prediction_figure(
                image=image_np,
                mask_true=mask_np,
                mask_pred=pred_np,
                save_path=save_path,
                title=f"Test prediction {saved + 1}"
            )

            print(f"Predicció guardada: {save_path}")

            saved += 1

            if saved >= CONFIG["num_predictions"]:
                break

    print("\nPrediccions generades correctament.")
    print("Total guardades:", saved)
    print("Carpeta de sortida:", CONFIG["output_dir"])

    if saved == 0:
        print("\nNo s'ha guardat cap predicció.")
        print("Pot ser que les primeres slices del test siguin buides.")
        print("Prova executant:")
        print("PYTHONPATH=. python generate_test_predictions.py --allow-empty-slices")


if __name__ == "__main__":
    main()
