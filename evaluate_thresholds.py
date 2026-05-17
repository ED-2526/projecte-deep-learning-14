import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import BraTSSegmentationDataset
from models.unet import UNet


# -----------------------------
# Configuració
# -----------------------------
CONFIG = {
    "root_dir": os.environ.get(
        "DATA_ROOT",
        "/home/edxnG14/laia/data/data"
    ),

    # IMPORTANT: model multimodal
    "modalities": ["flair", "t1", "t1ce", "t2"],
    "in_channels": 4,

    # IMPORTANT: totes les slices
    "only_tumor_slices": False,

    # Igual que al main.py
    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,

    "batch_size": 8,
    "num_workers": 2,

    # Canvia aquest path pel nom exacte del teu model guardat
    "model_path": "results/models/unet_multimodal_patient_split_25epochs_all_slices_all_patients.pth",
}


# -----------------------------
# Reproductibilitat
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Obtenir pacients
# -----------------------------
def get_case_ids(root_dir):
    case_ids = sorted([
        folder for folder in os.listdir(root_dir)
        if folder.startswith("BraTS20_Training_")
    ])

    if len(case_ids) == 0:
        raise ValueError(f"No s'han trobat pacients a: {root_dir}")

    return case_ids


# -----------------------------
# Split per pacients
# -----------------------------
def split_case_ids(case_ids, train_split, val_split, seed):
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


# -----------------------------
# Mètriques amb threshold variable
# -----------------------------
def compute_metrics(outputs, masks, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    masks = masks.view(-1)

    intersection = (preds * masks).sum()
    pred_sum = preds.sum()
    mask_sum = masks.sum()

    union = pred_sum + mask_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)

    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (mask_sum + smooth)

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )


# -----------------------------
# Avaluar un threshold
# -----------------------------
def evaluate_threshold(model, loader, device, threshold):
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice, iou, precision, recall = compute_metrics(
                outputs,
                masks,
                threshold=threshold
            )

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
            num_batches += 1

    return {
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\nCarregant pacients...")
    case_ids = get_case_ids(CONFIG["root_dir"])

    _, _, test_case_ids = split_case_ids(
        case_ids=case_ids,
        train_split=CONFIG["train_split"],
        val_split=CONFIG["val_split"],
        seed=CONFIG["seed"]
    )

    print("Pacients totals:", len(case_ids))
    print("Pacients test:", len(test_case_ids))

    print("\nCreant test dataset...")
    test_dataset = BraTSSegmentationDataset(
        root_dir=CONFIG["root_dir"],
        case_ids=test_case_ids,
        modalities=CONFIG["modalities"],
        only_tumor_slices=CONFIG["only_tumor_slices"],
        augment=False
    )

    print("Slices test:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    print("\nCarregant model...")
    model = UNet(
        in_channels=CONFIG["in_channels"],
        out_channels=1
    ).to(device)

    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(checkpoint)

    thresholds = [0.2, 0.3, 0.4, 0.5]

    print("\nAvaluació per threshold:")
    print("-" * 70)

    best_threshold = None
    best_dice = -1

    for threshold in thresholds:
        metrics = evaluate_threshold(
            model=model,
            loader=test_loader,
            device=device,
            threshold=threshold
        )

        print(f"Threshold: {threshold}")
        print(f"  Dice:      {metrics['dice']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print("-" * 70)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = threshold

    print(f"\nMillor threshold segons Dice: {best_threshold}")
    print(f"Millor Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
