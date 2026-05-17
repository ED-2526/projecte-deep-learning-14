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
        "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData"
    ),

    # Model multimodal
    "modalities": ["flair", "t1", "t1ce", "t2"],
    "in_channels": 4,

    # Totes les slices
    "only_tumor_slices": False,

    # Mateix split que main.py
    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,

    "batch_size": 1,
    "num_workers": 2,

    # IMPORTANT: canvia aquest nom si el teu checkpoint té un altre nom
    "model_path": "results/models/unet_multimodal_patient_split_20epochs_all_slices.pth",

    # Millor threshold trobat
    "threshold": 0.3,
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
# Obtenir IDs de pacients
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
# Mètriques per slice
# -----------------------------
def compute_slice_metrics(pred, mask, smooth=1e-6):
    pred = pred.float().view(-1)
    mask = mask.float().view(-1)

    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()

    union = pred_sum + mask_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (mask_sum + smooth)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "gt_pixels": mask_sum.item(),
        "pred_pixels": pred_sum.item(),
    }


# -----------------------------
# Categoria segons mida del tumor
# -----------------------------
def get_size_category(gt_pixels):
    if gt_pixels == 0:
        return "0 pixels"
    elif gt_pixels <= 10:
        return "1-10 pixels"
    elif gt_pixels <= 100:
        return "11-100 pixels"
    elif gt_pixels <= 500:
        return "101-500 pixels"
    elif gt_pixels <= 1000:
        return "501-1000 pixels"
    else:
        return ">1000 pixels"


# -----------------------------
# Inicialitzar estructura de resultats
# -----------------------------
def init_stats():
    categories = [
        "0 pixels",
        "1-10 pixels",
        "11-100 pixels",
        "101-500 pixels",
        "501-1000 pixels",
        ">1000 pixels",
    ]

    stats = {}

    for cat in categories:
        stats[cat] = {
            "count": 0,
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "gt_pixels": [],
            "pred_pixels": [],
            "false_negative_complete": 0,
            "false_positive": 0,
        }

    return stats


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

    print("Slices test totals:", len(test_dataset))

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
    model.eval()

    stats = init_stats()

    print("\nAnalitzant distribució de mida tumoral i rendiment per categoria...")

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > CONFIG["threshold"]).float()

            metrics = compute_slice_metrics(preds[0], masks[0])

            gt_pixels = metrics["gt_pixels"]
            pred_pixels = metrics["pred_pixels"]

            category = get_size_category(gt_pixels)

            stats[category]["count"] += 1
            stats[category]["dice"].append(metrics["dice"])
            stats[category]["iou"].append(metrics["iou"])
            stats[category]["precision"].append(metrics["precision"])
            stats[category]["recall"].append(metrics["recall"])
            stats[category]["gt_pixels"].append(gt_pixels)
            stats[category]["pred_pixels"].append(pred_pixels)

            # Cas greu: hi ha tumor real però el model prediu 0 píxels
            if gt_pixels > 0 and pred_pixels == 0:
                stats[category]["false_negative_complete"] += 1

            # Cas: no hi ha tumor real però el model prediu tumor
            if gt_pixels == 0 and pred_pixels > 0:
                stats[category]["false_positive"] += 1

            if idx % 500 == 0:
                print(f"Processades {idx}/{len(test_dataset)} slices...")

    total_slices = sum(stats[cat]["count"] for cat in stats)
    total_tumor_slices = sum(
        stats[cat]["count"] for cat in stats if cat != "0 pixels"
    )
    total_empty_slices = stats["0 pixels"]["count"]

    print("\n" + "=" * 80)
    print("RESUM GENERAL")
    print("=" * 80)
    print(f"Total slices test: {total_slices}")
    print(f"Slices sense tumor: {total_empty_slices}")
    print(f"Slices amb tumor: {total_tumor_slices}")

    if total_slices > 0:
        print(f"% slices sense tumor: {100 * total_empty_slices / total_slices:.2f}%")
        print(f"% slices amb tumor: {100 * total_tumor_slices / total_slices:.2f}%")

    print("\n" + "=" * 80)
    print("DISTRIBUCIÓ I RENDIMENT PER MIDA DE TUMOR")
    print("=" * 80)

    for category, values in stats.items():
        count = values["count"]

        print("\n" + "-" * 80)
        print(f"Categoria: {category}")
        print("-" * 80)
        print(f"Slices: {count}")

        if count == 0:
            continue

        print(f"Percentatge del test: {100 * count / total_slices:.2f}%")

        dice_mean = np.mean(values["dice"])
        iou_mean = np.mean(values["iou"])
        precision_mean = np.mean(values["precision"])
        recall_mean = np.mean(values["recall"])
        gt_mean = np.mean(values["gt_pixels"])
        pred_mean = np.mean(values["pred_pixels"])

        print(f"GT pixels mitjana:   {gt_mean:.2f}")
        print(f"Pred pixels mitjana: {pred_mean:.2f}")
        print(f"Dice mitjà:          {dice_mean:.4f}")
        print(f"IoU mitjà:           {iou_mean:.4f}")
        print(f"Precision mitjana:   {precision_mean:.4f}")
        print(f"Recall mitjà:        {recall_mean:.4f}")

        if category != "0 pixels":
            fn_complete = values["false_negative_complete"]
            print(f"Falsos negatius complets: {fn_complete}")
            print(f"% FN complets dins categoria: {100 * fn_complete / count:.2f}%")
        else:
            fp = values["false_positive"]
            print(f"Falsos positius en slices buides: {fp}")
            print(f"% FP dins slices buides: {100 * fp / count:.2f}%")

    print("\n" + "=" * 80)
    print("FI DE L'ANÀLISI")
    print("=" * 80)


if __name__ == "__main__":
    main()
