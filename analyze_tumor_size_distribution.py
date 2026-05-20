import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader

from utils.dataset import BraTSSegmentationDataset
from models.unet import UNet


CONFIG = {
    "root_dir": os.environ.get(
        "DATA_ROOT",
        "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData"
    ),

    "modalities": ["flair", "t1", "t1ce", "t2"],

    "segmentation_type": "multiclass",
    "in_channels": 4,
    "out_channels": 4,

    "only_tumor_slices": False,

    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,

    "batch_size": 1,
    "num_workers": 2,

    "model_path": "results/models/unet_multiclass_4modalities_20epochs_ce_dice.pth",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_case_ids(root_dir):
    case_ids = sorted([
        folder for folder in os.listdir(root_dir)
        if folder.startswith("BraTS20_Training_")
    ])

    if len(case_ids) == 0:
        raise ValueError(f"No s'han trobat pacients a: {root_dir}")

    return case_ids


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


def normalize_mask_shape(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    return mask.numpy()


def compute_binary_metrics_from_multiclass(pred, mask, smooth=1e-6):
    """
    Converteix multiclasse a binari:
        0 = background
        >0 = tumor

    Després calcula Dice, IoU, Precision i Recall binaris.
    """

    pred_tumor = (pred > 0).astype(np.float32)
    mask_tumor = (mask > 0).astype(np.float32)

    pred_flat = pred_tumor.reshape(-1)
    mask_flat = mask_tumor.reshape(-1)

    intersection = np.sum(pred_flat * mask_flat)
    pred_sum = np.sum(pred_flat)
    mask_sum = np.sum(mask_flat)

    union = pred_sum + mask_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (mask_sum + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_pixels": float(mask_sum),
        "pred_pixels": float(pred_sum),
    }


def compute_class_metrics(pred, mask, class_id, smooth=1e-6):
    """
    Mètriques one-vs-rest per una classe concreta.
    """

    pred_class = (pred == class_id).astype(np.float32)
    mask_class = (mask == class_id).astype(np.float32)

    pred_flat = pred_class.reshape(-1)
    mask_flat = mask_class.reshape(-1)

    intersection = np.sum(pred_flat * mask_flat)
    pred_sum = np.sum(pred_flat)
    mask_sum = np.sum(mask_flat)

    union = pred_sum + mask_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (mask_sum + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_pixels": float(mask_sum),
        "pred_pixels": float(pred_sum),
    }


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


def init_class_stats():
    class_stats = {}

    for class_id in [1, 2, 3]:
        class_stats[class_id] = {
            "count_present": 0,
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "gt_pixels": [],
            "pred_pixels": [],
            "false_negative_complete": 0,
        }

    return class_stats


def print_category_stats(stats):
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
    print("DISTRIBUCIÓ I RENDIMENT BINARI PER MIDA DE TUMOR")
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


def print_class_stats(class_stats):
    print("\n" + "=" * 80)
    print("MÈTRIQUES PER CLASSE MULTICLASSE")
    print("=" * 80)

    class_names = {
        1: "Necrosis / NCR-NET",
        2: "Edema",
        3: "Enhancing tumor",
    }

    for class_id, values in class_stats.items():
        print("\n" + "-" * 80)
        print(f"Classe {class_id}: {class_names[class_id]}")
        print("-" * 80)

        count = values["count_present"]
        print(f"Slices on la classe apareix a GT: {count}")

        if len(values["dice"]) == 0:
            print("No hi ha mostres per aquesta classe.")
            continue

        print(f"GT pixels mitjana:   {np.mean(values['gt_pixels']):.2f}")
        print(f"Pred pixels mitjana: {np.mean(values['pred_pixels']):.2f}")
        print(f"Dice mitjà:          {np.mean(values['dice']):.4f}")
        print(f"IoU mitjà:           {np.mean(values['iou']):.4f}")
        print(f"Precision mitjana:   {np.mean(values['precision']):.4f}")
        print(f"Recall mitjà:        {np.mean(values['recall']):.4f}")

        fn_complete = values["false_negative_complete"]
        if count > 0:
            print(f"Falsos negatius complets: {fn_complete}")
            print(f"% FN complets: {100 * fn_complete / count:.2f}%")


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
        augment=False,
        segmentation_type=CONFIG["segmentation_type"]
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
        out_channels=CONFIG["out_channels"]
    ).to(device)

    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    stats = init_stats()
    class_stats = init_class_stats()

    print("\nAnalitzant distribució de mida tumoral i mètriques per classe...")

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            pred_np = preds[0].detach().cpu().numpy()
            mask_np = normalize_mask_shape(masks[0])

            binary_metrics = compute_binary_metrics_from_multiclass(
                pred=pred_np,
                mask=mask_np
            )

            gt_pixels = binary_metrics["gt_pixels"]
            pred_pixels = binary_metrics["pred_pixels"]

            category = get_size_category(gt_pixels)

            stats[category]["count"] += 1
            stats[category]["dice"].append(binary_metrics["dice"])
            stats[category]["iou"].append(binary_metrics["iou"])
            stats[category]["precision"].append(binary_metrics["precision"])
            stats[category]["recall"].append(binary_metrics["recall"])
            stats[category]["gt_pixels"].append(gt_pixels)
            stats[category]["pred_pixels"].append(pred_pixels)

            if gt_pixels > 0 and pred_pixels == 0:
                stats[category]["false_negative_complete"] += 1

            if gt_pixels == 0 and pred_pixels > 0:
                stats[category]["false_positive"] += 1

            for class_id in [1, 2, 3]:
                class_metrics = compute_class_metrics(
                    pred=pred_np,
                    mask=mask_np,
                    class_id=class_id
                )

                class_gt_pixels = class_metrics["gt_pixels"]
                class_pred_pixels = class_metrics["pred_pixels"]

                if class_gt_pixels > 0:
                    class_stats[class_id]["count_present"] += 1
                    class_stats[class_id]["dice"].append(class_metrics["dice"])
                    class_stats[class_id]["iou"].append(class_metrics["iou"])
                    class_stats[class_id]["precision"].append(class_metrics["precision"])
                    class_stats[class_id]["recall"].append(class_metrics["recall"])
                    class_stats[class_id]["gt_pixels"].append(class_gt_pixels)
                    class_stats[class_id]["pred_pixels"].append(class_pred_pixels)

                    if class_pred_pixels == 0:
                        class_stats[class_id]["false_negative_complete"] += 1

            if idx % 500 == 0:
                print(f"Processades {idx}/{len(test_dataset)} slices...")

    print_category_stats(stats)
    print_class_stats(class_stats)

    print("\n" + "=" * 80)
    print("FI DE L'ANÀLISI")
    print("=" * 80)


if __name__ == "__main__":
    main()
