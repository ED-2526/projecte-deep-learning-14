import os
import random
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from utils.dataset import BraTSSegmentationDataset
from models.unet import UNet, ResUNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get(
            "DATA_ROOT",
            "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        choices=["unet", "resunet"]
    )

    parser.add_argument(
        "--segmentation-type",
        type=str,
        default="binary",
        choices=["binary"]
    )

    parser.add_argument("--threshold", type=float, default=0.3)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)

    return parser.parse_args()


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


def create_model(architecture, device):
    if architecture == "resunet":
        model = ResUNet(
            in_channels=4,
            out_channels=1
        ).to(device)
        print("Arquitectura utilitzada: ResUNet")
    else:
        model = UNet(
            in_channels=4,
            out_channels=1
        ).to(device)
        print("Arquitectura utilitzada: UNet")

    return model


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


def print_stats(stats):
    total_slices = sum(stats[cat]["count"] for cat in stats)
    total_empty_slices = stats["0 pixels"]["count"]
    total_tumor_slices = total_slices - total_empty_slices

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

        print(f"GT pixels mitjana:   {np.mean(values['gt_pixels']):.2f}")
        print(f"Pred pixels mitjana: {np.mean(values['pred_pixels']):.2f}")
        print(f"Dice mitjà:          {np.mean(values['dice']):.4f}")
        print(f"IoU mitjà:           {np.mean(values['iou']):.4f}")
        print(f"Precision mitjana:   {np.mean(values['precision']):.4f}")
        print(f"Recall mitjà:        {np.mean(values['recall']):.4f}")

        if category == "0 pixels":
            fp = values["false_positive"]
            print(f"Falsos positius en slices buides: {fp}")
            print(f"% FP dins slices buides: {100 * fp / count:.2f}%")
        else:
            fn = values["false_negative_complete"]
            print(f"Falsos negatius complets: {fn}")
            print(f"% FN complets dins categoria: {100 * fn / count:.2f}%")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\nCarregant pacients...")
    case_ids = get_case_ids(args.root_dir)

    _, _, test_case_ids = split_case_ids(
        case_ids=case_ids,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed
    )

    print("Pacients totals:", len(case_ids))
    print("Pacients test:", len(test_case_ids))

    print("\nCreant test dataset...")
    test_dataset = BraTSSegmentationDataset(
        root_dir=args.root_dir,
        case_ids=test_case_ids,
        modalities=["flair", "t1", "t1ce", "t2"],
        only_tumor_slices=False,
        augment=False,
        segmentation_type="binary"
    )

    print("Slices test totals:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("\nCarregant model...")
    print(f"Checkpoint carregat: {args.model_path}")
    print(f"Threshold utilitzat: {args.threshold}")

    model = create_model(args.architecture, device)
    checkpoint = torch.load(args.model_path, map_location=device)
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
            preds = (probs > args.threshold).float()

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

            if gt_pixels > 0 and pred_pixels == 0:
                stats[category]["false_negative_complete"] += 1

            if gt_pixels == 0 and pred_pixels > 0:
                stats[category]["false_positive"] += 1

            if idx % 500 == 0:
                print(f"Processades {idx}/{len(test_dataset)} slices...")

    print_stats(stats)

    print("\n" + "=" * 80)
    print("FI DE L'ANÀLISI")
    print("=" * 80)


if __name__ == "__main__":
    main()
