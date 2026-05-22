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

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5]
    )

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


def compute_metrics(preds, masks, smooth=1e-6):
    preds = preds.float().view(-1)
    masks = masks.float().view(-1)

    intersection = (preds * masks).sum()
    pred_sum = preds.sum()
    mask_sum = masks.sum()

    union = pred_sum + mask_sum - intersection

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (mask_sum + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item()


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

    print("Slices test:", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print("\nCarregant model...")
    print(f"Checkpoint carregat: {args.model_path}")

    model = create_model(args.architecture, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    best_threshold = None
    best_dice = -1.0

    print("\nAvaluació per threshold:")

    for threshold in args.thresholds:
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()

                dice, iou, precision, recall = compute_metrics(preds, masks)

                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)

        print("-" * 70)
        print(f"Threshold: {threshold}")
        print(f"  Dice:      {mean_dice:.4f}")
        print(f"  IoU:       {mean_iou:.4f}")
        print(f"  Precision: {mean_precision:.4f}")
        print(f"  Recall:    {mean_recall:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = threshold

    print("-" * 70)
    print(f"\nMillor threshold segons Dice: {best_threshold}")
    print(f"Millor Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
