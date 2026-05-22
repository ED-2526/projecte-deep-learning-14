import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

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

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True
    )

    parser.add_argument("--num-each", type=int, default=4)
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


def dice_per_slice(pred, mask, smooth=1e-6):
    pred = pred.float().view(-1)
    mask = mask.float().view(-1)

    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)

    return dice.item(), int(mask_sum.item()), int(pred_sum.item())


def save_prediction_figure(image, mask, pred, prob, dice, save_path, title):
    image = image.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if mask.ndim == 3:
        mask = mask[0]

    pred = pred.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()

    if pred.ndim == 3:
        pred = pred[0]

    if prob.ndim == 3:
        prob = prob[0]

    flair = image[0]

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(flair, cmap="gray")
    plt.title("MRI FLAIR")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(prob, cmap="gray")
    plt.title("Probability map")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Prediction\nDice={dice:.4f}")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(flair, cmap="gray")
    plt.imshow(pred, alpha=0.4)
    plt.title("Overlay prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

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

    print("\nCalculant Dice per cada slice tumoral del test...")

    results = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > args.threshold).float()

            dice, gt_pixels, pred_pixels = dice_per_slice(preds[0], masks[0])

            if gt_pixels <= 0:
                continue

            results.append({
                "idx": idx,
                "dice": dice,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "image": images[0].detach().cpu(),
                "mask": masks[0].detach().cpu(),
                "pred": preds[0].detach().cpu(),
                "prob": probs[0].detach().cpu(),
            })

            if idx % 500 == 0:
                print(f"Processades {idx}/{len(test_dataset)} slices...")

    print("\nSlices amb tumor real:", len(results))

    if len(results) == 0:
        raise ValueError("No s'ha trobat cap slice amb tumor real al test set.")

    print("\nOrdenant resultats només amb slices tumorals...")

    results_sorted = sorted(results, key=lambda x: x["dice"])

    n = args.num_each

    worst = results_sorted[:n]
    best = results_sorted[-n:]

    mid_start = max(0, len(results_sorted) // 2 - n // 2)
    middle = results_sorted[mid_start:mid_start + n]

    groups = {
        "worst": worst,
        "middle": middle,
        "best": best,
    }

    print("\nResum ràpid:")
    print(f"Pitjor Dice: {results_sorted[0]['dice']:.4f}")
    print(f"Dice mitjà aproximat: {results_sorted[len(results_sorted) // 2]['dice']:.4f}")
    print(f"Millor Dice: {results_sorted[-1]['dice']:.4f}")

    print("\nGuardant figures...")

    for group_name, group_results in groups.items():
        group_dir = os.path.join(args.out_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for i, item in enumerate(group_results, start=1):
            save_path = os.path.join(
                group_dir,
                f"{group_name}_{i:02d}_idx_{item['idx']}_dice_{item['dice']:.4f}_gt_{item['gt_pixels']}_pred_{item['pred_pixels']}.png"
            )

            title = (
                f"{group_name.upper()} | "
                f"idx={item['idx']} | "
                f"Dice={item['dice']:.4f} | "
                f"GT pixels={item['gt_pixels']} | "
                f"Pred pixels={item['pred_pixels']} | "
                f"threshold={args.threshold}"
            )

            save_prediction_figure(
                image=item["image"],
                mask=item["mask"],
                pred=item["pred"],
                prob=item["prob"],
                dice=item["dice"],
                save_path=save_path,
                title=title
            )

    print("\nFet!")
    print(f"Figures guardades a: {args.out_dir}")
    print("\nCarpetes generades:")
    print(f"- {args.out_dir}/best")
    print(f"- {args.out_dir}/middle")
    print(f"- {args.out_dir}/worst")


if __name__ == "__main__":
    main()
