import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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

    "out_dir": "results/ranked_predictions_tumor_only_multiclass",
    "num_each": 4,
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
    """
    Converteix la màscara a [H, W].
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    return mask.numpy()


def binary_dice_from_multiclass(pred, mask, smooth=1e-6):
    """
    Calcula Dice binari tumor/no tumor a partir de predicció multiclasse.

    pred:
        [H, W] amb classes 0,1,2,3

    mask:
        [H, W] amb classes 0,1,2,3
    """

    pred_tumor = (pred > 0).astype(np.float32)
    mask_tumor = (mask > 0).astype(np.float32)

    pred_flat = pred_tumor.reshape(-1)
    mask_flat = mask_tumor.reshape(-1)

    intersection = np.sum(pred_flat * mask_flat)
    pred_sum = np.sum(pred_flat)
    mask_sum = np.sum(mask_flat)

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)

    return float(dice), int(mask_sum), int(pred_sum)


def save_prediction_figure(image, mask, pred, dice, gt_pixels, pred_pixels, save_path, title):
    image = image.detach().cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = normalize_mask_shape(mask)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    flair = image[0]

    gt_tumor = (mask > 0).astype(np.float32)
    pred_tumor = (pred > 0).astype(np.float32)

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(flair, cmap="gray")
    plt.title("MRI FLAIR")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mask, cmap="tab10", vmin=0, vmax=3)
    plt.title("Ground Truth Multiclass")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(pred, cmap="tab10", vmin=0, vmax=3)
    plt.title(f"Prediction Multiclass\nBinary Dice={dice:.4f}")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(flair, cmap="gray")
    plt.imshow(gt_tumor, alpha=0.4)
    plt.title("Overlay GT tumor")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(flair, cmap="gray")
    plt.imshow(pred_tumor, alpha=0.4)
    plt.title("Overlay Pred tumor")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    set_seed(CONFIG["seed"])

    os.makedirs(CONFIG["out_dir"], exist_ok=True)

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

    print("\nCalculant Dice binari per cada slice tumoral del test...")
    results = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            image = images[0].detach().cpu()
            mask = masks[0].detach().cpu()
            pred = preds[0].detach().cpu()

            mask_np = normalize_mask_shape(mask)
            pred_np = pred.numpy()

            dice, gt_pixels, pred_pixels = binary_dice_from_multiclass(
                pred=pred_np,
                mask=mask_np
            )

            if gt_pixels <= 0:
                continue

            results.append({
                "idx": idx,
                "dice": dice,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "image": image,
                "mask": mask,
                "pred": pred,
            })

            if idx % 500 == 0:
                print(f"Processades {idx}/{len(test_dataset)} slices...")

    print("\nSlices amb tumor real:", len(results))

    if len(results) == 0:
        raise ValueError("No s'ha trobat cap slice amb tumor real al test set.")

    print("\nOrdenant resultats només amb slices tumorals...")
    results_sorted = sorted(results, key=lambda x: x["dice"])

    n = CONFIG["num_each"]

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
    print(f"Dice mitjà aproximat: {results_sorted[len(results_sorted)//2]['dice']:.4f}")
    print(f"Millor Dice: {results_sorted[-1]['dice']:.4f}")

    print("\nGuardant figures...")

    for group_name, group_results in groups.items():
        group_dir = os.path.join(CONFIG["out_dir"], group_name)
        os.makedirs(group_dir, exist_ok=True)

        for i, item in enumerate(group_results, start=1):
            save_path = os.path.join(
                group_dir,
                f"{group_name}_{i:02d}_idx_{item['idx']}_dice_{item['dice']:.4f}_gt_{item['gt_pixels']}_pred_{item['pred_pixels']}.png"
            )

            title = (
                f"{group_name.upper()} | "
                f"idx={item['idx']} | "
                f"Binary Dice={item['dice']:.4f} | "
                f"GT pixels={item['gt_pixels']} | "
                f"Pred pixels={item['pred_pixels']}"
            )

            save_prediction_figure(
                image=item["image"],
                mask=item["mask"],
                pred=item["pred"],
                dice=item["dice"],
                gt_pixels=item["gt_pixels"],
                pred_pixels=item["pred_pixels"],
                save_path=save_path,
                title=title
            )

    print("\nFet!")
    print(f"Figures guardades a: {CONFIG['out_dir']}")
    print("\nCarpetes generades:")
    print(f"- {CONFIG['out_dir']}/best")
    print(f"- {CONFIG['out_dir']}/middle")
    print(f"- {CONFIG['out_dir']}/worst")


if __name__ == "__main__":
    main()
