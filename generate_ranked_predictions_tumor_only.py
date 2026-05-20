import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    "model_path": "results/models/unet_multimodal_20epochs_bce_tversky.pth",
    
    # Millor threshold trobat amb evaluate_thresholds.py
    "threshold": 0.3,

    # Sortida
    "out_dir": "results/ranked_predictions_tumor_only",
    "num_each": 4,
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
# Dice per slice
# -----------------------------
def dice_per_slice(pred, mask, smooth=1e-6):
    pred = pred.float().view(-1)
    mask = mask.float().view(-1)

    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()

    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)
    return dice.item()


# -----------------------------
# Guardar figura
# -----------------------------
def save_prediction_figure(image, mask, pred, prob, dice, save_path, title):
    """
    image: tensor [4, H, W]
    mask: tensor [1, H, W]
    pred: tensor [1, H, W]
    prob: tensor [1, H, W]
    """

    image = image.cpu().numpy()
    mask = mask.cpu().numpy()[0]
    pred = pred.cpu().numpy()[0]
    prob = prob.cpu().numpy()[0]

    # Com que modalities = ["flair", "t1", "t1ce", "t2"],
    # el canal 0 és FLAIR.
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


# -----------------------------
# Main
# -----------------------------
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

    print("\nCalculant Dice per cada slice del test...")
    results = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > CONFIG["threshold"]).float()

            tumor_pixels = masks[0].sum().item()
            pred_pixels = preds[0].sum().item()

            # IMPORTANT:
            # Només guardem slices que tenen tumor real.
            # Així evitem que best/middle siguin slices buides.
            if tumor_pixels <= 0:
                continue

            dice = dice_per_slice(preds[0], masks[0])

            results.append({
                "idx": idx,
                "dice": dice,
                "tumor_pixels": tumor_pixels,
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
        raise ValueError("No s'ha trobat cap slice amb tumor al test set.")

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
                f"{group_name}_{i:02d}_idx_{item['idx']}_dice_{item['dice']:.4f}_gt_{int(item['tumor_pixels'])}_pred_{int(item['pred_pixels'])}.png"
            )

            title = (
                f"{group_name.upper()} | "
                f"slice idx={item['idx']} | "
                f"Dice={item['dice']:.4f} | "
                f"GT pixels={int(item['tumor_pixels'])} | "
                f"Pred pixels={int(item['pred_pixels'])} | "
                f"threshold={CONFIG['threshold']}"
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
    print(f"Figures guardades a: {CONFIG['out_dir']}")
    print("\nCarpetes generades:")
    print(f"- {CONFIG['out_dir']}/best")
    print(f"- {CONFIG['out_dir']}/middle")
    print(f"- {CONFIG['out_dir']}/worst")


if __name__ == "__main__":
    main()
