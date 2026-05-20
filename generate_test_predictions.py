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

    "out_dir": "results/predictions_multiclass",
    "num_examples": 12,
}


CLASS_NAMES = {
    0: "Background",
    1: "Necrosis / NCR-NET",
    2: "Edema",
    3: "Enhancing tumor",
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

    Pot venir com:
    - [1, H, W]
    - [H, W]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    return mask.numpy()


def save_multiclass_prediction(image, mask, pred, save_path, title):
    """
    image: tensor [4, H, W]
    mask: tensor [H, W] o [1, H, W]
    pred: tensor [H, W]
    """

    image = image.detach().cpu().numpy()
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
    plt.title("Prediction Multiclass")
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
        out_channels=CONFIG["out_channels"]
    ).to(device)

    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("\nGenerant prediccions multiclasse...")

    saved = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            image = images[0].detach().cpu()
            mask = masks[0].detach().cpu()
            pred = preds[0].detach().cpu()

            save_path = os.path.join(
                CONFIG["out_dir"],
                f"test_prediction_multiclass_{saved + 1:02d}_idx_{idx}.png"
            )

            title = f"Test example {saved + 1} | idx={idx}"

            save_multiclass_prediction(
                image=image,
                mask=mask,
                pred=pred,
                save_path=save_path,
                title=title
            )

            saved += 1

            if saved >= CONFIG["num_examples"]:
                break

    print("\nFet!")
    print(f"Prediccions guardades a: {CONFIG['out_dir']}")


if __name__ == "__main__":
    main()
