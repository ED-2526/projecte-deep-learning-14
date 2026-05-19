import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.visualization import save_prediction_figure
from utils.dataset import BraTSSegmentationDataset
from models.unet import UNet


CONFIG = {
    "root_dir": "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData",
    "modalities": ["flair", "t1", "t1ce", "t2"],
    "only_tumor_slices": False,
    "segmentation_type": "multiclass",

    "train_split": 0.8,
    "val_split": 0.1,
    "seed": 42,

    "in_channels": 4,
    "out_channels": 4,

    "batch_size": 1,
    "num_workers": 2,

    "model_path": "results/models/unet_multiclass_4modalities_20epochs_ce_dice.pth",
    "predictions_dir": "results/predictions_multiclass",
    "num_examples": 12,
}


def get_case_ids(root_dir):
    case_ids = sorted([
        folder for folder in os.listdir(root_dir)
        if folder.startswith("BraTS20_Training_")
    ])

    if len(case_ids) == 0:
        raise ValueError(f"No s'han trobat pacients a {root_dir}.")

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


def create_test_dataset(config):
    case_ids = get_case_ids(config["root_dir"])
    _, _, test_case_ids = split_case_ids(
        case_ids=case_ids,
        train_split=config["train_split"],
        val_split=config["val_split"],
        seed=config["seed"],
    )

    test_dataset = BraTSSegmentationDataset(
        root_dir=config["root_dir"],
        case_ids=test_case_ids,
        modalities=config["modalities"],
        only_tumor_slices=config["only_tumor_slices"],
        segmentation_type=config["segmentation_type"],
    )

    print("Pacients test:", len(test_case_ids))
    print("Slices test:", len(test_dataset))
    print("Exemples pacients test:", test_case_ids[:5])

    return test_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(CONFIG["predictions_dir"], exist_ok=True)

    test_dataset = create_test_dataset(CONFIG)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    model = UNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
    ).to(device)

    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    saved = 0

    with torch.no_grad():
        for _, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            if CONFIG["segmentation_type"] == "multiclass":
                preds = torch.argmax(logits, dim=1)  # [B, H, W]
                mask_np = masks[0].cpu().numpy()
                pred_np = preds[0].cpu().numpy()
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                mask_np = masks[0, 0].cpu().numpy()
                pred_np = preds[0, 0].cpu().numpy()

            modality_idx = CONFIG["modalities"].index("flair")
            image_np = images[0, modality_idx].cpu().numpy()

            # Evitem guardar slices completament buides.
            if np.sum(mask_np > 0) == 0:
                continue

            save_path = os.path.join(
                CONFIG["predictions_dir"],
                f"test_prediction_{saved + 1:02d}.png",
            )

            save_prediction_figure(
                image=image_np,
                mask=mask_np,
                pred=pred_np,
                save_path=save_path,
                title=f"Test example {saved + 1}",
                num_classes=CONFIG["out_channels"] if CONFIG["segmentation_type"] == "multiclass" else 2,
            )

            print(f"Predicció guardada a: {save_path}")
            saved += 1

            if saved >= CONFIG["num_examples"]:
                break

    print("\nPrediccions visuals generades correctament.")


if __name__ == "__main__":
    main()
