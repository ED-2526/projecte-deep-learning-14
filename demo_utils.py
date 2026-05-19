import os
import sys
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet import UNet

BINARY_CONFIG = {
    "model_path": "results/models/unet_flair_patient_split_20epochs.pth",
    "in_channels": 1,
    "out_channels": 1,
    "modalities": ["flair"],
    "segmentation_type": "binary",
    "loss": "BCEDiceLoss",
    "lr": "1e-4",
    "epochs": 20,
    "batch_size": 8,
    "seed": 42,
    "augmentation": "No",
    "test_dice": 0.7677,
    "test_iou": 0.6637,
}

MULTICLASS_CONFIG = {
    "model_path": "results/models/unet_multiclass_4modalities_20epochs_ce_dice.pth",
    "in_channels": 4,
    "out_channels": 4,
    "modalities": ["flair", "t1", "t1ce", "t2"],
    "segmentation_type": "multiclass",
    "loss": "CEDiceLoss (CE=0.5 + Dice=0.5)",
    "lr": "1e-4",
    "epochs": 20,
    "batch_size": 8,
    "seed": 42,
    "augmentation": "No",
    "test_dice": None,
    "test_iou": None,
}

CLASS_NAMES = {
    0: "Fons (Background)",
    1: "Necrosi / No realçat",
    2: "Edema peritumoral",
    3: "Tumor realçat",
}

CLASS_COLORS = {
    1: "#e63946",
    2: "#2dc653",
    3: "#f4a261",
}

BRATS_TO_MULTICLASS = {0: 0, 1: 1, 2: 2, 4: 3}

DATA_ROOT = "/home/edxnG14/laia/data/data"


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(mode):
    cfg = BINARY_CONFIG if mode == "binary" else MULTICLASS_CONFIG
    device = _get_device()
    path = cfg["model_path"]
    if not os.path.exists(path):
        return None, f"Checkpoint no trobat: {path}"
    try:
        model = UNet(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"])
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


def list_demo_patients():
    if not os.path.isdir(DATA_ROOT):
        return []
    return sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("BraTS20_Training_")])


def _normalize_slice(sl):
    sl = sl.astype(np.float32)
    if sl.max() > sl.min():
        sl = (sl - sl.min()) / (sl.max() - sl.min())
    return sl


def load_patient_slice(case_id, slice_idx, modalities):
    case_path = os.path.join(DATA_ROOT, case_id)
    channels = []
    for mod in modalities:
        nii_path = os.path.join(case_path, f"{case_id}_{mod}.nii")
        if not os.path.exists(nii_path):
            nii_path += ".gz"
        vol = nib.load(nii_path).get_fdata()
        sl = _normalize_slice(vol[:, :, slice_idx])
        channels.append(sl)

    seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
    if not os.path.exists(seg_path):
        seg_path += ".gz"
    seg_vol = nib.load(seg_path).get_fdata()
    seg_raw = seg_vol[:, :, slice_idx]
    seg_mapped = np.zeros_like(seg_raw, dtype=np.int64)
    for orig, new in BRATS_TO_MULTICLASS.items():
        seg_mapped[seg_raw == orig] = new

    return {
        "channels": np.stack(channels, axis=0),
        "seg": seg_mapped,
        "flair": channels[0],
        "all_channels": channels,
    }


def run_inference(model, image_np, mode):
    device = next(model.parameters()).device
    tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    if mode == "binary":
        pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.int64)
    else:
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    return pred


def build_overlay_figure(flair, pred, seg, mode, all_channels=None):
    has_gt = seg is not None
    modality_names = ["FLAIR", "T1", "T1ce", "T2"]
    cmap4 = plt.cm.get_cmap("tab10", 4)

    if mode == "binary":
        n_cols = 3 if has_gt else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#0e1117")

        axes[0].imshow(flair, cmap="gray")
        axes[0].set_title("FLAIR MRI", color="white", fontsize=13, pad=8)
        axes[0].axis("off")

        if has_gt:
            axes[1].imshow(flair, cmap="gray")
            gt_overlay = np.ma.masked_where(seg == 0, seg)
            axes[1].imshow(gt_overlay, cmap="Greens", vmin=0, vmax=1, alpha=0.6)
            axes[1].set_title("Ground Truth", color="white", fontsize=13, pad=8)
            axes[1].axis("off")
            ax_pred = axes[2]
        else:
            ax_pred = axes[1]

        ax_pred.imshow(flair, cmap="gray")
        pred_overlay = np.ma.masked_where(pred == 0, pred)
        ax_pred.imshow(pred_overlay, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
        ax_pred.set_title("Predicció del Model", color="white", fontsize=13, pad=8)
        ax_pred.axis("off")

    else:
        ch_count = len(all_channels) if all_channels else 1
        n_cols = ch_count + (2 if has_gt else 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#0e1117")

        for i, (ch, name) in enumerate(zip(all_channels or [flair], modality_names[:ch_count])):
            axes[i].imshow(ch, cmap="gray")
            axes[i].set_title(name, color="white", fontsize=11, pad=6)
            axes[i].axis("off")

        if has_gt:
            ax_gt = axes[ch_count]
            ax_gt.imshow(flair, cmap="gray")
            gt_ov = np.ma.masked_where(seg == 0, seg)
            ax_gt.imshow(gt_ov, cmap=cmap4, vmin=0, vmax=3, alpha=0.6)
            ax_gt.set_title("Ground Truth", color="white", fontsize=11, pad=6)
            ax_gt.axis("off")
            ax_pred = axes[ch_count + 1]
        else:
            ax_pred = axes[ch_count]

        ax_pred.imshow(flair, cmap="gray")
        pred_ov = np.ma.masked_where(pred == 0, pred)
        ax_pred.imshow(pred_ov, cmap=cmap4, vmin=0, vmax=3, alpha=0.6)
        ax_pred.set_title("Predicció del Model", color="white", fontsize=11, pad=6)
        ax_pred.axis("off")

        patches = [
            mpatches.Patch(color=cmap4(i / 3), label=CLASS_NAMES[i])
            for i in [1, 2, 3]
        ]
        ax_pred.legend(
            handles=patches,
            loc="lower right",
            fontsize=8,
            framealpha=0.7,
            facecolor="#1a1a2e",
            labelcolor="white",
        )

    fig.tight_layout(pad=1.5)
    return fig


def compute_stats(pred, mode):
    total = pred.size
    if mode == "binary":
        tumor_px = int((pred > 0).sum())
        bg_px = total - tumor_px
        return {
            "Tumor": tumor_px / total * 100,
            "Background": bg_px / total * 100,
        }
    else:
        stats = {}
        for cls_id, cls_name in CLASS_NAMES.items():
            px = int((pred == cls_id).sum())
            stats[cls_name] = px / total * 100
        return stats
