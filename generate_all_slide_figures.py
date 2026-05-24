import os
import json
import glob
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ============================================================
# CONFIGURACIÓ GENERAL
# ============================================================

CONFIG = {
    # Ruta a la carpeta que conté els pacients BraTS20_Training_xxx
    "DATA_ROOT": "/home/edxnG14/laia/data/MICCAI_BraTS2020_TrainingData",

    # Pacient que s'utilitzarà per generar figures de dades
    "CASE_ID": "BraTS20_Training_091",

    # Carpeta de sortida
    "OUT_DIR": "results/figures_slides",

    # Histories reals que tens dins results/history/
    "HISTORY_CANDIDATES": {
        "baseline_binary": [
            "results/history/unet_flair_patient_split_20epochs_history.json",
        ],

        "baseline_binary_all_slices": [
            "results/history/unet_flair_patient_split_20epochs_all_slices_history.json",
        ],

        "baseline_binary_aug": [
            "results/history/unet_flair_patient_split_20epochs_aug_history.json",
        ],

        "multimodal_binary": [
            "results/history/unet_multimodal_20epochs_bce_tversky_history.json",
            "results/history/unet_multimodal_patient_split_20epochs_all_slices_history.json",
        ],

        "weighted_binary": [
            "results/history/unet_binary_4modalities_20epochs_bce_dice_weighted_sampler_history.json",
        ],

        "multiclass_unet": [
            "results/history/unet_multiclass_4modalities_20epochs_ce_dice_history.json",
        ],

        "weighted_multiclass": [
            "results/history/unet_multiclass_4modalities_20epochs_ce_dice_weighted_sampler_history.json",
        ],

        "resunet_binary": [
            "results/history/resunet_binary_4modalities_20epochs_bce_dice_history.json",
        ],

        "resunet_multiclass": [
            "results/history/resunet_multiclass_4modalities_20epochs_ce_dice_history.json",
        ],
    },

    # Si vols triar manualment 3 prediccions bones/intermèdies/dolentes,
    # posa les rutes aquí. Si ho deixes buit, l'script les buscarà automàticament.
    "PREDICTION_IMAGES": {
        "good": "",
        "medium": "",
        "bad": "",
    },

    # Carpetes on buscar prediccions si no s'han especificat manualment
    "PREDICTION_SEARCH_DIRS": [
        "results/predictions",
        "results/figures",
        "results",
    ],
}


# ============================================================
# UTILITATS GENERALS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_existing_path(candidates):
    """
    Rep una llista de rutes o patrons glob i retorna el primer fitxer existent.
    """
    if isinstance(candidates, str):
        candidates = [candidates]

    for candidate in candidates:
        matches = sorted(glob.glob(candidate))
        if matches:
            return matches[0]

        if os.path.exists(candidate):
            return candidate

    return None


def create_text_placeholder(title, subtitle, output_path, figsize=(10, 6)):
    """
    Crea una imatge placeholder si falta algun fitxer.
    Això evita que l'script peti si encara no tens algun experiment.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.text(
        0.5, 0.62,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold"
    )

    ax.text(
        0.5, 0.38,
        subtitle,
        ha="center",
        va="center",
        fontsize=14
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_history_by_name(exp_name):
    candidates = CONFIG["HISTORY_CANDIDATES"].get(exp_name, [])
    history_path = resolve_existing_path(candidates)

    if history_path is None:
        return None, None

    try:
        history = load_json(history_path)
        return history, history_path
    except Exception as e:
        print(f"No s'ha pogut carregar el history {history_path}: {e}")
        return None, history_path


def get_series(history, possible_keys):
    """
    Busca una sèrie dins el JSON usant diferents noms possibles.
    """
    if history is None:
        return None

    for key in possible_keys:
        if key in history and isinstance(history[key], list) and len(history[key]) > 0:
            return history[key]

    return None


def get_metric_for_comparison(history, metric_type="dice"):
    """
    Prioritat per les gràfiques comparatives:
    1. Mètrica de test si existeix.
    2. Millor mètrica de validation.
    3. Millor mètrica de train.

    Això és millor per la presentació perquè el test representa
    millor la generalització final.
    """
    if history is None:
        return None

    if metric_type == "dice":
        test_keys = ["test_dice", "test_dice_score"]
        val_keys = ["val_dice", "valid_dice", "val_dice_score"]
        train_keys = ["train_dice", "dice", "train_dice_score"]
        reduce_fn = np.max

    elif metric_type == "iou":
        test_keys = ["test_iou", "test_iou_score"]
        val_keys = ["val_iou", "valid_iou", "val_iou_score"]
        train_keys = ["train_iou", "iou", "train_iou_score"]
        reduce_fn = np.max

    elif metric_type == "loss":
        test_keys = ["test_loss"]
        val_keys = ["val_loss", "valid_loss"]
        train_keys = ["train_loss", "loss"]
        reduce_fn = np.min

    else:
        return None

    # 1. Test metric: pot ser escalar o llista
    for key in test_keys:
        if key in history:
            value = history[key]
            if isinstance(value, list):
                return float(value[-1])
            return float(value)

    # 2. Best validation metric
    series = get_series(history, val_keys)
    if series is not None:
        return float(reduce_fn(series))

    # 3. Best train metric
    series = get_series(history, train_keys)
    if series is not None:
        return float(reduce_fn(series))

    return None


# ============================================================
# UTILITATS MRI
# ============================================================

def load_nifti(path):
    return nib.load(path).get_fdata()


def normalize_slice(slice_2d):
    """
    Normalitza una slice entre 0 i 1 per visualitzar millor.
    """
    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val == 0:
        return slice_2d

    return (slice_2d - min_val) / (max_val - min_val)


def find_best_tumor_slice(seg):
    """
    Selecciona la slice amb més píxels de tumor.
    Això fa que la visualització sigui més clara.
    """
    tumor_pixels_per_slice = [
        np.sum(seg[:, :, i] > 0)
        for i in range(seg.shape[2])
    ]

    return int(np.argmax(tumor_pixels_per_slice))


def load_patient_modalities():
    """
    Carrega FLAIR, T1, T1CE, T2 i SEG d'un pacient.
    Retorna la millor slice per visualitzar tumor.
    """
    data_root = CONFIG["DATA_ROOT"]
    case_id = CONFIG["CASE_ID"]
    case_path = os.path.join(data_root, case_id)

    flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
    t1_path = os.path.join(case_path, f"{case_id}_t1.nii")
    t1ce_path = os.path.join(case_path, f"{case_id}_t1ce.nii")
    t2_path = os.path.join(case_path, f"{case_id}_t2.nii")
    seg_path = os.path.join(case_path, f"{case_id}_seg.nii")

    required_paths = [flair_path, t1_path, t1ce_path, t2_path, seg_path]

    for p in required_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No s'ha trobat el fitxer: {p}")

    flair = load_nifti(flair_path)
    t1 = load_nifti(t1_path)
    t1ce = load_nifti(t1ce_path)
    t2 = load_nifti(t2_path)
    seg = load_nifti(seg_path)

    slice_idx = find_best_tumor_slice(seg)

    seg_slice = seg[:, :, slice_idx]
    binary_mask = (seg_slice > 0).astype(np.float32)

    return {
        "flair": normalize_slice(flair[:, :, slice_idx]),
        "t1": normalize_slice(t1[:, :, slice_idx]),
        "t1ce": normalize_slice(t1ce[:, :, slice_idx]),
        "t2": normalize_slice(t2[:, :, slice_idx]),
        "seg": seg_slice,
        "binary_mask": binary_mask,
        "slice_idx": slice_idx,
        "case_id": case_id,
    }


# ============================================================
# SLIDE 3 — DADES I TRANSFORMACIÓ
# ============================================================

def generate_slide03_data_figures():
    """
    Genera:
    - slide03_modalitats_i_mascara.png
    - slide03_conversio_mascara.png
    """
    out_dir = CONFIG["OUT_DIR"]
    patient = load_patient_modalities()

    # --------------------------------------------------------
    # Slide 3A: FLAIR | T1 | T1CE | T2 | Màscara binària
    # --------------------------------------------------------
    plt.figure(figsize=(18, 5))

    titles = ["FLAIR", "T1", "T1CE", "T2", "Màscara binària"]
    images = [
        patient["flair"],
        patient["t1"],
        patient["t1ce"],
        patient["t2"],
        patient["binary_mask"],
    ]

    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(images[i], cmap="gray")
        ax.set_title(titles[i], fontsize=16)
        ax.axis("off")

    plt.suptitle(
        f"Modalitats MRI i màscara binària - {patient['case_id']} - Slice {patient['slice_idx']}",
        fontsize=18
    )

    plt.tight_layout()

    output_path = os.path.join(out_dir, "slide03_modalitats_i_mascara.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Slide 3B: Màscara multiclasse -> binària
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(patient["seg"], cmap="viridis")
    ax1.set_title("Màscara original multiclasse\n0, 1, 2, 4", fontsize=16)
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(patient["binary_mask"], cmap="gray")
    ax2.set_title("Màscara binària\n0 = fons, 1 = tumor", fontsize=16)
    ax2.axis("off")

    plt.suptitle("Conversió de màscara multiclasse a binària", fontsize=18)
    plt.tight_layout()

    output_path = os.path.join(out_dir, "slide03_conversio_mascara.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# SLIDE 4 — U-NET I RESUNET
# ============================================================

def draw_box(ax, xy, width, height, text, fontsize=12):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.8,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(rect)

    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )


def generate_slide04_architectures():
    """
    Genera:
    - slide04_unet_i_resunet.png
    """
    out_dir = CONFIG["OUT_DIR"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------
    ax = axes[0]
    ax.axis("off")
    ax.set_title("U-Net", fontsize=20, fontweight="bold")

    draw_box(ax, (0.05, 0.70), 0.18, 0.10, "Input MRI", 13)
    draw_box(ax, (0.08, 0.55), 0.14, 0.08, "Encoder 1", 12)
    draw_box(ax, (0.10, 0.42), 0.10, 0.07, "Encoder 2", 11)
    draw_box(ax, (0.115, 0.32), 0.07, 0.06, "Bottleneck", 10)

    draw_box(ax, (0.35, 0.42), 0.10, 0.07, "Decoder 2", 11)
    draw_box(ax, (0.33, 0.55), 0.14, 0.08, "Decoder 1", 12)
    draw_box(ax, (0.32, 0.70), 0.18, 0.10, "Output Mask", 13)

    # Fletxes encoder
    ax.annotate("", xy=(0.15, 0.63), xytext=(0.15, 0.70),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.15, 0.49), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.15, 0.38), xytext=(0.15, 0.42),
                arrowprops=dict(arrowstyle="->", lw=2))

    # Fletxes decoder
    ax.annotate("", xy=(0.35, 0.45), xytext=(0.185, 0.35),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.40, 0.55), xytext=(0.40, 0.49),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.40, 0.70), xytext=(0.40, 0.63),
                arrowprops=dict(arrowstyle="->", lw=2))

    # Skip connections
    ax.annotate("", xy=(0.35, 0.59), xytext=(0.22, 0.59),
                arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--"))
    ax.annotate("", xy=(0.35, 0.455), xytext=(0.20, 0.455),
                arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--"))

    ax.text(0.27, 0.80, "Skip connections", fontsize=12)

    # --------------------------------------------------------
    # ResUNet
    # --------------------------------------------------------
    ax = axes[1]
    ax.axis("off")
    ax.set_title("ResUNet", fontsize=20, fontweight="bold")

    draw_box(ax, (0.05, 0.70), 0.20, 0.10, "Input MRI", 13)
    draw_box(ax, (0.08, 0.55), 0.18, 0.10, "Residual block", 12)
    draw_box(ax, (0.08, 0.38), 0.18, 0.10, "Residual block", 12)
    draw_box(ax, (0.40, 0.38), 0.18, 0.10, "Residual block", 12)
    draw_box(ax, (0.40, 0.55), 0.18, 0.10, "Residual block", 12)
    draw_box(ax, (0.42, 0.70), 0.20, 0.10, "Output Mask", 13)

    ax.annotate("", xy=(0.17, 0.65), xytext=(0.17, 0.70),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.17, 0.48), xytext=(0.17, 0.55),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.40, 0.43), xytext=(0.26, 0.43),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.49, 0.55), xytext=(0.49, 0.48),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.52, 0.70), xytext=(0.52, 0.65),
                arrowprops=dict(arrowstyle="->", lw=2))

    # Connexió residual
    ax.annotate("", xy=(0.26, 0.595), xytext=(0.08, 0.595),
                arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--"))
    ax.text(0.11, 0.62, "Connexió residual", fontsize=11)

    # Skip connection
    ax.annotate("", xy=(0.40, 0.60), xytext=(0.26, 0.60),
                arrowprops=dict(arrowstyle="->", lw=1.8, linestyle="--"))
    ax.text(0.31, 0.63, "Skip connection", fontsize=11)

    plt.suptitle("Esquema simplificat de U-Net i ResUNet", fontsize=22)
    plt.tight_layout()

    output_path = os.path.join(out_dir, "slide04_unet_i_resunet.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# SLIDE 6 — BASELINE I DATA LEAKAGE
# ============================================================

def generate_slide06_split_scheme():
    """
    Genera:
    - slide06_split_slices_vs_pacient.png
    """
    out_dir = CONFIG["OUT_DIR"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    ax.text(
        0.5, 0.95,
        "Split per slices vs split per pacient",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold"
    )

    # Incorrecte
    left_box = patches.FancyBboxPatch(
        (0.05, 0.12),
        0.40,
        0.72,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor="#ffe6e6"
    )
    ax.add_patch(left_box)

    ax.text(0.25, 0.78, "Incorrecte: split per slices",
            ha="center", fontsize=18, fontweight="bold")

    ax.text(
        0.25, 0.70,
        "El mateix pacient pot aparèixer\na train i validation",
        ha="center",
        fontsize=14
    )

    ax.text(0.10, 0.58, "Pacient A", ha="left", fontsize=15, fontweight="bold")
    ax.text(0.12, 0.49, "Slice 40  →  TRAIN", ha="left", fontsize=14)
    ax.text(0.12, 0.41, "Slice 41  →  VALIDATION", ha="left", fontsize=14)
    ax.text(0.12, 0.33, "Slice 42  →  TEST", ha="left", fontsize=14)

    ax.text(
        0.25, 0.22,
        "Risc de data leakage:\nvalidació massa optimista",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    # Correcte
    right_box = patches.FancyBboxPatch(
        (0.55, 0.12),
        0.40,
        0.72,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor="#e6ffe6"
    )
    ax.add_patch(right_box)

    ax.text(0.75, 0.78, "Correcte: split per pacient",
            ha="center", fontsize=18, fontweight="bold")

    ax.text(
        0.75, 0.70,
        "Totes les slices d'un pacient\nvan al mateix conjunt",
        ha="center",
        fontsize=14
    )

    ax.text(0.60, 0.58, "Pacient A  →  TRAIN", ha="left", fontsize=14)
    ax.text(0.60, 0.49, "Pacient B  →  VALIDATION", ha="left", fontsize=14)
    ax.text(0.60, 0.40, "Pacient C  →  TEST", ha="left", fontsize=14)

    ax.text(
        0.75, 0.25,
        "Avaluació més realista:\ngeneralització a pacients nous",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    output_path = os.path.join(out_dir, "slide06_split_slices_vs_pacient.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_slide06_baseline_curves():
    """
    Genera:
    - slide06_corbes_baseline.png
    """
    out_dir = CONFIG["OUT_DIR"]
    history, history_path = load_history_by_name("baseline_binary")
    output_path = os.path.join(out_dir, "slide06_corbes_baseline.png")

    if history is None:
        create_text_placeholder(
            "Corbes baseline no disponibles",
            "No s'ha trobat el history del baseline.",
            output_path
        )
        return

    metrics = [
        ("Loss", ["train_loss", "loss"], ["val_loss", "valid_loss"]),
        ("Dice", ["train_dice", "dice"], ["val_dice", "valid_dice"]),
        ("IoU", ["train_iou", "iou"], ["val_iou", "valid_iou"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, train_keys, val_keys) in zip(axes, metrics):
        train_series = get_series(history, train_keys)
        val_series = get_series(history, val_keys)

        if train_series is not None:
            epochs = range(1, len(train_series) + 1)
            ax.plot(epochs, train_series, marker="o", label="Train")

        if val_series is not None:
            epochs = range(1, len(val_series) + 1)
            ax.plot(epochs, val_series, marker="o", label="Validation")

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

    plt.suptitle("Corbes d'aprenentatge - Baseline U-Net binària", fontsize=20)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# BAR CHARTS DE COMPARACIÓ
# ============================================================

def generate_bar_comparison(experiments, title, output_path):
    """
    Genera gràfica de barres amb Dice i IoU.

    experiments = [
        {"label": "...", "dice": 0.77, "iou": 0.66}
    ]
    """
    available = [
        e for e in experiments
        if e["dice"] is not None or e["iou"] is not None
    ]

    if len(available) == 0:
        create_text_placeholder(
            "Comparació no disponible",
            "No hi ha histories suficients per generar aquesta figura.",
            output_path
        )
        return

    labels = [e["label"] for e in available]
    dice_vals = [e["dice"] if e["dice"] is not None else 0 for e in available]
    iou_vals = [e["iou"] if e["iou"] is not None else 0 for e in available]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, dice_vals, width, label="Dice")
    ax.bar(x + width / 2, iou_vals, width, label="IoU")

    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Valor", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=12)

    for i, v in enumerate(dice_vals):
        ax.text(i - width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)

    for i, v in enumerate(iou_vals):
        ax.text(i + width / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# SLIDE 7 — MÉS DADES D'ENTRADA
# ============================================================

def generate_slide07_input_scheme():
    """
    Genera:
    - slide07_input_1_vs_4_canals.png
    """
    out_dir = CONFIG["OUT_DIR"]
    patient = load_patient_modalities()

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))

    # Fila 1: FLAIR only
    axes[0, 0].imshow(patient["flair"], cmap="gray")
    axes[0, 0].set_title("FLAIR", fontsize=14)
    axes[0, 0].axis("off")

    for j in range(1, 4):
        axes[0, j].axis("off")

    axes[0, 1].text(
        0.5, 0.5,
        "Input baseline:\n1 canal",
        ha="center",
        va="center",
        fontsize=16
    )

    # Fila 2: 4 modalitats
    imgs = [patient["flair"], patient["t1"], patient["t1ce"], patient["t2"]]
    titles = ["FLAIR", "T1", "T1CE", "T2"]

    for j in range(4):
        axes[1, j].imshow(imgs[j], cmap="gray")
        axes[1, j].set_title(titles[j], fontsize=14)
        axes[1, j].axis("off")

    fig.text(
        0.5, 0.95,
        "1 modalitat vs 4 modalitats d'entrada",
        ha="center",
        fontsize=20,
        fontweight="bold"
    )

    fig.text(
        0.5, 0.90,
        "Experiment multimodal",
        ha="center",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    output_path = os.path.join(out_dir, "slide07_input_1_vs_4_canals.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_slide07_multimodal_comparison():
    """
    Genera:
    - slide07_comparacio_multimodal.png
    """
    out_dir = CONFIG["OUT_DIR"]

    baseline, _ = load_history_by_name("baseline_binary")
    multimodal, _ = load_history_by_name("multimodal_binary")

    experiments = [
        {
            "label": "U-Net FLAIR",
            "dice": get_metric_for_comparison(baseline, "dice"),
            "iou": get_metric_for_comparison(baseline, "iou"),
        },
        {
            "label": "U-Net 4 modalitats",
            "dice": get_metric_for_comparison(multimodal, "dice"),
            "iou": get_metric_for_comparison(multimodal, "iou"),
        },
    ]

    output_path = os.path.join(out_dir, "slide07_comparacio_multimodal.png")

    generate_bar_comparison(
        experiments,
        "Comparació: FLAIR vs 4 modalitats",
        output_path
    )


# ============================================================
# SLIDE 8 — WEIGHTED SAMPLER
# ============================================================

def generate_slide08_weighted_scheme():
    """
    Genera:
    - slide08_weighted_sampler_esquema.png
    """
    out_dir = CONFIG["OUT_DIR"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    ax.text(0.5, 0.93, "Weighted sampler", ha="center",
            fontsize=22, fontweight="bold")

    ax.text(
        0.5, 0.87,
        "Idea: equilibrar millor les mostres que veu el model",
        ha="center",
        fontsize=14
    )

    # Sense weighted sampler
    left_box = patches.FancyBboxPatch(
        (0.08, 0.18),
        0.34,
        0.55,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor="#f7f7f7"
    )
    ax.add_patch(left_box)

    ax.text(0.25, 0.67, "Sense weighted sampler",
            ha="center", fontsize=17, fontweight="bold")

    ax.text(0.25, 0.57, "Moltes mostres fàcils", ha="center", fontsize=13)
    ax.text(0.25, 0.51, "Molt fons", ha="center", fontsize=13)
    ax.text(0.25, 0.45, "Menys exemples difícils", ha="center", fontsize=13)

    for y in [0.38, 0.33, 0.28, 0.23]:
        circ = patches.Circle((0.19, y), 0.025, facecolor="#cccccc", edgecolor="black")
        ax.add_patch(circ)

    for y in [0.38, 0.28]:
        circ = patches.Circle((0.31, y), 0.025, facecolor="#999999", edgecolor="black")
        ax.add_patch(circ)

    # Amb weighted sampler
    right_box = patches.FancyBboxPatch(
        (0.58, 0.18),
        0.34,
        0.55,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor="#eef7ee"
    )
    ax.add_patch(right_box)

    ax.text(0.75, 0.67, "Amb weighted sampler",
            ha="center", fontsize=17, fontweight="bold")

    ax.text(0.75, 0.57, "Mostreig més equilibrat", ha="center", fontsize=13)
    ax.text(0.75, 0.51, "Més presència d'exemples", ha="center", fontsize=13)
    ax.text(0.75, 0.45, "informatius o difícils", ha="center", fontsize=13)

    positions = [
        (0.68, 0.38),
        (0.76, 0.38),
        (0.84, 0.38),
        (0.68, 0.28),
        (0.76, 0.28),
        (0.84, 0.28),
    ]

    colors = ["#cccccc", "#cccccc", "#999999", "#cccccc", "#999999", "#999999"]

    for (x, y), c in zip(positions, colors):
        circ = patches.Circle((x, y), 0.025, facecolor=c, edgecolor="black")
        ax.add_patch(circ)

    ax.text(
        0.5, 0.08,
        "Objectiu: reduir el biaix cap al fons i millorar la segmentació del tumor",
        ha="center",
        fontsize=13
    )

    output_path = os.path.join(out_dir, "slide08_weighted_sampler_esquema.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_slide08_weighted_comparison():
    """
    Genera:
    - slide08_comparacio_weighted_sampler.png
    """
    out_dir = CONFIG["OUT_DIR"]

    multimodal, _ = load_history_by_name("multimodal_binary")
    weighted_bin, _ = load_history_by_name("weighted_binary")

    multiclass_unet, _ = load_history_by_name("multiclass_unet")
    weighted_multi, _ = load_history_by_name("weighted_multiclass")

    experiments = [
        {
            "label": "U-Net 4 modalitats",
            "dice": get_metric_for_comparison(multimodal, "dice"),
            "iou": get_metric_for_comparison(multimodal, "iou"),
        },
        {
            "label": "U-Net 4 modalitats\n+ WS",
            "dice": get_metric_for_comparison(weighted_bin, "dice"),
            "iou": get_metric_for_comparison(weighted_bin, "iou"),
        },
        {
            "label": "U-Net multiclasse",
            "dice": get_metric_for_comparison(multiclass_unet, "dice"),
            "iou": get_metric_for_comparison(multiclass_unet, "iou"),
        },
        {
            "label": "U-Net multiclasse\n+ WS",
            "dice": get_metric_for_comparison(weighted_multi, "dice"),
            "iou": get_metric_for_comparison(weighted_multi, "iou"),
        },
    ]

    output_path = os.path.join(out_dir, "slide08_comparacio_weighted_sampler.png")

    generate_bar_comparison(
        experiments,
        "Comparació d'experiments amb weighted sampler",
        output_path
    )


# ============================================================
# SLIDE 9 — RESUNET
# ============================================================

def generate_slide09_resunet_comparison():
    """
    Genera:
    - slide09_comparacio_resunet.png
    """
    out_dir = CONFIG["OUT_DIR"]

    multimodal, _ = load_history_by_name("multimodal_binary")
    resunet_bin, _ = load_history_by_name("resunet_binary")

    multiclass_unet, _ = load_history_by_name("multiclass_unet")
    resunet_multi, _ = load_history_by_name("resunet_multiclass")

    experiments = [
        {
            "label": "U-Net 4 modalitats",
            "dice": get_metric_for_comparison(multimodal, "dice"),
            "iou": get_metric_for_comparison(multimodal, "iou"),
        },
        {
            "label": "ResUNet binari\n4 modalitats",
            "dice": get_metric_for_comparison(resunet_bin, "dice"),
            "iou": get_metric_for_comparison(resunet_bin, "iou"),
        },
        {
            "label": "U-Net multiclasse",
            "dice": get_metric_for_comparison(multiclass_unet, "dice"),
            "iou": get_metric_for_comparison(multiclass_unet, "iou"),
        },
        {
            "label": "ResUNet multiclasse\n4 modalitats",
            "dice": get_metric_for_comparison(resunet_multi, "dice"),
            "iou": get_metric_for_comparison(resunet_multi, "iou"),
        },
    ]

    output_path = os.path.join(out_dir, "slide09_comparacio_resunet.png")

    generate_bar_comparison(
        experiments,
        "Comparació U-Net vs ResUNet",
        output_path
    )


# ============================================================
# SLIDE 10 — PREDICCIONS QUALITATIVES
# ============================================================

def find_prediction_images():
    """
    Troba 3 imatges de prediccions.
    Prioritat:
    1. Les especificades manualment a CONFIG.
    2. Imatges trobades automàticament a results/predictions o results/figures.
    """
    manual = CONFIG["PREDICTION_IMAGES"]
    found = []

    for key in ["good", "medium", "bad"]:
        p = manual.get(key, "")
        if p and os.path.exists(p):
            found.append((key, p))

    if len(found) == 3:
        return found

    candidates = []

    for d in CONFIG["PREDICTION_SEARCH_DIRS"]:
        if os.path.exists(d):
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
                candidates.extend(
                    glob.glob(os.path.join(d, "**", ext), recursive=True)
                )

    preferred = []

    for p in sorted(candidates):
        lower = os.path.basename(p).lower()
        if any(word in lower for word in ["prediction", "pred", "overlay", "sample", "test"]):
            preferred.append(p)

    if len(preferred) >= 3:
        return [
            ("Cas 1", preferred[0]),
            ("Cas 2", preferred[1]),
            ("Cas 3", preferred[2]),
        ]

    if len(candidates) >= 3:
        candidates = sorted(candidates)
        return [
            ("Cas 1", candidates[0]),
            ("Cas 2", candidates[1]),
            ("Cas 3", candidates[2]),
        ]

    return []


def generate_slide10_predictions():
    """
    Genera:
    - slide10_prediccions_qualitatives.png
    """
    out_dir = CONFIG["OUT_DIR"]
    output_path = os.path.join(out_dir, "slide10_prediccions_qualitatives.png")

    pred_images = find_prediction_images()

    if len(pred_images) < 3:
        create_text_placeholder(
            "Prediccions qualitatives no disponibles",
            "Especifica 3 imatges a CONFIG['PREDICTION_IMAGES']\no guarda prediccions a results/predictions/",
            output_path,
            figsize=(12, 6)
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Exemple bo", "Exemple intermedi", "Exemple difícil"]

    for ax, (_, path), title in zip(axes, pred_images, titles):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    plt.suptitle("Anàlisi qualitatiu de les prediccions", fontsize=20)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# SLIDE 11 — RESUM I COMPARACIÓ GLOBAL
# ============================================================

def generate_slide11_summary():
    """
    Genera:
    - slide11_resum_experiments.png

    Inclou Dice, IoU i guany/pèrdua respecte al baseline.
    """
    out_dir = CONFIG["OUT_DIR"]
    output_path = os.path.join(out_dir, "slide11_resum_experiments.png")

    experiments_info = [
        ("U-Net FLAIR", "baseline_binary"),
        ("U-Net FLAIR all slices", "baseline_binary_all_slices"),
        ("U-Net FLAIR + aug", "baseline_binary_aug"),
        ("U-Net 4 modalitats", "multimodal_binary"),
        ("U-Net 4 modalitats + WS", "weighted_binary"),
        ("U-Net multiclasse", "multiclass_unet"),
        ("U-Net multiclasse + WS", "weighted_multiclass"),
        ("ResUNet binari 4 mod.", "resunet_binary"),
        ("ResUNet multiclasse 4 mod.", "resunet_multiclass"),
    ]

    baseline_history, _ = load_history_by_name("baseline_binary")
    baseline_dice = get_metric_for_comparison(baseline_history, "dice")
    baseline_iou = get_metric_for_comparison(baseline_history, "iou")

    rows = []

    for label, key in experiments_info:
        history, _ = load_history_by_name(key)

        dice = get_metric_for_comparison(history, "dice")
        iou = get_metric_for_comparison(history, "iou")

        if dice is not None and baseline_dice is not None:
            delta_dice = dice - baseline_dice
            delta_dice_txt = f"{delta_dice:+.3f}"
        else:
            delta_dice_txt = "N/D"

        if iou is not None and baseline_iou is not None:
            delta_iou = iou - baseline_iou
            delta_iou_txt = f"{delta_iou:+.3f}"
        else:
            delta_iou_txt = "N/D"

        rows.append([
            label,
            f"{dice:.3f}" if dice is not None else "N/D",
            delta_dice_txt,
            f"{iou:.3f}" if iou is not None else "N/D",
            delta_iou_txt,
        ])

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")

    ax.set_title(
        "Resum dels experiments i guany respecte al baseline",
        fontsize=20,
        pad=20
    )

    table = ax.table(
        cellText=rows,
        colLabels=["Experiment", "Dice", "Δ Dice", "IoU", "Δ IoU"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_slide11_global_comparison():
    """
    Genera:
    - slide11_comparacio_global_test_metrics.png
    """
    out_dir = CONFIG["OUT_DIR"]

    experiments_info = [
        ("U-Net FLAIR", "baseline_binary"),
        ("U-Net FLAIR\nall slices", "baseline_binary_all_slices"),
        ("U-Net FLAIR\n+ aug", "baseline_binary_aug"),
        ("U-Net\n4 modalitats", "multimodal_binary"),
        ("U-Net 4 mod.\n+ WS", "weighted_binary"),
        ("U-Net\nmulticlasse", "multiclass_unet"),
        ("U-Net multic.\n+ WS", "weighted_multiclass"),
        ("ResUNet bin.\n4 mod.", "resunet_binary"),
        ("ResUNet multic.\n4 mod.", "resunet_multiclass"),
    ]

    experiments = []

    for label, key in experiments_info:
        history, _ = load_history_by_name(key)

        dice = get_metric_for_comparison(history, "dice")
        iou = get_metric_for_comparison(history, "iou")

        experiments.append({
            "label": label,
            "dice": dice,
            "iou": iou,
        })

    output_path = os.path.join(out_dir, "slide11_comparacio_global_test_metrics.png")

    generate_bar_comparison(
        experiments,
        "Comparació global d'experiments",
        output_path
    )


# ============================================================
# EXECUCIÓ PRINCIPAL
# ============================================================

def main():
    ensure_dir(CONFIG["OUT_DIR"])

    print(f"\nGuardant figures a: {CONFIG['OUT_DIR']}\n")

    # Slide 3
    try:
        print("Generant figures slide 3: dades i transformació...")
        generate_slide03_data_figures()
    except Exception as e:
        print(f"Error generant slide 3: {e}")
        create_text_placeholder(
            "Slide 3 no generada",
            f"Error amb les dades MRI:\n{str(e)}",
            os.path.join(CONFIG["OUT_DIR"], "slide03_modalitats_i_mascara.png")
        )

    # Slide 4
    print("Generant figura slide 4: arquitectures...")
    generate_slide04_architectures()

    # Slide 6
    print("Generant figures slide 6: baseline i data leakage...")
    generate_slide06_split_scheme()
    generate_slide06_baseline_curves()

    # Slide 7
    try:
        print("Generant figures slide 7: entrada 1 canal vs 4 canals...")
        generate_slide07_input_scheme()
    except Exception as e:
        print(f"Error generant slide 7 input: {e}")
        create_text_placeholder(
            "Slide 7 input no generada",
            str(e),
            os.path.join(CONFIG["OUT_DIR"], "slide07_input_1_vs_4_canals.png")
        )

    print("Generant comparació slide 7: FLAIR vs 4 modalitats...")
    generate_slide07_multimodal_comparison()

    # Slide 8
    print("Generant figures slide 8: weighted sampler...")
    generate_slide08_weighted_scheme()
    generate_slide08_weighted_comparison()

    # Slide 9
    print("Generant figura slide 9: U-Net vs ResUNet...")
    generate_slide09_resunet_comparison()

    # Slide 10
    print("Generant figura slide 10: prediccions qualitatives...")
    generate_slide10_predictions()

    # Slide 11
    print("Generant figura slide 11: resum d'experiments...")
    generate_slide11_summary()

    print("Generant comparació global slide 11...")
    generate_slide11_global_comparison()

    print("\nTotes les figures s'han generat correctament.")
    print(f"Revisa la carpeta: {CONFIG['OUT_DIR']}\n")


if __name__ == "__main__":
    main()
