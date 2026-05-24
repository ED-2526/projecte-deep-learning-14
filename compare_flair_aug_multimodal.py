import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓ
# ============================================================

HISTORY_DIR = "results/history"

HISTORIES = {
    "unet_flair": os.path.join(
        HISTORY_DIR,
        "unet_flair_patient_split_20epochs_history.json"
    ),
    "unet_flair_aug": os.path.join(
        HISTORY_DIR,
        "unet_flair_patient_split_20epochs_aug_history.json"
    ),
    "unet_multimodal": os.path.join(
        HISTORY_DIR,
        "unet_multimodal_20epochs_bce_tversky_history.json"
    ),
}

OUT_DIR = "results/figures_flair_aug_multimodal"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# FUNCIONS AUXILIARS
# ============================================================

def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_series(history, possible_keys):
    for key in possible_keys:
        if key in history and isinstance(history[key], list) and len(history[key]) > 0:
            return history[key]
    return None


def get_metric(history, metric_type="dice"):
    """
    Prioritat:
    1. test metric si existeix
    2. millor validació
    3. últim valor de train
    """

    if metric_type == "loss":
        test_keys = ["test_loss"]
        val_keys = ["val_loss", "valid_loss"]
        train_keys = ["train_loss", "loss"]
        reduce_fn = np.min

    elif metric_type == "dice":
        test_keys = ["test_dice", "test_dice_score"]
        val_keys = ["val_dice", "valid_dice", "val_dice_score"]
        train_keys = ["train_dice", "dice"]
        reduce_fn = np.max

    elif metric_type == "iou":
        test_keys = ["test_iou", "test_iou_score"]
        val_keys = ["val_iou", "valid_iou", "val_iou_score"]
        train_keys = ["train_iou", "iou"]
        reduce_fn = np.max

    else:
        raise ValueError(f"Metric type desconegut: {metric_type}")

    # 1. Test metric
    for key in test_keys:
        if key in history:
            value = history[key]
            if isinstance(value, list):
                return float(value[-1])
            return float(value)

    # 2. Validation
    val_series = get_series(history, val_keys)
    if val_series is not None:
        return float(reduce_fn(val_series))

    # 3. Train
    train_series = get_series(history, train_keys)
    if train_series is not None:
        return float(train_series[-1])

    return None


def get_metric_source(history, metric_type="dice"):
    """
    Retorna d'on surt la mètrica:
    - Test
    - Best validation
    - Train final
    """

    if metric_type == "loss":
        test_keys = ["test_loss"]
        val_keys = ["val_loss", "valid_loss"]
        train_keys = ["train_loss", "loss"]

    elif metric_type == "dice":
        test_keys = ["test_dice", "test_dice_score"]
        val_keys = ["val_dice", "valid_dice", "val_dice_score"]
        train_keys = ["train_dice", "dice"]

    elif metric_type == "iou":
        test_keys = ["test_iou", "test_iou_score"]
        val_keys = ["val_iou", "valid_iou", "val_iou_score"]
        train_keys = ["train_iou", "iou"]

    else:
        return "N/D"

    for key in test_keys:
        if key in history:
            return "Test"

    for key in val_keys:
        if key in history:
            return "Best Val"

    for key in train_keys:
        if key in history:
            return "Train final"

    return "N/D"


def print_summary(histories):
    print("\nRESUM DE MÈTRIQUES")
    print("=" * 95)
    print(
        f"{'Experiment':<25} "
        f"{'Loss':<10} "
        f"{'Dice':<10} "
        f"{'IoU':<10} "
        f"{'Origen':<12}"
    )
    print("-" * 95)

    for name, history in histories.items():
        loss = get_metric(history, "loss")
        dice = get_metric(history, "dice")
        iou = get_metric(history, "iou")
        source = get_metric_source(history, "dice")

        print(
            f"{name:<25} "
            f"{loss if loss is not None else 0:<10.4f} "
            f"{dice if dice is not None else 0:<10.4f} "
            f"{iou if iou is not None else 0:<10.4f} "
            f"{source:<12}"
        )

    print("=" * 95)


def plot_curves(histories, metric_name, train_key, val_key, ylabel, title, output_name):
    """
    Dibuixa corbes train/validation per diversos experiments.
    """

    plt.figure(figsize=(11, 6))

    for exp_name, history in histories.items():
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            plt.plot(
                epochs,
                history[train_key],
                linestyle="--",
                marker="o",
                label=f"{exp_name} - train"
            )

        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            plt.plot(
                epochs,
                history[val_key],
                marker="o",
                label=f"{exp_name} - validation"
            )

    plt.title(title, fontsize=16)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(OUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada: {output_path}")


def plot_bar_comparison(experiments, title, output_name):
    """
    Gràfica de barres amb Loss, Dice i IoU.
    """

    labels = list(experiments.keys())

    loss_values = [get_metric(experiments[label], "loss") for label in labels]
    dice_values = [get_metric(experiments[label], "dice") for label in labels]
    iou_values = [get_metric(experiments[label], "iou") for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, dice_values, width, label="Dice")
    plt.bar(x, iou_values, width, label="IoU")
    plt.bar(x + width, loss_values, width, label="Loss")

    plt.xticks(x, labels, rotation=15, ha="right", fontsize=11)
    plt.ylabel("Valor", fontsize=13)
    plt.title(title, fontsize=16)
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=11)

    for values, offset in [
        (dice_values, -width),
        (iou_values, 0),
        (loss_values, width),
    ]:
        for i, value in enumerate(values):
            if value is not None:
                plt.text(
                    x[i] + offset,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=9
                )

    plt.tight_layout()

    output_path = os.path.join(OUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada: {output_path}")


def plot_delta_table(reference_name, experiments, output_name):
    """
    Crea una taula amb guany/pèrdua respecte a un experiment de referència.
    """

    ref_history = experiments[reference_name]
    ref_loss = get_metric(ref_history, "loss")
    ref_dice = get_metric(ref_history, "dice")
    ref_iou = get_metric(ref_history, "iou")

    rows = []

    for name, history in experiments.items():
        loss = get_metric(history, "loss")
        dice = get_metric(history, "dice")
        iou = get_metric(history, "iou")

        delta_loss = loss - ref_loss if loss is not None and ref_loss is not None else None
        delta_dice = dice - ref_dice if dice is not None and ref_dice is not None else None
        delta_iou = iou - ref_iou if iou is not None and ref_iou is not None else None

        rows.append([
            name,
            f"{loss:.3f}" if loss is not None else "N/D",
            f"{delta_loss:+.3f}" if delta_loss is not None else "N/D",
            f"{dice:.3f}" if dice is not None else "N/D",
            f"{delta_dice:+.3f}" if delta_dice is not None else "N/D",
            f"{iou:.3f}" if iou is not None else "N/D",
            f"{delta_iou:+.3f}" if delta_iou is not None else "N/D",
        ])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    ax.set_title(
        f"Guany/pèrdua respecte a {reference_name}",
        fontsize=17,
        pad=20
    )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Experiment",
            "Loss",
            "Δ Loss",
            "Dice",
            "Δ Dice",
            "IoU",
            "Δ IoU",
        ],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    output_path = os.path.join(OUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada: {output_path}")


def select_best_flair(histories):
    """
    Selecciona el millor FLAIR entre normal i augmentació segons Dice.
    """
    candidates = {
        "U-Net FLAIR": histories["U-Net FLAIR"],
        "U-Net FLAIR + aug": histories["U-Net FLAIR + aug"],
    }

    best_name = None
    best_dice = -1

    for name, history in candidates.items():
        dice = get_metric(history, "dice")
        if dice is not None and dice > best_dice:
            best_dice = dice
            best_name = name

    return best_name, candidates[best_name]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    histories = {
        "U-Net FLAIR": load_history(HISTORIES["unet_flair"]),
        "U-Net FLAIR + aug": load_history(HISTORIES["unet_flair_aug"]),
        "U-Net multimodal": load_history(HISTORIES["unet_multimodal"]),
    }

    print_summary(histories)

    # ========================================================
    # COMPARACIÓ 1:
    # U-Net FLAIR normal vs U-Net FLAIR + augmentació
    # ========================================================

    flair_aug_histories = {
        "U-Net FLAIR": histories["U-Net FLAIR"],
        "U-Net FLAIR + aug": histories["U-Net FLAIR + aug"],
    }

    plot_bar_comparison(
        experiments=flair_aug_histories,
        title="Comparació U-Net FLAIR: normal vs augmentació",
        output_name="01_bar_flair_vs_flair_aug.png"
    )

    plot_delta_table(
        reference_name="U-Net FLAIR",
        experiments=flair_aug_histories,
        output_name="02_table_delta_flair_aug.png"
    )

    plot_curves(
        histories=flair_aug_histories,
        metric_name="loss",
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Loss: U-Net FLAIR normal vs augmentació",
        output_name="03_loss_flair_vs_flair_aug.png"
    )

    plot_curves(
        histories=flair_aug_histories,
        metric_name="dice",
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Dice: U-Net FLAIR normal vs augmentació",
        output_name="04_dice_flair_vs_flair_aug.png"
    )

    plot_curves(
        histories=flair_aug_histories,
        metric_name="iou",
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title="IoU: U-Net FLAIR normal vs augmentació",
        output_name="05_iou_flair_vs_flair_aug.png"
    )

    # ========================================================
    # COMPARACIÓ 2:
    # Millor U-Net FLAIR vs U-Net multimodal
    # ========================================================

    best_flair_name, best_flair_history = select_best_flair(histories)

    flair_vs_multimodal_histories = {
        best_flair_name: best_flair_history,
        "U-Net multimodal": histories["U-Net multimodal"],
    }

    print(f"\nMillor model FLAIR segons Dice: {best_flair_name}")

    plot_bar_comparison(
        experiments=flair_vs_multimodal_histories,
        title=f"Comparació {best_flair_name} vs U-Net multimodal",
        output_name="06_bar_best_flair_vs_multimodal.png"
    )

    plot_delta_table(
        reference_name=best_flair_name,
        experiments=flair_vs_multimodal_histories,
        output_name="07_table_delta_best_flair_multimodal.png"
    )

    plot_curves(
        histories=flair_vs_multimodal_histories,
        metric_name="loss",
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title=f"Loss: {best_flair_name} vs U-Net multimodal",
        output_name="08_loss_best_flair_vs_multimodal.png"
    )

    plot_curves(
        histories=flair_vs_multimodal_histories,
        metric_name="dice",
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title=f"Dice: {best_flair_name} vs U-Net multimodal",
        output_name="09_dice_best_flair_vs_multimodal.png"
    )

    plot_curves(
        histories=flair_vs_multimodal_histories,
        metric_name="iou",
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title=f"IoU: {best_flair_name} vs U-Net multimodal",
        output_name="10_iou_best_flair_vs_multimodal.png"
    )

    print("\nFigures generades a:")
    print(OUT_DIR)
