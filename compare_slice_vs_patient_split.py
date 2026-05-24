import json
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

HISTORY_DIR = "results/history"

PATIENT_SPLIT_HISTORY = os.path.join(
    HISTORY_DIR,
    "unet_flair_patient_split_20epochs_history.json"
)

SLICE_SPLIT_HISTORY = os.path.join(
    HISTORY_DIR,
    "unet_flair_patient_split_20epochs_all_slices_history.json"
)

OUT_DIR = "results/figures_split_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el fichero: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_best(values, mode="max"):
    if values is None:
        return None

    if isinstance(values, list):
        if len(values) == 0:
            return None
        return max(values) if mode == "max" else min(values)

    return values


def get_last(values):
    if values is None:
        return None

    if isinstance(values, list):
        if len(values) == 0:
            return None
        return values[-1]

    return values


def get_metric(history, key, mode="max"):
    """
    Devuelve:
    - valor de test si existe,
    - si no, mejor valor de validation,
    - si no, último valor de train.
    """

    test_key = f"test_{key}"
    val_key = f"val_{key}"
    train_key = f"train_{key}"

    if test_key in history:
        return get_last(history[test_key])

    if val_key in history:
        return get_best(history[val_key], mode=mode)

    if train_key in history:
        return get_last(history[train_key])

    return None


def plot_metric_comparison(
    patient_history,
    slice_history,
    train_key,
    val_key,
    title,
    ylabel,
    output_name
):
    """
    Genera una gráfica con:
    - train patient split
    - val patient split
    - train all slices
    - val all slices
    """

    plt.figure(figsize=(10, 6))

    if train_key in patient_history:
        epochs = range(1, len(patient_history[train_key]) + 1)
        plt.plot(
            epochs,
            patient_history[train_key],
            marker="o",
            label="Train - split pacient"
        )

    if val_key in patient_history:
        epochs = range(1, len(patient_history[val_key]) + 1)
        plt.plot(
            epochs,
            patient_history[val_key],
            marker="o",
            label="Validation - split pacient"
        )

    if train_key in slice_history:
        epochs = range(1, len(slice_history[train_key]) + 1)
        plt.plot(
            epochs,
            slice_history[train_key],
            marker="o",
            linestyle="--",
            label="Train - all slices"
        )

    if val_key in slice_history:
        epochs = range(1, len(slice_history[val_key]) + 1)
        plt.plot(
            epochs,
            slice_history[val_key],
            marker="o",
            linestyle="--",
            label="Validation - all slices"
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_path = os.path.join(OUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada: {output_path}")


def print_summary_table(patient_history, slice_history):
    """
    Imprime una tabla resumen con Loss, Dice e IoU.
    """

    experiments = {
        "Split pacient": patient_history,
        "All slices": slice_history,
    }

    print("\nResumen comparativo")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Best Val Loss':<15} {'Best Val Dice':<15} {'Best Val IoU':<15} {'Test Dice':<12} {'Test IoU':<12}")
    print("-" * 80)

    for name, history in experiments.items():
        best_val_loss = get_best(history.get("val_loss"), mode="min")
        best_val_dice = get_best(history.get("val_dice"), mode="max")
        best_val_iou = get_best(history.get("val_iou"), mode="max")

        test_dice = history.get("test_dice", None)
        test_iou = history.get("test_iou", None)

        if isinstance(test_dice, list):
            test_dice = test_dice[-1]
        if isinstance(test_iou, list):
            test_iou = test_iou[-1]

        print(
            f"{name:<20} "
            f"{best_val_loss if best_val_loss is not None else 'N/D':<15.4f} "
            f"{best_val_dice if best_val_dice is not None else 'N/D':<15.4f} "
            f"{best_val_iou if best_val_iou is not None else 'N/D':<15.4f} "
            f"{test_dice if test_dice is not None else 'N/D':<12.4f} "
            f"{test_iou if test_iou is not None else 'N/D':<12.4f}"
        )

    print("=" * 80)


def generate_bar_summary(patient_history, slice_history):
    """
    Genera una gráfica de barras con Best Val Dice, Best Val IoU y Best Val Loss.
    """

    labels = ["Split pacient", "All slices"]

    best_val_dice = [
        get_best(patient_history.get("val_dice"), mode="max"),
        get_best(slice_history.get("val_dice"), mode="max"),
    ]

    best_val_iou = [
        get_best(patient_history.get("val_iou"), mode="max"),
        get_best(slice_history.get("val_iou"), mode="max"),
    ]

    best_val_loss = [
        get_best(patient_history.get("val_loss"), mode="min"),
        get_best(slice_history.get("val_loss"), mode="min"),
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 6))

    plt.bar(x - width, best_val_dice, width, label="Best Val Dice")
    plt.bar(x, best_val_iou, width, label="Best Val IoU")
    plt.bar(x + width, best_val_loss, width, label="Best Val Loss")

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel("Valor", fontsize=13)
    plt.title("Comparació split per pacient vs all slices", fontsize=16)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=11)

    for i, values in enumerate([best_val_dice, best_val_iou, best_val_loss]):
        offset = [-width, 0, width][i]
        for j, value in enumerate(values):
            if value is not None:
                plt.text(
                    x[j] + offset,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=10
                )

    plt.tight_layout()

    output_path = os.path.join(OUT_DIR, "summary_split_patient_vs_all_slices.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura guardada: {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    patient_history = load_history(PATIENT_SPLIT_HISTORY)
    slice_history = load_history(SLICE_SPLIT_HISTORY)

    print_summary_table(patient_history, slice_history)

    plot_metric_comparison(
        patient_history=patient_history,
        slice_history=slice_history,
        train_key="train_loss",
        val_key="val_loss",
        title="Loss: split per pacient vs all slices",
        ylabel="Loss",
        output_name="loss_split_patient_vs_all_slices.png"
    )

    plot_metric_comparison(
        patient_history=patient_history,
        slice_history=slice_history,
        train_key="train_dice",
        val_key="val_dice",
        title="Dice: split per pacient vs all slices",
        ylabel="Dice Score",
        output_name="dice_split_patient_vs_all_slices.png"
    )

    plot_metric_comparison(
        patient_history=patient_history,
        slice_history=slice_history,
        train_key="train_iou",
        val_key="val_iou",
        title="IoU: split per pacient vs all slices",
        ylabel="IoU",
        output_name="iou_split_patient_vs_all_slices.png"
    )

    generate_bar_summary(patient_history, slice_history)

    print("\nListo. Figuras guardadas en:")
    print(OUT_DIR)
