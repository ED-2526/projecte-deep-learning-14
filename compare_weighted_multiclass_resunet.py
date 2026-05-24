import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓ
# ============================================================

HISTORY_DIR = "results/history"

HISTORIES = {
    # Experiment 5: weighted sampler en binari amb 4 modalitats
    "unet_4modalities": os.path.join(
        HISTORY_DIR,
        "unet_multimodal_20epochs_bce_tversky_history.json"
    ),
    "unet_4modalities_ws": os.path.join(
        HISTORY_DIR,
        "unet_binary_4modalities_20epochs_bce_dice_weighted_sampler_history.json"
    ),

    # Experiment 6: multiclasse
    "unet_multiclass": os.path.join(
        HISTORY_DIR,
        "unet_multiclass_4modalities_20epochs_ce_dice_history.json"
    ),
    "unet_multiclass_ws": os.path.join(
        HISTORY_DIR,
        "unet_multiclass_4modalities_20epochs_ce_dice_weighted_sampler_history.json"
    ),

    # Experiment 7: ResUNet
    "resunet_binary": os.path.join(
        HISTORY_DIR,
        "resunet_binary_4modalities_20epochs_bce_dice_history.json"
    ),
    "resunet_multiclass": os.path.join(
        HISTORY_DIR,
        "resunet_multiclass_4modalities_20epochs_ce_dice_history.json"
    ),
}

OUT_DIR = "results/figures_weighted_multiclass_resunet"
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
    2. millor validation
    3. últim train
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

    # 2. Best validation
    val_series = get_series(history, val_keys)
    if val_series is not None:
        return float(reduce_fn(val_series))

    # 3. Train final
    train_series = get_series(history, train_keys)
    if train_series is not None:
        return float(train_series[-1])

    return None


def get_metric_source(history, metric_type="dice"):
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
    print("=" * 110)
    print(
        f"{'Experiment':<35} "
        f"{'Loss':<10} "
        f"{'Dice':<10} "
        f"{'IoU':<10} "
        f"{'Origen':<12}"
    )
    print("-" * 110)

    for name, history in histories.items():
        loss = get_metric(history, "loss")
        dice = get_metric(history, "dice")
        iou = get_metric(history, "iou")
        source = get_metric_source(history, "dice")

        print(
            f"{name:<35} "
            f"{loss if loss is not None else 0:<10.4f} "
            f"{dice if dice is not None else 0:<10.4f} "
            f"{iou if iou is not None else 0:<10.4f} "
            f"{source:<12}"
        )

    print("=" * 110)


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

    plt.figure(figsize=(11, 6))

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
    Taula amb guany/pèrdua respecte a un experiment de referència.
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


def plot_curves(histories, train_key, val_key, ylabel, title, output_name):
    """
    Corbes train/validation per diversos experiments.
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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    histories = {
        "U-Net 4 modalitats": load_history(HISTORIES["unet_4modalities"]),
        "U-Net 4 modalitats + WS": load_history(HISTORIES["unet_4modalities_ws"]),
        "U-Net multiclasse": load_history(HISTORIES["unet_multiclass"]),
        "U-Net multiclasse + WS": load_history(HISTORIES["unet_multiclass_ws"]),
        "ResUNet binari 4 mod.": load_history(HISTORIES["resunet_binary"]),
        "ResUNet multiclasse": load_history(HISTORIES["resunet_multiclass"]),
    }

    print_summary(histories)

    # ========================================================
    # EXPERIMENT 5: WEIGHTED SAMPLER EN BINARI 4 MODALITATS
    # U-Net 4 modalitats vs U-Net 4 modalitats + WS
    # ========================================================

    weighted_binary_comparison = {
        "U-Net 4 modalitats": histories["U-Net 4 modalitats"],
        "U-Net 4 modalitats + WS": histories["U-Net 4 modalitats + WS"],
    }

    plot_bar_comparison(
        experiments=weighted_binary_comparison,
        title="Experiment 5: U-Net 4 modalitats vs weighted sampler",
        output_name="01_bar_4modalities_vs_weighted_sampler.png"
    )

    plot_delta_table(
        reference_name="U-Net 4 modalitats",
        experiments=weighted_binary_comparison,
        output_name="02_table_delta_4modalities_weighted_sampler.png"
    )

    plot_curves(
        histories=weighted_binary_comparison,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Loss: U-Net 4 modalitats vs weighted sampler",
        output_name="03_loss_4modalities_vs_weighted_sampler.png"
    )

    plot_curves(
        histories=weighted_binary_comparison,
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Dice: U-Net 4 modalitats vs weighted sampler",
        output_name="04_dice_4modalities_vs_weighted_sampler.png"
    )

    plot_curves(
        histories=weighted_binary_comparison,
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title="IoU: U-Net 4 modalitats vs weighted sampler",
        output_name="05_iou_4modalities_vs_weighted_sampler.png"
    )

    # ========================================================
    # EXPERIMENT 6: MULTICLASSE
    # U-Net multiclasse vs U-Net multiclasse + WS
    # ========================================================

    multiclass_comparison = {
        "U-Net multiclasse": histories["U-Net multiclasse"],
        "U-Net multiclasse + WS": histories["U-Net multiclasse + WS"],
    }

    plot_bar_comparison(
        experiments=multiclass_comparison,
        title="Experiment 6: U-Net multiclasse vs weighted sampler",
        output_name="06_bar_multiclass_vs_weighted_sampler.png"
    )

    plot_delta_table(
        reference_name="U-Net multiclasse",
        experiments=multiclass_comparison,
        output_name="07_table_delta_multiclass_weighted_sampler.png"
    )

    plot_curves(
        histories=multiclass_comparison,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Loss: multiclasse vs multiclasse + weighted sampler",
        output_name="08_loss_multiclass_vs_weighted_sampler.png"
    )

    plot_curves(
        histories=multiclass_comparison,
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Dice: multiclasse vs multiclasse + weighted sampler",
        output_name="09_dice_multiclass_vs_weighted_sampler.png"
    )

    plot_curves(
        histories=multiclass_comparison,
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title="IoU: multiclasse vs multiclasse + weighted sampler",
        output_name="10_iou_multiclass_vs_weighted_sampler.png"
    )

    # ========================================================
    # EXPERIMENT 7: RESUNET
    # Comparació binària i multiclasse
    # ========================================================

    resunet_binary_comparison = {
        "U-Net 4 modalitats": histories["U-Net 4 modalitats"],
        "ResUNet binari 4 mod.": histories["ResUNet binari 4 mod."],
    }

    plot_bar_comparison(
        experiments=resunet_binary_comparison,
        title="Experiment 7A: U-Net vs ResUNet en binari",
        output_name="11_bar_unet_vs_resunet_binary.png"
    )

    plot_delta_table(
        reference_name="U-Net 4 modalitats",
        experiments=resunet_binary_comparison,
        output_name="12_table_delta_unet_resunet_binary.png"
    )

    resunet_multiclass_comparison = {
        "U-Net multiclasse": histories["U-Net multiclasse"],
        "ResUNet multiclasse": histories["ResUNet multiclasse"],
    }

    plot_bar_comparison(
        experiments=resunet_multiclass_comparison,
        title="Experiment 7B: U-Net vs ResUNet en multiclasse",
        output_name="13_bar_unet_vs_resunet_multiclass.png"
    )

    plot_delta_table(
        reference_name="U-Net multiclasse",
        experiments=resunet_multiclass_comparison,
        output_name="14_table_delta_unet_resunet_multiclass.png"
    )

    plot_curves(
        histories=resunet_binary_comparison,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Loss: U-Net vs ResUNet binari",
        output_name="15_loss_unet_vs_resunet_binary.png"
    )

    plot_curves(
        histories=resunet_binary_comparison,
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Dice: U-Net vs ResUNet binari",
        output_name="16_dice_unet_vs_resunet_binary.png"
    )

    plot_curves(
        histories=resunet_binary_comparison,
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title="IoU: U-Net vs ResUNet binari",
        output_name="17_iou_unet_vs_resunet_binary.png"
    )

    plot_curves(
        histories=resunet_multiclass_comparison,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Loss: U-Net vs ResUNet multiclasse",
        output_name="18_loss_unet_vs_resunet_multiclass.png"
    )

    plot_curves(
        histories=resunet_multiclass_comparison,
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Dice: U-Net vs ResUNet multiclasse",
        output_name="19_dice_unet_vs_resunet_multiclass.png"
    )

    plot_curves(
        histories=resunet_multiclass_comparison,
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU",
        title="IoU: U-Net vs ResUNet multiclasse",
        output_name="20_iou_unet_vs_resunet_multiclass.png"
    )

    print("\nFigures generades a:")
    print(OUT_DIR)
