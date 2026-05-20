import os
import json
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓ
# ============================================================

HISTORY_PATH =  "results/history/unet_multiclass_4modalities_20epochs_ce_dice_history.json"
OUTPUT_DIR = "results/figures_multiclase"


# ============================================================
# FUNCIONS
# ============================================================

def load_history(history_path):
    """
    Carrega l'historial d'entrenament guardat en format JSON.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No s'ha trobat el fitxer d'historial: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    return history


def plot_metric(history, train_key, val_key, title, ylabel, save_path):
    """
    Genera una gràfica comparant una mètrica de train i validation.
    """
    plt.figure(figsize=(8, 5))

    if train_key in history:
        plt.plot(history[train_key], label=train_key)

    if val_key in history:
        plt.plot(history[val_key], label=val_key)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Gràfica guardada: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = load_history(HISTORY_PATH)

    plot_metric(
        history=history,
        train_key="train_loss",
        val_key="val_loss",
        title="Training vs Validation Loss",
        ylabel="Loss",
        save_path=os.path.join(OUTPUT_DIR, "loss_curve.png")
    )

    plot_metric(
        history=history,
        train_key="train_dice",
        val_key="val_dice",
        title="Training vs Validation Dice",
        ylabel="Dice Score",
        save_path=os.path.join(OUTPUT_DIR, "dice_curve.png")
    )

    plot_metric(
        history=history,
        train_key="train_iou",
        val_key="val_iou",
        title="Training vs Validation IoU",
        ylabel="IoU",
        save_path=os.path.join(OUTPUT_DIR, "iou_curve.png")
    )

    print("\nGràfiques generades correctament.")
    print(f"Carpeta de sortida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
