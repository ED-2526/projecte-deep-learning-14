import os
import json
import argparse
import matplotlib.pyplot as plt


def load_history(history_path):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No existeix el fitxer history: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    return history


def get_metric(history, key):
    """
    Retorna una mètrica del history si existeix.
    Si no existeix, retorna None.
    """
    return history.get(key, None)


def plot_curve(train_values, val_values, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_values) + 1)

    plt.plot(epochs, train_values, marker="o", label=f"train_{ylabel.lower()}")
    plt.plot(epochs, val_values, marker="o", label=f"val_{ylabel.lower()}")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_single_curve(values, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(values) + 1)

    plt.plot(epochs, values, marker="o", label=ylabel)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history-path",
        type=str,
        default="results/history/unet_multiclass_4modalities_20epochs_ce_dice_history.json",
        help="Ruta al fitxer history JSON."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figures_multiclass",
        help="Carpeta on guardar les figures."
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    history = load_history(args.history_path)

    train_loss = get_metric(history, "train_loss")
    val_loss = get_metric(history, "val_loss")

    train_dice = get_metric(history, "train_dice")
    val_dice = get_metric(history, "val_dice")

    train_iou = get_metric(history, "train_iou")
    val_iou = get_metric(history, "val_iou")

    if train_loss is not None and val_loss is not None:
        plot_curve(
            train_values=train_loss,
            val_values=val_loss,
            title="Training Loss vs Validation Loss",
            ylabel="Loss",
            save_path=os.path.join(args.out_dir, "loss_curve.png")
        )

    if train_dice is not None and val_dice is not None:
        plot_curve(
            train_values=train_dice,
            val_values=val_dice,
            title="Training Dice vs Validation Dice",
            ylabel="Dice",
            save_path=os.path.join(args.out_dir, "dice_curve.png")
        )

    if train_iou is not None and val_iou is not None:
        plot_curve(
            train_values=train_iou,
            val_values=val_iou,
            title="Training IoU vs Validation IoU",
            ylabel="IoU",
            save_path=os.path.join(args.out_dir, "iou_curve.png")
        )

    print("\nGràfiques generades correctament:")
    print(args.out_dir)

    if "test_loss" in history:
        print("\nResultats test guardats al history:")
        print(f"Test Loss: {history.get('test_loss')}")
        print(f"Test Dice: {history.get('test_dice')}")
        print(f"Test IoU:  {history.get('test_iou')}")


if __name__ == "__main__":
    main()
