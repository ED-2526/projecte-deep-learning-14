import os
import json

from utils.visualization import plot_metric


def main():
    history_path = "results/history/unet_flair_patient_split_20epochs_all_slices_history.json"
    figures_dir = "results/figures"

    os.makedirs(figures_dir, exist_ok=True)

    with open(history_path, "r") as f:
        history = json.load(f)

    plot_metric(
        history=history,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Training Loss vs Validation Loss",
        save_path=os.path.join(figures_dir, "loss_curve.png")
    )

    plot_metric(
        history=history,
        train_key="train_dice",
        val_key="val_dice",
        ylabel="Dice Score",
        title="Training Dice vs Validation Dice",
        save_path=os.path.join(figures_dir, "dice_curve.png")
    )

    plot_metric(
        history=history,
        train_key="train_iou",
        val_key="val_iou",
        ylabel="IoU Score",
        title="Training IoU vs Validation IoU",
        save_path=os.path.join(figures_dir, "iou_curve.png")
    )

    print("Gràfiques generades correctament:")
    print(os.path.join(figures_dir, "loss_curve.png"))
    print(os.path.join(figures_dir, "dice_curve.png"))
    print(os.path.join(figures_dir, "iou_curve.png"))


if __name__ == "__main__":
    main()
