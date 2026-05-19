import os
import numpy as np
import matplotlib.pyplot as plt


def plot_metric(history, train_key, val_key, ylabel, title, save_path):
    """Genera una gràfica comparant una mètrica de train i validation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history[train_key]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[train_key], marker="o", label=train_key)
    plt.plot(epochs, history[val_key], marker="o", label=val_key)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()


def _prepare_display_mask(mask):
    mask = np.asarray(mask).squeeze()
    return mask


def save_prediction_figure(image, mask, pred, save_path, title=None, num_classes=None):
    """
    Guarda una figura amb imatge MRI, màscara real, predicció i overlay.

    Funciona tant per binari com per multiclasse:
        - binari: valors 0/1
        - multiclasse: valors 0/1/2/3
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image = np.asarray(image).squeeze()
    mask = _prepare_display_mask(mask)
    pred = _prepare_display_mask(pred)

    is_multiclass = num_classes is not None and num_classes > 2
    cmap = "tab10" if is_multiclass else "gray"
    vmin = 0
    vmax = (num_classes - 1) if is_multiclass else 1

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI FLAIR")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image, cmap="gray")
    # En overlay fem transparent el fons perquè destaquin les classes tumorals.
    overlay = np.ma.masked_where(pred == 0, pred)
    plt.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.45)
    plt.title("Overlay prediction")
    plt.axis("off")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
