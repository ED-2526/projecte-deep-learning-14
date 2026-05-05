import os
import matplotlib.pyplot as plt


def plot_metric(history, train_key, val_key, ylabel, title, save_path):
    """
    Genera una gràfica comparant una mètrica de train i validation.

    Parameters
    ----------
    history : dict
        Diccionari amb les mètriques de l'entrenament.
    train_key : str
        Clau de la mètrica de train dins de history.
    val_key : str
        Clau de la mètrica de validation dins de history.
    ylabel : str
        Nom de l'eix Y.
    title : str
        Títol de la figura.
    save_path : str
        Ruta on es guardarà la figura.
    """
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


def save_prediction_figure(image, mask, pred, save_path, title=None):
    """
    Guarda una figura amb:
        1. Imatge MRI
        2. Màscara real
        3. Màscara predita
        4. Overlay de la predicció sobre la imatge

    Parameters
    ----------
    image : np.ndarray
        Imatge MRI amb forma [1, H, W] o [H, W].
    mask : np.ndarray
        Màscara real amb forma [1, H, W] o [H, W].
    pred : np.ndarray
        Màscara predita amb forma [1, H, W] o [H, W].
    save_path : str
        Ruta on es guardarà la figura.
    title : str, optional
        Títol general de la figura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image = image.squeeze()
    mask = mask.squeeze()
    pred = pred.squeeze()

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI FLAIR")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image, cmap="gray")
    plt.imshow(pred, alpha=0.4)
    plt.title("Overlay prediction")
    plt.axis("off")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
