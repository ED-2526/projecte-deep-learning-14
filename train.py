import torch
from tqdm.auto import tqdm
import wandb

from utils.metrics import dice_score, iou_score


def get_config_value(config, key, default=None):
    """
    Permet llegir valors tant si config és un dict com si és wandb.config.
    """
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entrena el model durant una epoch.
    """
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mètriques
        total_loss += loss.item()
        total_dice += dice_score(outputs.detach(), masks.detach())
        total_iou += iou_score(outputs.detach(), masks.detach())

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    return avg_loss, avg_dice, avg_iou


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Avalua el model sobre el conjunt de validació.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    return avg_loss, avg_dice, avg_iou


def train(model, train_loader, val_loader, criterion, optimizer, config, device, save_path="best_model.pth"):
    """
    Entrenament complet del model.

    Guarda el millor model segons el Dice Score de validació.
    """
    epochs = get_config_value(config, "epochs", 10)

    best_val_dice = 0.0
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    try:
        wandb.watch(model, criterion, log="all", log_freq=10)
    except Exception:
        pass

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_dice, val_iou = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | Val IoU:   {val_iou:.4f}")

        try:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou
            })
        except Exception:
            pass

        # Guardar el millor model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"Millor model guardat amb Val Dice = {best_val_dice:.4f}")

    return history
