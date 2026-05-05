# Importem PyTorch.
# Ens cal per moure tensors a GPU/CPU, calcular gradients i guardar models.
import torch

# Importem tqdm per mostrar barres de progrés durant train i validation.
# tqdm.auto intenta adaptar-se bé tant a terminal com a notebooks.
from tqdm.auto import tqdm

# Importem wandb per registrar mètriques de l'entrenament.
# En el nostre projecte s'utilitza en mode offline.
import wandb

# Importem les mètriques de segmentació que hem definit a utils/metrics.py.
# Les utilitzarem per calcular Dice i IoU durant train i validation.
from utils.metrics import dice_score, iou_score


def get_config_value(config, key, default=None):
    """
    Permet llegir valors tant si config és un diccionari normal
    com si és un objecte wandb.config.

    Args:
        config: configuració de l'experiment.
        key: nom del paràmetre que volem llegir.
        default: valor per defecte si el paràmetre no existeix.

    Returns:
        valor associat a key.
    """

    # Si config és un diccionari de Python, llegim el valor amb get().
    if isinstance(config, dict):
        return config.get(key, default)

    # Si config no és un dict, assumim que pot ser un objecte tipus wandb.config.
    # En aquest cas llegim l'atribut amb getattr().
    return getattr(config, key, default)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entrena el model durant una sola epoch.

    Args:
        model: xarxa neuronal que volem entrenar.
        train_loader: DataLoader amb les dades d'entrenament.
        criterion: funció de loss.
        optimizer: optimitzador que actualitza els pesos.
        device: CPU o GPU.

    Returns:
        avg_loss: loss mitjana de train.
        avg_dice: Dice mitjà de train.
        avg_iou: IoU mitjana de train.
    """

    # Posem el model en mode entrenament.
    # Això és important per capes com BatchNorm o Dropout.
    model.train()

    # Inicialitzem acumuladors per calcular mitjanes al final de l'epoch.
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    # Recorrem tots els batches del DataLoader d'entrenament.
    # tqdm mostra una barra de progrés.
    for images, masks in tqdm(train_loader, desc="Training", leave=False):

        # Movem les imatges al dispositiu corresponent.
        # Si hi ha GPU, aniran a cuda; si no, a CPU.
        images = images.to(device)

        # Movem les màscares al mateix dispositiu que les imatges.
        masks = masks.to(device)

        # ============================================================
        # Forward pass
        # ============================================================

        # Passem les imatges pel model.
        # El model retorna logits amb forma [batch, 1, H, W].
        outputs = model(images)

        # Calculem la loss comparant la predicció amb la màscara real.
        # En el nostre projecte, criterion sol ser BCEDiceLoss.
        loss = criterion(outputs, masks)

        # ============================================================
        # Backward pass
        # ============================================================

        # Posem els gradients acumulats a zero.
        # PyTorch acumula gradients per defecte, per això cal netejar-los a cada batch.
        optimizer.zero_grad()

        # Calculem els gradients de la loss respecte als pesos del model.
        loss.backward()

        # Actualitzem els pesos del model utilitzant els gradients.
        optimizer.step()

        # ============================================================
        # Mètriques del batch
        # ============================================================

        # Afegim la loss del batch a l'acumulador.
        # loss.item() converteix el tensor loss en un número Python.
        total_loss += loss.item()

        # Calculem el Dice del batch.
        # detach() separa outputs i masks del graf de gradients.
        # Les mètriques no han d'afectar el backpropagation.
        total_dice += dice_score(outputs.detach(), masks.detach())

        # Calculem la IoU del batch.
        total_iou += iou_score(outputs.detach(), masks.detach())

    # Calculem la loss mitjana de tota l'epoch.
    avg_loss = total_loss / len(train_loader)

    # Calculem el Dice mitjà de tota l'epoch.
    avg_dice = total_dice / len(train_loader)

    # Calculem la IoU mitjana de tota l'epoch.
    avg_iou = total_iou / len(train_loader)

    # Retornem les mètriques mitjanes de train.
    return avg_loss, avg_dice, avg_iou


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Avalua el model sobre un conjunt de validació o test.

    Aquesta funció no actualitza pesos.
    Només fa forward pass i calcula loss, Dice i IoU.

    Args:
        model: model entrenat o en entrenament.
        val_loader: DataLoader de validació o test.
        criterion: funció de loss.
        device: CPU o GPU.

    Returns:
        avg_loss: loss mitjana.
        avg_dice: Dice mitjà.
        avg_iou: IoU mitjana.
    """

    # Posem el model en mode avaluació.
    # Això és important per BatchNorm i Dropout.
    model.eval()

    # Inicialitzem acumuladors.
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    # Durant la validació no volem calcular gradients.
    # Això redueix memòria i accelera el procés.
    with torch.no_grad():

        # Recorrem tots els batches del DataLoader de validació.
        for images, masks in tqdm(val_loader, desc="Validation", leave=False):

            # Movem imatges i màscares al dispositiu.
            images = images.to(device)
            masks = masks.to(device)

            # Fem forward pass.
            outputs = model(images)

            # Calculem la loss.
            loss = criterion(outputs, masks)

            # Acumulem la loss.
            total_loss += loss.item()

            # Calculem i acumulem el Dice.
            # Aquí no cal detach perquè ja estem dins torch.no_grad().
            total_dice += dice_score(outputs, masks)

            # Calculem i acumulem la IoU.
            total_iou += iou_score(outputs, masks)

    # Calculem mitjanes de validació.
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    # Retornem les mètriques mitjanes.
    return avg_loss, avg_dice, avg_iou


def train(model, train_loader, val_loader, criterion, optimizer, config, device, save_path="best_model.pth"):
    """
    Entrenament complet del model.

    Aquesta funció entrena durant diverses epochs, valida després de cada epoch,
    guarda l'historial de mètriques i desa el millor model segons el Dice de validació.

    Args:
        model: model que volem entrenar.
        train_loader: DataLoader de train.
        val_loader: DataLoader de validation.
        criterion: funció de loss.
        optimizer: optimitzador.
        config: configuració de l'experiment.
        device: CPU o GPU.
        save_path: ruta on es guardarà el millor checkpoint.

    Returns:
        history: diccionari amb l'evolució de les mètriques.
    """

    # Llegim el nombre d'epochs des de config.
    # Si no existeix, utilitzem 10 per defecte.
    epochs = get_config_value(config, "epochs", 10)

    # Guardarem aquí el millor Dice de validació vist fins al moment.
    best_val_dice = 0.0

    # Diccionari per guardar l'evolució de loss, Dice i IoU.
    # Aquest history després es pot utilitzar per generar gràfiques.
    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    # Intentem registrar el model amb wandb.
    # Si wandb falla, no volem que l'entrenament s'aturi.
    try:
        wandb.watch(model, criterion, log="all", log_freq=10)
    except Exception:
        pass

    # Bucle principal d'entrenament.
    # Es repeteix una vegada per cada epoch.
    for epoch in range(epochs):

        # Mostrem el número d'epoch actual.
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ============================================================
        # Entrenament d'una epoch
        # ============================================================

        # Entrenem el model sobre tot el train_loader.
        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # ============================================================
        # Validació d'una epoch
        # ============================================================

        # Avaluem el model sobre el val_loader.
        val_loss, val_dice, val_iou = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # ============================================================
        # Guardar mètriques a history
        # ============================================================

        # Afegim les mètriques de train.
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)

        # Afegim les mètriques de validació.
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # Mostrem les mètriques per pantalla.
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | Val IoU:   {val_iou:.4f}")

        # ============================================================
        # Registrar mètriques a wandb
        # ============================================================

        # Guardem les mètriques a wandb.
        # Està dins try/except perquè si wandb no està disponible,
        # l'entrenament igualment ha de continuar.
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

        # ============================================================
        # Guardar el millor model
        # ============================================================

        # Si el Dice de validació actual és millor que el millor vist fins ara,
        # guardem aquest model com a millor checkpoint.
        if val_dice > best_val_dice:

            # Actualitzem el millor Dice.
            best_val_dice = val_dice

            # Guardem només els pesos del model.
            # Aquesta és la pràctica habitual en PyTorch.
            torch.save(model.state_dict(), save_path)

            # Informem per pantalla que s'ha guardat un nou millor model.
            print(f"Millor model guardat amb Val Dice = {best_val_dice:.4f}")

    # Quan acaben totes les epochs, retornem l'historial complet.
    return history
