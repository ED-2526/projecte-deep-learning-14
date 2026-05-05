# Importem sys i os per poder importar mòduls del projecte des de notebooks/.
import sys
import os

# Calculem la ruta de la carpeta principal del projecte.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Afegim la carpeta principal al path de Python.
# Això permet importar utils, models i train.py.
sys.path.append(project_root)


# Importem PyTorch.
import torch

# Importem DataLoader per crear batches.
# random_split per dividir un Dataset en train i validation.
# Subset per agafar només una part petita del Dataset.
from torch.utils.data import DataLoader, random_split, Subset

# Importem el nostre Dataset BraTS.
from utils.dataset import BraTSSegmentationDataset

# Importem la loss combinada BCE + Dice.
from utils.losses import BCEDiceLoss

# Importem la U-Net.
from models.unet import UNet

# Importem la funció train, que conté el bucle d'entrenament.
from train import train


# Ruta principal del dataset Training.
root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"


# ============================================================
# Seleccionar dispositiu: GPU o CPU
# ============================================================

# Seleccionem GPU si PyTorch detecta CUDA.
# Si no hi ha GPU disponible, fem servir CPU.
#
# Nota:
# Al codi original hi havia un typo: "cudxza".
# La forma correcta és "cuda".
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mostrem quin dispositiu s'utilitzarà.
print("Device:", device)


# ============================================================
# Crear el Dataset
# ============================================================

# Creem el Dataset complet a partir del dataset BraTS2020.
dataset = BraTSSegmentationDataset(
    root_dir=root_dir,

    # Utilitzem només la modalitat FLAIR.
    modality="flair",

    # Agafem només slices amb tumor per aquesta prova inicial.
    only_tumor_slices=True
)


# ============================================================
# Crear un subconjunt petit per provar ràpid
# ============================================================

# Agafem només les primeres 32 mostres del Dataset.
# Això no és per entrenar un model final.
# Només serveix per comprovar que el pipeline complet funciona ràpidament.
small_dataset = Subset(dataset, range(32))


# Calculem quantes mostres seran per train.
# Utilitzem el 80% del subconjunt petit.
train_size = int(0.8 * len(small_dataset))

# La resta de mostres seran per validation.
val_size = len(small_dataset) - train_size


# Dividim el subconjunt petit en train i validation.
# Aquí fem random_split perquè és només una prova tècnica.
# En el pipeline final, per evitar data leakage, fem split per pacient.
train_dataset, val_dataset = random_split(
    small_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


# ============================================================
# Crear DataLoaders
# ============================================================

# DataLoader d'entrenament.
train_loader = DataLoader(

    # Dataset de train.
    train_dataset,

    # Batch size petit perquè és una prova ràpida.
    batch_size=2,

    # Barregem les mostres durant l'entrenament.
    shuffle=True,

    # num_workers=0 per evitar problemes en entorns locals o notebooks.
    num_workers=0
)


# DataLoader de validació.
val_loader = DataLoader(

    # Dataset de validation.
    val_dataset,

    # Mateix batch size.
    batch_size=2,

    # No cal barrejar en validació.
    shuffle=False,

    num_workers=0
)


# ============================================================
# Crear model, loss i optimizer
# ============================================================

# Creem la U-Net.
# in_channels=1 perquè l'entrada és FLAIR.
# out_channels=1 perquè la sortida és una màscara binària.
# .to(device) mou el model a GPU o CPU.
model = UNet(in_channels=1, out_channels=1).to(device)


# Definim la loss.
# BCEDiceLoss combina BCEWithLogitsLoss i Dice Loss.
# És adequada per segmentació de tumors perquè hi ha molt desequilibri entre fons i tumor.
criterion = BCEDiceLoss()


# Definim l'optimizer Adam.
# model.parameters() són els pesos que s'han d'actualitzar.
# lr=1e-4 és el learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Configuració mínima per aquesta prova.
# Només entrenem 1 epoch perquè no volem obtenir resultats finals,
# només comprovar que el bucle funciona.
config = {
    "epochs": 1
}


# ============================================================
# Executar entrenament de prova
# ============================================================

# Cridem la funció train.
# Aquesta funció farà:
#   1. forward pass
#   2. càlcul de loss
#   3. backward pass
#   4. optimizer.step()
#   5. validació
#   6. càlcul de mètriques
#   7. guardat del millor model
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    config=config,
    device=device,

    # Guardem el model de prova en aquest fitxer.
    # No és el model final, només un checkpoint de test.
    save_path="test_best_model.pth"
)


# Si arribem aquí, vol dir que tot el pipeline ha funcionat.
print("Training de prova completat.")

# Mostrem l'historial retornat per la funció train.
# Normalment conté loss, Dice i IoU de train i validation.
print(history)
