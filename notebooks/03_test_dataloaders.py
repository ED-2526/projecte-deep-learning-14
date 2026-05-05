# Importem sys i os per poder modificar el path de Python.
# Això és necessari perquè aquest fitxer està dins de notebooks/
# i volem importar fitxers que estan a la carpeta arrel del projecte.
import sys
import os

# Calculem la ruta absoluta de la carpeta arrel del projecte.
# os.path.dirname(__file__) dona la carpeta actual, és a dir, notebooks/.
# ".." puja un nivell, fins a la carpeta principal del projecte.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Afegim la carpeta principal al path de Python.
# Així podem importar utils.dataset encara que el script estigui dins notebooks/.
sys.path.append(project_root)


# Importem PyTorch.
import torch

# random_split serveix per dividir un Dataset en subconjunts.
# DataLoader serveix per crear batches a partir d'un Dataset.
from torch.utils.data import random_split, DataLoader

# Importem el nostre Dataset personalitzat.
from utils.dataset import BraTSSegmentationDataset


# Ruta principal del dataset Training.
root_dir = "/Users/laiaalcaldemaria/Desktop/descarga/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Definim el batch size.
# batch_size=4 vol dir que cada batch tindrà 4 imatges i 4 màscares.
batch_size = 4


# Creem el Dataset complet.
dataset = BraTSSegmentationDataset(
    root_dir=root_dir,
    modality="flair",
    only_tumor_slices=True
)


# Calculem el nombre total de mostres.
# Recordem que cada mostra és una slice 2D.
total_size = len(dataset)

# Calculem el 80% de mostres per entrenament.
train_size = int(0.8 * total_size)

# Calculem el 10% de mostres per validació.
val_size = int(0.1 * total_size)

# La resta de mostres seran per test.
# Ho fem així per evitar perdre mostres per l'arrodoniment dels enters.
test_size = total_size - train_size - val_size


# Dividim el Dataset en train, validation i test.
# Aquesta divisió és aleatòria però reproduïble gràcies a manual_seed(42).
# IMPORTANT: aquí encara dividim per slices, no per pacients.
# Això ens va servir com a prova inicial, però després ho vam corregir al pipeline final.
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)


# Creem el DataLoader d'entrenament.
train_loader = DataLoader(

    # Dataset d'entrenament.
    train_dataset,

    # Nombre de mostres per batch.
    batch_size=batch_size,

    # shuffle=True perquè durant l'entrenament volem barrejar les mostres.
    # Això ajuda que el model no aprengui un ordre fix.
    shuffle=True,

    # num_workers=0 fa que les dades es carreguin al procés principal.
    # És més lent, però més simple i segur per proves locals.
    num_workers=0
)


# Creem el DataLoader de validació.
val_loader = DataLoader(

    # Dataset de validació.
    val_dataset,

    # Mateix batch size.
    batch_size=batch_size,

    # En validació no cal barrejar les dades perquè no entrenem.
    shuffle=False,

    num_workers=0
)


# Creem el DataLoader de test.
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,

    # En test tampoc cal shuffle.
    shuffle=False,
    num_workers=0
)


# Mostrem quantes mostres hi ha en total.
print("Total mostres:", total_size)

# Mostrem quantes mostres han quedat a train.
print("Train:", len(train_dataset))

# Mostrem quantes mostres han quedat a validation.
print("Validation:", len(val_dataset))

# Mostrem quantes mostres han quedat a test.
print("Test:", len(test_dataset))


# Agafem el primer batch del DataLoader d'entrenament.
# iter(train_loader) crea un iterador.
# next(...) obté el primer batch.
images, masks = next(iter(train_loader))


# Mostrem la forma del batch d'imatges.
# Esperem torch.Size([4, 1, 240, 240]).
# Això vol dir:
#   4 imatges
#   1 canal FLAIR
#   240x240 píxels
print("Batch images:", images.shape)

# Mostrem la forma del batch de màscares.
# Ha de coincidir amb la forma de les imatges.
print("Batch masks:", masks.shape)

# Comprovem el valor mínim del batch d'imatges.
# Ens ajuda a verificar que la normalització funciona.
print("Valor mínim images:", images.min().item())

# Comprovem el valor màxim del batch d'imatges.
print("Valor màxim images:", images.max().item())

# Comprovem que les màscares continuen sent binàries.
# Esperem valors 0 i 1.
print("Valors únics masks:", masks.unique())
