# Segmentación de Tumores Cerebrales con U-Net

Este proyecto entrena una red neuronal U-Net para **detectar y segmentar tumores cerebrales** en imágenes MRI del dataset BraTS2020. Dado un scan cerebral, el modelo predice qué píxeles corresponden a tumor y cuáles no (segmentación binaria).

## ¿Qué hace exactamente?

1. Carga imágenes MRI cerebrales 3D en formato NIfTI (`.nii`)
2. Las divide en cortes 2D (slices), usando solo los que contienen tumor
3. Entrena una U-Net para predecir una máscara binaria: tumor (1) o no tumor (0)
4. Evalúa el modelo con métricas Dice e IoU
5. Guarda el mejor modelo y las gráficas de entrenamiento

---

## Estructura del proyecto

```
projecte-deep-learning-14/
│
├── main.py                      # Script principal: configura y lanza el entrenamiento
├── train.py                     # Bucle de entrenamiento y validación
├── generate_test_predictions.py # Genera imágenes visuales de las predicciones
├── plot_training_history.py     # Genera gráficas de loss/Dice/IoU
│
├── models/
│   └── unet.py                  # Arquitectura U-Net 2D
│
├── utils/
│   ├── dataset.py               # Carga y preprocesa el dataset BraTS2020
│   ├── losses.py                # Función de pérdida BCEDiceLoss
│   ├── metrics.py               # Métricas: Dice Score e IoU
│   └── visualization.py        # Funciones para generar figuras
│
├── notebooks/                   # Scripts de exploración y prueba por componentes
├── results/
│   ├── models/                  # Pesos del modelo guardados (.pth)
│   ├── history/                 # Historial de entrenamiento en JSON
│   ├── figures/                 # Gráficas de entrenamiento
│   └── predictions/             # Imágenes de predicciones visuales
│
└── environment.yml              # Dependencias del entorno conda
```

---

## Requisitos previos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o Anaconda instalado
- El dataset BraTS2020 descargado localmente (ver sección siguiente)
- GPU recomendada (funciona también en CPU, pero más lento)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd projecte-deep-learning-14
```

### 2. Crear el entorno conda

```bash
conda env create --file environment.yml
```

### 3. Activar el entorno

```bash
conda activate xnap-example
```

También necesitas instalar `nibabel` para leer archivos NIfTI:

```bash
pip install nibabel matplotlib
```

---

## Datos: BraTS2020

El dataset debe estar organizado así:

```
/tu/ruta/al/data/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_seg.nii
│   └── ...
├── BraTS20_Training_002/
│   ├── BraTS20_Training_002_flair.nii
│   ├── BraTS20_Training_002_seg.nii
│   └── ...
└── ...
```

Cada paciente tiene su propia carpeta con varios archivos `.nii`. El proyecto usa:
- `*_flair.nii` — imagen MRI de entrada (modalidad FLAIR)
- `*_seg.nii` — máscara de segmentación real (ground truth)

Puedes descargar el dataset desde [BraTS2020 en Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) o desde el [portal oficial de BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html).

---

## Cómo ejecutar

### Paso 1 — Configurar la ruta del dataset

Abre `main.py` y cambia `root_dir` por la ruta donde tienes los datos:

```python
config = {
    "root_dir": "/ruta/a/tu/carpeta/con/pacientes",  # <-- cambia esto
    ...
}
```

### Paso 2 — Entrenar el modelo

```bash
python main.py
```

Esto hará automáticamente:
- Dividir los pacientes en train (80%), validación (10%) y test (10%)
- Entrenar la U-Net durante 20 épocas
- Guardar el mejor modelo en `results/models/`
- Guardar el historial de métricas en `results/history/`

Verás en la terminal el progreso época a época:

```
Epoch 1/20
Train Loss: 0.2341 | Train Dice: 0.7812 | Train IoU: 0.6401
Val Loss:   0.1923 | Val Dice:   0.8134 | Val IoU:   0.6872
Millor model guardat amb Val Dice = 0.8134
```

### Paso 3 — Ver las gráficas de entrenamiento

```bash
python plot_training_history.py
```

Genera en `results/figures/`:
- `loss_curve.png` — evolución de la pérdida
- `dice_curve.png` — evolución del Dice Score
- `iou_curve.png` — evolución del IoU

### Paso 4 — Ver predicciones visuales

```bash
python generate_test_predictions.py
```

Guarda en `results/predictions/` imágenes con tres columnas:
- Imagen MRI original
- Máscara real (ground truth)
- Predicción del modelo

---

## ¿Cómo funciona la U-Net?

La U-Net es una arquitectura clásica para segmentación de imágenes médicas. Tiene forma de U:

```
Entrada (imagen MRI)
        |
   [Encoder] — comprime la imagen, extrae características
        |
   [Bottleneck] — representación más compacta
        |
   [Decoder] — reconstruye la máscara pixel a pixel
        |
  Salida (máscara predicha)
```

Las **skip connections** conectan cada nivel del encoder con el decoder correspondiente, permitiendo al modelo combinar información de contexto global con detalle espacial local.

---

## Métricas

| Métrica | Descripción | Rango |
|---------|-------------|-------|
| **Dice Score** | Mide el solapamiento entre predicción y ground truth | 0–1 (1 = perfecto) |
| **IoU** | Intersección sobre la unión de las regiones | 0–1 (1 = perfecto) |

### Resultados del baseline

| Conjunto | Dice | IoU |
|----------|------|-----|
| Train | 0.8986 | 0.8177 |
| Validación | 0.9057 | 0.8285 |

---

## Seguimiento del entrenamiento con Weights & Biases

El proyecto usa [wandb](https://wandb.ai) en modo **offline** por defecto. Las métricas se guardan localmente en la carpeta `wandb/`.

Si quieres verlas online, crea una cuenta en wandb.ai y cambia en `main.py`:

```python
"wandb_mode": "online"   # en vez de "offline"
```

Luego ejecuta `wandb login` antes de entrenar.

---

## Posibles errores frecuentes

**`No s'han trobat pacients`** — La ruta `root_dir` no es correcta o las carpetas no empiezan por `BraTS20_Training_`.

**`CUDA out of memory`** — Reduce el `batch_size` en `main.py` (prueba con 4 o 2).

**`ModuleNotFoundError: nibabel`** — Ejecuta `pip install nibabel`.

---

## Autoras

Laia Camara, Laia Alcalde, Elena Gutiérrez, Cristina Huanca

Xarxes Neuronals i Aprenentatge Profund  
Grau d'Enginyeria de Dades, UAB, 2026
