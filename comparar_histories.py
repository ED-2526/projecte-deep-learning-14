#!/usr/bin/env python3
"""
comparar_histories.py

Script per comparar tots els fitxers *_history.json generats durant els experiments.

Què fa:
  1. Llegeix tots els JSON de results/history.
  2. Calcula mètriques resum per experiment:
     - millor epoch segons val_dice
     - best val_dice / val_iou
     - train/val final
     - gap train-val
     - test_dice / test_iou / test_loss si existeixen
  3. Genera una taula CSV amb tots els resultats.
  4. Genera gràfiques comparatives.
  5. Genera un informe Markdown amb conclusions automàtiques.

Ús recomanat des de l'arrel del projecte:

    PYTHONPATH=. python comparar_histories.py

O especificant carpetes:

    PYTHONPATH=. python comparar_histories.py \
        --history-dir results/history \
        --out-dir results/comparison_report

Sortides:
    results/comparison_report/comparison_summary.csv
    results/comparison_report/comparison_report.md
    results/comparison_report/plots/*.png
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuració de noms bonics segons els fitxers que hem anat generant
# ---------------------------------------------------------------------
EXPERIMENT_INFO = {
    "unet_flair_patient_split_20epochs_history.json": {
        "label": "FLAIR - tumor slices",
        "entrada": "FLAIR",
        "slices": "Només tumor",
        "loss": "BCE + Dice",
        "notes": "Baseline amb split per pacient i només slices amb tumor.",
    },
    "unet_flair_patient_split_20epochs_aug_history.json": {
        "label": "FLAIR - augmentation",
        "entrada": "FLAIR",
        "slices": "Només tumor",
        "loss": "BCE + Dice",
        "notes": "Data augmentation. Redueix memorització però pot empitjorar test.",
    },
    "unet_flair_patient_split_20epochs_all_slices_history.json": {
        "label": "FLAIR - all slices",
        "entrada": "FLAIR",
        "slices": "Totes",
        "loss": "BCE + Dice",
        "notes": "Inclou slices amb i sense tumor. Avaluació més realista.",
    },
    "unet_multimodal_patient_split_20epochs_all_slices_history.json": {
        "label": "Multimodal - all slices",
        "entrada": "FLAIR+T1+T1CE+T2",
        "slices": "Totes",
        "loss": "BCE + Dice",
        "notes": "Model multimodal amb 4 canals. Normalment és el millor model global.",
    },
    "unet_multimodal_20epochs_bce_tversky_history.json": {
        "label": "Multimodal - BCE+Tversky",
        "entrada": "FLAIR+T1+T1CE+T2",
        "slices": "Totes",
        "loss": "BCE + Tversky",
        "notes": "Prova per penalitzar més falsos negatius i millorar tumors petits.",
    },
    "unet_multimodal_all_patients_25epochs_weighted_sampling_history.json": {
        "label": "Multimodal - weighted sampling",
        "entrada": "FLAIR+T1+T1CE+T2",
        "slices": "Totes",
        "loss": "BCE + Dice",
        "notes": "Mostreig ponderat per donar més pes a tumors petits.",
    },
}


# ---------------------------------------------------------------------
# Funcions auxiliars
# ---------------------------------------------------------------------
def safe_float(x: Any) -> Optional[float]:
    """Converteix a float si es pot; si no, retorna None."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt(x: Optional[float], digits: int = 4) -> str:
    """Format curt per taules."""
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def get_list(history: Dict[str, Any], key: str) -> List[float]:
    """Retorna una llista de floats d'una clau del history, o [] si no existeix."""
    values = history.get(key, [])
    if values is None:
        return []
    return [float(v) for v in values]


def readable_name(filename: str) -> Dict[str, str]:
    """Retorna metadata llegible d'un experiment."""
    if filename in EXPERIMENT_INFO:
        return EXPERIMENT_INFO[filename]

    stem = filename.replace("_history.json", "").replace(".json", "")
    label = stem.replace("unet_", "").replace("_", " ")

    return {
        "label": label,
        "entrada": "No especificada",
        "slices": "No especificat",
        "loss": "No especificada",
        "notes": "Experiment detectat automàticament a partir del nom del fitxer.",
    }


def load_histories(history_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Carrega tots els JSON de history_dir."""
    if not history_dir.exists():
        raise FileNotFoundError(f"No existeix la carpeta: {history_dir}")

    paths = sorted(history_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No s'han trobat fitxers JSON a: {history_dir}")

    histories = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            histories[path.name] = json.load(f)

    return histories


def summarize_history(filename: str, history: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula resum numèric d'un history."""
    info = readable_name(filename)

    train_loss = get_list(history, "train_loss")
    val_loss = get_list(history, "val_loss")
    train_dice = get_list(history, "train_dice")
    val_dice = get_list(history, "val_dice")
    train_iou = get_list(history, "train_iou")
    val_iou = get_list(history, "val_iou")

    epochs = max(len(train_loss), len(val_loss), len(train_dice), len(val_dice), len(train_iou), len(val_iou))

    best_epoch = None
    best_val_dice = None
    best_val_iou = None
    best_train_dice = None
    gap_best = None

    if val_dice:
        best_idx = int(np.argmax(val_dice))
        best_epoch = best_idx + 1
        best_val_dice = val_dice[best_idx]
        if best_idx < len(val_iou):
            best_val_iou = val_iou[best_idx]
        if best_idx < len(train_dice):
            best_train_dice = train_dice[best_idx]
            gap_best = best_train_dice - best_val_dice

    final_train_dice = train_dice[-1] if train_dice else None
    final_val_dice = val_dice[-1] if val_dice else None
    final_train_iou = train_iou[-1] if train_iou else None
    final_val_iou = val_iou[-1] if val_iou else None
    final_train_loss = train_loss[-1] if train_loss else None
    final_val_loss = val_loss[-1] if val_loss else None

    gap_final = None
    if final_train_dice is not None and final_val_dice is not None:
        gap_final = final_train_dice - final_val_dice

    return {
        "file": filename,
        "label": info["label"],
        "entrada": info["entrada"],
        "slices": info["slices"],
        "loss": info["loss"],
        "notes": info["notes"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_train_dice": best_train_dice,
        "gap_best": gap_best,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_dice": final_train_dice,
        "final_val_dice": final_val_dice,
        "final_train_iou": final_train_iou,
        "final_val_iou": final_val_iou,
        "gap_final": gap_final,
        "test_loss": safe_float(history.get("test_loss")),
        "test_dice": safe_float(history.get("test_dice")),
        "test_iou": safe_float(history.get("test_iou")),
    }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Guarda la taula resum en CSV."""
    fieldnames = [
        "label",
        "file",
        "entrada",
        "slices",
        "loss",
        "epochs",
        "best_epoch",
        "best_val_dice",
        "best_val_iou",
        "gap_best",
        "final_train_dice",
        "final_val_dice",
        "gap_final",
        "test_loss",
        "test_dice",
        "test_iou",
        "notes",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ordena experiments: primer els coneguts, després la resta."""
    order = list(EXPERIMENT_INFO.keys())

    def key(row: Dict[str, Any]) -> tuple:
        filename = row["file"]
        if filename in order:
            return (0, order.index(filename))
        return (1, filename)

    return sorted(rows, key=key)


# ---------------------------------------------------------------------
# Gràfiques
# ---------------------------------------------------------------------
def save_current_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_test_metrics(rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    valid = [r for r in rows if r["test_dice"] is not None or r["test_iou"] is not None]
    if not valid:
        return

    labels = [r["label"] for r in valid]
    x = np.arange(len(valid))
    width = 0.35

    test_dice = [r["test_dice"] if r["test_dice"] is not None else 0 for r in valid]
    test_iou = [r["test_iou"] if r["test_iou"] is not None else 0 for r in valid]

    plt.figure(figsize=(max(8, len(valid) * 1.6), 5))
    plt.bar(x - width / 2, test_dice, width, label="Test Dice")
    plt.bar(x + width / 2, test_iou, width, label="Test IoU")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Valor")
    plt.title("Comparació de resultats finals en test")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for i, value in enumerate(test_dice):
        if value > 0:
            plt.text(i - width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    for i, value in enumerate(test_iou):
        if value > 0:
            plt.text(i + width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)

    save_current_fig(plots_dir / "01_test_metrics.png")


def plot_best_val_metrics(rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    best_dice = [r["best_val_dice"] if r["best_val_dice"] is not None else 0 for r in rows]
    best_iou = [r["best_val_iou"] if r["best_val_iou"] is not None else 0 for r in rows]

    plt.figure(figsize=(max(8, len(rows) * 1.6), 5))
    plt.bar(x - width / 2, best_dice, width, label="Best Val Dice")
    plt.bar(x + width / 2, best_iou, width, label="Best Val IoU")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Valor")
    plt.title("Millor validació per experiment")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    save_current_fig(plots_dir / "02_best_validation_metrics.png")


def plot_val_dice_curves(histories: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    plt.figure(figsize=(10, 5.5))

    for row in rows:
        history = histories[row["file"]]
        val_dice = get_list(history, "val_dice")
        if not val_dice:
            continue
        epochs = range(1, len(val_dice) + 1)
        plt.plot(epochs, val_dice, marker="o", markersize=3, linewidth=1.7, label=row["label"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("Evolució del Validation Dice")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_current_fig(plots_dir / "03_val_dice_curves.png")


def plot_val_iou_curves(histories: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    plt.figure(figsize=(10, 5.5))

    for row in rows:
        history = histories[row["file"]]
        val_iou = get_list(history, "val_iou")
        if not val_iou:
            continue
        epochs = range(1, len(val_iou) + 1)
        plt.plot(epochs, val_iou, marker="o", markersize=3, linewidth=1.7, label=row["label"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation IoU")
    plt.title("Evolució del Validation IoU")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save_current_fig(plots_dir / "04_val_iou_curves.png")


def plot_gap(rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    labels = [r["label"] for r in rows]
    values = [r["gap_final"] if r["gap_final"] is not None else 0 for r in rows]

    plt.figure(figsize=(max(8, len(rows) * 1.6), 5))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Train Dice final - Val Dice final")
    plt.title("Gap train-validation al final de l'entrenament")
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)

    save_current_fig(plots_dir / "05_final_gap.png")


def plot_single_history_curves(histories: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    single_dir = plots_dir / "per_experiment"
    single_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        history = histories[row["file"]]
        label_safe = row["label"].replace(" ", "_").replace("/", "_")

        train_dice = get_list(history, "train_dice")
        val_dice = get_list(history, "val_dice")
        train_iou = get_list(history, "train_iou")
        val_iou = get_list(history, "val_iou")
        train_loss = get_list(history, "train_loss")
        val_loss = get_list(history, "val_loss")

        # Dice
        if train_dice and val_dice:
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(train_dice) + 1), train_dice, marker="o", markersize=3, label="Train Dice")
            plt.plot(range(1, len(val_dice) + 1), val_dice, marker="o", markersize=3, label="Val Dice")
            plt.title(f"Dice - {row['label']}")
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.ylim(0, 1.05)
            plt.grid(alpha=0.3)
            plt.legend()
            save_current_fig(single_dir / f"{label_safe}_dice.png")

        # IoU
        if train_iou and val_iou:
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(train_iou) + 1), train_iou, marker="o", markersize=3, label="Train IoU")
            plt.plot(range(1, len(val_iou) + 1), val_iou, marker="o", markersize=3, label="Val IoU")
            plt.title(f"IoU - {row['label']}")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.ylim(0, 1.05)
            plt.grid(alpha=0.3)
            plt.legend()
            save_current_fig(single_dir / f"{label_safe}_iou.png")

        # Loss
        if train_loss and val_loss:
            plt.figure(figsize=(8, 4.5))
            plt.plot(range(1, len(train_loss) + 1), train_loss, marker="o", markersize=3, label="Train Loss")
            plt.plot(range(1, len(val_loss) + 1), val_loss, marker="o", markersize=3, label="Val Loss")
            plt.title(f"Loss - {row['label']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(alpha=0.3)
            plt.legend()
            save_current_fig(single_dir / f"{label_safe}_loss.png")


def generate_plots(histories: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_test_metrics(rows, plots_dir)
    plot_best_val_metrics(rows, plots_dir)
    plot_val_dice_curves(histories, rows, plots_dir)
    plot_val_iou_curves(histories, rows, plots_dir)
    plot_gap(rows, plots_dir)
    plot_single_history_curves(histories, rows, plots_dir)


# ---------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------
def markdown_table(headers: List[str], data: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in data:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_markdown_report(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    report_path = out_dir / "comparison_report.md"

    # Millors models segons test i val
    rows_with_test = [r for r in rows if r["test_dice"] is not None]
    best_test = max(rows_with_test, key=lambda r: r["test_dice"]) if rows_with_test else None
    best_val = max(rows, key=lambda r: r["best_val_dice"] if r["best_val_dice"] is not None else -1)

    table_data = []
    for r in rows:
        table_data.append([
            r["label"],
            r["entrada"],
            r["slices"],
            r["loss"],
            str(r["epochs"]),
            str(r["best_epoch"] or "-"),
            fmt(r["best_val_dice"]),
            fmt(r["best_val_iou"]),
            fmt(r["test_dice"]),
            fmt(r["test_iou"]),
            fmt(r["gap_final"]),
        ])

    md = []
    md.append("# Comparació d'experiments XNAP\n")
    md.append("Aquest informe s'ha generat automàticament a partir dels fitxers `*_history.json` de `results/history`.\n")

    md.append("## Resum global\n")
    md.append(markdown_table(
        ["Experiment", "Entrada", "Slices", "Loss", "Epochs", "Best epoch", "Best Val Dice", "Best Val IoU", "Test Dice", "Test IoU", "Gap final"],
        table_data,
    ))
    md.append("\n")

    if best_test:
        md.append("## Millor model segons test\n")
        md.append(
            f"El millor model segons **Test Dice** és **{best_test['label']}**, "
            f"amb Test Dice = **{fmt(best_test['test_dice'])}** i Test IoU = **{fmt(best_test['test_iou'])}**.\n"
        )

    if best_val:
        md.append("## Millor model segons validació\n")
        md.append(
            f"El millor model segons **Best Val Dice** és **{best_val['label']}**, "
            f"a l'epoch {best_val['best_epoch']}, amb Val Dice = **{fmt(best_val['best_val_dice'])}** "
            f"i Val IoU = **{fmt(best_val['best_val_iou'])}**.\n"
        )

    md.append("## Gràfiques generades\n")
    md.append("Les gràfiques principals es troben a la carpeta `plots/`:\n")
    md.append("- `01_test_metrics.png`: comparació de Test Dice i Test IoU.\n")
    md.append("- `02_best_validation_metrics.png`: millor validació per experiment.\n")
    md.append("- `03_val_dice_curves.png`: evolució del Validation Dice.\n")
    md.append("- `04_val_iou_curves.png`: evolució del Validation IoU.\n")
    md.append("- `05_final_gap.png`: diferència final entre train i validation.\n")
    md.append("- `plots/per_experiment/`: corbes de loss, Dice i IoU per cada experiment.\n")

    md.append("## Interpretació automàtica\n")
    md.append("- Si el gap train-validation és alt, hi ha senyal d'overfitting.\n")
    md.append("- Si Test Dice i Test IoU milloren en el model multimodal, vol dir que les modalitats T1, T1CE i T2 aporten informació útil respecte a FLAIR només.\n")
    md.append("- Si Tversky Loss millora el recall o redueix falsos negatius en anàlisis posteriors, pot ser útil per tumors petits, encara que el Dice global no pugi gaire.\n")
    md.append("- La comparació correcta no ha de mirar només el Test Dice global: també cal mirar threshold, prediccions visuals i rendiment per mida tumoral.\n")

    md.append("## Notes per experiment\n")
    for r in rows:
        md.append(f"### {r['label']}\n")
        md.append(f"- Fitxer: `{r['file']}`\n")
        md.append(f"- Notes: {r['notes']}\n")
        md.append(f"- Best Val Dice: {fmt(r['best_val_dice'])}\n")
        md.append(f"- Test Dice: {fmt(r['test_dice'])}\n")
        md.append(f"- Test IoU: {fmt(r['test_iou'])}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compara tots els histories JSON dels experiments.")
    parser.add_argument("--history-dir", type=str, default="results/history", help="Carpeta on hi ha els *_history.json")
    parser.add_argument("--out-dir", type=str, default="results/comparison_report", help="Carpeta de sortida")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPARACIÓ D'HISTORIES")
    print("=" * 80)
    print(f"History dir: {history_dir}")
    print(f"Output dir:  {out_dir}")

    histories = load_histories(history_dir)
    rows = [summarize_history(filename, history) for filename, history in histories.items()]
    rows = sort_rows(rows)

    csv_path = out_dir / "comparison_summary.csv"
    save_csv(rows, csv_path)
    generate_plots(histories, rows, plots_dir)
    generate_markdown_report(rows, out_dir)

    print("\nFitxers analitzats:")
    for r in rows:
        print(f"- {r['file']} -> {r['label']}")

    print("\nResum:")
    for r in rows:
        print(
            f"{r['label']}: "
            f"Best Val Dice={fmt(r['best_val_dice'])}, "
            f"Test Dice={fmt(r['test_dice'])}, "
            f"Test IoU={fmt(r['test_iou'])}, "
            f"Gap final={fmt(r['gap_final'])}"
        )

    print("\nSortides generades:")
    print(f"- {csv_path}")
    print(f"- {out_dir / 'comparison_report.md'}")
    print(f"- {plots_dir}")
    print("\nFet.")


if __name__ == "__main__":
    main()
