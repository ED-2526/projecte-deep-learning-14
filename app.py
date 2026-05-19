import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import nibabel as nib

from demo_utils import (
    BINARY_CONFIG,
    MULTICLASS_CONFIG,
    CLASS_NAMES,
    CLASS_COLORS,
    load_model,
    list_demo_patients,
    load_patient_slice,
    run_inference,
    build_overlay_figure,
    compute_stats,
    _normalize_slice,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor Detection — Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .pipeline-box {
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            min-height: 75px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .pipeline-arrow {
            font-size: 28px;
            color: #888;
            display: flex;
            align-items: center;
            justify-content: center;
            padding-top: 20px;
        }
        .metric-label { font-size: 13px; color: #aaa; margin-bottom: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Cached model loader ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Carregant model...")
def get_model(mode):
    return load_model(mode)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Detection")
    st.markdown("*Projecte Deep Learning — BraTS 2020*")
    st.divider()

    mode = st.radio(
        "Tipus de detecció",
        ["binary", "multiclass"],
        format_func=lambda x: "Binari (FLAIR)" if x == "binary" else "Multiclasse (4 MRI)",
        key="mode",
    )

    cfg = BINARY_CONFIG if mode == "binary" else MULTICLASS_CONFIG

    st.divider()

    source = st.radio(
        "Font de la imatge",
        ["demo", "upload_nifti", "upload_png"],
        format_func=lambda x: {
            "demo": "Pacient del dataset",
            "upload_nifti": "Pujar NIfTI",
            "upload_png": "Pujar PNG",
        }[x],
    )

    loaded_data = None
    input_error = None

    if source == "demo":
        patients = list_demo_patients()
        if not patients:
            st.error(f"No s'han trobat pacients a:\n`{__import__('demo_utils').DATA_ROOT}`")
        else:
            patient_id = st.selectbox("Pacient", patients)
            max_slice = 154
            slice_idx = st.slider("Índex de tall (axial)", 0, max_slice, 80)

    elif source == "upload_nifti":
        if mode == "binary":
            flair_file = st.file_uploader("FLAIR (.nii / .nii.gz)", type=["nii", "gz"])
            uploaded_files = {"flair": flair_file}
        else:
            uploaded_files = {}
            for mod in ["flair", "t1", "t1ce", "t2"]:
                uploaded_files[mod] = st.file_uploader(
                    f"{mod.upper()} (.nii / .nii.gz)", type=["nii", "gz"], key=mod
                )
        slice_idx = st.slider("Índex de tall (axial)", 0, 154, 80)

    else:
        png_file = st.file_uploader("Imatge PNG (greyscale, ~240×240)", type=["png", "jpg", "jpeg"])

    st.divider()

    model, model_error = get_model(mode)
    if model_error:
        st.error(f"Error carregant model:\n{model_error}")
    else:
        import torch
        device = next(model.parameters()).device
        st.success(f"Model carregat · {str(device).upper()}")
        st.caption(f"`{cfg['model_path']}`")

    run_btn = st.button(
        "▶ Executar predicció",
        disabled=(model is None),
        use_container_width=True,
        type="primary",
    )

# ─── Header ──────────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("Detecció de Tumors Cerebrals amb U-Net")
    mode_label = "Binari (FLAIR → tumor/fons)" if mode == "binary" else "Multiclasse (4 modalitats → 4 classes)"
    st.markdown(f"**Mode actiu:** {mode_label}")

# ─── Pipeline flow ────────────────────────────────────────────────────────────

with st.expander("Flux de la xarxa neuronal", expanded=True):
    if mode == "binary":
        steps = [
            ("#1d3557", "📥 Input MRI", "FLAIR\n1 canal · ~240×240"),
            ("#457b9d", "⚙️ Pre-processament", "Normalització\n[0, 1] per tall"),
            ("#e63946", "🔽 Encoder U-Net", "4 nivells\n32→64→128→256→512"),
            ("#2a9d8f", "🔼 Decoder U-Net", "Skip connections\nUp-conv ×4"),
            ("#264653", "📤 Predicció", "Sigmoid > 0.5\nMàscara binària"),
        ]
    else:
        steps = [
            ("#1d3557", "📥 Input MRI", "4 modalitats\nFLAIR·T1·T1ce·T2"),
            ("#457b9d", "⚙️ Pre-processament", "Normalització\n[0, 1] per tall"),
            ("#e63946", "🔽 Encoder U-Net", "4 nivells\n32→64→128→256→512"),
            ("#2a9d8f", "🔼 Decoder U-Net", "Skip connections\nUp-conv ×4"),
            ("#264653", "📤 Predicció", "Argmax → 4 classes\nFons·Nec·Edema·Tumor"),
        ]

    cols = st.columns([3, 0.4, 3, 0.4, 3, 0.4, 3, 0.4, 3])
    step_positions = [0, 2, 4, 6, 8]
    arrow_positions = [1, 3, 5, 7]

    for idx, (col_i, (color, title, detail)) in enumerate(zip(step_positions, steps)):
        with cols[col_i]:
            detail_html = detail.replace("\n", "<br>")
            st.markdown(
                f"""<div class="pipeline-box" style="background:{color}">
                    <div style="font-size:15px">{title}</div>
                    <div style="font-size:11px;opacity:0.85;margin-top:5px">{detail_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    for arrow_col in arrow_positions:
        with cols[arrow_col]:
            st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────

def load_nifti_bytes(file_obj):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        vol = nib.load(tmp_path).get_fdata()
    finally:
        os.unlink(tmp_path)
    return vol


data = None
if source == "demo" and patients:
    try:
        data = load_patient_slice(patient_id, slice_idx, cfg["modalities"])
    except Exception as e:
        st.error(f"Error carregant el pacient: {e}")

elif source == "upload_nifti":
    missing = [k for k, v in uploaded_files.items() if v is None]
    if missing:
        if any(v for v in uploaded_files.values()):
            st.warning(f"Falten fitxers: {', '.join(missing).upper()}")
    else:
        try:
            channels = []
            for mod in cfg["modalities"]:
                vol = load_nifti_bytes(uploaded_files[mod])
                sl = _normalize_slice(vol[:, :, slice_idx])
                channels.append(sl)
            data = {
                "channels": np.stack(channels, axis=0),
                "seg": None,
                "flair": channels[0],
                "all_channels": channels,
            }
        except Exception as e:
            st.error(f"Error llegint NIfTI: {e}")

elif source == "upload_png" and png_file is not None:
    try:
        img = Image.open(png_file).convert("L").resize((240, 240), Image.BILINEAR)
        arr = _normalize_slice(np.array(img))
        data = {
            "channels": arr[np.newaxis, :, :],
            "seg": None,
            "flair": arr,
            "all_channels": [arr],
        }
        st.info("PNG carregat com a FLAIR (escala de grisos, 240×240).")
    except Exception as e:
        st.error(f"Error llegint PNG: {e}")

# ─── Input preview ────────────────────────────────────────────────────────────

if data is not None:
    st.subheader("Vista prèvia de l'entrada")
    if mode == "binary" or len(data["all_channels"]) == 1:
        st.image(
            (data["flair"] * 255).astype(np.uint8),
            caption="FLAIR",
            width=240,
        )
    else:
        mod_names = ["FLAIR", "T1", "T1ce", "T2"]
        preview_cols = st.columns(4)
        for col, ch, name in zip(preview_cols, data["all_channels"], mod_names):
            col.image(
                (ch * 255).astype(np.uint8),
                caption=name,
                use_container_width=True,
            )

# ─── Prediction ───────────────────────────────────────────────────────────────

if run_btn:
    if data is None:
        st.error("No hi ha dades carregades. Selecciona un pacient o puja un fitxer.")
    elif model is None:
        st.error("El model no s'ha pogut carregar.")
    else:
        with st.spinner("Executant inferència..."):
            pred = run_inference(model, data["channels"], mode)
        st.session_state["pred"] = pred
        st.session_state["prediction_ready"] = True
        st.session_state["pred_data"] = data
        st.session_state["pred_mode"] = mode

if st.session_state.get("prediction_ready") and st.session_state.get("pred_mode") == mode:
    pred = st.session_state["pred"]
    saved_data = st.session_state["pred_data"]

    st.subheader("Resultats de la segmentació")

    fig = build_overlay_figure(
        flair=saved_data["flair"],
        pred=pred,
        seg=saved_data.get("seg"),
        mode=mode,
        all_channels=saved_data.get("all_channels"),
    )
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    col_stats, col_params = st.columns([1, 1])

    with col_stats:
        st.subheader("Estadístiques de la predicció")
        stats = compute_stats(pred, mode)

        if mode == "binary":
            for label, pct in stats.items():
                color = CLASS_COLORS.get(1, "#e63946") if label != "Background" else "#555"
                st.markdown(f"**{label}**: `{pct:.1f}%`")
                bar_val = min(pct / 100 * (4 if label != "Background" else 1.2), 1.0)
                st.progress(bar_val)
        else:
            for cls_id, cls_name in CLASS_NAMES.items():
                pct = stats.get(cls_name, 0.0)
                color = CLASS_COLORS.get(cls_id, "#888")
                st.markdown(
                    f'<span style="color:{color};font-weight:bold">■</span> **{cls_name}**: `{pct:.1f}%`',
                    unsafe_allow_html=True,
                )
                bar_val = min(pct / 100 * (4 if cls_id != 0 else 1.2), 1.0)
                st.progress(bar_val)

    with col_params:
        st.subheader("Hiperparàmetres del model")
        col_left, col_right = st.columns(2)

        def _hp(label, value, col):
            col.markdown(
                f'<div style="background:#1e2130;border-radius:8px;padding:10px 12px;margin-bottom:8px">'
                f'<div style="font-size:11px;color:#888;margin-bottom:2px">{label}</div>'
                f'<div style="font-size:15px;font-weight:bold;color:#e8eaf6">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _hp("Arquitectura", "U-Net 2D", col_left)
        _hp("Funció de pèrdua", cfg["loss"], col_right)
        _hp("Canals entrada / sortida", f"{cfg['in_channels']} → {cfg['out_channels']}", col_left)
        _hp("Learning rate", cfg["lr"], col_right)
        _hp("Modalitats", ", ".join(cfg["modalities"]).upper(), col_left)
        _hp("Epochs / Batch", f"{cfg['epochs']} / {cfg['batch_size']}", col_right)
        _hp("Test Dice", f"{cfg['test_dice']:.4f}" if cfg["test_dice"] else "N/A", col_left)
        _hp("Test IoU", f"{cfg['test_iou']:.4f}" if cfg["test_iou"] else "N/A", col_right)

elif not st.session_state.get("prediction_ready"):
    st.info("Selecciona una imatge i prem **▶ Executar predicció** per veure els resultats.")
