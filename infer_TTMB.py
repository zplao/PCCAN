# -*- coding: utf-8 -*-
"""
infer_TTMB.py

Inference / evaluation script for the GXU bearing zero-shot compound GNN backend.

V17 change: all frequently used runtime parameters are defined in this file, and counterfactual intervention follows the component-space logic of the MFD-BG inference script.  Because the GXU GNN model forward method requires a PyG Data/Batch object with a ``signal`` attribute, the counterfactual spectrum is injected into a fresh Data object by synchronously replacing ``signal`` and ``x`` before prediction.  This avoids the v16 raw-tensor error while ensuring that the spectrum fed to the model is the displayed zeroed-component spectrum.

Expected project files in the same directory or Python path:
    common.py
    dataset_gnn_ordered.py
    models_gnn_ordered.py
    backend_utils_gnn.py  (optional; only used for protocol split if available)

Typical usage:
    python infer_TTMB.py \
        --cache_root ./multicomponent_results_TTMB \
        --ckpt ./zs_outputs_TTMB/best_gnn_backend_cf.pt \
        --out_dir ./infer_outputs_TTMB \
        --eval_mode all

Evaluate Task-A held-out real compounds only:
    python infer_zs_cf_gnn.py --cache_root ... --ckpt ... --eval_mode taskA_unseen

Evaluate Task-B held-out seen + unseen split, using the same seed/proxy settings:
    python infer_zs_cf_gnn.py --cache_root ... --ckpt ... --eval_mode taskB_all

Important:
    This script does not train the model. It only loads a trained checkpoint and performs
    forward inference. If labels are available in the cache names, it will also report metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG_FONT_FAMILY = "Times New Roman"
FIG_FONT_SIZE = 22
plt.rcParams["font.family"] = FIG_FONT_FAMILY
plt.rcParams["font.serif"] = [FIG_FONT_FAMILY]
plt.rcParams["font.size"] = FIG_FONT_SIZE
plt.rcParams["axes.titlesize"] = FIG_FONT_SIZE
plt.rcParams["axes.labelsize"] = FIG_FONT_SIZE
plt.rcParams["xtick.labelsize"] = FIG_FONT_SIZE
plt.rcParams["ytick.labelsize"] = FIG_FONT_SIZE
plt.rcParams["legend.fontsize"] = FIG_FONT_SIZE
plt.rcParams["figure.titlesize"] = FIG_FONT_SIZE
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.default"] = "it"

from common import FAULTS, ensure_dir, multihot_to_name, exact_match_from_arrays, seed_everything
from dataset_gnn import V11OrderCacheReaderGNN, RealOrderDataset
from models_gnn import GNNCompoundNet

try:
    from backend_utils_gnn import build_protocol_splits
except Exception:  # older project versions may not have it
    build_protocol_splits = None


# ============================================================
# Runtime defaults
# ============================================================
# These defaults let you run this file directly in PyCharm without passing
# command-line arguments.  Edit them here if your local folders differ.
DEFAULT_CACHE_ROOT = "./multicomponent_results_TTMB"
DEFAULT_CKPT = "./zs_outputs_TTMB/best_gnn_backend_cf.pt"
DEFAULT_OUT_DIR = "./infer_outputs_TTMB"
DEFAULT_EVAL_MODE = "taskB_all"  # taskA_unseen, taskB_all
DEFAULT_DECODE_MODE = "threshold"
DEFAULT_CF_PREDICT_MODE = "data"  # data = PyG Data path with signal/x replaced; tensor is diagnostic only


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_arg(ckpt_args: Dict, cli_args, name: str, default):
    v = getattr(cli_args, name, None)
    if v is not None:
        return v
    return ckpt_args.get(name, default)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))).astype(np.float32)


def _svg_path_for(out_path: Path) -> Path:
    out_path = Path(out_path)
    return out_path.with_suffix(".svg")


def save_figure_png_svg(fig, out_path: Path, dpi: int = 220, **savefig_kwargs) -> Tuple[Path, Path]:
    """Save a Matplotlib figure as both PNG and SVG using the same basename."""
    out_path = Path(out_path)
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    svg_path = _svg_path_for(png_path)
    fig.savefig(png_path, dpi=dpi, **savefig_kwargs)
    fig.savefig(svg_path, **savefig_kwargs)
    return png_path, svg_path



def extract_embedding_np(aux, expected_batch: int | None = None) -> np.ndarray | None:
    """Robustly extract a batch embedding tensor from model auxiliary output.

    Some backends return (logits, embedding), while others return
    (logits, aux_dict).  This helper prevents inference from failing when
    the second output is a dict containing attention weights, node features,
    graph embeddings, etc.  If no suitable embedding is found, return None;
    the caller can then fall back to logits/probabilities for t-SNE.
    """
    def _as_array(v):
        if torch.is_tensor(v):
            arr = v.detach().cpu().numpy().astype(np.float32)
            if expected_batch is not None:
                if arr.ndim == 1 and expected_batch == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim >= 2 and arr.shape[0] == expected_batch:
                    return arr.reshape(expected_batch, -1)
                return None
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim >= 2:
                arr = arr.reshape(arr.shape[0], -1)
            return arr
        return None

    arr = _as_array(aux)
    if arr is not None:
        return arr

    if isinstance(aux, dict):
        preferred = [
            'embedding', 'embeddings', 'emb', 'graph_embedding', 'graph_emb',
            'global_embedding', 'global_emb', 'pooled', 'pooled_feat',
            'features', 'feat', 'z', 'h', 'readout'
        ]
        for k in preferred:
            if k in aux:
                arr = _as_array(aux[k])
                if arr is not None:
                    return arr
        # Fall back to the first tensor-like value with the correct batch dimension.
        for v in aux.values():
            arr = extract_embedding_np(v, expected_batch=expected_batch)
            if arr is not None:
                return arr

    if isinstance(aux, (list, tuple)):
        for v in aux:
            arr = extract_embedding_np(v, expected_batch=expected_batch)
            if arr is not None:
                return arr

    return None


def robust_norm_1d_local(x: np.ndarray) -> np.ndarray:
    """Robust normalization consistent with training-time spectrum preprocessing."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    med = np.nanmedian(x)
    y = x - med
    mad = np.nanmedian(np.abs(y))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.nanstd(y) + 1e-8
    return (y / scale).astype(np.float32)


def _sample_faults_from_label(label: str) -> List[str]:
    label = str(label)
    if label.lower() == "normal" or label.strip() == "":
        return []
    return [f for f in FAULTS if f in label.split("+")]



def _interp_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if len(arr) == length:
        return arr.astype(np.float32)
    if len(arr) <= 1:
        return np.zeros(length, dtype=np.float32)
    old = np.linspace(0.0, 1.0, len(arr))
    new = np.linspace(0.0, 1.0, length)
    return np.interp(new, old, arr).astype(np.float32)


def _get_component_array(reader, sample_index: int, fault: str, length: int,
                         preferred_keys=("display", "clean", "masked_clean")) -> np.ndarray:
    """Return one cached component in the original component-amplitude space.

    Counterfactual removal must be performed before robust normalization.  The old
    implementation subtracted an unnormalized component from reader.input_spectrum,
    which is already normalized; consequently the removed component was not visibly
    suppressed in the plotted counterfactual spectrum.
    """
    s = reader.samples[int(sample_index)]
    comp = None
    if isinstance(s, dict):
        specs = s.get("component_specs", {})
        if isinstance(specs, dict) and fault in specs and isinstance(specs[fault], dict):
            for k in preferred_keys:
                if k in specs[fault] and specs[fault][k] is not None:
                    comp = specs[fault][k]
                    break
        if comp is None and "display_components" in s and fault in s["display_components"] and "display" in preferred_keys:
            comp = s["display_components"][fault]
        if comp is None and "masked_clean_components" in s and fault in s["masked_clean_components"]:
            comp = s["masked_clean_components"][fault]
    if comp is None:
        return np.zeros(length, dtype=np.float32)
    return _interp_to_length(comp, length)


def _get_intervention_component(reader, sample_index: int, fault: str, length: int) -> np.ndarray:
    """Return the component subtracted for do(C_f=0)."""
    s = reader.samples[int(sample_index)]
    comp = None
    if isinstance(s, dict):
        mcc = s.get("masked_clean_components", {})
        if isinstance(mcc, dict) and fault in mcc:
            comp = mcc[fault]
        if comp is None:
            specs = s.get("component_specs", {})
            if isinstance(specs, dict) and fault in specs and isinstance(specs[fault], dict):
                comp = specs[fault].get("masked_clean", None)
    if comp is None:
        return np.zeros(length, dtype=np.float32)
    return _interp_to_length(comp, length)


def _rebuild_raw_factual_spectrum(reader, sample_index: int, active_faults: List[str], length: int) -> np.ndarray:
    raw = np.zeros(length, dtype=np.float32)
    for f in active_faults:
        raw += _get_component_array(reader, sample_index, f, length, preferred_keys=("display", "clean", "masked_clean"))
    if not np.any(np.isfinite(raw)) or float(np.nanmax(np.abs(raw))) < 1e-12:
        _, x = _get_input_spectrum(reader, sample_index)
        raw = _interp_to_length(x, length)
    return np.maximum(0.0, raw.astype(np.float32))


def _make_counterfactual_raw_spectrum(reader, sample_index: int, active_faults: List[str],
                                      remove_fault: str, length: int, alpha: float = 1.0) -> np.ndarray:
    """Construct X^{do(C_remove=0)} by explicitly zeroing the removed component.

    V14 important change for GXU:
    The previous implementation used

        raw_cf = raw_fact - alpha * masked_clean_component[remove_fault]

    where ``raw_fact`` was rebuilt mainly from display/clean components but the
    subtracted term was often a much narrower ``masked_clean`` component.  In the
    counterfactual figures this produced only thin vertical notches and did not
    truly remove the target fault contribution before prediction.

    Here we follow the MFD-BG component-space intervention idea more strictly:
    the factual spectrum is represented as the sum of active cached fault
    components, and the intervention ``do(C_remove=0)`` is obtained by rebuilding
    the spectrum with the removed component omitted, i.e.

        X_cf(o) = sum_{f in active_faults, f != remove_fault} C_f(o).

    Therefore the intervened component is exactly set to zero in the spectrum
    fed to the model.  ``alpha`` is kept only for command-line compatibility and
    is intentionally not used in this zeroing mode.
    """
    raw_cf = np.zeros(length, dtype=np.float32)
    for f in active_faults:
        if str(f) == str(remove_fault):
            continue
        raw_cf += _get_component_array(
            reader,
            sample_index,
            f,
            length,
            preferred_keys=("display", "clean", "masked_clean"),
        )

    # If all remaining components are absent, keep a valid all-zero intervention
    # rather than falling back to the original input.  This is necessary for a
    # faithful do(C_f=0) operation on single-component residual cases.
    raw_cf = np.nan_to_num(raw_cf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.maximum(0.0, raw_cf)


def _tensor_with_reference_shape(values: np.ndarray, reference, device: torch.device | None = None) -> torch.Tensor:
    """Create a tensor from a 1-D spectrum using the shape of an existing Data attribute.

    PyG ``Data`` attributes such as ``signal`` or ``x`` may be stored as ``[L]``
    or ``[1, L]`` before batching.  The GXU model reads ``batch.signal``; thus
    replacing only ``data.x`` is insufficient when ``data.signal`` still contains
    the original factual spectrum.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if torch.is_tensor(reference):
        ref_shape = tuple(reference.shape)
        ref_dtype = reference.dtype if reference.dtype.is_floating_point else torch.float32
        if int(np.prod(ref_shape)) == arr.size:
            out = torch.from_numpy(arr.reshape(ref_shape)).to(dtype=ref_dtype)
        elif len(ref_shape) >= 1 and ref_shape[-1] == arr.size:
            shape = (1,) * (len(ref_shape) - 1) + (arr.size,)
            out = torch.from_numpy(arr.reshape(shape)).to(dtype=ref_dtype)
        else:
            out = torch.from_numpy(arr[None, :]).to(dtype=ref_dtype)
    else:
        out = torch.from_numpy(arr[None, :]).to(dtype=torch.float32)
    if device is not None:
        out = out.to(device)
    return out


def _replace_data_spectrum_fields(data, x_spec: np.ndarray) -> object:
    """Replace every spectrum field that may be used by the GXU PyG model.

    The original v8/v14/v16 scripts modified only ``data.x``.  In the current
    GXU backend the model accesses ``data.signal`` during ``forward``.  If
    ``signal`` is not synchronized with the counterfactual spectrum, the plotted
    left-panel spectrum changes but the right-panel probability is still computed
    from the original factual signal.
    """
    x_norm = robust_norm_1d_local(x_spec)
    # Update the model input signal.  The local GXU model raises
    # "Tensor object has no attribute signal", confirming that ``signal`` is
    # the field consumed by forward().
    if hasattr(data, "signal"):
        ref = getattr(data, "signal")
        try:
            setattr(data, "signal", _tensor_with_reference_shape(x_norm, ref))
        except Exception:
            setattr(data, "signal", torch.from_numpy(x_norm[None, :].astype(np.float32)))

    # Update x only when it is also a spectrum-shaped field.  If ``x`` stores
    # node features in a local implementation, leaving it unchanged is safer.
    if hasattr(data, "x"):
        ref = getattr(data, "x")
        if torch.is_tensor(ref):
            shape = tuple(ref.shape)
            if int(np.prod(shape)) == x_norm.size or (len(shape) >= 1 and shape[-1] == x_norm.size):
                setattr(data, "x", _tensor_with_reference_shape(x_norm, ref))
        else:
            setattr(data, "x", torch.from_numpy(x_norm[None, :].astype(np.float32)))
    # Optional aliases in older local dataset implementations.
    for attr in ("input_spectrum", "spectrum", "order_amp"):
        if hasattr(data, attr):
            ref = getattr(data, attr)
            if torch.is_tensor(ref) and int(np.prod(tuple(ref.shape))) == x_norm.size:
                setattr(data, attr, _tensor_with_reference_shape(x_norm, ref))
    return data


def predict_prob_gxu_data(model, reader, sample_index: int, x_spec: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """PyG-Data prediction path with faithful counterfactual spectrum injection.

    This mirrors the MFD-BG script at the intervention level: first rebuild the
    component-space counterfactual spectrum, then normalize it, then feed it to
    the model.  The difference is only the container type: GXU ``GNNCompoundNet``
    requires a PyG ``Data/Batch`` object and reads ``batch.signal``, so both
    ``signal`` and ``x`` are replaced before batching.
    """
    base_data = RealOrderDataset(reader, [int(sample_index)])[0]
    try:
        data = base_data.clone()
    except Exception:
        data = copy.deepcopy(base_data)
    data = _replace_data_spectrum_fields(data, x_spec)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        logits, emb = model(batch)
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    if logits_np.ndim == 1:
        logits_np = logits_np.reshape(1, -1)
    probs = sigmoid_np(logits_np)[0]
    emb_arr = extract_embedding_np(emb, expected_batch=1)
    emb_np = emb_arr[0] if emb_arr is not None else logits_np[0]
    return probs, emb_np


def _run_model_on_tensor(model, xb: torch.Tensor):
    """Run a model on a tensor while accepting common output conventions."""
    out = model(xb)
    if isinstance(out, tuple):
        if len(out) == 0:
            raise RuntimeError("Model returned an empty tuple.")
        logits = out[0]
        aux = out[1] if len(out) > 1 else logits
    else:
        logits, aux = out, out
    return logits, aux


def predict_prob_gxu_direct_tensor(model, x_spec: np.ndarray, device: torch.device, prefer_3d: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Direct tensor prediction path for counterfactual spectra.

    This intentionally mirrors the MFD-BG inference code:

        x = robust_norm_1d_local(x_spec)
        xb = torch.from_numpy(x[None, None, :]).to(device)
        logits, emb = model(xb)

    Some local GXU model implementations may accept ``[B,L]`` rather than
    ``[B,1,L]``; therefore v15 first tries the MFD-BG-compatible 3-D form and
    then the 2-D form.  If both fail, an explicit error is raised because the
    loaded GXU ``GNNCompoundNet`` does not implement a tensor forward path.
    """
    x = robust_norm_1d_local(x_spec)
    candidates = []
    if prefer_3d:
        candidates.append(torch.from_numpy(x[None, None, :].astype(np.float32)).to(device))
        candidates.append(torch.from_numpy(x[None, :].astype(np.float32)).to(device))
    else:
        candidates.append(torch.from_numpy(x[None, :].astype(np.float32)).to(device))
        candidates.append(torch.from_numpy(x[None, None, :].astype(np.float32)).to(device))

    errors = []
    with torch.no_grad():
        for xb in candidates:
            try:
                logits, emb = _run_model_on_tensor(model, xb)
                logits_np = logits.detach().cpu().numpy().astype(np.float32)
                if logits_np.ndim == 1:
                    logits_np = logits_np.reshape(1, -1)
                probs = sigmoid_np(logits_np)[0]
                emb_arr = extract_embedding_np(emb, expected_batch=1)
                emb_np = emb_arr[0] if emb_arr is not None else logits_np[0]
                return probs, emb_np
            except Exception as e:
                errors.append(f"shape={tuple(xb.shape)}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Direct tensor counterfactual prediction failed. The loaded GXU "
        "GNNCompoundNet appears to require a PyG Data/Batch object rather than "
        "a raw tensor. To truly use the MFD-BG-style tensor path, the GXU "
        "model forward method must support tensor inputs. Tried:\n  " + "\n  ".join(errors)
    )


def predict_prob_gxu_cf(model, reader, sample_index: int, x_spec: np.ndarray, device: torch.device, mode: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Counterfactual prediction dispatcher.

    ``data`` is the default because the GXU model expects ``batch.signal``.
    ``tensor`` is kept only as a diagnostic option; if tensor forward is not
    supported, the function automatically falls back to the PyG Data path with
    ``signal`` and ``x`` replaced.
    """
    mode = str(mode).lower()
    if mode == "data":
        return predict_prob_gxu_data(model, reader, sample_index, x_spec, device)
    if mode == "tensor":
        try:
            return predict_prob_gxu_direct_tensor(model, x_spec, device)
        except Exception:
            return predict_prob_gxu_data(model, reader, sample_index, x_spec, device)
    raise ValueError(f"Unsupported counterfactual prediction mode: {mode}")


def _draw_cf_bar(ax, probs: np.ndarray, title: str, prob_ref_line: float = 0.5):
    x = np.arange(len(FAULTS))
    ax.set_axisbelow(True)
    ax.bar(x, probs, width=0.45, color=FAULT_BAR_COLORS[:len(FAULTS)])
    # ax.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(_display_fault_names())
    ax.set_title(title)
    ax.set_xlabel("Fault type")
    ax.grid(True, axis="y", alpha=0.20, zorder=0)
    for j, p in enumerate(probs):
        p = float(p)
        # if p > 0.8:
        #     y_text = 0.98
        #     va = "top"
        # else:
        #     y_text = min(0.98, p + 0.035)
        #     va = "bottom"
        y_text = min(0.98, p + 0.035)
        va = "bottom"
        ax.text(j, y_text, f"{p:.4f}", ha="center", va=va, fontsize=16)


def _scale_for_plot(stages_raw: List[np.ndarray]) -> List[np.ndarray]:
    fact = np.asarray(stages_raw[0], dtype=np.float32)
    scale = float(np.nanmax(fact)) if fact.size else 1.0
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return [np.asarray(x, dtype=np.float32) / scale for x in stages_raw]


def plot_counterfactual_interventions_by_label(reader, model, rows: List[dict], out_dir: Path,
                                               device: torch.device, max_per_label: int = 1,
                                               alpha: float = 1.0, prob_ref_line: float = 0.5,
                                               theory_info: Dict[str, Dict[str, object]] | None = None,
                                               max_theory_lines_per_fault: int = 12,
                                               cf_predict_mode: str = "data"):
    """For each compound fault type, remove one active causal component at a time.

    Interventions are performed in the cached component-amplitude space and only then
    robust-normalized for model prediction.  In v17 the probability bars are computed after the displayed counterfactual
    spectrum has been injected into both ``data.signal`` and ``data.x``.  This
    is necessary because the GXU GNN reads ``batch.signal`` rather than a raw
    tensor.
    """
    ensure_dir(out_dir)
    compound_rows = [r for r in rows if len(_sample_faults_from_label(r.get("true_label", ""))) >= 2]
    if not compound_rows:
        return
    grouped: Dict[str, List[dict]] = {}
    for r in compound_rows:
        grouped.setdefault(str(r["true_label"]), []).append(r)
    index_rows = []
    for label in sorted(grouped.keys()):
        candidates = [r for r in grouped[label] if int(r.get("exact", 0)) == 1] or grouped[label]
        candidates = candidates[:max(1, int(max_per_label))]
        for sample_no, r in enumerate(candidates):
            idx = int(r["index"])
            active_faults = _sample_faults_from_label(r["true_label"])
            axis = _get_reader_axis(reader, len(reader.samples[idx]["input_spectrum"]))
            raw_fact = _rebuild_raw_factual_spectrum(reader, idx, active_faults, len(axis))
            p_fact, _ = predict_prob_gxu_cf(model, reader, idx, raw_fact, device, mode=cf_predict_mode)

            stage_names = ["Factual input"]
            stage_raw = [raw_fact]
            stage_probs = [p_fact]
            for f in active_faults:
                raw_cf = _make_counterfactual_raw_spectrum(reader, idx, active_faults, f, len(axis), alpha)
                p_cf, _ = predict_prob_gxu_cf(model, reader, idx, raw_cf, device, mode=cf_predict_mode)
                stage_names.append(rf"Counterfactual intervention: $do(C_{{\mathrm{{{_display_fault_name(f)}}}}}=0)$")
                stage_raw.append(raw_cf)
                stage_probs.append(p_cf)

            stage_plot = _scale_for_plot(stage_raw)
            n = len(stage_names)
            fig, axes = plt.subplots(n, 2, figsize=(12, max(3.2, 2.8 * n)), squeeze=False,
                                     gridspec_kw={"width_ratios": [3.2, 1.35]})
            for i, (stage_name, spec_plot, probs) in enumerate(zip(stage_names, stage_plot, stage_probs)):
                axes[i, 0].plot(axis, spec_plot, color="#1f77b4" if i == 0 else "#d62728", lw=1.25)
                # axes[i, 0].set_xlim(float(np.nanmin(axis)), float(np.nanmax(axis)))
                axes[i, 0].set_xlim(0.0, 50.01)
                axes[i, 0].set_xticks(np.arange(0.0, 50.01, 5))
                axes[i, 0].set_xlabel("Order")
                axes[i, 0].set_ylabel("Amplitude")
                # axes[i, 0].set_title(f"{stage_name} | true={_display_fault_label(r['true_label'])}")
                axes[i, 0].set_title(f"{stage_name}")
                _overlay_theory_orders(axes[i, 0], theory_info, active_faults, max_lines_per_fault=max_theory_lines_per_fault)
                axes[i, 0].set_ylim(0.0, 1.05)
                axes[i, 0].set_yticks(np.arange(0.0, 1.01, 0.2))
                axes[i, 0].grid(True, alpha=0.25)
                _draw_cf_bar(axes[i, 1], probs, "Predicted probabilities", prob_ref_line=prob_ref_line)
            # fig.suptitle(f"Counterfactual intervention analysis: {_display_fault_label(label)}", fontsize=FIG_FONT_SIZE, y=1.01)
            fig.tight_layout()
            suffix = f"_{sample_no+1}" if len(candidates) > 1 else ""
            out_path = out_dir / f"counterfactual_{_safe_filename(label)}{suffix}.png"
            save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            index_rows.append({"label": label, "sample_index": idx, "name": r["name"], "figure": str(out_path), "svg_figure": str(_svg_path_for(out_path))})
    with open(out_dir / "counterfactual_figures_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "sample_index", "name", "figure", "svg_figure"])
        writer.writeheader(); writer.writerows(index_rows)


def plot_tsne_embeddings(embeddings: np.ndarray, labels: List[str], out_path: Path, max_points: int = 1200, seed: int = 42):
    if embeddings is None or len(embeddings) < 3:
        return
    X = np.asarray(embeddings, dtype=np.float32)
    labels = list(labels)
    if len(X) > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=int(max_points), replace=False)
        X = X[idx]
        labels = [labels[i] for i in idx]
    perplexity = max(2, min(30, (len(X) - 1) // 3))
    try:
        Z = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(X)
    except Exception:
        return
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for i, lab in enumerate(uniq):
        m = np.asarray([x == lab for x in labels])
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.78, label=_display_fault_label(lab), color=cmap(i % 10))
    ax.set_title("t-SNE visualization of inference embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=FIG_FONT_SIZE, loc="best", frameon=True)
    fig.tight_layout()
    save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_seed_list(seed_list: str, seed: int, num_runs: int) -> List[int]:
    if seed_list:
        return [int(x.strip()) for x in str(seed_list).split(",") if x.strip()]
    return [int(seed) + i for i in range(max(1, int(num_runs)))]

def labels_to_name_from_multihot(y: np.ndarray) -> str:
    try:
        return multihot_to_name(y)
    except Exception:
        y = np.asarray(y).reshape(-1)
        active = [FAULTS[i] for i in range(min(len(FAULTS), len(y))) if int(y[i]) == 1]
        return "+".join(active) if active else "Normal"


def build_candidate_multihots(mode: str) -> List[np.ndarray]:
    """Candidate-label decoding, optional. Threshold decoding is the default."""
    mode = str(mode).lower()
    candidates: List[List[str]] = []
    if mode == "taska":
        candidates = [["IR", "OR"], ["IR", "Ball"], ["OR", "Ball"], ["IR", "OR", "Ball"]]
    elif mode == "taskb":
        candidates = [[], ["IR"], ["OR"], ["Ball"], ["IR", "OR"], ["IR", "Ball"], ["OR", "Ball"], ["IR", "OR", "Ball"]]
    else:
        return []
    out = []
    for labs in candidates:
        y = np.zeros(len(FAULTS), dtype=np.float32)
        for i, f in enumerate(FAULTS):
            if f in labs:
                y[i] = 1.0
        out.append(y)
    return out


def decode_probs(probs: np.ndarray, thr: Sequence[float], decode_mode: str = "threshold") -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    decode_mode = str(decode_mode).lower()
    if decode_mode == "threshold":
        return (probs >= np.asarray(thr, dtype=np.float32)[None, :]).astype(np.float32)

    candidates = build_candidate_multihots(decode_mode.replace("candidate_", ""))
    if not candidates:
        raise ValueError(f"Unsupported decode_mode={decode_mode}. Use threshold, candidate_taska, or candidate_taskb.")
    C = np.stack(candidates, axis=0).astype(np.float32)  # [C,K]
    eps = 1e-6
    logp = np.log(np.clip(probs, eps, 1.0 - eps))
    log1p = np.log(np.clip(1.0 - probs, eps, 1.0 - eps))
    scores = logp @ C.T + log1p @ (1.0 - C).T  # [N,C]
    idx = np.argmax(scores, axis=1)
    return C[idx].astype(np.float32)


def choose_indices(reader: V11OrderCacheReaderGNN, args) -> List[int]:
    n = len(reader.samples)
    mode = str(args.eval_mode).lower()
    if mode == "all":
        return list(range(n))
    if mode == "seen":
        return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) <= 1]
    if mode == "singles":
        return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) == 1]
    if mode == "normal":
        return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) == 0]
    if mode in {"unseen", "compounds", "taska_unseen"}:
        return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) >= 2]

    if mode.startswith("taskb") or mode.startswith("taska"):
        if build_protocol_splits is not None:
            task = "A" if mode.startswith("taska") else "B"
            splits = build_protocol_splits(reader, task=task, proxy_ratio=args.proxy_ratio,
                                           seen_test_ratio=args.seen_test_ratio, seed=args.seed)
            if mode in {"taska_unseen", "taska", "taska_all"}:
                return list(splits.get("real_test_idx", []))
            if mode == "taskb_seen":
                return list(splits.get("seen_test_idx", []))
            if mode in {"taskb_unseen", "taskb_real"}:
                return list(splits.get("unseen_test_idx", splits.get("real_test_idx", [])))
            if mode in {"taskb_all", "taskb"}:
                seen = list(splits.get("seen_test_idx", []))
                unseen = list(splits.get("unseen_test_idx", splits.get("real_test_idx", [])))
                return seen + unseen
        # fallback without split helper
        if mode.startswith("taska"):
            return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) >= 2]
        return list(range(n))

    raise ValueError(f"Unknown eval_mode={args.eval_mode}")


def plot_confusion(y_true_names: List[str], y_pred_names: List[str], out_path: Path):
    labels = sorted(list(set(y_true_names) | set(y_pred_names)))
    if not labels:
        return
    cm = confusion_matrix(y_true_names, y_pred_names, labels=labels)
    fig, ax = plt.subplots(figsize=(5 + 0.6 * len(labels), 4 + 0.45 * len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    display_labels = [_display_fault_label(lab) for lab in labels]
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Inference confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=FIG_FONT_SIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure_png_svg(fig, out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(name: str) -> str:
    name = str(name).replace("+", "_plus_").replace("/", "_").replace("\\", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _get_reader_axis(reader, spectrum_len: int) -> np.ndarray:
    axis = getattr(reader, "target_axis", None)
    if axis is not None:
        axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        if len(axis) == spectrum_len:
            return axis
    max_order = getattr(reader, "max_order", None)
    if max_order is None:
        try:
            max_order = float(np.nanmax(axis)) if axis is not None and len(axis) > 1 else float(spectrum_len - 1)
        except Exception:
            max_order = float(spectrum_len - 1)
    return np.linspace(0.0, float(max_order), spectrum_len, dtype=np.float32)


def _get_input_spectrum(reader, sample_index: int) -> Tuple[np.ndarray, np.ndarray]:
    s = reader.samples[int(sample_index)]
    x = None
    for key in ("input_spectrum", "display_spectrum", "order_amp", "x"):
        if isinstance(s, dict) and key in s:
            x = s[key]
            break
    if x is None and isinstance(s, dict) and "input" in s:
        x = s["input"]
    if x is None:
        raise KeyError(f"Cannot find input spectrum in reader.samples[{sample_index}]. Available keys: {list(s.keys()) if isinstance(s, dict) else type(s)}")
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    axis = _get_reader_axis(reader, len(x))
    return axis, x


FAULT_DISPLAY_NAMES = {"IR": "IF", "OR": "OF", "Ball": "RF"}
FAULT_BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def _fault_color_map() -> Dict[str, str]:
    return {f: FAULT_BAR_COLORS[i % len(FAULT_BAR_COLORS)] for i, f in enumerate(FAULTS)}


def _display_fault_name(name: str) -> str:
    return FAULT_DISPLAY_NAMES.get(str(name), str(name))


def _display_fault_label(label: str) -> str:
    label = str(label)
    if label.lower() == "normal" or label.strip() == "":
        return "Normal"
    return "+".join(_display_fault_name(part) for part in label.split("+"))


def _harmonic_centers(base_order: float, max_order: float, max_harmonics: int) -> List[float]:
    centers = []
    try:
        base = float(base_order)
    except Exception:
        return centers
    if not np.isfinite(base) or base <= 0:
        return centers
    for k in range(1, int(max_harmonics) + 1):
        c = k * base
        if c <= float(max_order) + 1e-9:
            centers.append(float(c))
    return centers


def build_theory_order_info_gxu(max_order: float, max_harmonics: int = 12,
                                ir_order: float = 10.2273, or_order: float = 7.7727,
                                ball_order: float = 3.5985) -> Dict[str, Dict[str, object]]:
    """Build theoretical order markers for GXU bearing figures."""
    color_map = _fault_color_map()
    bases = {'IR': float(ir_order), 'OR': float(or_order), 'Ball': float(ball_order)}
    info: Dict[str, Dict[str, object]] = {}
    for f in FAULTS:
        if f in bases:
            rho = bases[f]
            info[f] = {
                'centers': _harmonic_centers(rho, max_order, max_harmonics),
                'text': f'{_display_fault_name(f)}: rho={rho:.3f}, m*rho',
                'color': color_map.get(f, 'k'),
            }
    return info


def _overlay_theory_orders(ax, theory_info: Dict[str, Dict[str, object]] | None,
                           faults: Sequence[str] | None = None,
                           max_lines_per_fault: int = 12, annotate: bool = True):
    """Overlay theoretical order markers on a spectrum axis."""
    if not theory_info:
        return
    if faults is None:
        faults = list(FAULTS)
    faults = [f for f in faults if f in theory_info]
    if not faults:
        return
    x0, x1 = ax.get_xlim()
    text_lines = []
    handles = []
    labels = []
    for f in faults:
        item = theory_info.get(f, {})
        centers = [float(c) for c in item.get('centers', []) if x0 <= float(c) <= x1]
        if not centers:
            continue
        color = str(item.get('color', '0.3'))
        shown = centers[:max(1, int(max_lines_per_fault))]
        for j, c in enumerate(shown):
            line = ax.axvline(c, color=color, linestyle='--', linewidth=0.8, alpha=0.55, zorder=0)
            if j == 0:
                handles.append(line); labels.append(f'{_display_fault_name(f)} theory order')
        if len(centers) > len(shown):
            text_lines.append(str(item.get('text', f'{_display_fault_name(f)}: theory order')) + f' (first {len(shown)}/{len(centers)})')
        else:
            text_lines.append(str(item.get('text', f'{_display_fault_name(f)}: theory order')))
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=18, framealpha=0.88, ncol=min(2, len(handles)))
    # if annotate and text_lines:
    #     ax.text(0.01, 0.98, 'Theoretical orders\n' + '\n'.join(text_lines),
    #             transform=ax.transAxes, ha='left', va='top', fontsize=FIG_FONT_SIZE,
    #             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.7', alpha=0.82))


def _display_fault_names() -> List[str]:
    return [FAULT_DISPLAY_NAMES.get(f, f) for f in FAULTS]


def _select_representative_rows(rows: List[dict], max_examples: int = 12) -> List[dict]:
    """Select one representative sample per true label first, then fill remaining slots."""
    if not rows:
        return []
    selected: List[dict] = []
    used = set()
    # Prefer correct predictions for each true label.
    for r in rows:
        key = str(r.get("true_label", "Unknown"))
        if key in used or int(r.get("exact", 0)) != 1:
            continue
        selected.append(r)
        used.add(key)
        if len(selected) >= max_examples:
            return selected
    # If some labels have no correct prediction, still show one sample.
    for r in rows:
        key = str(r.get("true_label", "Unknown"))
        if key not in used:
            selected.append(r)
            used.add(key)
        if len(selected) >= max_examples:
            return selected
    # Fill if needed.
    seen_ids = set(id(x) for x in selected)
    for r in rows:
        if id(r) not in seen_ids:
            selected.append(r)
        if len(selected) >= max_examples:
            break
    return selected


def _draw_probability_row(reader, row: dict, ax_spec, ax_prob, prob_ref_line: float = 0.5,
                          theory_info: Dict[str, Dict[str, object]] | None = None,
                          max_theory_lines_per_fault: int = 12):
    axis, spec = _get_input_spectrum(reader, int(row["index"]))
    probs = np.asarray([row[f"p_{f}"] for f in FAULTS], dtype=float)

    ax_spec.plot(axis, spec, linewidth=1.35, color="#1f77b4")
    ax_spec.set_xlabel("Order")
    ax_spec.set_ylabel("Normalized amplitude")
    ax_spec.set_title(f"Input order spectrum | true={_display_fault_label(row['true_label'])} | pred={_display_fault_label(row['pred_label'])}")
    active_faults = _sample_faults_from_label(row.get('true_label', ''))
    _overlay_theory_orders(ax_spec, theory_info, active_faults, max_lines_per_fault=max_theory_lines_per_fault)
    ax_spec.grid(True, alpha=0.25)
    try:
        ax_spec.set_xlim(float(np.nanmin(axis)), float(np.nanmax(axis)))
    except Exception:
        pass

    x = np.arange(len(FAULTS))
    ax_prob.bar(x, probs, color=FAULT_BAR_COLORS[:len(FAULTS)])
    ax_prob.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0)
    ax_prob.set_ylim(0.0, 1.08)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(_display_fault_names())
    ax_prob.set_title("Model predicted probabilities")
    ax_prob.grid(True, axis="y", alpha=0.20)
    for j, p in enumerate(probs):
        ax_prob.text(j, min(1.04, p + 0.035), f"{p:.2f}", ha="center", va="bottom", fontsize=FIG_FONT_SIZE, fontweight="bold")


def plot_probability_examples_with_spectrum(reader, rows: List[dict], out_path: Path, max_examples: int = 12, prob_ref_line: float = 0.5,
                                            theory_info: Dict[str, Dict[str, object]] | None = None,
                                            max_theory_lines_per_fault: int = 12):
    """Combined figure: one row per representative true label, same layout as requested."""
    selected = _select_representative_rows(rows, max_examples=max_examples)
    if not selected:
        return
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(15.2, max(3.4, 3.15 * n)), squeeze=False,
                             gridspec_kw={"width_ratios": [3.3, 1.35]})
    for i, r in enumerate(selected):
        _draw_probability_row(reader, r, axes[i, 0], axes[i, 1], prob_ref_line=prob_ref_line, theory_info=theory_info, max_theory_lines_per_fault=max_theory_lines_per_fault)
    fig.tight_layout()
    save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_probability_examples_by_label(reader, rows: List[dict], out_dir: Path, max_per_label: int = 1, prob_ref_line: float = 0.5,
                                       theory_info: Dict[str, Dict[str, object]] | None = None,
                                       max_theory_lines_per_fault: int = 12):
    """Save exactly one PNG for each true fault label/type.

    Each PNG uses the requested two-column layout: left input order spectrum, right predicted
    probabilities for all fault attributes. If max_per_label > 1, the samples for the same
    true label are stacked as multiple rows in the same PNG rather than generating many
    separate files.
    """
    if not rows:
        return
    ensure_dir(out_dir)
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(str(r.get("true_label", "Unknown")), []).append(r)

    summary = []
    max_per_label = max(1, int(max_per_label))
    for label in sorted(grouped.keys()):
        label_rows = grouped[label]
        correct = [r for r in label_rows if int(r.get("exact", 0)) == 1]
        candidates = correct if correct else label_rows
        candidates = sorted(candidates, key=lambda r: max(float(r[f"p_{f}"]) for f in FAULTS), reverse=True)[:max_per_label]

        n = len(candidates)
        out_path = out_dir / f"probability_{_safe_filename(label)}.png"
        fig, axes = plt.subplots(n, 2, figsize=(15.2, max(3.4, 3.15 * n)), squeeze=False,
                                 gridspec_kw={"width_ratios": [3.3, 1.35]})
        for i, r in enumerate(candidates):
            _draw_probability_row(reader, r, axes[i, 0], axes[i, 1], prob_ref_line=prob_ref_line, theory_info=theory_info, max_theory_lines_per_fault=max_theory_lines_per_fault)
            summary.append({"label": label, "index": r["index"], "name": r["name"], "path": r["path"], "figure": str(out_path), "svg_figure": str(_svg_path_for(out_path))})
        fig.tight_layout()
        save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    with open(out_dir / "probability_figures_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["label", "index", "name", "path", "figure", "svg_figure"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary:
            writer.writerow(r)


def plot_probability_examples(rows: List[dict], out_path: Path, max_examples: int = 12, prob_ref_line: float = 0.5):
    """Bar-only summary, retained as an auxiliary quick check."""
    selected = _select_representative_rows(rows, max_examples=max_examples)
    if not selected:
        return
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(8.8, max(2.2, 1.7 * n)), squeeze=False)
    for ax, r in zip(axes[:, 0], selected):
        probs = np.asarray([r[f"p_{f}"] for f in FAULTS], dtype=float)
        ax.bar(np.arange(len(FAULTS)), probs, color=FAULT_BAR_COLORS[:len(FAULTS)])
        ax.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(np.arange(len(FAULTS)))
        ax.set_xticklabels(_display_fault_names())
        ax.set_ylabel("Probability")
        ax.set_title(f"{r['name']} | true={_display_fault_label(r['true_label'])} | pred={_display_fault_label(r['pred_label'])}", fontsize=FIG_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        for j, p in enumerate(probs):
            ax.text(j, min(1.04, p + 0.035), f"{p:.2f}", ha="center", va="bottom", fontsize=FIG_FONT_SIZE, fontweight="bold")
    fig.tight_layout()
    save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("GXU GNN backend inference")
    ap.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to best_gnn_backend_cf.pt / best model checkpoint")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--eval_mode", type=str, default=DEFAULT_EVAL_MODE,
                    help="all, seen, singles, normal, unseen/compounds, taskA_unseen, taskB_seen, taskB_unseen, taskB_all")
    ap.add_argument("--decode_mode", type=str, default=DEFAULT_DECODE_MODE, choices=["threshold", "candidate_taska", "candidate_taskb"])
    ap.add_argument("--plot_max_examples", type=int, default=12, help="Max examples in the combined probability figure.")
    ap.add_argument("--per_label_plot_max", type=int, default=1, help="Number of spectrum+probability figures saved for each true label.")
    ap.add_argument("--cf_plot_max_per_label", type=int, default=1, help="Number of counterfactual intervention examples saved for each compound label.")
    ap.add_argument("--cf_alpha", type=float, default=1.0, help="Kept for compatibility. V14 zeroes the removed component, so this value is not used in counterfactual figures.")
    ap.add_argument("--cf_predict_mode", type=str, default=DEFAULT_CF_PREDICT_MODE, choices=["tensor", "data"],
                    help="Counterfactual probability path. tensor directly feeds the intervened spectrum to the model, following MFD-BG; data uses the legacy PyG Data path.")
    ap.add_argument("--show_theory_orders", type=int, default=1, choices=[0, 1], help="Overlay theoretical fault-order markers on spectrum figures.")
    ap.add_argument("--theory_max_harmonics", type=int, default=12, help="Maximum bearing harmonics used for theoretical order markers.")
    ap.add_argument("--theory_max_lines_per_fault", type=int, default=12, help="Maximum vertical theory markers displayed for each fault in one subplot.")
    ap.add_argument("--ir_order", type=float, default=10.2273, help="Theoretical IR/BPFI order for marker overlay.")
    ap.add_argument("--or_order", type=float, default=7.7727, help="Theoretical OR/BPFO order for marker overlay.")
    ap.add_argument("--ball_order", type=float, default=3.5985, help="Theoretical Ball/BSF order for marker overlay.")
    ap.add_argument("--tsne_max_points", type=int, default=1200, help="Maximum number of points used in t-SNE visualization.")
    ap.add_argument("--num_seed_runs", type=int, default=1, help="Repeat inference split evaluation with different random seeds; figures are saved only for the main seed.")
    ap.add_argument("--seed_list", type=str, default="", help="Comma-separated seeds. Overrides --num_seed_runs when provided.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)  # aligned with train_zs_cf_gnn_GXU_standalonecache_ordered_v20.py
    ap.add_argument("--proxy_ratio", type=float, default=0.25)
    ap.add_argument("--seen_test_ratio", type=float, default=0.25)
    # Optional overrides. If omitted, values are read from checkpoint args when possible.
    ap.add_argument("--keep_bins", type=int, default=None)
    ap.add_argument("--max_order", type=float, default=None)
    ap.add_argument("--input_spectrum_kind", type=str, default=None, choices=[None, "clean", "display"])
    ap.add_argument("--mask_mode", type=str, default=None, choices=[None, "peak_locked", "fixed_theory"])
    ap.add_argument("--ir_or_half_width", type=float, default=None)
    ap.add_argument("--ball_half_width", type=float, default=None)
    ap.add_argument("--search_half_width", type=float, default=None)
    ap.add_argument("--min_rel_peak", type=float, default=None)
    ap.add_argument("--max_harmonics_ir_or", type=int, default=None)
    ap.add_argument("--max_harmonics_ball", type=int, default=None)
    ap.add_argument("--stem_branch_ch", type=int, default=None)
    ap.add_argument("--node_dim", type=int, default=None)
    ap.add_argument("--gat_hidden", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--num_gnn_layers", type=int, default=None)
    ap.add_argument("--radius_bins", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--threshold_IR", type=float, default=None)
    ap.add_argument("--threshold_OR", type=float, default=None)
    ap.add_argument("--threshold_Ball", type=float, default=None)
    args = ap.parse_args()
    # Backward-compatible safeguard in case an older argparse block is reused.
    if not hasattr(args, "cf_predict_mode"):
        args.cf_predict_mode = DEFAULT_CF_PREDICT_MODE

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = safe_torch_load(Path(args.ckpt), map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    keep_bins = int(get_arg(ckpt_args, args, "keep_bins", 2048))
    max_order = float(get_arg(ckpt_args, args, "max_order", 50.0))
    input_spectrum_kind = get_arg(ckpt_args, args, "input_spectrum_kind", "display")
    mask_mode = get_arg(ckpt_args, args, "mask_mode", "peak_locked")
    theory_info = build_theory_order_info_gxu(max_order, max_harmonics=args.theory_max_harmonics,
                                             ir_order=args.ir_order, or_order=args.or_order,
                                             ball_order=args.ball_order) if int(args.show_theory_orders) else {}

    reader = V11OrderCacheReaderGNN(
        cache_root=args.cache_root,
        keep_bins=keep_bins,
        max_order=max_order,
        input_spectrum_kind=input_spectrum_kind,
        mask_mode=mask_mode,
        ir_or_half_width=float(get_arg(ckpt_args, args, "ir_or_half_width", 0.06)),
        ball_half_width=float(get_arg(ckpt_args, args, "ball_half_width", 0.04)),
        search_half_width=float(get_arg(ckpt_args, args, "search_half_width", 0.12)),
        min_rel_peak=float(get_arg(ckpt_args, args, "min_rel_peak", 0.12)),
        max_harmonics_ir_or=int(get_arg(ckpt_args, args, "max_harmonics_ir_or", 0)),
        max_harmonics_ball=int(get_arg(ckpt_args, args, "max_harmonics_ball", 0)),
    )
    indices = choose_indices(reader, args)
    if len(indices) == 0:
        raise RuntimeError(f"No samples selected by eval_mode={args.eval_mode}")
    ds = RealOrderDataset(reader, indices)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GNNCompoundNet(
        max_order=max_order,
        keep_bins=keep_bins,
        stem_branch_ch=int(get_arg(ckpt_args, args, "stem_branch_ch", 16)),
        node_dim=int(get_arg(ckpt_args, args, "node_dim", 64)),
        gat_hidden=int(get_arg(ckpt_args, args, "gat_hidden", 32)),
        num_heads=int(get_arg(ckpt_args, args, "num_heads", 4)),
        num_gnn_layers=int(get_arg(ckpt_args, args, "num_gnn_layers", 2)),
        radius_bins=int(get_arg(ckpt_args, args, "radius_bins", 8)),
        dropout=float(get_arg(ckpt_args, args, "dropout", 0.10)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    if getattr(model, "graph_node_order", "") != "ascending_theoretical_order":
        raise RuntimeError("Loaded GXU model is not using the ordered pre-GAT graph. "
                           "Please use models_gnn_ordered_v19.py / dataset_gnn_ordered_v20.py and retrain the checkpoint.")
    if hasattr(model, "nodes"):
        with open(out_dir / "ordered_graph_nodes_inference.json", "w", encoding="utf-8") as f:
            json.dump(getattr(model, "nodes"), f, ensure_ascii=False, indent=2)
    thr = (
        float(get_arg(ckpt_args, args, "threshold_IR", 0.9)),
        float(get_arg(ckpt_args, args, "threshold_OR", 0.9)),
        float(get_arg(ckpt_args, args, "threshold_Ball", 0.9)),
    )

    logits_all, probs_all, y_true_all, y_pred_all, embeddings_all = [], [], [], [], []
    names_all, paths_all, true_names_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, emb = model(batch)
            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            emb_np = extract_embedding_np(emb, expected_batch=logits_np.shape[0])
            embeddings_all.append(emb_np if emb_np is not None else logits_np)
            probs_np = sigmoid_np(logits_np)
            pred_np = decode_probs(probs_np, thr=thr, decode_mode=args.decode_mode)
            y_np = batch.y.detach().cpu().numpy().astype(np.float32)
            if y_np.ndim == 1:
                y_np = y_np.reshape(1, -1)
            logits_all.append(logits_np)
            probs_all.append(probs_np)
            y_true_all.append(y_np)
            y_pred_all.append(pred_np)
            names_all.extend(list(batch.name))
            paths_all.extend(list(batch.path))
            true_names_all.extend(list(batch.label_name))

    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = np.concatenate(probs_all, axis=0)
    embeddings_all = np.concatenate(embeddings_all, axis=0) if embeddings_all else np.zeros((len(probs_all), 1), dtype=np.float32)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    pred_names = [labels_to_name_from_multihot(v) for v in y_pred_all]
    true_names = [labels_to_name_from_multihot(v) for v in y_true_all]
    exact = exact_match_from_arrays(y_true_all, y_pred_all)

    rows: List[dict] = []
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["index", "name", "path", "true_label", "pred_label", "exact"] + [f"p_{x}" for x in FAULTS] + [f"y_{x}" for x in FAULTS] + [f"pred_{x}" for x in FAULTS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in range(len(names_all)):
            row = {
                "index": int(indices[k]),
                "name": names_all[k],
                "path": paths_all[k],
                "true_label": true_names[k],
                "pred_label": pred_names[k],
                "exact": int(np.all(y_true_all[k] == y_pred_all[k])),
            }
            for j, fault in enumerate(FAULTS):
                row[f"p_{fault}"] = float(probs_all[k, j])
                row[f"y_{fault}"] = int(y_true_all[k, j])
                row[f"pred_{fault}"] = int(y_pred_all[k, j])
            writer.writerow(row)
            rows.append(row)

    rep = classification_report(true_names, pred_names, labels=sorted(list(set(true_names) | set(pred_names))), digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(rep, encoding="utf-8")
    plot_confusion(true_names, pred_names, out_dir / "confusion_matrix.png")
    plot_probability_examples(rows, out_dir / "prediction_probability_bars.png", max_examples=args.plot_max_examples)
    plot_probability_examples_with_spectrum(reader, rows, out_dir / "prediction_probability_examples.png", max_examples=args.plot_max_examples, theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault)
    plot_probability_examples_by_label(reader, rows, out_dir / "prediction_probability_by_label", max_per_label=args.per_label_plot_max, theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault)
    plot_counterfactual_interventions_by_label(reader, model, rows, out_dir / "counterfactual_intervention_by_label",
                                               device=device, max_per_label=args.cf_plot_max_per_label, alpha=args.cf_alpha,
                                               theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault,
                                               cf_predict_mode=args.cf_predict_mode)
    plot_tsne_embeddings(embeddings_all, true_names, out_dir / "tsne_embeddings.png", max_points=args.tsne_max_points, seed=args.seed)
    metrics = {
        "eval_mode": args.eval_mode,
        "decode_mode": args.decode_mode,
        "num_samples": int(len(indices)),
        "exact_match_accuracy": float(exact),
        "thresholds": dict(zip(FAULTS, map(float, thr))),
        "cache_root": str(args.cache_root),
        "ckpt": str(args.ckpt),
        "cf_predict_mode": str(args.cf_predict_mode),
    }

    seen_mask = np.asarray([len(reader.samples[i].get("faults", [])) <= 1 for i in indices], dtype=bool)
    unseen_mask = np.asarray([len(reader.samples[i].get("faults", [])) >= 2 for i in indices], dtype=bool)
    if seen_mask.any():
        metrics["seen_exact"] = float(exact_match_from_arrays(y_true_all[seen_mask], y_pred_all[seen_mask]))
    if unseen_mask.any():
        metrics["unseen_exact"] = float(exact_match_from_arrays(y_true_all[unseen_mask], y_pred_all[unseen_mask]))
    if "seen_exact" in metrics and "unseen_exact" in metrics:
        s_val, u_val = metrics["seen_exact"], metrics["unseen_exact"]
        metrics["harmonic_mean"] = float(0.0 if s_val + u_val <= 1e-12 else 2 * s_val * u_val / (s_val + u_val))

    seeds = parse_seed_list(args.seed_list, args.seed, args.num_seed_runs)
    repeated_rows = []
    for sd in seeds:
        tmp_args = argparse.Namespace(**vars(args))
        tmp_args.seed = int(sd)
        idx_sd = choose_indices(reader, tmp_args)
        ds_sd = RealOrderDataset(reader, idx_sd)
        loader_sd = DataLoader(ds_sd, batch_size=args.batch_size, shuffle=False, num_workers=0)
        yy, pp = [], []
        with torch.no_grad():
            for batch in loader_sd:
                batch = batch.to(device)
                logits, _ = model(batch)
                probs_np = sigmoid_np(logits.detach().cpu().numpy().astype(np.float32))
                pred_np = decode_probs(probs_np, thr=thr, decode_mode=args.decode_mode)
                y_np = batch.y.detach().cpu().numpy().astype(np.float32)
                if y_np.ndim == 1:
                    y_np = y_np.reshape(1, -1)
                yy.append(y_np); pp.append(pred_np)
        yy = np.concatenate(yy, axis=0); pp = np.concatenate(pp, axis=0)
        ex = exact_match_from_arrays(yy, pp)
        seen_mask_sd = np.asarray([len(reader.samples[i].get("faults", [])) <= 1 for i in idx_sd], dtype=bool)
        unseen_mask_sd = np.asarray([len(reader.samples[i].get("faults", [])) >= 2 for i in idx_sd], dtype=bool)
        row = {"seed": int(sd), "num_samples": int(len(idx_sd)), "exact": float(ex)}
        if seen_mask_sd.any():
            row["seen_exact"] = float(exact_match_from_arrays(yy[seen_mask_sd], pp[seen_mask_sd]))
        if unseen_mask_sd.any():
            row["unseen_exact"] = float(exact_match_from_arrays(yy[unseen_mask_sd], pp[unseen_mask_sd]))
        if "seen_exact" in row and "unseen_exact" in row:
            row["harmonic_mean"] = float(0.0 if row["seen_exact"] + row["unseen_exact"] <= 1e-12 else 2 * row["seen_exact"] * row["unseen_exact"] / (row["seen_exact"] + row["unseen_exact"]))
        repeated_rows.append(row)
    if repeated_rows:
        fields = ["seed", "num_samples", "exact", "seen_exact", "unseen_exact", "harmonic_mean"]
        with open(out_dir / "repeated_seed_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in repeated_rows:
                writer.writerow({k: r.get(k, "") for k in fields})
        metrics["repeated_seed_metrics"] = repeated_rows
    (out_dir / "infer_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Selected samples: {len(indices)} | eval_mode={args.eval_mode} | decode_mode={args.decode_mode}")
    print(f"Counterfactual prediction mode: {args.cf_predict_mode}")
    print(f"Exact-match accuracy: {exact:.4f}")
    if "seen_exact" in metrics:
        print(f"Seen exact-match accuracy: {metrics['seen_exact']:.4f}")
    if "unseen_exact" in metrics:
        print(f"Unseen exact-match accuracy: {metrics['unseen_exact']:.4f}")
    if "harmonic_mean" in metrics:
        print(f"Harmonic mean: {metrics['harmonic_mean']:.4f}")
    print(rep)
    print(f"Saved predictions: {csv_path}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
