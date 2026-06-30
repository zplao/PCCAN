# -*- coding: utf-8 -*-
"""
infer_BGMF.py

Inference / evaluation script for the MFD bearing+gear zero-shot compound GNN backend
with the v12 physics-guided harmonic-atom cache and dual GTB extraction/intervention prior.

Expected project files in the same directory or Python path:
    common_bg.py
    dataset_gnn_bg.py
    models_gnn_bg.py
    backend_utils_gnn_bg.py  (optional; only used for protocol split if available)

Typical usage:
    python infer_BGMF.py \
        --cache_root ./multicomponent_results_BGMF \
        --ckpt ./zs_outputs_BGMF/best_model.pt \
        --out_dir ./infer_outputs_BGMF \
        --eval_mode taskB_all

Evaluate Task-A held-out real compounds only:
    python infer_zs_cf_gnn_MFD_BG_ordered_v24.py --eval_mode taskA_unseen

Evaluate Task-B held-out seen + real unseen compounds:
    python infer_zs_cf_gnn_MFD_BG_ordered_v24.py --eval_mode taskB_all
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
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

from common_bg import FAULTS, PAIR_LABELS_BG, ensure_dir, multihot_to_name, exact_match_from_arrays, seed_everything
from dataset_gnn_bg import BGOrderCacheReader, RealOrderDataset
from models_gnn_bg import GNNCompoundNetBG, build_fault_node_map

try:
    from backend_utils_gnn_bg import build_protocol_splits
except Exception:
    build_protocol_splits = None


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_arg(ckpt_args: Dict, cli_args, name: str, default):
    """Read an argument from CLI first, then checkpoint args, then default.

    The current MFD-BG training script stores ``meta`` in the checkpoint and may
    not store all CLI arguments.  Architecture/cache options therefore fall back
    to checkpoint args when present and otherwise to the defaults used by the
    v12 training script.
    """
    v = getattr(cli_args, name, None)
    if v is not None:
        return v
    return ckpt_args.get(name, default)


def extract_checkpoint_state(ckpt):
    """Return a model state_dict from several common checkpoint formats."""
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_module_prefix(state: Dict) -> Dict:
    if not isinstance(state, dict):
        return state
    if not any(str(k).startswith("module.") for k in state.keys()):
        return state
    return {str(k).replace("module.", "", 1): v for k, v in state.items()}


def load_state_dict_compatible(model, state, strict: bool = True):
    """Load checkpoints saved with or without DataParallel prefixes."""
    try:
        return model.load_state_dict(state, strict=strict)
    except RuntimeError:
        state2 = _strip_module_prefix(state)
        return model.load_state_dict(state2, strict=strict)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
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

    Important: this function intentionally avoids reader.samples[i]['input_spectrum'],
    because that array is already robust-normalized.  Counterfactual removal must be
    performed before robust normalization, otherwise the removed component is on a
    different scale and the spectrum will visually remain almost unchanged.
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
    """Return the component that should be subtracted for do(C_f=0).

    For GTB this prefers masked_clean_components['GTB'], which may exclude pure GMF
    main orders and only keep fault-specific shaft orders and GMF sidebands.
    """
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
    """Rebuild the factual spectrum from cached physical components before normalization."""
    raw = np.zeros(length, dtype=np.float32)
    for f in active_faults:
        raw += _get_component_array(reader, sample_index, f, length, preferred_keys=("display", "clean", "masked_clean"))
    if not np.any(np.isfinite(raw)) or float(np.nanmax(np.abs(raw))) < 1e-12:
        _, x = _get_input_spectrum(reader, sample_index)
        raw = _interp_to_length(x, length)
    return np.maximum(0.0, raw.astype(np.float32))


def _make_counterfactual_raw_spectrum(reader, sample_index: int, active_faults: List[str],
                                      remove_fault: str, length: int, alpha: float) -> np.ndarray:
    """Construct X^{do(C_remove=0)} in the unnormalized component space."""
    raw_fact = _rebuild_raw_factual_spectrum(reader, sample_index, active_faults, length)
    comp = _get_intervention_component(reader, sample_index, remove_fault, length)
    raw_cf = raw_fact - float(alpha) * comp.astype(np.float32)
    return np.maximum(0.0, raw_cf.astype(np.float32))


def predict_prob_mfd(model, x_spec: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    x = robust_norm_1d_local(x_spec)
    xb = torch.from_numpy(x[None, None, :].astype(np.float32)).to(device)
    with torch.no_grad():
        logits, emb = model(xb)
    probs = sigmoid_np(logits.detach().cpu().numpy().astype(np.float32))[0]
    emb_arr = extract_embedding_np(emb, expected_batch=1)
    emb_np = emb_arr[0] if emb_arr is not None else logits.detach().cpu().numpy().astype(np.float32)[0]
    return probs, emb_np


def _draw_cf_bar(ax, probs: np.ndarray, title: str, prob_ref_line: float = 0.5):
    x = np.arange(len(FAULTS))
    ax.set_axisbelow(True)
    ax.bar(x, probs, width=0.45, color=FAULT_BAR_COLORS[:len(FAULTS)], zorder=3)
    # Keep the reference line disabled by default for cleaner publication figures.
    # ax.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0, zorder=4)
    ax.set_ylim(0.0, 1.18)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(_display_fault_names())
    ax.set_title(title)
    ax.set_xlabel("Fault type")
    ax.grid(True, axis="y", alpha=0.20, zorder=0)
    for j, p in enumerate(probs):
        p = float(p)
        y_text = min(1.10, p + 0.02)
        ax.text(j, y_text, f"{p:.4f}", ha="center", va="bottom", fontsize=15)


def _scale_for_plot(stages_raw: List[np.ndarray]) -> List[np.ndarray]:
    # Use the factual maximum as a common scale so that removed peaks are visually comparable.
    fact = np.asarray(stages_raw[0], dtype=np.float32)
    scale = float(np.nanmax(fact)) if fact.size else 1.0
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return [np.asarray(x, dtype=np.float32) / scale for x in stages_raw]


def plot_counterfactual_interventions_by_label(reader, model, rows: List[dict], out_dir: Path,
                                               device: torch.device, max_per_label: int = 1,
                                               alpha: float = 1.0, prob_ref_line: float = 0.5,
                                               theory_info: Dict[str, Dict[str, object]] | None = None,
                                               max_theory_lines_per_fault: int = 12):
    """For each compound fault type, remove one active causal component at a time.

    The previous implementation subtracted an unnormalized cached component from an
    already robust-normalized input spectrum.  That scale mismatch made the
    counterfactual spectrum almost identical to the factual spectrum.  This version
    rebuilds the factual spectrum from cached component amplitudes, performs the
    intervention in the same unnormalized component space, and only then applies the
    training-time robust normalization before model prediction.
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
            p_fact, _ = predict_prob_mfd(model, raw_fact, device)

            stage_names = ["Factual input"]
            stage_raw = [raw_fact]
            stage_probs = [p_fact]
            for f in active_faults:
                raw_cf = _make_counterfactual_raw_spectrum(reader, idx, active_faults, f, len(axis), alpha)
                p_cf, _ = predict_prob_mfd(model, raw_cf, device)
                stage_names.append(rf"Counterfactual intervention: $do(C_{{\mathrm{{{_display_fault_name(f)}}}}}=0)$")
                stage_raw.append(raw_cf)
                stage_probs.append(p_cf)

            stage_plot = _scale_for_plot(stage_raw)
            n = len(stage_names)
            fig, axes = plt.subplots(n, 2, figsize=(12, max(3.2, 2.8 * n)), squeeze=False,
                                     gridspec_kw={"width_ratios": [3.2, 1.35]})
            for i, (stage_name, spec_plot, probs) in enumerate(zip(stage_names, stage_plot, stage_probs)):
                axes[i, 0].plot(axis, spec_plot, color="#1f77b4" if i == 0 else "#d62728", lw=1.25)
                # x_max = float(getattr(reader, "max_order", np.nanmax(axis)))
                x_max = 130.0
                if not np.isfinite(x_max) or x_max <= 0:
                    x_max = float(np.nanmax(axis))
                axes[i, 0].set_xlim(0.0, x_max)
                tick_step = 20.0 if x_max > 80.0 else 5.0
                axes[i, 0].set_xticks(np.arange(0.0, x_max + 0.01, tick_step))
                axes[i, 0].set_xlabel("Order")
                axes[i, 0].set_ylabel("Amplitude")
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


def _multihot_from_faults(faults: Sequence[str]) -> np.ndarray:
    y = np.zeros(len(FAULTS), dtype=np.float32)
    for i, f in enumerate(FAULTS):
        if f in faults:
            y[i] = 1.0
    return y


def build_candidate_multihots(mode: str) -> List[np.ndarray]:
    mode = str(mode).lower()
    if mode == "taska":
        pairs = list(PAIR_LABELS_BG)
        return [_multihot_from_faults(list(p)) for p in pairs]
    if mode == "taskb":
        candidates: List[List[str]] = [[]] + [[f] for f in FAULTS] + [list(p) for p in PAIR_LABELS_BG]
        return [_multihot_from_faults(x) for x in candidates]
    return []


def decode_probs(probs: np.ndarray, thr: Sequence[float], decode_mode: str = "threshold") -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    decode_mode = str(decode_mode).lower()
    if decode_mode == "threshold":
        return (probs >= np.asarray(thr, dtype=np.float32)[None, :]).astype(np.float32)
    candidates = build_candidate_multihots(decode_mode.replace("candidate_", ""))
    if not candidates:
        raise ValueError(f"Unsupported decode_mode={decode_mode}. Use threshold, candidate_taska, or candidate_taskb.")
    C = np.stack(candidates, axis=0).astype(np.float32)
    eps = 1e-6
    logp = np.log(np.clip(probs, eps, 1.0 - eps))
    log1p = np.log(np.clip(1.0 - probs, eps, 1.0 - eps))
    scores = logp @ C.T + log1p @ (1.0 - C).T
    idx = np.argmax(scores, axis=1)
    return C[idx].astype(np.float32)


def maybe_apply_gtb_cf_components(reader: BGOrderCacheReader):
    """Optional: align with training script behavior for counterfactual components.
    It does not change input_spectrum, but keeps reader metadata consistent if downstream
    visualization relies on masked_clean_components['GTB'].
    """
    applied = 0
    for s in reader.samples:
        path = Path(s.get('path', ''))
        if not path.exists():
            continue
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if 'GTB_cf_component_order_amp_clean' not in d:
            continue
        axis_key = 'GTB_cf_component_order_axis' if 'GTB_cf_component_order_axis' in d else 'GTB_component_order_axis'
        axis = np.asarray(d[axis_key], dtype=np.float32).reshape(-1)
        amp = np.asarray(d['GTB_cf_component_order_amp_clean'], dtype=np.float32).reshape(-1)
        cf_interp = np.interp(reader.target_axis, axis, amp, left=0.0, right=0.0).astype(np.float32)
        if 'masked_clean_components' in s:
            s['masked_clean_components']['GTB'] = cf_interp
        if 'component_specs' in s and 'GTB' in s['component_specs']:
            s['component_specs']['GTB']['masked_clean'] = cf_interp
        if 'adaptive_centers' in s:
            if 'GTB_cf_centers' in d:
                s['adaptive_centers']['GTB_cf'] = np.asarray(d['GTB_cf_centers'], dtype=np.float32).reshape(-1).tolist()
            if 'GTB_extract_centers' in d:
                s['adaptive_centers']['GTB_extract'] = np.asarray(d['GTB_extract_centers'], dtype=np.float32).reshape(-1).tolist()
        applied += 1
    return applied


def unique_keep_order(ids):
    seen = set()
    out = []
    for i in ids:
        i = int(i)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def is_normal_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    fs = s.get("faults", [])
    lab = str(s.get("label_name", s.get("label", "")))
    return len(fs) == 0 or lab.lower() == "normal"


def is_single_fault_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    fs = s.get("faults", [])
    return len(fs) == 1 and fs[0] in FAULTS


def is_compound_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    return len(s.get("faults", [])) >= 2


def choose_indices(reader: BGOrderCacheReader, args) -> List[int]:
    """Choose inference indices with the same Task-B split semantics as the v24 training script.

    For Task B, held-out Normal samples are explicitly merged into the seen split,
    while the unseen split contains only real compound samples.
    """
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

    if mode.startswith("task"):
        if build_protocol_splits is not None:
            task = "A" if mode.startswith("taska") else "B"
            splits = build_protocol_splits(
                reader, task=task, proxy_ratio=args.proxy_ratio,
                seen_test_ratio=args.seen_test_ratio, seed=args.seed
            )
            normal_test_idx = list(splits.get("normal_test_idx", []))
            seen_test_idx = unique_keep_order(list(splits.get("seen_test_idx", [])) + normal_test_idx)
            real_test_idx = list(splits.get("unseen_test_idx", splits.get("real_test_idx", [])))

            # Robust fallback for older split helpers: remove Normal from real/unseen
            # and put it back into Task-B seen evaluation.
            if any(is_normal_index(reader, i) for i in real_test_idx):
                normal_from_real = [i for i in real_test_idx if is_normal_index(reader, i)]
                real_test_idx = [i for i in real_test_idx if is_compound_index(reader, i)]
                if task == "B":
                    seen_test_idx = unique_keep_order(seen_test_idx + normal_from_real)

            if mode in {"taska", "taska_all", "taska_unseen"}:
                return list(real_test_idx)
            if mode == "taskb_seen":
                return list(seen_test_idx)
            if mode in {"taskb_unseen", "taskb_real"}:
                return list(real_test_idx)
            if mode in {"taskb", "taskb_all"}:
                return unique_keep_order(list(seen_test_idx) + list(real_test_idx))
        # fallback
        if mode.startswith("taska"):
            return [i for i, s in enumerate(reader.samples) if len(s.get("faults", [])) >= 2]
        return list(range(n))
    raise ValueError(f"Unknown eval_mode={args.eval_mode}")


def plot_confusion(y_true_names: List[str], y_pred_names: List[str], out_path: Path):
    labels = sorted(list(set(y_true_names) | set(y_pred_names)))
    if not labels:
        return
    cm = confusion_matrix(y_true_names, y_pred_names, labels=labels)
    fig, ax = plt.subplots(figsize=(5 + 0.62 * len(labels), 4 + 0.48 * len(labels)))
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


FAULT_DISPLAY_NAMES = {"IR": "IF", "OR": "OF", "Ball": "RF", "GTB": "GTB"}
FAULT_BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _fault_color_map() -> Dict[str, str]:
    return {f: FAULT_BAR_COLORS[i % len(FAULT_BAR_COLORS)] for i, f in enumerate(FAULTS)}


def _display_fault_name(name: str) -> str:
    return FAULT_DISPLAY_NAMES.get(str(name), str(name))


def _display_fault_label(label: str) -> str:
    label = str(label)
    if label.lower() == "normal" or label.strip() == "":
        return "Normal"
    return "+".join(_display_fault_name(part) for part in label.split("+"))


def _harmonic_centers(base_order: float, max_order: float, max_harmonics: int = 0) -> List[float]:
    """Return harmonic centers up to max_order.

    max_harmonics <= 0 means automatic full coverage up to max_order.
    This is important for MFD-BG, whose display range may be 160 orders.
    """
    centers: List[float] = []
    try:
        base = float(base_order)
        xmax = float(max_order)
    except Exception:
        return centers
    if not np.isfinite(base) or base <= 0 or not np.isfinite(xmax) or xmax <= 0:
        return centers

    if int(max_harmonics) <= 0:
        k_max = int(np.floor(xmax / base))
    else:
        k_max = int(max_harmonics)

    for k in range(1, k_max + 1):
        c = k * base
        if c <= xmax + 1e-9:
            centers.append(float(c))
    return centers


def build_theory_order_info_mfd(meta: Dict, max_order: float, max_harmonics: int = 0) -> Dict[str, Dict[str, object]]:
    """Build theoretical order markers for MFD-BG figures.

    Bearing faults are shown as harmonic families m*rho_a.
    Gear fault is shown using the front-end extraction centers, i.e. shaft-order
    family, GMF carriers, and GMF sidebands.  The GTB counterfactual branch can
    still remove only the cf-centers; the plot intentionally shows the physical
    theory centers used to interpret the spectrum.
    """
    color_map = _fault_color_map()
    info: Dict[str, Dict[str, object]] = {}
    bearing_orders = meta.get('bearing_orders', {}) if isinstance(meta, dict) else {}
    for f in ['IR', 'OR', 'Ball']:
        if f in FAULTS and f in bearing_orders:
            rho = float(bearing_orders[f])
            info[f] = {
                'centers': _harmonic_centers(rho, max_order, max_harmonics),
                'text': f'{_display_fault_name(f)}: rho={rho:.3f}, m*rho',
                'color': color_map.get(f, 'k'),
            }
    gear_info = meta.get('gear_info', {}) if isinstance(meta, dict) else {}
    if 'GTB' in FAULTS and isinstance(gear_info, dict):
        centers = gear_info.get('extract_centers', gear_info.get('centers', []))
        centers = [float(c) for c in centers if np.isfinite(float(c)) and 0.0 < float(c) <= float(max_order) + 1e-9]
        rho_g = gear_info.get('rho_g', None)
        rho_gmf = gear_info.get('rho_GMF', None)
        if rho_g is not None and rho_gmf is not None:
            txt = f'{_display_fault_name("GTB")}: rho_g={float(rho_g):.3f}, GMF={float(rho_gmf):.1f}, sidebands'
        else:
            txt = f'{_display_fault_name("GTB")}: shaft orders, GMF and sidebands'
        info['GTB'] = {'centers': centers, 'text': txt, 'color': color_map.get('GTB', 'r')}
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
        if int(max_lines_per_fault) <= 0:
            shown = centers
        else:
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
    ax_prob.set_axisbelow(True)
    ax_prob.bar(x, probs, width=0.45, color=FAULT_BAR_COLORS[:len(FAULTS)], zorder=3)
    ax_prob.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0, zorder=4)
    ax_prob.set_ylim(0.0, 1.08)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(_display_fault_names())
    ax_prob.set_title("Model predicted probabilities")
    ax_prob.grid(True, axis="y", alpha=0.20, zorder=0)
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
        ax.set_axisbelow(True)
        ax.bar(np.arange(len(FAULTS)), probs, width=0.45, color=FAULT_BAR_COLORS[:len(FAULTS)], zorder=3)
        ax.axhline(float(prob_ref_line), linestyle="--", color="0.55", linewidth=1.0)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(np.arange(len(FAULTS)))
        ax.set_xticklabels(_display_fault_names())
        ax.set_ylabel("Probability")
        ax.set_title(f"{r['name']} | true={_display_fault_label(r['true_label'])} | pred={_display_fault_label(r['pred_label'])}", fontsize=FIG_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        for j, p in enumerate(probs):
            ax.text(j, min(1.04, p + 0.035), f"{p:.2f}", ha="center", va="bottom", fontsize=FIG_FONT_SIZE, fontweight="bold")
    fig.tight_layout()
    save_figure_png_svg(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("MFD-BG GNN backend inference")
    ap.add_argument("--cache_root", type=str, default="./multicomponent_results_BGMF")
    ap.add_argument("--ckpt", type=str, default="./zs_outputs_BGMF/best_model.pt", help="Path to best_model.pt")
    ap.add_argument("--out_dir", type=str, default="./infer_outputs_NGMF")
    ap.add_argument("--strict_load", type=int, default=1, choices=[0, 1], help="Use strict checkpoint loading. Set 0 only for diagnostic compatibility checks.")
    ap.add_argument("--eval_mode", type=str, default="taskA_unseen",
                    help="all, seen, singles, normal, unseen/compounds, taskA_unseen, taskB_seen, taskB_unseen, taskB_all")
    ap.add_argument("--decode_mode", type=str, default="threshold", choices=["threshold", "candidate_taska", "candidate_taskb"])
    ap.add_argument("--plot_max_examples", type=int, default=12, help="Max examples in the combined probability figure.")
    ap.add_argument("--per_label_plot_max", type=int, default=1, help="Number of spectrum+probability figures saved for each true label.")
    ap.add_argument("--cf_plot_max_per_label", type=int, default=20, help="Number of counterfactual intervention examples saved for each compound label.")
    ap.add_argument("--cf_alpha", type=float, default=1.0, help="Subtraction strength for counterfactual component removal.")
    ap.add_argument("--show_theory_orders", type=int, default=1, choices=[0, 1], help="Overlay theoretical fault-order markers on spectrum figures.")
    ap.add_argument("--theory_max_harmonics", type=int, default=0, help="Maximum bearing harmonics used for theoretical order markers. 0 means automatic full coverage up to max_order.")
    ap.add_argument("--theory_max_lines_per_fault", type=int, default=0, help="Maximum vertical theory markers displayed for each fault in one subplot. 0 means show all generated markers.")
    ap.add_argument("--tsne_max_points", type=int, default=1200, help="Maximum number of points used in t-SNE visualization.")
    ap.add_argument("--num_seed_runs", type=int, default=1, help="Repeat inference split evaluation with different random seeds; figures are saved only for the main seed.")
    ap.add_argument("--seed_list", type=str, default="", help="Comma-separated seeds. Overrides --num_seed_runs when provided.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proxy_ratio", type=float, default=0.25)
    ap.add_argument("--seen_test_ratio", type=float, default=0.25)
    ap.add_argument("--apply_gtb_cf", type=int, default=1)
    # Optional overrides; omitted values are read from checkpoint/meta/default.
    ap.add_argument("--keep_bins", type=int, default=None)
    ap.add_argument("--max_order", type=float, default=None)
    ap.add_argument("--input_spectrum_kind", type=str, default=None, choices=[None, "clean", "display"])
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
    ap.add_argument("--threshold_GTB", type=float, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not Path(args.cache_root).exists():
        raise FileNotFoundError(f"Cache root not found: {args.cache_root}")
    ckpt = safe_torch_load(Path(args.ckpt), map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and "args" in ckpt else {}
    state = extract_checkpoint_state(ckpt)

    meta_path = Path(args.cache_root) / "bg_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif isinstance(ckpt, dict) and "meta" in ckpt:
        meta = ckpt["meta"]
    else:
        raise FileNotFoundError(f"Cannot find bg_meta.json under {args.cache_root}, and checkpoint has no meta.")
    bearing_orders = meta["bearing_orders"]
    gear_info = meta["gear_info"]

    keep_bins = int(get_arg(ckpt_args, args, "keep_bins", 2048))
    max_order = float(get_arg(ckpt_args, args, "max_order", meta.get("max_order", 160.0)))
    input_spectrum_kind = get_arg(ckpt_args, args, "input_spectrum_kind", "display")
    theory_info = build_theory_order_info_mfd(meta, max_order, max_harmonics=args.theory_max_harmonics) if int(args.show_theory_orders) else {}

    reader = BGOrderCacheReader(cache_root=args.cache_root, keep_bins=keep_bins, max_order=max_order,
                                input_spectrum_kind=input_spectrum_kind)
    if int(args.apply_gtb_cf):
        applied = maybe_apply_gtb_cf_components(reader)
    else:
        applied = 0

    indices = choose_indices(reader, args)
    if len(indices) == 0:
        raise RuntimeError(f"No samples selected by eval_mode={args.eval_mode}")
    ds = RealOrderDataset(reader, indices)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gear_centers = [c for c in gear_info.get("extract_centers", gear_info.get("centers", [])) if float(c) <= max_order]
    node_map = build_fault_node_map(bearing_orders, gear_centers, max_order=max_order)
    model = GNNCompoundNetBG(
        node_map=node_map,
        max_order=max_order,
        keep_bins=keep_bins,
        stem_branch_ch=int(get_arg(ckpt_args, args, "stem_branch_ch", 16)),
        node_dim=int(get_arg(ckpt_args, args, "node_dim", 64)),
        gat_hidden=int(get_arg(ckpt_args, args, "gat_hidden", 32)),
        num_heads=int(get_arg(ckpt_args, args, "num_heads", 4)),
        num_gnn_layers=int(get_arg(ckpt_args, args, "num_gnn_layers", 2)),
        radius_bins=int(get_arg(ckpt_args, args, "radius_bins", 8)),
        dropout=float(get_arg(ckpt_args, args, "dropout", 0.15)),
    ).to(device)
    load_state_dict_compatible(model, state, strict=bool(args.strict_load))
    model.eval()

    if getattr(model, "graph_node_order", "") != "ascending_theoretical_order":
        raise RuntimeError("Loaded MFD-BG model is not using the ordered pre-GAT graph. "
                           "Please use models_gnn_bg_ordered_v19.py and retrain the checkpoint.")
    if hasattr(model, "nodes"):
        with open(out_dir / "ordered_graph_nodes_inference.json", "w", encoding="utf-8") as f:
            json.dump(getattr(model, "nodes"), f, ensure_ascii=False, indent=2)
    thr = (
        float(get_arg(ckpt_args, args, "threshold_IR", 0.80)),
        float(get_arg(ckpt_args, args, "threshold_OR", 0.80)),
        float(get_arg(ckpt_args, args, "threshold_Ball", 0.80)),
        float(get_arg(ckpt_args, args, "threshold_GTB", 0.80)),
    )

    probs_all, y_true_all, y_pred_all, embeddings_all = [], [], [], []
    names_all, paths_all, true_names_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(device)
            logits, emb = model(xb)
            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            emb_np = extract_embedding_np(emb, expected_batch=logits_np.shape[0])
            embeddings_all.append(emb_np if emb_np is not None else logits_np)
            probs_np = sigmoid_np(logits_np)
            pred_np = decode_probs(probs_np, thr=thr, decode_mode=args.decode_mode)
            y_np = batch["y"].detach().cpu().numpy().astype(np.float32)
            if y_np.ndim == 1:
                y_np = y_np.reshape(1, -1)
            probs_all.append(probs_np)
            y_true_all.append(y_np)
            y_pred_all.append(pred_np)
            names_all.extend(list(batch["name"]))
            paths_all.extend(list(batch["path"]))
            true_names_all.extend(list(batch["label_name"]))

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

    labels = sorted(list(set(true_names) | set(pred_names)))
    rep = classification_report(true_names, pred_names, labels=labels, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(rep, encoding="utf-8")
    plot_confusion(true_names, pred_names, out_dir / "confusion_matrix.png")
    plot_probability_examples(rows, out_dir / "prediction_probability_bars.png", max_examples=args.plot_max_examples)
    plot_probability_examples_with_spectrum(reader, rows, out_dir / "prediction_probability_examples.png", max_examples=args.plot_max_examples, theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault)
    plot_probability_examples_by_label(reader, rows, out_dir / "prediction_probability_by_label", max_per_label=args.per_label_plot_max, theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault)
    plot_counterfactual_interventions_by_label(reader, model, rows, out_dir / "counterfactual_intervention_by_label",
                                               device=device, max_per_label=args.cf_plot_max_per_label, alpha=args.cf_alpha,
                                               theory_info=theory_info, max_theory_lines_per_fault=args.theory_max_lines_per_fault)
    plot_tsne_embeddings(embeddings_all, true_names, out_dir / "tsne_embeddings.png", max_points=args.tsne_max_points, seed=args.seed)

    metrics = {
        "eval_mode": args.eval_mode,
        "decode_mode": args.decode_mode,
        "num_samples": int(len(indices)),
        "exact_match_accuracy": float(exact),
        "thresholds": dict(zip(FAULTS, map(float, thr))),
        "cache_root": str(args.cache_root),
        "ckpt": str(args.ckpt),
        "gtb_cf_components_applied": int(applied),
        "gear_info": gear_info,
        "keep_bins": int(keep_bins),
        "max_order": float(max_order),
        "input_spectrum_kind": str(input_spectrum_kind),
    }
    # Task-B style summary when selected set contains seen and unseen labels.
    seen_mask = np.asarray([len(reader.samples[i].get("faults", [])) <= 1 for i in indices], dtype=bool)
    unseen_mask = np.asarray([len(reader.samples[i].get("faults", [])) >= 2 for i in indices], dtype=bool)
    if seen_mask.any():
        metrics["seen_exact"] = float(exact_match_from_arrays(y_true_all[seen_mask], y_pred_all[seen_mask]))
    if unseen_mask.any():
        metrics["unseen_exact"] = float(exact_match_from_arrays(y_true_all[unseen_mask], y_pred_all[unseen_mask]))
    if "seen_exact" in metrics and "unseen_exact" in metrics:
        s, u = metrics["seen_exact"], metrics["unseen_exact"]
        metrics["harmonic_mean"] = float(0.0 if s + u <= 1e-12 else 2 * s * u / (s + u))

    # Repeated split evaluation with different random seeds. This does not retrain the model;
    # it only rebuilds the held-out split for protocol-dependent eval_mode values.
    seeds = parse_seed_list(args.seed_list, args.seed, args.num_seed_runs)
    repeated_rows = []
    for sd in seeds:
        tmp_args = argparse.Namespace(**vars(args))
        tmp_args.seed = int(sd)
        idx_sd = choose_indices(reader, tmp_args)
        if len(idx_sd) == 0:
            continue
        ds_sd = RealOrderDataset(reader, idx_sd)
        loader_sd = DataLoader(ds_sd, batch_size=args.batch_size, shuffle=False, num_workers=0)
        yy, pp = [], []
        with torch.no_grad():
            for batch in loader_sd:
                xb = batch["x"].to(device)
                logits, _ = model(xb)
                probs_np = sigmoid_np(logits.detach().cpu().numpy().astype(np.float32))
                pred_np = decode_probs(probs_np, thr=thr, decode_mode=args.decode_mode)
                y_np = batch["y"].detach().cpu().numpy().astype(np.float32)
                if y_np.ndim == 1:
                    y_np = y_np.reshape(1, -1)
                yy.append(y_np); pp.append(pred_np)
        if not yy:
            continue
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
    print(f"GTB cf components applied: {applied}")
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
