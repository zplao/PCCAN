from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit

from common_bg import FAULTS, PAIR_LABELS_BG, labels_to_multihot, multihot_to_name, robust_norm_1d, seed_everything, ensure_dir, exact_match_from_arrays
from dataset_gnn_bg import BGOrderCacheReader, RealOrderDataset
from models_gnn_bg import GNNCompoundNetBG, build_fault_node_map
from ordered_gat_utils import export_node_mapping_csv, build_node_mapping_rows
from backend_utils_gnn_bg import (
    build_protocol_splits,
    evaluate_dataset,
    collect_outputs,
    save_prediction_records,
    plot_prediction_examples,
    plot_masked_examples,
    plot_graph_nodes,
)


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def maybe_apply_gtb_cf_components(reader: BGOrderCacheReader):
    """If the front-end cached a separate GTB counterfactual component, use it only for
    masked_clean_components (i.e., intervention/proxy synthesis), while keeping the input
    spectrum and graph/node centers based on the extraction representation.
    """
    applied = 0
    for s in reader.samples:
        path = Path(s['path'])
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


def check_reader_model_order_consistency(reader: BGOrderCacheReader, model: GNNCompoundNetBG, out_dir: Path) -> None:
    """Validate that dataset-side ordered graph metadata matches the model graph.

    dataset_gnn_bg_ordered_v20 exposes reader.nodes, reader.order_centers,
    reader.base_edge_index_np and reader.graph_node_order.  The MFD-BG model
    still constructs its graph internally from node_map, so this check makes
    the synchronization explicit and fails early if the two sides diverge.
    """
    if getattr(reader, 'graph_node_order', None) != 'ascending_theoretical_order':
        raise RuntimeError(
            "dataset_gnn_bg_ordered_v20 did not expose ascending_theoretical_order. "
            "Please use dataset_gnn_bg_ordered_v20.py for this training script."
        )
    if getattr(model, 'graph_node_order', None) != 'ascending_theoretical_order':
        raise RuntimeError(
            "models_gnn_bg_ordered_v19 did not expose ascending_theoretical_order. "
            "Please use the ordered MFD-BG model file."
        )

    reader_orders = np.asarray(getattr(reader, 'order_centers', []), dtype=np.float32).reshape(-1)
    model_orders = np.asarray(getattr(model, 'order_centers', []), dtype=np.float32).reshape(-1)
    if reader_orders.size == 0:
        raise RuntimeError(
            "reader.order_centers is empty. dataset_gnn_bg_ordered_v20 could not build graph metadata. "
            "Check that bg_meta.json exists under cache_root."
        )
    if reader_orders.shape != model_orders.shape or not np.allclose(reader_orders, model_orders, atol=1e-5, rtol=1e-5):
        raise RuntimeError(
            "Reader-side ordered graph and model-side ordered graph are inconsistent.\n"
            f"reader orders shape={reader_orders.shape}, model orders shape={model_orders.shape}\n"
            "This usually means dataset_gnn_bg_ordered_v20.py and models_gnn_bg_ordered_v19.py "
            "are not using the same ordered_pre_gat_graph_utils version."
        )

    reader_edges = np.asarray(getattr(reader, 'base_edge_index_np', np.empty((2, 0))), dtype=np.int64)
    model_edges = np.asarray(getattr(model, 'edge_index_np', np.empty((2, 0))), dtype=np.int64)
    if reader_edges.shape != model_edges.shape or set(map(tuple, reader_edges.T.tolist())) != set(map(tuple, model_edges.T.tolist())):
        raise RuntimeError(
            "Reader-side edge_index and model-side edge_index are inconsistent. "
            "Please keep dataset_gnn_bg_ordered_v20.py, models_gnn_bg_ordered_v19.py and "
            "ordered_pre_gat_graph_utils_v3.py synchronized."
        )

    with open(out_dir / 'ordered_graph_consistency_check.json', 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_module': 'dataset_gnn_bg_ordered_v20',
            'model_module': 'models_gnn_bg_ordered_v19',
            'graph_node_order': 'ascending_theoretical_order',
            'num_nodes': int(model_orders.size),
            'num_edge_records': int(model_edges.shape[1]) if model_edges.ndim == 2 else 0,
            'max_abs_order_difference': float(np.max(np.abs(reader_orders - model_orders))) if model_orders.size else 0.0,
        }, f, ensure_ascii=False, indent=2)
    print('Dataset/model ordered graph consistency check passed. Saved ordered_graph_consistency_check.json.')


class BGCounterfactualTrainDataset(Dataset):
    def __init__(self, reader: BGOrderCacheReader, train_single_indices: Sequence[int],
                 virtual_per_epoch: int = 4096, noise_std: float = 0.003, shift_bins: int = 1,
                 amp_jitter=(0.97, 1.03), background_floor_scale: float = 0.04, seed: int = 42):
        self.reader = reader
        self.real_ids = list(train_single_indices)
        self.virtual_per_epoch = int(virtual_per_epoch)
        self.noise_std = float(noise_std)
        self.shift_bins = int(shift_bins)
        self.amp_jitter = amp_jitter
        self.background_floor_scale = float(background_floor_scale)
        self.rng = np.random.default_rng(seed)
        self.pool = {f: [i for i in self.real_ids if reader.samples[i]['faults'] == [f]] for f in FAULTS}
        self.base_faults = [f for f in FAULTS if len(self.pool[f]) > 0]

    def __len__(self):
        return self.virtual_per_epoch

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32).copy()
        sc = float(self.rng.uniform(*self.amp_jitter))
        sh = int(self.rng.integers(-self.shift_bins, self.shift_bins + 1))
        y = np.roll(y * sc, sh)
        if self.noise_std > 0:
            y += self.rng.normal(0.0, self.noise_std * (np.std(y) + 1e-8), size=y.shape).astype(np.float32)
        return np.maximum(0.0, y).astype(np.float32)

    def _make_item(self, x: np.ndarray, faults: List[str], name: str):
        return {
            'x': torch.from_numpy(robust_norm_1d(x)[None, :].astype(np.float32)),
            'y': torch.from_numpy(labels_to_multihot(faults).astype(np.float32)),
            'name': name,
            'label_name': '+'.join(faults) if faults else 'Normal',
        }

    def __getitem__(self, idx):
        base_fault = str(self.rng.choice(self.base_faults))
        base_idx = int(self.rng.choice(self.pool[base_fault]))
        base_s = self.reader.samples[base_idx]
        x_base = base_s['input_spectrum'].astype(np.float32).copy()
        base_item = self._make_item(x_base, [base_fault], f'BASE_{base_s["name"]}')

        comp = base_s['masked_clean_components'].get(base_fault, np.zeros_like(x_base, dtype=np.float32))
        x_sub = np.maximum(0.0, x_base - 0.85 * comp)
        x_sub += self.background_floor_scale * x_base
        sub_item = self._make_item(x_sub, [], f'SUB_{base_s["name"]}')

        if base_fault in ['IR', 'OR', 'GTB']:
            candidate_pairs = [p for p in PAIR_LABELS_BG if base_fault in p]
            pair = tuple(candidate_pairs[self.rng.integers(0, len(candidate_pairs))])
            other_fault = pair[1] if pair[0] == base_fault else pair[0]
            other_idx = int(self.rng.choice(self.pool[other_fault]))
            other_s = self.reader.samples[other_idx]
            x_add = x_base.astype(np.float32).copy()
            x_add += self._jitter(other_s['masked_clean_components'][other_fault].astype(np.float32))
            x_add += self.background_floor_scale * 0.5 * (base_s['input_spectrum'].astype(np.float32) + other_s['input_spectrum'].astype(np.float32))
            add_faults = sorted(list(pair), key=lambda x: ['IR', 'OR', 'Ball', 'GTB'].index(x))
        else:
            x_add = x_base.astype(np.float32).copy() + self.background_floor_scale * x_base
            add_faults = [base_fault]
        add_item = self._make_item(x_add, add_faults, 'ADD_' + '+'.join(add_faults))
        return {'base': base_item, 'sub': sub_item, 'add': add_item}


class BGProxyCompoundDataset(Dataset):
    def __init__(self, reader: BGOrderCacheReader, proxy_single_indices: Sequence[int],
                 samples_per_pair: int = 160, noise_std: float = 0.003, shift_bins: int = 1,
                 amp_jitter=(0.97, 1.03), background_floor_scale: float = 0.04, seed: int = 123):
        self.samples = []
        self.rng = np.random.default_rng(seed)
        pool = {f: [i for i in proxy_single_indices if reader.samples[i]['faults'] == [f]] for f in FAULTS}
        for pair in PAIR_LABELS_BG:
            if len(pool[pair[0]]) == 0 or len(pool[pair[1]]) == 0:
                continue
            for _ in range(int(samples_per_pair)):
                floors = []
                x = np.zeros_like(reader.target_axis, dtype=np.float32)
                for f in pair:
                    idx = int(self.rng.choice(pool[f]))
                    s = reader.samples[idx]
                    comp = s['masked_clean_components'][f].astype(np.float32)
                    sc = float(self.rng.uniform(*amp_jitter))
                    sh = int(self.rng.integers(-shift_bins, shift_bins + 1))
                    x += np.roll(comp * sc, sh)
                    floors.append(s['input_spectrum'])
                x += background_floor_scale * np.mean(np.stack(floors, axis=0), axis=0).astype(np.float32)
                if noise_std > 0:
                    x += self.rng.normal(0.0, noise_std * (np.std(x) + 1e-8), size=x.shape).astype(np.float32)
                self.samples.append({
                    'x': torch.from_numpy(robust_norm_1d(x)[None, :].astype(np.float32)),
                    'y': torch.from_numpy(labels_to_multihot(list(pair)).astype(np.float32)),
                    'name': 'SYN_' + '+'.join(pair),
                    'label_name': '+'.join(pair),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cf_collate(batch):
    def stack_part(key: str):
        return {
            'x': torch.stack([b[key]['x'] for b in batch], dim=0),
            'y': torch.stack([b[key]['y'] for b in batch], dim=0),
            'name': [b[key]['name'] for b in batch],
            'label_name': [b[key]['label_name'] for b in batch],
        }
    return {'base': stack_part('base'), 'sub': stack_part('sub'), 'add': stack_part('add')}


def make_loader(ds, batch_size, shuffle=False, collate_fn=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)




def is_normal_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    fs = s.get('faults', [])
    lab = str(s.get('label_name', s.get('label', '')))
    return len(fs) == 0 or lab.lower() == 'normal'


def is_single_fault_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    fs = s.get('faults', [])
    return len(fs) == 1 and fs[0] in FAULTS


def is_compound_index(reader, idx: int) -> bool:
    s = reader.samples[int(idx)]
    return len(s.get('faults', [])) >= 2


def unique_keep_order(ids):
    seen = set()
    out = []
    for i in ids:
        i = int(i)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out

def pair_name(pair: Tuple[str, str]) -> str:
    order = {f: i for i, f in enumerate(FAULTS)}
    pair = tuple(sorted(pair, key=lambda x: order[x]))
    return '+'.join(pair)


def save_taskA_compound_report(y_true: np.ndarray, y_pred: np.ndarray, target_pairs: Sequence[Tuple[str, str]], out_path: Path) -> str:
    target_names = [pair_name(tuple(p)) for p in target_pairs]
    true_names = [multihot_to_name(v) for v in y_true]
    pred_names = [multihot_to_name(v) for v in y_pred]
    rep = classification_report(true_names, pred_names, labels=target_names, target_names=target_names, digits=4, zero_division=0)
    out_path.write_text(rep, encoding='utf-8')
    return rep


def plot_taskA_compound_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, target_pairs: Sequence[Tuple[str, str]], out_path: Path):
    ensure_dir(out_path.parent)
    labels = [pair_name(tuple(p)) for p in target_pairs]
    true_names = [multihot_to_name(v) for v in y_true]
    pred_names = [multihot_to_name(v) for v in y_pred]
    if not labels:
        return
    cm = confusion_matrix(true_names, pred_names, labels=labels)
    fig, ax = plt.subplots(figsize=(5 + 0.8 * len(labels), 4 + 0.6 * len(labels)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix (held-out real compounds only)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def save_general_report(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> str:
    true_names = [multihot_to_name(v) for v in y_true]
    pred_names = [multihot_to_name(v) for v in y_pred]
    labels = sorted(list(set(true_names) | set(pred_names)))
    rep = classification_report(true_names, pred_names, labels=labels, target_names=labels, digits=4, zero_division=0)
    out_path.write_text(rep, encoding='utf-8')
    return rep


def plot_general_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    ensure_dir(out_path.parent)
    true_names = [multihot_to_name(v) for v in y_true]
    pred_names = [multihot_to_name(v) for v in y_pred]
    labels = sorted(list(set(true_names) | set(pred_names)))
    if not labels:
        return
    cm = confusion_matrix(true_names, pred_names, labels=labels)
    fig, ax = plt.subplots(figsize=(5 + 0.6 * len(labels), 4 + 0.5 * len(labels)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='MFD bearing+gear zero-shot compound GNN backend')
    ap.add_argument('--cache_root', type=str, default='./multicomponent_results_BGMF')
    ap.add_argument('--out_dir', type=str, default='./zs_outputs_BGMF')
    ap.add_argument('--task', type=str, default='A', choices=['A', 'B'])
    ap.add_argument('--keep_bins', type=int, default=2048)
    ap.add_argument('--max_order', type=float, default=160.0)
    ap.add_argument('--input_spectrum_kind', type=str, default='display', choices=['clean', 'display'])
    ap.add_argument('--proxy_ratio', type=float, default=0.25)
    ap.add_argument('--seen_test_ratio', type=float, default=0.25)
    ap.add_argument('--virtual_per_epoch', type=int, default=1024)
    ap.add_argument('--proxy_val_per_pair', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--stem_branch_ch', type=int, default=16)
    ap.add_argument('--node_dim', type=int, default=64)
    ap.add_argument('--gat_hidden', type=int, default=32)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--num_gnn_layers', type=int, default=2)
    ap.add_argument('--radius_bins', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--pos_weight_or', type=float, default=1.0)
    ap.add_argument('--pos_weight_ball', type=float, default=1.0)
    ap.add_argument('--pos_weight_gtb', type=float, default=1.0)
    ap.add_argument('--threshold_IR', type=float, default=0.8)
    ap.add_argument('--threshold_OR', type=float, default=0.8)
    ap.add_argument('--threshold_Ball', type=float, default=0.8)
    ap.add_argument('--threshold_GTB', type=float, default=0.8)
    ap.add_argument('--lambda_cfn', type=float, default=0.20)
    ap.add_argument('--lambda_cfp', type=float, default=0.50)
    ap.add_argument('--noise_std', type=float, default=0.003)
    ap.add_argument('--shift_bins', type=int, default=1)
    ap.add_argument('--background_floor_scale', type=float, default=0.04)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)  # 42, 2021, 2023, 2024, 2025, 2026
    args = ap.parse_args()
    args.graph_node_order = "ascending_theoretical_order"

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    meta_path = Path(args.cache_root) / 'bg_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'Cannot find bg_meta.json under {args.cache_root}. Please run the BG front-end first.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    bearing_orders = meta['bearing_orders']
    gear_info = meta['gear_info']

    reader = BGOrderCacheReader(cache_root=args.cache_root, keep_bins=args.keep_bins, max_order=args.max_order,
                                input_spectrum_kind=args.input_spectrum_kind)
    applied_cf = maybe_apply_gtb_cf_components(reader)
    if applied_cf > 0:
        print(f'Applied GTB counterfactual-only masked components to {applied_cf} cached windows (GTB extract representation kept for inputs / graph nodes).')
    splits = build_protocol_splits(reader, task=args.task, proxy_ratio=args.proxy_ratio, seen_test_ratio=args.seen_test_ratio, seed=args.seed)
    train_idx = list(splits['train_idx'])
    proxy_idx = list(splits['proxy_idx'])

    # Corrected protocol split:
    #   seen_test_idx  = held-out single-fault samples + Normal samples
    #   unseen_test_idx / real_test_idx = held-out real compound samples only
    normal_test_idx = list(splits.get('normal_test_idx', []))
    seen_test_idx = unique_keep_order(list(splits.get('seen_test_idx', [])) + normal_test_idx)
    real_test_idx = list(splits.get('unseen_test_idx', splits.get('real_test_idx', [])))
    # Robust fallback in case an older backend_utils is accidentally imported.
    if any(is_normal_index(reader, i) for i in real_test_idx):
        normal_from_real = [i for i in real_test_idx if is_normal_index(reader, i)]
        real_test_idx = [i for i in real_test_idx if is_compound_index(reader, i)]
        if args.task == 'B':
            seen_test_idx = unique_keep_order(seen_test_idx + normal_from_real)
            normal_test_idx = unique_keep_order(normal_test_idx + normal_from_real)
    seen_single_test_idx = [i for i in seen_test_idx if is_single_fault_index(reader, i)]
    seen_normal_test_idx = [i for i in seen_test_idx if is_normal_index(reader, i)]

    if args.task == 'A':
        eval_idx = real_test_idx
        print(f'Train singles: {len(train_idx)} | Proxy-source singles: {len(proxy_idx)} | Held-out real compounds: {len(real_test_idx)} | Task A')
    else:
        eval_idx = unique_keep_order(list(seen_test_idx) + list(real_test_idx))
        print(
            f'Train singles: {len(train_idx)} | Proxy-source singles: {len(proxy_idx)} | '
            f'Held-out seen singles: {len(seen_single_test_idx)} | Held-out seen Normal: {len(seen_normal_test_idx)} | '
            f'Held-out seen total: {len(seen_test_idx)} | Held-out unseen real compounds: {len(real_test_idx)} | Task B'
        )
    print('Protocol: no real unseen compound sample is used during training, early stopping, or model selection.')
    print('Target unseen pair set:', PAIR_LABELS_BG)
    print(f"Gear order prior: rho_g={gear_info['rho_g']:.6f}, rho_GMF={gear_info['rho_GMF']:.6f}")
    if 'extract_centers' in gear_info and 'cf_centers' in gear_info:
        print(f"GTB graph centers (extract): {gear_info['extract_centers']}")
        print(f"GTB intervention centers (cf): {gear_info['cf_centers']}")

    cf_train_ds = BGCounterfactualTrainDataset(reader, train_idx, virtual_per_epoch=args.virtual_per_epoch,
                                               noise_std=args.noise_std, shift_bins=args.shift_bins,
                                               background_floor_scale=args.background_floor_scale, seed=args.seed)
    factual_train_ds = RealOrderDataset(reader, train_idx)
    proxy_compound_ds = BGProxyCompoundDataset(reader, proxy_idx, samples_per_pair=args.proxy_val_per_pair,
                                               noise_std=args.noise_std, shift_bins=args.shift_bins,
                                               background_floor_scale=args.background_floor_scale, seed=args.seed + 1)
    proxy_seen_ds = RealOrderDataset(reader, proxy_idx)
    proxy_seen_loader = make_loader(proxy_seen_ds, args.batch_size, shuffle=False)

    eval_ds = RealOrderDataset(reader, eval_idx)
    real_compound_ds = RealOrderDataset(reader, real_test_idx)
    seen_eval_ds = RealOrderDataset(reader, seen_test_idx)
    seen_single_eval_ds = RealOrderDataset(reader, seen_single_test_idx)
    normal_eval_ds = RealOrderDataset(reader, seen_normal_test_idx)

    cf_train_loader = make_loader(cf_train_ds, args.batch_size, shuffle=True, collate_fn=cf_collate)
    factual_train_loader = make_loader(factual_train_ds, args.batch_size, shuffle=False)
    proxy_loader = make_loader(proxy_compound_ds, args.batch_size, shuffle=False)
    eval_loader = make_loader(eval_ds, args.batch_size, shuffle=False)
    real_compound_loader = make_loader(real_compound_ds, args.batch_size, shuffle=False)
    seen_eval_loader = make_loader(seen_eval_ds, args.batch_size, shuffle=False)
    seen_single_eval_loader = make_loader(seen_single_eval_ds, args.batch_size, shuffle=False)
    normal_eval_loader = make_loader(normal_eval_ds, args.batch_size, shuffle=False)

    gear_centers = [c for c in gear_info.get('extract_centers', gear_info.get('centers', [])) if c <= args.max_order]
    node_map = build_fault_node_map(bearing_orders, gear_centers, max_order=args.max_order)
    model = GNNCompoundNetBG(node_map=node_map, max_order=args.max_order, keep_bins=args.keep_bins,
                             stem_branch_ch=args.stem_branch_ch, node_dim=args.node_dim,
                             gat_hidden=args.gat_hidden, num_heads=args.num_heads,
                             num_gnn_layers=args.num_gnn_layers, radius_bins=args.radius_bins,
                             dropout=args.dropout).to(device)
    check_reader_model_order_consistency(reader, model, out_dir)
    export_node_mapping_csv(model.nodes, out_dir / 'ordered_graph_nodes.csv')
    with open(out_dir / 'ordered_graph_nodes.json', 'w', encoding='utf-8') as f:
        json.dump({
            'graph_node_order': 'ascending_theoretical_order',
            'nodes': build_node_mapping_rows(model.nodes),
        }, f, ensure_ascii=False, indent=2)
    print('Pre-GAT graph node order: ascending theoretical order. Node mapping saved to ordered_graph_nodes.csv/json.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([1.0, args.pos_weight_or, args.pos_weight_ball, args.pos_weight_gtb], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    thr = (args.threshold_IR, args.threshold_OR, args.threshold_Ball, args.threshold_GTB)

    plot_graph_nodes(model, out_dir / 'verify_graph_nodes.png')
    plot_masked_examples(reader, eval_idx[:min(len(eval_idx), 12)], out_dir / 'verify_masked_order_spectra.png', max_per_label=1)

    best_metric = -1.0
    best_path = out_dir / 'best_model.pt'
    wait = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_meter, base_meter, cfn_meter, cfp_meter = [], [], [], []
        ys_train, ps_train = [], []
        for batch in cf_train_loader:
            optimizer.zero_grad()
            xb = batch['base']['x'].to(device)
            yb = batch['base']['y'].to(device)
            xs = batch['sub']['x'].to(device)
            ys = batch['sub']['y'].to(device)
            xa = batch['add']['x'].to(device)
            ya = batch['add']['y'].to(device)
            lb, _ = model(xb)
            ls, _ = model(xs)
            la, _ = model(xa)
            loss_base = bce(lb, yb)
            loss_cfn = bce(ls, ys)
            loss_cfp = bce(la, ya)
            loss = loss_base + args.lambda_cfn * loss_cfn + args.lambda_cfp * loss_cfp
            loss.backward()
            optimizer.step()
            loss_meter.append(float(loss.item()))
            base_meter.append(float(loss_base.item()))
            cfn_meter.append(float(loss_cfn.item()))
            cfp_meter.append(float(loss_cfp.item()))
            logits_np = lb.detach().cpu().numpy().astype(np.float32)
            y_np = yb.detach().cpu().numpy().astype(np.float32)
            probs = expit(logits_np)
            # probs = 1.0 / (1.0 + np.exp(-logits_np))
            pred = (probs >= np.asarray(thr, dtype=np.float32)[None, :]).astype(np.float32)
            ys_train.append(y_np)
            ps_train.append(pred)
        train_exact = exact_match_from_arrays(np.concatenate(ys_train, axis=0), np.concatenate(ps_train, axis=0)) if ys_train else 0.0
        proxy_exact, _, _ = evaluate_dataset(model, proxy_loader, device, thr=thr)
        proxy_seen_exact, _, _ = evaluate_dataset(model, proxy_seen_loader, device, thr=thr)
        log = {'epoch': ep, 'train_loss': float(np.mean(loss_meter)), 'base_loss': float(np.mean(base_meter)),
               'cfn_loss': float(np.mean(cfn_meter)), 'cfp_loss': float(np.mean(cfp_meter)),
               'train_exact': float(train_exact), 'proxy_exact': float(proxy_exact)}
        history.append(log)
        print(f"Epoch {ep:03d}/{args.epochs} | train {log['train_loss']:.4f} (base {log['base_loss']:.4f}, cfn {log['cfn_loss']:.4f}, cfp {log['cfp_loss']:.4f}) | train exact {train_exact:.4f} | proxy-val exact {proxy_exact:.4f}")
        if args.task == 'A':
            metric = proxy_exact
        else:
            # proxy_H = 0.0 if (train_exact + proxy_exact) <= 1e-12 else 2 * train_exact * proxy_exact / (
            #             train_exact + proxy_exact)
            proxy_H = 0.0 if (proxy_seen_exact + proxy_exact) <= 1e-12 else 2 * proxy_seen_exact * proxy_exact / (
                        proxy_seen_exact + proxy_exact)
            metric = proxy_H
        # metric = proxy_exact    # proxy_exact
        if metric > best_metric + 1e-9:
            best_metric = metric
            wait = 0
            torch.save({
                'model': model.state_dict(),
                'history': history,
                'meta': meta,
                'graph_node_order': 'ascending_theoretical_order',
                'dataset_module': 'dataset_gnn_bg_ordered_v20',
                'model_module': 'models_gnn_bg_ordered_v19',
                'ordered_graph_nodes': build_node_mapping_rows(model.nodes),
            }, best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'Early stopping at epoch {ep} due to no proxy-val improvement in {args.patience} epochs.')
                break

    if best_path.exists():
        ckpt = safe_torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    if args.task == 'A':
        final_exact, y_true, y_pred = evaluate_dataset(model, real_compound_loader, device, thr=thr)
        print(f'\nFinal held-out real-compound exact-match accuracy: {final_exact:.4f}')
        rep = save_taskA_compound_report(y_true, y_pred, PAIR_LABELS_BG, out_dir / 'classification_report.txt')
        print(rep)
        heldout_names = {pair_name(tuple(p)) for p in PAIR_LABELS_BG}
        invalid = sum(1 for row in y_pred if multihot_to_name(row) not in heldout_names)
        print(f'Invalid predictions outside held-out label set: {invalid}/{len(y_pred)} ({(invalid / max(1, len(y_pred))):.4f})')
        recs = collect_outputs(model, real_compound_loader, device, thr=thr)
        plot_cm_fn = lambda: plot_taskA_compound_confusion_matrix(y_true, y_pred, PAIR_LABELS_BG, out_dir / 'confusion_matrix.png')
    else:
        all_exact, y_true, y_pred = evaluate_dataset(model, eval_loader, device, thr=thr)
        seen_exact, y_seen_t, y_seen_p = evaluate_dataset(model, seen_eval_loader, device, thr=thr)
        seen_single_exact, y_seen_single_t, y_seen_single_p = evaluate_dataset(model, seen_single_eval_loader, device, thr=thr)
        normal_exact, y_norm_t, y_norm_p = evaluate_dataset(model, normal_eval_loader, device, thr=thr)
        unseen_exact, y_un_t, y_un_p = evaluate_dataset(model, real_compound_loader, device, thr=thr)
        H = 0.0 if (seen_exact + unseen_exact) <= 1e-12 else 2 * seen_exact * unseen_exact / (seen_exact + unseen_exact)
        print(f'\nTask B exact-match accuracy: {all_exact:.4f}')
        print(f'Seen exact-match accuracy (Normal + single faults): {seen_exact:.4f}')
        print(f'  Seen single-fault exact-match accuracy: {seen_single_exact:.4f}')
        print(f'  Seen Normal exact-match accuracy: {normal_exact:.4f}')
        print(f'Unseen real-compound exact-match accuracy: {unseen_exact:.4f}')
        print(f'Harmonic mean: {H:.4f}')
        rep = save_general_report(y_true, y_pred, out_dir / 'classification_report.txt')
        print(rep)
        save_general_report(y_un_t, y_un_p, out_dir / 'classification_report_unseen_real_compounds.txt')
        save_general_report(y_seen_t, y_seen_p, out_dir / 'classification_report_seen_normal_plus_singles.txt')
        if len(y_norm_t) > 0:
            save_general_report(y_norm_t, y_norm_p, out_dir / 'classification_report_seen_normal.txt')
        recs = collect_outputs(model, eval_loader, device, thr=thr)
        (out_dir / 'gzsl_harmonic_mean.txt').write_text(
            f'Seen_Normal_plus_singles={seen_exact:.6f}\n'
            f'Seen_single_faults={seen_single_exact:.6f}\n'
            f'Seen_Normal={normal_exact:.6f}\n'
            f'Unseen_real_compounds={unseen_exact:.6f}\n'
            f'H={H:.6f}\n', encoding='utf-8')
        (out_dir / 'taskB_split_counts.json').write_text(json.dumps({
            'seen_single_test': len(seen_single_test_idx),
            'seen_normal_test': len(seen_normal_test_idx),
            'seen_total': len(seen_test_idx),
            'unseen_real_compounds': len(real_test_idx),
            'taskB_eval_total': len(eval_idx),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        plot_cm_fn = lambda: plot_general_confusion_matrix(y_true, y_pred, out_dir / 'confusion_matrix.png')

    save_prediction_records(recs, out_dir)
    plot_prediction_examples(reader, recs, out_dir / 'prediction_probability_examples.png')
    plot_cm_fn()
    (out_dir / 'train_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'backend_meta.json').write_text(json.dumps({
        'thresholds': {'IR': args.threshold_IR, 'OR': args.threshold_OR, 'Ball': args.threshold_Ball, 'GTB': args.threshold_GTB},
        'pair_labels_bg': PAIR_LABELS_BG,
        'meta': meta,
        'graph_node_order': 'ascending_theoretical_order',
        'dataset_module': 'dataset_gnn_bg',
        'model_module': 'models_gnn_bg',
        'normal_as_seen': True,
        'ordered_graph_nodes': build_node_mapping_rows(model.nodes),
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
