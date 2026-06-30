# -*- coding: utf-8 -*-
"""
dataset_gnn_ordered_v20.py

GXU bearing dataset module for the ordered pre-GAT graph backend.
This file corresponds to the original dataset_gnn.py branch.

It keeps the PyG Data-based API:
    - V11OrderCacheReaderGNN
    - RealOrderDataset
    - CounterfactualTrainDataset
    - ProxyCompoundDataset
    - cf_triplet_collate

Graph convention:
    All pre-GAT order nodes are globally sorted by ascending theoretical order.
    The same order is used by Data.edge_index, model.order_centers, adjacency
    matrix visualization, and node-attribution figures.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from common import FAULTS, ORDER_MAP, labels_to_multihot, robust_norm_1d
from ordered_gat_utils import build_gxu_ordered_graph


def parse_weak_label(name: str) -> List[str]:
    s = Path(name).stem.lower().replace('-', '_').replace(' ', '_')
    if 'normal' in s or re.search(r'(^|_)n($|_)', s):
        return []
    faults = []
    if 'ir' in s or 'inner' in s or 'bpfi' in s:
        faults.append('IR')
    if 'or' in s or 'outer' in s or 'bpfo' in s:
        faults.append('OR')
    if 'ball' in s or 'bsf' in s or re.search(r'(^|_)b($|_)', s):
        faults.append('Ball')
    if 'c_iob' in s or s.endswith('_iob') or s == 'ciob':
        faults += ['IR', 'OR', 'Ball']
    if 'c_io' in s or s.endswith('_io') or s == 'cio':
        faults += ['IR', 'OR']
    if 'c_ib' in s or s.endswith('_ib') or s == 'cib':
        faults += ['IR', 'Ball']
    if 'c_ob' in s or s.endswith('_ob') or s == 'cob':
        faults += ['OR', 'Ball']
    ordered = []
    for f in FAULTS:
        if f in set(faults):
            ordered.append(f)
    return ordered


def build_fixed_theory_mask(axis: np.ndarray, fault: str, base_order: float, max_order: float,
                            ir_or_half_width: float = 0.06, ball_half_width: float = 0.04,
                            max_harmonics_ir_or: int = 0, max_harmonics_ball: int = 0) -> Tuple[np.ndarray, List[float]]:
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    half = ball_half_width if fault == 'Ball' else ir_or_half_width
    auto_h = int(np.floor(float(max_order) / max(float(base_order), 1e-8)))
    raw_max_h = max_harmonics_ball if fault == 'Ball' else max_harmonics_ir_or
    max_h = auto_h if int(raw_max_h) <= 0 else min(int(raw_max_h), auto_h)
    mask = np.zeros_like(axis, dtype=np.float32)
    centers = []
    for h in range(1, max_h + 1):
        c = float(base_order * h)
        if c > max_order:
            break
        centers.append(c)
        mask[(axis >= c - half) & (axis <= c + half)] = 1.0
    return mask.astype(np.float32), centers


def build_peak_locked_fault_mask(axis: np.ndarray, clean_spec: np.ndarray, fault: str, base_order: float, max_order: float,
                                 ir_or_half_width: float = 0.06, ball_half_width: float = 0.04,
                                 search_half_width: float = 0.12, min_rel_peak: float = 0.12,
                                 max_harmonics_ir_or: int = 0, max_harmonics_ball: int = 0,
                                 floor_quantile: float = 0.80) -> Tuple[np.ndarray, List[float]]:
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    clean = np.asarray(clean_spec, dtype=np.float32).reshape(-1)
    clean_s = np.convolve(clean, np.ones(7, dtype=np.float32) / 7.0, mode='same').astype(np.float32)
    if clean_s.max() <= 0:
        return np.zeros_like(axis, dtype=np.float32), []
    auto_h = int(np.floor(float(max_order) / max(float(base_order), 1e-8)))
    raw_max_h = max_harmonics_ball if fault == 'Ball' else max_harmonics_ir_or
    max_h = auto_h if int(raw_max_h) <= 0 else min(int(raw_max_h), auto_h)
    half = ball_half_width if fault == 'Ball' else ir_or_half_width
    floor = float(np.quantile(clean_s, floor_quantile))
    global_max = float(clean_s.max())
    mask = np.zeros_like(axis, dtype=np.float32)
    centers: List[float] = []
    for h in range(1, max_h + 1):
        theory = float(base_order * h)
        if theory > max_order:
            break
        lo = max(0.0, theory - search_half_width)
        hi = min(max_order, theory + search_half_width)
        idx = np.where((axis >= lo) & (axis <= hi))[0]
        if idx.size == 0:
            c = theory
        else:
            local = clean_s[idx]
            j = int(np.argmax(local))
            peak_val = float(local[j])
            weak_thr = max(0.25 * min_rel_peak * global_max, 0.25 * floor)
            c = float(axis[idx[j]]) if peak_val >= weak_thr else theory
        centers.append(c)
        mask[(axis >= c - half) & (axis <= c + half)] = 1.0
    return mask.astype(np.float32), centers


def _safe_norm(x: np.ndarray, tiny: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if np.max(np.abs(x)) < tiny:
        return np.zeros_like(x, dtype=np.float32)
    return robust_norm_1d(x)


class V11OrderCacheReaderGNN:
    def __init__(self, cache_root: str | Path, keep_bins: int = 2048, max_order: float = 50.0,
                 input_spectrum_kind: str = 'display', mask_mode: str = 'peak_locked',
                 ir_or_half_width: float = 0.06, ball_half_width: float = 0.04,
                 search_half_width: float = 0.12, min_rel_peak: float = 0.12,
                 max_harmonics_ir_or: int = 0, max_harmonics_ball: int = 0,
                 cross_order_distance: float = 0.75):
        self.cache_root = Path(cache_root)
        self.keep_bins = int(keep_bins)
        self.max_order = float(max_order)
        self.input_spectrum_kind = input_spectrum_kind
        self.target_axis = np.linspace(0.0, self.max_order, self.keep_bins, dtype=np.float32)
        self.samples: List[dict] = []
        self.nodes, self.order_centers, self.base_edge_index_np, self.fault_mask_np = build_gxu_ordered_graph(
            order_map=ORDER_MAP,
            max_order=self.max_order,
            cross_order_distance=cross_order_distance,
            fault_order=FAULTS,
        )
        self.num_nodes = len(self.nodes)
        self.base_edge_index = torch.from_numpy(self.base_edge_index_np).long()
        self.graph_node_order = "ascending_theoretical_order"
        files = sorted(self.cache_root.glob('**/components.npz'))
        for fp in files:
            d = np.load(fp, allow_pickle=True)
            base_name = fp.parent.parent.name if fp.parent.parent.name else fp.parent.name
            faults = parse_weak_label(base_name)
            comps: Dict[str, Dict[str, np.ndarray | List[float]]] = {}
            for f in FAULTS:
                ak = f'{f}_component_order_axis'
                ck = f'{f}_component_order_amp_clean'
                dk = f'{f}_component_order_amp_display'
                if ak not in d or ck not in d:
                    continue
                axis = np.asarray(d[ak], dtype=np.float32).reshape(-1)
                clean = np.interp(self.target_axis, axis, np.asarray(d[ck], dtype=np.float32).reshape(-1), left=0.0, right=0.0).astype(np.float32)
                disp = np.interp(self.target_axis, axis, np.asarray(d[dk], dtype=np.float32).reshape(-1), left=0.0, right=0.0).astype(np.float32) if dk in d else clean.copy()
                if mask_mode == 'peak_locked':
                    mask, centers = build_peak_locked_fault_mask(self.target_axis, clean, f, ORDER_MAP[f], self.max_order,
                                                                 ir_or_half_width, ball_half_width, search_half_width,
                                                                 min_rel_peak, max_harmonics_ir_or, max_harmonics_ball)
                else:
                    mask, centers = build_fixed_theory_mask(self.target_axis, f, ORDER_MAP[f], self.max_order,
                                                            ir_or_half_width, ball_half_width,
                                                            max_harmonics_ir_or, max_harmonics_ball)
                masked_clean = (clean * mask).astype(np.float32)
                residual_display = np.maximum(0.0, disp - masked_clean).astype(np.float32)
                comps[f] = {
                    'clean': clean,
                    'display': disp,
                    'mask': mask,
                    'centers': centers,
                    'masked_clean': masked_clean,
                    'residual_display': residual_display,
                }
            input_spectrum = np.zeros_like(self.target_axis, dtype=np.float32)
            for f in faults:
                if f in comps:
                    input_spectrum += comps[f][self.input_spectrum_kind]
            self.samples.append({
                'path': str(fp),
                'name': fp.parent.name,
                'base_name': base_name,
                'faults': faults,
                'label_name': '+'.join(faults) if faults else 'Normal',
                'y': labels_to_multihot(faults),
                'order_axis': self.target_axis.copy(),
                'input_spectrum': robust_norm_1d(input_spectrum),
                'component_specs': comps,
                'masked_clean_components': {f: comps[f]['masked_clean'] for f in comps},
                'residual_display_components': {f: comps[f]['residual_display'] for f in comps},
                'adaptive_centers': {f: comps[f]['centers'] for f in comps},
            })

    def split_indices(self, task='A'):
        train_single = [i for i, s in enumerate(self.samples) if len(s['faults']) == 1]
        if task.upper() == 'A':
            real_test = [i for i, s in enumerate(self.samples) if len(s['faults']) >= 2]
        else:
            real_test = [i for i, s in enumerate(self.samples) if len(s['faults']) != 1]
        return train_single, real_test

    def data_from_arrays(self, x: np.ndarray, y: np.ndarray, name: str, label_name: str, path: str = '', faults: Sequence[str] | None = None) -> Data:
        x = _safe_norm(np.asarray(x, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32).reshape(1, -1)
        data = Data(
            edge_index=self.base_edge_index.clone(),
            signal=torch.from_numpy(x[None, :].astype(np.float32)),
            y=torch.from_numpy(y.astype(np.float32)),
            order_axis=torch.from_numpy(self.target_axis.astype(np.float32)),
        )
        data.num_nodes = self.num_nodes
        data.name = name
        data.label_name = label_name
        data.path = path
        data.faults = list(faults) if faults is not None else parse_weak_label(label_name)
        return data

    def sample_to_data(self, s: dict) -> Data:
        return self.data_from_arrays(s['input_spectrum'], s['y'], s['name'], s['label_name'], path=s['path'], faults=s['faults'])


class RealOrderDataset(Dataset):
    def __init__(self, reader: V11OrderCacheReaderGNN, indices: Sequence[int]):
        self.reader = reader
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.reader.samples[self.indices[idx]]
        return self.reader.sample_to_data(s)


class CounterfactualTrainDataset(Dataset):
    """
    Explicit counterfactual triplets for Task A/B training on top of the current GNN backend.

    Returns a dict with three graphs:
      - base: factual single-fault observation x_base -> y_base
      - sub:  negative counterfactual after intervention do(C=0) on the base fault component -> 0-vector
      - add:  positive counterfactual after adding one or more other fault causal components -> y_add
    """
    def __init__(self, reader: V11OrderCacheReaderGNN, train_single_indices: Sequence[int],
                 virtual_per_epoch: int = 4096, triple_ratio: float = 0.25,
                 noise_std: float = 0.004, shift_bins: int = 2, amp_jitter=(0.95, 1.05),
                 background_floor_scale: float = 0.06, seed: int = 42):
        self.reader = reader
        self.real_ids = list(train_single_indices)
        self.virtual_per_epoch = int(virtual_per_epoch)
        self.triple_ratio = float(triple_ratio)
        self.noise_std = float(noise_std)
        self.shift_bins = int(shift_bins)
        self.amp_jitter = amp_jitter
        self.background_floor_scale = float(background_floor_scale)
        self.rng = np.random.default_rng(seed)
        self.pool = {f: [i for i in self.real_ids if reader.samples[i]['faults'] == [f]] for f in FAULTS}
        self.real_items = [self.reader.sample_to_data(self.reader.samples[i]) for i in self.real_ids]

    def __len__(self):
        return max(len(self.real_items), self.virtual_per_epoch)

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        sc = float(self.rng.uniform(*self.amp_jitter))
        sh = int(self.rng.integers(-self.shift_bins, self.shift_bins + 1))
        y = np.roll(x * sc, sh)
        if self.noise_std > 0:
            y += self.rng.normal(0.0, self.noise_std * (np.std(y) + 1e-8), size=y.shape).astype(np.float32)
        return np.maximum(0.0, y).astype(np.float32)

    def _compose_base_sub_add(self):
        base_fault = str(self.rng.choice(FAULTS))
        base_idx = int(self.rng.choice(self.pool[base_fault]))
        base_s = self.reader.samples[base_idx]

        x_base = base_s['input_spectrum'].astype(np.float32).copy()
        base_label = [base_fault]
        y_base = labels_to_multihot(base_label)

        # Negative counterfactual: do(C=0) on the base fault-related causal component.
        # Since the current reader uses processed order spectra as inputs, we intervene by
        # zeroing/removing the masked causal component while keeping the residual display floor.
        x_sub = base_s['residual_display_components'][base_fault].astype(np.float32).copy()
        # very weak background preservation to avoid exact all-zero collapse after normalization
        x_sub += self.background_floor_scale * np.maximum(0.0, base_s['input_spectrum']).astype(np.float32) * 0.15
        y_sub = np.zeros(len(FAULTS), dtype=np.float32)

        # Positive counterfactual: add one or more other causal components from different single-fault samples.
        others = [f for f in FAULTS if f != base_fault]
        if float(self.rng.random()) < self.triple_ratio:
            add_faults = others
        else:
            add_faults = [str(self.rng.choice(others))]
        x_add = x_base.astype(np.float32).copy()
        added = [base_fault]
        floors = [base_s['input_spectrum'].astype(np.float32)]
        for f in add_faults:
            idx = int(self.rng.choice(self.pool[f]))
            s = self.reader.samples[idx]
            comp = s['masked_clean_components'][f].astype(np.float32)
            x_add += self._jitter(comp)
            floors.append(s['input_spectrum'].astype(np.float32))
            added.append(f)
        if floors:
            x_add += self.background_floor_scale * np.mean(np.stack(floors, axis=0), axis=0).astype(np.float32)
        y_add = labels_to_multihot(added)

        base_data = self.reader.data_from_arrays(x_base, y_base, name=f'BASE_{base_s["name"]}', label_name='+'.join(base_label), path='factual', faults=base_label)
        sub_data = self.reader.data_from_arrays(x_sub, y_sub, name=f'SUB_{base_s["name"]}', label_name='Normal', path='counterfactual_sub', faults=[])
        add_data = self.reader.data_from_arrays(x_add, y_add, name='ADD_' + '+'.join(added), label_name='+'.join(added), path='counterfactual_add', faults=added)
        return {'base': base_data, 'sub': sub_data, 'add': add_data}

    def __getitem__(self, idx):
        return self._compose_base_sub_add()


class ProxyCompoundDataset(Dataset):
    def __init__(self, reader: V11OrderCacheReaderGNN, proxy_single_indices: Sequence[int],
                 samples_per_combo: int = 96, triple_ratio: float = 0.25,
                 noise_std: float = 0.004, shift_bins: int = 2, amp_jitter=(0.95, 1.05),
                 background_floor_scale: float = 0.06, seed: int = 123):
        self.reader = reader
        self.samples = []
        self.rng = np.random.default_rng(seed)
        pool = {f: [i for i in proxy_single_indices if reader.samples[i]['faults'] == [f]] for f in FAULTS}
        combos = [(['IR', 'OR'], samples_per_combo), (['IR', 'Ball'], samples_per_combo), (['OR', 'Ball'], samples_per_combo)]
        if triple_ratio > 0:
            combos.append((['IR', 'OR', 'Ball'], max(1, int(round(samples_per_combo * triple_ratio)))))
        for flist, n in combos:
            if any(len(pool[f]) == 0 for f in flist):
                continue
            for _ in range(n):
                x = np.zeros_like(reader.target_axis, dtype=np.float32)
                floors = []
                for f in flist:
                    idx = int(self.rng.choice(pool[f]))
                    s = reader.samples[idx]
                    comp = s['masked_clean_components'][f].astype(np.float32)
                    sc = float(self.rng.uniform(*amp_jitter))
                    sh = int(self.rng.integers(-shift_bins, shift_bins + 1))
                    x += np.roll(comp * sc, sh)
                    floors.append(s['input_spectrum'])
                x += background_floor_scale * np.mean(np.stack(floors, axis=0), axis=0).astype(np.float32)
                x += self.rng.normal(0.0, noise_std * (np.std(x) + 1e-8), size=x.shape).astype(np.float32)
                x = _safe_norm(np.maximum(0.0, x).astype(np.float32))
                y = labels_to_multihot(flist)
                self.samples.append(reader.data_from_arrays(x, y, name='SYN_' + '+'.join(flist), label_name='+'.join(flist), path='proxy', faults=flist))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cf_triplet_collate(batch):
    return {
        'base': Batch.from_data_list([b['base'] for b in batch]),
        'sub': Batch.from_data_list([b['sub'] for b in batch]),
        'add': Batch.from_data_list([b['add'] for b in batch]),
    }
