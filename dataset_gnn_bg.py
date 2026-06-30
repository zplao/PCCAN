# -*- coding: utf-8 -*-
"""
dataset_gnn_bg_ordered_v20.py

MFD-BG dataset module for the ordered pre-GAT graph backend.  This file
corresponds to the original dataset_gnn_bg.py branch.

Compared with dataset_gnn_bg.py, this version keeps the same tensor-based
sample API but additionally exposes ordered graph metadata when bg_meta.json is
available:
    - reader.nodes
    - reader.order_centers
    - reader.base_edge_index_np
    - reader.fault_mask_np
    - reader.graph_node_order = "ascending_theoretical_order"

The model still constructs and uses its own ordered graph through
models_gnn_bg_ordered_v19.py.  The reader-side graph metadata is provided only
for consistency checking, visualization, and downstream scripts that want the
same ordered node convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Optional
import json
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from common_bg import FAULTS, labels_to_multihot, robust_norm_1d
from ordered_gat_utils import (
    build_mfd_bg_node_map,
    build_nodes_from_node_map,
    build_edges_from_ordered_nodes,
    build_fault_mask,
)


def parse_weak_label_bg(name: str) -> List[str]:
    s = Path(name).stem.lower().replace('-', '_').replace(' ', '_')
    if 'normal' in s or re.search(r'(^|_)n($|_)', s):
        return []
    faults: List[str] = []
    if 'ir' in s or 'inner' in s or 'bpfi' in s:
        faults.append('IR')
    if 'or' in s or 'outer' in s or 'bpfo' in s:
        faults.append('OR')
    if 'ball' in s or 'bsf' in s or re.search(r'(^|_)b($|_)', s):
        faults.append('Ball')
    if 'gtb' in s or 'gear' in s or 'tooth' in s or 'broken' in s or 'tb' in s:
        faults.append('GTB')
    # explicit compound codes
    if 'c_io' in s or s.endswith('_io') or s == 'cio':
        faults += ['IR', 'OR']
    if 'c_itb' in s or s.endswith('_itb') or s == 'citb':
        faults += ['IR', 'GTB']
    if 'c_otb' in s or s.endswith('_otb') or s == 'cotb':
        faults += ['OR', 'GTB']
    if 's_gtb' in s:
        faults += ['GTB']
    ordered: List[str] = []
    seen = set(faults)
    for f in FAULTS:
        if f in seen:
            ordered.append(f)
    return ordered


def _load_bg_meta(cache_root: Path, meta: Optional[dict] = None) -> Optional[dict]:
    if isinstance(meta, dict):
        return meta
    meta_path = Path(cache_root) / 'bg_meta.json'
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding='utf-8'))
    return None


class BGOrderCacheReader:
    def __init__(self, cache_root: str | Path, keep_bins: int = 4096, max_order: float = 120.0,
                 input_spectrum_kind: str = 'display', meta: Optional[dict] = None,
                 cross_order_distance: float = 0.80):
        self.cache_root = Path(cache_root)
        self.keep_bins = int(keep_bins)
        self.max_order = float(max_order)
        self.input_spectrum_kind = input_spectrum_kind
        self.target_axis = np.linspace(0.0, self.max_order, self.keep_bins, dtype=np.float32)
        self.samples: List[dict] = []

        # Ordered graph metadata for consistency / visualization.
        self.meta = _load_bg_meta(self.cache_root, meta=meta)
        self.nodes: List[dict] = []
        self.order_centers = np.asarray([], dtype=np.float32)
        self.base_edge_index_np = np.empty((2, 0), dtype=np.int64)
        self.fault_mask_np = np.zeros((len(FAULTS), 0), dtype=np.float32)
        self.graph_node_order = "ascending_theoretical_order"
        if isinstance(self.meta, dict):
            try:
                bearing_orders = self.meta['bearing_orders']
                gear_info = self.meta['gear_info']
                gear_centers = [float(c) for c in gear_info.get('extract_centers', gear_info.get('centers', []))
                                if np.isfinite(float(c)) and float(c) <= self.max_order]
                node_map = build_mfd_bg_node_map(bearing_orders, gear_centers, max_order=self.max_order)
                self.nodes = build_nodes_from_node_map(node_map, fault_order=FAULTS)
                self.order_centers = np.asarray([float(n['order']) for n in self.nodes], dtype=np.float32)
                self.base_edge_index_np = build_edges_from_ordered_nodes(
                    self.nodes,
                    fault_order=FAULTS,
                    cross_order_distance=float(cross_order_distance),
                    add_self_loops=True,
                )
                self.fault_mask_np = build_fault_mask(self.nodes, fault_order=FAULTS)
            except Exception as exc:
                print(f"[WARN] Failed to build reader-side ordered graph metadata: {type(exc).__name__}: {exc}")
        self.num_nodes = len(self.nodes)

        files = sorted(self.cache_root.glob('**/components.npz'))
        for fp in files:
            d = np.load(fp, allow_pickle=True)
            base_name = fp.parent.name
            faults = parse_weak_label_bg(base_name)
            comps: Dict[str, Dict[str, np.ndarray]] = {}
            for f in FAULTS:
                ak = f'{f}_component_order_axis'
                ck = f'{f}_component_order_amp_clean'
                dk = f'{f}_component_order_amp_display'
                if ak not in d or ck not in d:
                    continue
                axis = np.asarray(d[ak], dtype=np.float32).reshape(-1)
                clean = np.interp(
                    self.target_axis, axis, np.asarray(d[ck], dtype=np.float32).reshape(-1),
                    left=0.0, right=0.0,
                ).astype(np.float32)
                if dk in d:
                    disp = np.interp(
                        self.target_axis, axis, np.asarray(d[dk], dtype=np.float32).reshape(-1),
                        left=0.0, right=0.0,
                    ).astype(np.float32)
                else:
                    disp = clean.copy()
                floor = np.zeros_like(clean)
                fk = f'{f}_component_order_floor'
                if fk in d:
                    floor = np.interp(
                        self.target_axis, axis, np.asarray(d[fk], dtype=np.float32).reshape(-1),
                        left=0.0, right=0.0,
                    ).astype(np.float32)
                centers = np.asarray(d.get(f'{f}_centers', []), dtype=np.float32).reshape(-1).tolist() if f'{f}_centers' in d else []
                comps[f] = {
                    'clean': clean,
                    'display': disp,
                    'floor': floor,
                    'masked_clean': clean.copy(),
                    'centers': centers,
                }

            input_spectrum = np.zeros_like(self.target_axis, dtype=np.float32)
            for f in faults:
                if f in comps:
                    input_spectrum += comps[f][self.input_spectrum_kind]
            input_spectrum = robust_norm_1d(input_spectrum)

            self.samples.append({
                'path': str(fp),
                'name': fp.parent.name,
                'base_name': base_name,
                'faults': faults,
                'label_name': '+'.join(faults) if faults else 'Normal',
                'y': labels_to_multihot(faults),
                'order_axis': self.target_axis.copy(),
                'input_spectrum': input_spectrum,
                'component_specs': comps,
                'masked_clean_components': {f: comps[f]['masked_clean'] for f in comps},
                'display_components': {f: comps[f]['display'] for f in comps},
                'adaptive_centers': {f: comps[f]['centers'] for f in comps},
            })

    def split_indices(self, task='A'):
        train_single = [i for i, s in enumerate(self.samples) if len(s['faults']) == 1 and s['faults']]
        if task.upper() == 'A':
            real_test = [i for i, s in enumerate(self.samples) if len(s['faults']) >= 2]
        else:
            real_test = [i for i, s in enumerate(self.samples) if len(s['faults']) != 1]
        return train_single, real_test


class RealOrderDataset(Dataset):
    def __init__(self, reader: BGOrderCacheReader, indices: Sequence[int]):
        self.reader = reader
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.reader.samples[self.indices[idx]]
        return {
            'x': torch.from_numpy(s['input_spectrum'][None, :].astype(np.float32)),
            'y': torch.from_numpy(s['y'].astype(np.float32)),
            'name': s['name'],
            'label_name': s['label_name'],
            'path': s['path'],
        }
