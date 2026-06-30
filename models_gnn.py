
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from common import FAULTS, ORDER_MAP
from ordered_gat_utils import build_gxu_ordered_graph


class MultiScaleConvStem(nn.Module):
    def __init__(self, in_ch: int = 1, branch_ch: int = 16, dropout: float = 0.10):
        super().__init__()
        ks = [3, 7, 15, 31]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, k, padding=k // 2, bias=False),
                nn.InstanceNorm1d(branch_ch, affine=True),
                nn.GELU(),
            ) for k in ks
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(branch_ch * len(ks), branch_ch * len(ks), kernel_size=1, bias=False),
            nn.InstanceNorm1d(branch_ch * len(ks), affine=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_ch = branch_ch * len(ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.mix(y)


class NodePool(nn.Module):
    def __init__(self, order_centers, max_order: float = 50.0,
                 keep_bins: int = 2048, radius_bins: int = 8,
                 stem_ch: int = 64, out_dim: int = 64):
        super().__init__()
        idx = torch.clamp(torch.round(torch.tensor(order_centers) / max_order * (keep_bins - 1)).long(), 0, keep_bins - 1)
        self.register_buffer('center_idx', idx, persistent=False)
        self.radius_bins = int(radius_bins)
        self.proj = nn.Sequential(
            nn.Linear(stem_ch * 2 + 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, stem_feat: torch.Tensor) -> torch.Tensor:
        # stem_feat: [B, C, L]
        B, C, L = stem_feat.shape
        nodes = []
        for cidx in self.center_idx.tolist():
            lo = max(0, int(cidx) - self.radius_bins)
            hi = min(L, int(cidx) + self.radius_bins + 1)
            local = stem_feat[:, :, lo:hi]
            meanf = local.mean(dim=-1)
            maxf = local.max(dim=-1).values
            rel_pos = torch.full((B, 1), float(cidx) / max(L - 1, 1), device=stem_feat.device)
            width = torch.full((B, 1), float(hi - lo) / max(L, 1), device=stem_feat.device)
            feat = torch.cat([meanf, maxf, rel_pos, width], dim=1)
            nodes.append(self.proj(feat))
        return torch.stack(nodes, dim=1)


class PyGDataBatchGNNCompoundNet(nn.Module):
    """
    Standard torch_geometric.data.Data / Batch flow.
    Each graph in the batch contains:
      - edge_index: static harmonic graph topology
      - signal: graph-level order spectrum [1, L]
      - num_nodes: number of harmonic nodes
    The model converts signal -> node embeddings with multi-scale conv + node pooling,
    then uses batched edge_index / batch from PyG Batch directly.
    """
    def __init__(self, max_order: float = 50.0, keep_bins: int = 2048,
                 stem_branch_ch: int = 16, node_dim: int = 64,
                 gat_hidden: int = 32, num_heads: int = 4,
                 num_gnn_layers: int = 2, radius_bins: int = 8,
                 dropout: float = 0.10, cross_order_distance: float = 0.75):
        super().__init__()
        self.nodes, self.order_centers, edge_index, fault_mask = build_gxu_ordered_graph(
            order_map=ORDER_MAP,
            max_order=max_order,
            cross_order_distance=cross_order_distance,
            fault_order=FAULTS,
        )
        self.num_nodes = len(self.nodes)
        self.edge_index_np = edge_index
        self.graph_node_order = "ascending_theoretical_order"
        self.register_buffer('fault_mask', torch.from_numpy(fault_mask), persistent=False)
        self.stem = MultiScaleConvStem(in_ch=1, branch_ch=stem_branch_ch, dropout=dropout)
        self.node_pool = NodePool(self.order_centers, max_order=max_order, keep_bins=keep_bins,
                                  radius_bins=radius_bins, stem_ch=self.stem.out_ch, out_dim=node_dim)

        convs: List[GATConv] = []
        norms: List[nn.Module] = []
        in_dim = node_dim
        for _ in range(num_gnn_layers):
            conv = GATConv(
                in_channels=in_dim,
                out_channels=gat_hidden,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
                bias=True,
            )
            convs.append(conv)
            in_dim = gat_hidden * num_heads
            norms.append(nn.LayerNorm(in_dim))
        self.gnn_layers = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        readout_dim = in_dim * (1 + len(FAULTS))
        self.cls = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(FAULTS)),
        )

    def encode(self, data):
        signal = data.signal
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        elif signal.dim() == 3 and signal.size(1) != 1:
            pass
        stem = self.stem(signal)
        node_feat = self.node_pool(stem)  # [B, N, D]
        B, N, D = node_feat.shape
        h = node_feat.reshape(B * N, D)
        edge_index = data.edge_index
        batch = data.batch
        attn_maps = []
        for conv, norm in zip(self.gnn_layers, self.norms):
            h_out, attn_info = conv(h, edge_index, return_attention_weights=True)
            h = norm(F.gelu(h_out))
            h = self.dropout(h)
            attn_maps.append(attn_info)

        global_feat = global_mean_pool(h, batch)
        node_pos = torch.arange(h.size(0), device=h.device) % self.num_nodes
        fault_feats = []
        for fi in range(self.fault_mask.shape[0]):
            mask = self.fault_mask[fi, node_pos].to(h.dtype)
            denom = global_mean_pool(mask.unsqueeze(-1), batch).clamp_min(1e-6)
            feat = global_mean_pool(h * mask.unsqueeze(-1), batch) / denom
            fault_feats.append(feat)
        z = torch.cat([global_feat] + fault_feats, dim=1)
        return z, {
            'node_feat': h,
            'edge_index': edge_index,
            'batch': batch,
            'attn_maps': attn_maps,
        }

    def forward(self, data):
        z, aux = self.encode(data)
        logits = self.cls(z)
        return logits, aux


GNNCompoundNet = PyGDataBatchGNNCompoundNet
