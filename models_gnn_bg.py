from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_bg import FAULTS
from ordered_gat_utils import (
    build_mfd_bg_node_map,
    build_nodes_from_node_map,
    build_edges_from_ordered_nodes,
    build_fault_mask,
)


def build_fault_node_map(bearing_orders: Dict[str, float], gear_centers: List[float], max_order: float = 120.0) -> Dict[str, List[float]]:
    """Return the per-fault node map.

    The returned dictionary is intentionally still grouped by fault for
    compatibility with previous training / inference code. The actual node
    order used by the GAT is imposed in build_graph_from_node_map, where all
    nodes are globally sorted by ascending theoretical order.
    """
    return build_mfd_bg_node_map(bearing_orders=bearing_orders, gear_centers=gear_centers, max_order=max_order)


def build_graph_from_node_map(node_map: Dict[str, List[float]], cross_order_distance: float = 0.80):
    """Build an order-sorted pre-GAT graph from a node map.

    Compared with the previous grouped-by-fault construction, this function
    first sorts all IR/OR/Ball/GTB nodes by theoretical order. The same ordered
    node list is then used to construct:
        - order_centers for NodePool;
        - edge_index / adj_mask for GAT;
        - fault_mask for attribute-wise readout;
        - visualization and adjacency-matrix ordering.
    """
    nodes = build_nodes_from_node_map(node_map, fault_order=FAULTS)
    edge_index = build_edges_from_ordered_nodes(
        nodes,
        fault_order=FAULTS,
        cross_order_distance=cross_order_distance,
        add_self_loops=True,
    )
    fault_mask = build_fault_mask(nodes, fault_order=FAULTS)
    orders = np.asarray([float(n['order']) for n in nodes], dtype=np.float32)
    return nodes, orders, edge_index, fault_mask


class MultiScaleConvStem(nn.Module):
    def __init__(self, in_ch=1, branch_ch=16, dropout=0.10):
        super().__init__()
        ks = [3, 7, 15, 31]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, k, padding=k // 2),
                nn.InstanceNorm1d(branch_ch, affine=True),
                nn.GELU(),
            ) for k in ks
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(branch_ch * len(ks), branch_ch * len(ks), kernel_size=1),
            nn.InstanceNorm1d(branch_ch * len(ks), affine=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_ch = branch_ch * len(ks)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.mix(y)


class NodePool(nn.Module):
    def __init__(self, order_centers: np.ndarray, max_order: float, keep_bins: int, radius_bins: int, stem_ch: int, out_dim: int):
        super().__init__()
        centers = np.asarray(order_centers, dtype=np.float32)
        idx = np.clip(np.round(centers / max_order * (keep_bins - 1)).astype(np.int64), 0, keep_bins - 1)
        self.register_buffer('center_idx', torch.from_numpy(idx), persistent=False)
        self.radius_bins = int(radius_bins)
        self.proj = nn.Sequential(
            nn.Linear(stem_ch * 2 + 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, stem_feat):
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


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.10, concat: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor):
        B, N, _ = x.shape
        h = self.lin(x).view(B, N, self.num_heads, self.out_dim)
        src = (h * self.attn_src[None, None, :, :]).sum(dim=-1)
        dst = (h * self.attn_dst[None, None, :, :]).sum(dim=-1)
        e = self.leaky(src[:, :, None, :] + dst[:, None, :, :])
        mask = adj_mask[None, :, :, None]
        e = e.masked_fill(~mask, -1e9)
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        out = torch.einsum('bijh,bjhf->bihf', alpha, h)
        if self.concat:
            out = out.reshape(B, N, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=2)
        return out, alpha


class GNNCompoundNetBG(nn.Module):
    def __init__(self, node_map: Dict[str, List[float]], max_order: float = 120.0, keep_bins: int = 4096,
                 stem_branch_ch: int = 24, node_dim: int = 64, gat_hidden: int = 32,
                 num_heads: int = 4, num_gnn_layers: int = 2, radius_bins: int = 10,
                 dropout: float = 0.10):
        super().__init__()
        self.node_map = node_map
        self.nodes, orders, edge_index, fault_mask = build_graph_from_node_map(node_map)
        self.num_nodes = len(self.nodes)
        self.edge_index_np = edge_index
        self.graph_node_order = "ascending_theoretical_order"
        self.order_centers = orders
        adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool)
        for i, j in edge_index.T.tolist():
            adj[i, j] = True
        self.register_buffer('adj_mask', adj, persistent=False)
        self.register_buffer('fault_mask', torch.from_numpy(fault_mask), persistent=False)
        self.stem = MultiScaleConvStem(in_ch=1, branch_ch=stem_branch_ch, dropout=dropout)
        self.node_pool = NodePool(orders, max_order=max_order, keep_bins=keep_bins,
                                  radius_bins=radius_bins, stem_ch=self.stem.out_ch, out_dim=node_dim)
        layers = []
        in_dim = node_dim
        for _ in range(num_gnn_layers):
            gat = GraphAttentionLayer(in_dim=in_dim, out_dim=gat_hidden, num_heads=num_heads, dropout=dropout, concat=True)
            layers.append(gat)
            in_dim = gat_hidden * num_heads
        self.gnn_layers = nn.ModuleList(layers)
        readout_dim = in_dim * (1 + len(FAULTS))
        self.cls = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(FAULTS)),
        )

    def encode(self, x: torch.Tensor):
        stem = self.stem(x)
        h = self.node_pool(stem)
        attn_maps = []
        for gat in self.gnn_layers:
            h_new, alpha = gat(h, self.adj_mask)
            h = F.gelu(h_new)
            attn_maps.append(alpha)
        global_feat = h.mean(dim=1)
        fault_feats = []
        for fi in range(self.fault_mask.shape[0]):
            mask = self.fault_mask[fi][None, :, None].to(h.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            feat = (h * mask).sum(dim=1) / denom
            fault_feats.append(feat)
        z = torch.cat([global_feat] + fault_feats, dim=1)
        return z, {'node_feat': h, 'attn_maps': attn_maps}

    def forward(self, x: torch.Tensor):
        z, aux = self.encode(x)
        logits = self.cls(z)
        return logits, aux
