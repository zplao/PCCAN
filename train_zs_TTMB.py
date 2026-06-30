from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common import seed_everything, ensure_dir
from dataset_gnn import (
    V11OrderCacheReaderGNN,
    CounterfactualTrainDataset,
    ProxyCompoundDataset,
    RealOrderDataset,
    cf_triplet_collate,
)
from models_gnn import GNNCompoundNet
from ordered_gat_utils import export_node_mapping_csv, build_node_mapping_rows
from backend_utils_gnn import (
    build_protocol_splits,
    harmonic_mean,
    evaluate_dataset,
    collect_outputs,
    save_classification_report,
    save_gzsl_summary,
    save_prediction_records,
    plot_verify_train_inputs,
    plot_graph_nodes,
    plot_masked_examples,
    plot_confusion_matrix_figure,
    plot_tsne_figure,
)


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    ap = argparse.ArgumentParser(description='Train GNN backend with explicit counterfactual intervention branches from TTMB harmonic-atom cache')
    ap.add_argument('--cache_root', type=str, default="./multicomponent_results_TTMB")
    ap.add_argument('--out_dir', type=str, default='./zs_outputs_TTMB')
    ap.add_argument('--task', type=str, default='A', choices=['A', 'B'])
    ap.add_argument('--keep_bins', type=int, default=2048)
    ap.add_argument('--max_order', type=float, default=50.0)
    ap.add_argument('--input_spectrum_kind', type=str, default='display', choices=['clean', 'display'])
    ap.add_argument('--mask_mode', type=str, default='peak_locked', choices=['peak_locked', 'fixed_theory'])
    ap.add_argument('--ir_or_half_width', type=float, default=0.06)
    ap.add_argument('--ball_half_width', type=float, default=0.04)
    ap.add_argument('--search_half_width', type=float, default=0.12)
    ap.add_argument('--min_rel_peak', type=float, default=0.12)
    ap.add_argument('--max_harmonics_ir_or', type=int, default=0)
    ap.add_argument('--max_harmonics_ball', type=int, default=0)
    ap.add_argument('--proxy_ratio', type=float, default=0.25)
    ap.add_argument('--seen_test_ratio', type=float, default=0.25)
    ap.add_argument('--virtual_per_epoch', type=int, default=4096)
    ap.add_argument('--proxy_val_per_pair', type=int, default=96)
    ap.add_argument('--triple_ratio', type=float, default=0.25)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--stem_branch_ch', type=int, default=16)
    ap.add_argument('--node_dim', type=int, default=64)
    ap.add_argument('--gat_hidden', type=int, default=32)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--num_gnn_layers', type=int, default=2)
    ap.add_argument('--radius_bins', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.10)
    ap.add_argument('--pos_weight_or', type=float, default=1.0)
    ap.add_argument('--pos_weight_ball', type=float, default=1.0)
    ap.add_argument('--threshold_IR', type=float, default=0.9)
    ap.add_argument('--threshold_OR', type=float, default=0.9)
    ap.add_argument('--threshold_Ball', type=float, default=0.9)
    ap.add_argument('--lambda_cfn', type=float, default=0.2, help='weight of negative counterfactual intervention loss')  # 0.3
    ap.add_argument('--lambda_cfp', type=float, default=0.5, help='weight of positive counterfactual intervention loss')  # 1.0
    ap.add_argument('--noise_std', type=float, default=0.004)
    ap.add_argument('--shift_bins', type=int, default=2)
    ap.add_argument('--background_floor_scale', type=float, default=0.06)
    ap.add_argument('--eval_real_test_after_train', type=int, default=1, choices=[0, 1])
    ap.add_argument('--tsne_max_points', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=42)  # 42, 2021, 2023, 2024, 2025, 2026
    args = ap.parse_args()
    args.graph_node_order = "ascending_theoretical_order"

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    reader = V11OrderCacheReaderGNN(
        cache_root=args.cache_root,
        keep_bins=args.keep_bins,
        max_order=args.max_order,
        input_spectrum_kind=args.input_spectrum_kind,
        mask_mode=args.mask_mode,
        ir_or_half_width=args.ir_or_half_width,
        ball_half_width=args.ball_half_width,
        search_half_width=args.search_half_width,
        min_rel_peak=args.min_rel_peak,
        max_harmonics_ir_or=args.max_harmonics_ir_or,
        max_harmonics_ball=args.max_harmonics_ball,
    )
    splits = build_protocol_splits(reader, task=args.task, proxy_ratio=args.proxy_ratio,
                                   seen_test_ratio=args.seen_test_ratio, seed=args.seed)
    train_idx = splits['train_idx']
    proxy_idx = splits['proxy_idx']
    seen_test_idx = splits['seen_test_idx']
    unseen_test_idx = splits['unseen_test_idx']
    real_test_idx = splits['real_test_idx']

    if args.task.upper() == 'A':
        print(f'Train singles: {len(train_idx)} | Proxy-source singles: {len(proxy_idx)} | Held-out real compounds: {len(real_test_idx)} | Task A')
    else:
        print(
            f'Train seen samples (Normal + singles): {len(train_idx)} | Proxy-source seen samples: {len(proxy_idx)} | '
            f'Held-out seen samples (Normal + singles): {len(seen_test_idx)} | Held-out unseen real compounds: {len(unseen_test_idx)} | '
            f'Held-out total: {len(real_test_idx)} | Task B'
        )
    print('Protocol: no real unseen compound sample is used during training, early stopping, or model selection.')

    cf_train_ds = CounterfactualTrainDataset(
        reader, train_idx,
        virtual_per_epoch=args.virtual_per_epoch,
        triple_ratio=args.triple_ratio,
        noise_std=args.noise_std,
        shift_bins=args.shift_bins,
        background_floor_scale=args.background_floor_scale,
        seed=args.seed,
    )
    factual_train_ds = RealOrderDataset(reader, train_idx)
    proxy_compound_ds = ProxyCompoundDataset(
        reader, proxy_idx,
        samples_per_combo=args.proxy_val_per_pair,
        triple_ratio=args.triple_ratio,
        noise_std=args.noise_std,
        shift_bins=args.shift_bins,
        background_floor_scale=args.background_floor_scale,
        seed=args.seed + 1,
    )
    proxy_seen_ds = RealOrderDataset(reader, proxy_idx)
    seen_test_ds = RealOrderDataset(reader, seen_test_idx)
    unseen_test_ds = RealOrderDataset(reader, unseen_test_idx)
    real_test_ds = RealOrderDataset(reader, real_test_idx)

    cf_train_loader = DataLoader(cf_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=cf_triplet_collate)
    factual_train_loader = DataLoader(factual_train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    proxy_compound_loader = DataLoader(proxy_compound_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    proxy_seen_loader = DataLoader(proxy_seen_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    seen_test_loader = DataLoader(seen_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unseen_test_loader = DataLoader(unseen_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    real_test_loader = DataLoader(real_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GNNCompoundNet(max_order=args.max_order, keep_bins=args.keep_bins,
                           stem_branch_ch=args.stem_branch_ch, node_dim=args.node_dim,
                           gat_hidden=args.gat_hidden, num_heads=args.num_heads,
                           num_gnn_layers=args.num_gnn_layers, radius_bins=args.radius_bins,
                           dropout=args.dropout).to(device)
    export_node_mapping_csv(model.nodes, out_dir / 'ordered_graph_nodes.csv')
    with open(out_dir / 'ordered_graph_nodes.json', 'w', encoding='utf-8') as f:
        json.dump({
            'graph_node_order': 'ascending_theoretical_order',
            'nodes': build_node_mapping_rows(model.nodes),
        }, f, ensure_ascii=False, indent=2)
    print('Pre-GAT graph node order: ascending theoretical order. Node mapping saved to ordered_graph_nodes.csv/json.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([1.0, args.pos_weight_or, args.pos_weight_ball], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_zero = nn.BCEWithLogitsLoss()
    thr = (args.threshold_IR, args.threshold_OR, args.threshold_Ball)

    plot_verify_train_inputs(reader, train_idx, out_dir / 'verify_train_input_windows.png')
    plot_graph_nodes(model, out_dir / 'verify_graph_nodes.png')
    plot_masked_examples(reader, real_test_idx, out_dir / 'verify_masked_order_spectra.png', max_per_label=1)

    best_metric = -1.0
    best_path = out_dir / 'best_gnn_backend_cf.pt'
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_base = 0.0
        total_cfn = 0.0
        total_cfp = 0.0
        n = 0
        for batch in cf_train_loader:
            base_batch = batch['base'].to(device)
            sub_batch = batch['sub'].to(device)
            add_batch = batch['add'].to(device)

            y_base = base_batch.y.view(-1, len(pos_weight)).to(device)
            y_sub = sub_batch.y.view(-1, len(pos_weight)).to(device)
            y_add = add_batch.y.view(-1, len(pos_weight)).to(device)

            optimizer.zero_grad()
            logits_base, _ = model(base_batch)
            logits_sub, _ = model(sub_batch)
            logits_add, _ = model(add_batch)

            loss_base = criterion(logits_base, y_base)
            loss_cfn = criterion_zero(logits_sub, y_sub)
            loss_cfp = criterion(logits_add, y_add)
            loss = loss_base + args.lambda_cfn * loss_cfn + args.lambda_cfp * loss_cfp
            loss.backward()
            optimizer.step()

            bs = y_base.size(0)
            total_loss += float(loss.item()) * bs
            total_base += float(loss_base.item()) * bs
            total_cfn += float(loss_cfn.item()) * bs
            total_cfp += float(loss_cfp.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)
        train_base = total_base / max(n, 1)
        train_cfn = total_cfn / max(n, 1)
        train_cfp = total_cfp / max(n, 1)
        train_exact, _, _ = evaluate_dataset(model, factual_train_loader, device, thr=thr)

        if args.task.upper() == 'A':
            proxy_unseen_exact, _, _ = evaluate_dataset(model, proxy_compound_loader, device, thr=thr)
            selection_metric = proxy_unseen_exact
            print(
                f'Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} '
                f'(base {train_base:.4f}, cfn {train_cfn:.4f}, cfp {train_cfp:.4f}) '
                f'| train exact {train_exact:.4f} | proxy-val exact {proxy_unseen_exact:.4f}'
            )
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'loss_base': train_base,
                'loss_cfn': train_cfn,
                'loss_cfp': train_cfp,
                'train_exact': train_exact,
                'proxy_unseen_exact': proxy_unseen_exact,
            })
        else:
            proxy_seen_exact, _, _ = evaluate_dataset(model, proxy_seen_loader, device, thr=thr)
            proxy_unseen_exact, _, _ = evaluate_dataset(model, proxy_compound_loader, device, thr=thr)
            proxy_H = harmonic_mean(proxy_seen_exact, proxy_unseen_exact)
            selection_metric = proxy_H
            print(
                f'Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} '
                f'(base {train_base:.4f}, cfn {train_cfn:.4f}, cfp {train_cfp:.4f}) '
                f'| train exact {train_exact:.4f} | proxy-seen exact {proxy_seen_exact:.4f} '
                f'| proxy-unseen exact {proxy_unseen_exact:.4f} | proxy-H {proxy_H:.4f}'
            )
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'loss_base': train_base,
                'loss_cfn': train_cfn,
                'loss_cfp': train_cfp,
                'train_exact': train_exact,
                'proxy_seen_exact': proxy_seen_exact,
                'proxy_unseen_exact': proxy_unseen_exact,
                'proxy_harmonic_mean': proxy_H,
            })

        if selection_metric > best_metric:
            best_metric = selection_metric
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'args': vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= 25:
                if args.task.upper() == 'A':
                    print(f'Early stopping at epoch {epoch} due to no proxy-val improvement in 25 epochs.')
                else:
                    print(f'Early stopping at epoch {epoch} due to no proxy-H improvement in 25 epochs.')
                break

    with open(out_dir / 'train_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    ckpt = safe_torch_load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    if args.eval_real_test_after_train:
        all_out = collect_outputs(model, real_test_loader, device, thr=thr)
        real_exact = all_out['exact']
        print(f'\nFinal held-out real-test exact-match accuracy: {real_exact:.4f}')

        if args.task.upper() == 'B':
            seen_out = collect_outputs(model, seen_test_loader, device, thr=thr)
            unseen_out = collect_outputs(model, unseen_test_loader, device, thr=thr)
            seen_exact = seen_out['exact']
            unseen_exact = unseen_out['exact']
            H = harmonic_mean(seen_exact, unseen_exact)
            print(f'Seen exact-match accuracy   : {seen_exact:.4f}')
            print(f'Unseen exact-match accuracy : {unseen_exact:.4f}')
            print(f'Harmonic mean (H)           : {H:.4f}')
            save_gzsl_summary(seen_exact, unseen_exact, H, out_dir / 'gzsl_harmonic_mean.txt', prefix='Task B GZSL')

        rep = save_classification_report(all_out['y_true'], all_out['y_pred'], out_dir / 'classification_report.txt')
        print(rep)
        save_prediction_records(model, real_test_ds, device, out_dir, thr=thr)
        plot_confusion_matrix_figure(all_out['true_names'], all_out['pred_names'], out_dir / 'confusion_matrix.png', title=f'Task {args.task} held-out confusion matrix')
        plot_tsne_figure(all_out['embeddings'], all_out['true_names'], all_out['pred_names'], out_dir / 'tsne_embeddings.png', max_points=args.tsne_max_points, seed=args.seed)

        metrics = {
            'task': args.task,
            'overall_exact': float(real_exact),
            'num_test_samples': int(len(real_test_ds)),
        }
        if args.task.upper() == 'B':
            metrics.update({
                'num_seen_test': int(len(seen_test_ds)),
                'num_unseen_test': int(len(unseen_test_ds)),
                'seen_exact': float(seen_exact),
                'unseen_exact': float(unseen_exact),
                'harmonic_mean': float(H),
            })
        with open(out_dir / 'eval_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
