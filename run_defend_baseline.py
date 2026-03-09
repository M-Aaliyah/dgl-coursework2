#!/usr/bin/env python3
"""
DEFEND Baseline — 3-Fold CV with v2-consistent evaluation metrics.
"""

import os, sys, time, warnings, random, gc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SLURM
import matplotlib.pyplot as plt
import networkx as nx
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

#  Path configuration 
DATA_DIR = os.environ.get('DATA_DIR', 'data')
sys.path.insert(0, os.environ.get('VECTORIZER_DIR', '.'))
from MatrixVectorizer import MatrixVectorizer

#  Reproducibility 
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA available — using GPU.')
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')
    print('CUDA not available — using CPU.')

#  Load data 
lr_train = pd.read_csv(os.path.join(DATA_DIR, 'lr_train.csv')).values.astype(np.float32)
hr_train = pd.read_csv(os.path.join(DATA_DIR, 'hr_train.csv')).values.astype(np.float32)
lr_test  = pd.read_csv(os.path.join(DATA_DIR, 'lr_test.csv')).values.astype(np.float32)
print(f'LR Train : {lr_train.shape}')
print(f'HR Train : {hr_train.shape}')
print(f'LR Test  : {lr_test.shape}')

lr_train = np.nan_to_num(np.clip(lr_train, 0, None))
hr_train = np.nan_to_num(np.clip(hr_train, 0, None))
lr_test  = np.nan_to_num(np.clip(lr_test,  0, None))

#  Constants 
N_LR     = 160
N_HR     = 268
N_LR_VEC = 12720
N_HR_VEC = 35778


# Data Utilities
def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)

    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)
    elif node_feature_init == 'identity':
        node_feat = torch.eye(n_nodes, device=edge_attr.device)
    else:
        raise ValueError(f"Unsupported node feature init: {node_feature_init}")

    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0).to(edge_attr.device)
    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    return pyg_graph


def create_dual_graph(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    row, col = torch.triu_indices(n, n, offset=1)
    all_edges = torch.stack([row, col], dim=1).to(adjacency_matrix.device)
    actual_edges_mask = adjacency_matrix[row, col].nonzero().view(-1)
    actual_edges = all_edges[actual_edges_mask]
    max_possible_edges = row.size(0)

    edge_to_nodes = torch.zeros((max_possible_edges, n), dtype=torch.float,
                                 device=adjacency_matrix.device)
    edge_to_nodes[actual_edges_mask, actual_edges[:, 0]] = 1.0
    edge_to_nodes[actual_edges_mask, actual_edges[:, 1]] = 1.0

    shared_nodes_matrix = edge_to_nodes @ edge_to_nodes.t()
    shared_nodes_matrix.fill_diagonal_(0)
    edge_index = shared_nodes_matrix.nonzero(as_tuple=False).t().contiguous()

    node_feat_matrix = torch.zeros((max_possible_edges, 1), dtype=torch.float,
                                    device=adjacency_matrix.device)
    node_feat_matrix[actual_edges_mask] = adjacency_matrix[
        actual_edges[:, 0], actual_edges[:, 1]
    ].view(-1, 1).float()

    torch.cuda.empty_cache()
    gc.collect()
    return edge_index, node_feat_matrix


def revert_dual(node_feat, n_nodes):
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float, device=node_feat.device)
    row, col = torch.triu_indices(n_nodes, n_nodes, offset=1)
    adj[row, col] = node_feat.view(-1)
    adj[col, row] = node_feat.view(-1)
    return adj


def anti_vectorize_batch(vectors, N):
    return np.stack([MatrixVectorizer.anti_vectorize(v, N) for v in vectors])

def vectorize_batch(matrices):
    return np.stack([MatrixVectorizer.vectorize(m) for m in matrices])

def postprocess(pred_matrices):
    pred_matrices = np.clip(pred_matrices, 0, None)
    for i in range(len(pred_matrices)):
        np.fill_diagonal(pred_matrices[i], 0)
        pred_matrices[i] = (pred_matrices[i] + pred_matrices[i].T) / 2
    return pred_matrices


#  Precompute adjacency matrices & PyG graphs 
print('Anti-vectorizing data into adjacency matrices...')
lr_matrices_train = anti_vectorize_batch(lr_train, N_LR)
hr_matrices_train = anti_vectorize_batch(hr_train, N_HR)

lr_tensors_train = [torch.tensor(m, dtype=torch.float, device=device) for m in lr_matrices_train]
hr_tensors_train = [torch.tensor(m, dtype=torch.float, device=device) for m in hr_matrices_train]

print('Building PyG graphs...')
source_pyg_all = [create_pyg_graph(m, N_LR, node_feature_init='adj') for m in lr_tensors_train]
target_pyg_all = [create_pyg_graph(m, N_HR, node_feature_init='adj') for m in hr_tensors_train]
print(f'Built {len(source_pyg_all)} source, {len(target_pyg_all)} target graphs')

#  Precompute dual graph 
print('Precomputing dual graph edge index from fully-connected HR graph...')
fully_connected_hr = torch.ones(N_HR, N_HR, device=device) - torch.eye(N_HR, device=device)
dual_edge_index, _ = create_dual_graph(fully_connected_hr)
dual_edge_index = dual_edge_index.to(device)
print(f'  Dual graph: {N_HR_VEC} nodes, {dual_edge_index.shape[1]} edges')
del fully_connected_hr
torch.cuda.empty_cache()
gc.collect()


# Model Definitions (faithful DEFEND reproduction)
class LA(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1,
                 dropout=0.2, beta=False, min_max_scale=True, multi_dim_edge=False,
                 binarize=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0
        self.min_max_scale = min_max_scale
        self.multi_dim_edge = multi_dim_edge
        self.binarize = binarize
        self.conv1 = TransformerConv(n_source_nodes, n_target_nodes // num_heads,
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        if self.multi_dim_edge:
            xt = x.unsqueeze(1) + x.unsqueeze(2)
            xt = xt.permute(1, 2, 0)
        else:
            xt = x.T @ x
            if self.min_max_scale:
                xt_min = torch.min(xt)
                xt_max = torch.max(xt)
                xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)
            if self.binarize:
                xt = (xt - torch.mean(xt)) / torch.std(xt)
                xt = torch.sigmoid(xt)
        return xt


class DualLearner(nn.Module):
    def __init__(self, in_dim, out_dim=1, num_heads=4,
                 dropout=0.2, beta=False, min_max_scale=True, binarize=False):
        super().__init__()
        if out_dim == 1:
            num_heads = 1
        else:
            assert out_dim % num_heads == 0
        self.min_max_scale = min_max_scale
        self.binarize = binarize
        self.conv1 = TransformerConv(in_dim, out_dim // num_heads,
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)
        if self.min_max_scale:
            xt_min = torch.min(xt)
            xt_max = torch.max(xt)
            xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)
        if self.binarize:
            xt = (xt - torch.mean(xt)) / torch.std(xt)
            xt = torch.sigmoid(xt)
        return xt


class DualModel(nn.Module):
    def __init__(self, n_source_nodes, n_target_nodes,
                 num_heads=4, edge_dim=1, dropout=0.2, beta=False,
                 min_max_scale=True, dual_node_in_dim=1, dual_node_out_dim=1):
        super().__init__()
        self.n_target_nodes = n_target_nodes
        self.n_ut = n_target_nodes * (n_target_nodes - 1) // 2
        self.node_init_model = LA(
            n_source_nodes, n_target_nodes,
            num_heads=num_heads, edge_dim=edge_dim,
            dropout=dropout, beta=beta,
            min_max_scale=min_max_scale,
            multi_dim_edge=False, binarize=False)
        self.dual_learner = DualLearner(
            dual_node_in_dim, dual_node_out_dim,
            num_heads=num_heads, dropout=dropout,
            beta=beta, min_max_scale=min_max_scale,
            binarize=False)

    def get_dual_node_init(self, source_data):
        dual_node_init_x = self.node_init_model(source_data)
        ut_mask = torch.triu(
            torch.ones(self.n_target_nodes, self.n_target_nodes), diagonal=1
        ).bool().to(dual_node_init_x.device)
        dual_x = torch.masked_select(dual_node_init_x, ut_mask)
        dual_x = dual_x.view(self.n_ut, -1)
        return dual_x

    def forward(self, source_data, dual_edge_index):
        dual_node_init = self.get_dual_node_init(source_data)
        dual_target_x = self.dual_learner(dual_node_init, dual_edge_index)
        return dual_target_x



# Training & Prediction
def train_one_epoch(model, optimizer, source_pyg_list, target_mat_list,
                    dual_edge_index, n_target, batch_size=16):
    model.train()
    n_samples = len(source_pyg_list)
    indices = np.random.permutation(n_samples)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_loss = 0.0
        optimizer.zero_grad()

        for idx in batch_idx:
            source_data = source_pyg_list[idx]
            target_mat = target_mat_list[idx]
            pred_dual = model(source_data, dual_edge_index)
            gt_dual = target_mat[
                torch.triu_indices(n_target, n_target, offset=1)[0],
                torch.triu_indices(n_target, n_target, offset=1)[1]
            ].view(-1, 1)
            loss = F.l1_loss(pred_dual, gt_dual)
            batch_loss += loss

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def predict_matrices(model, source_pyg_list, dual_edge_index, n_target):
    model.eval()
    predictions = []
    for source_data in source_pyg_list:
        pred_dual = model(source_data, dual_edge_index)
        pred_adj = revert_dual(pred_dual, n_target)
        pred_np = pred_adj.cpu().numpy()
        pred_np = np.clip(pred_np, 0, None)
        np.fill_diagonal(pred_np, 0)
        pred_np = (pred_np + pred_np.T) / 2
        predictions.append(pred_np)
    return np.array(predictions)



# Evaluation — MATCHING metrics exactly
def compute_all_metrics(pred_matrices, gt_vectors, n_hr=N_HR):
    """Same 8 metrics as defend_brain_sr_v2.ipynb:
    MAE, PCC, JSD, MAE_PC, MAE_EC, MAE_BC, MAE_CC (distance=None), MAE_Clust
    """
    n_test = len(pred_matrices)
    pred_matrices = postprocess(pred_matrices.copy())

    mae_bc, mae_ec, mae_pc = [], [], []
    mae_cc, mae_clust = [], []

    gt_mats = anti_vectorize_batch(gt_vectors, n_hr)

    for i in range(n_test):
        pg = nx.from_numpy_array(pred_matrices[i], edge_attr='weight')
        gg = nx.from_numpy_array(gt_mats[i], edge_attr='weight')

        # Betweenness centrality
        p_bc = list(nx.betweenness_centrality(pg, weight='weight').values())
        g_bc = list(nx.betweenness_centrality(gg, weight='weight').values())
        mae_bc.append(mean_absolute_error(p_bc, g_bc))

        # Eigenvector centrality
        try:
            p_ec = list(nx.eigenvector_centrality(pg, weight='weight', max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            p_ec = [0.0] * n_hr
        try:
            g_ec = list(nx.eigenvector_centrality(gg, weight='weight', max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            g_ec = [0.0] * n_hr
        mae_ec.append(mean_absolute_error(p_ec, g_ec))

        # PageRank centrality
        p_pc = list(nx.pagerank(pg, weight='weight').values())
        g_pc = list(nx.pagerank(gg, weight='weight').values())
        mae_pc.append(mean_absolute_error(p_pc, g_pc))

        # Closeness centrality — distance=None, matching ours version
        p_cc = list(nx.closeness_centrality(pg, distance=None).values())
        g_cc = list(nx.closeness_centrality(gg, distance=None).values())
        mae_cc.append(mean_absolute_error(p_cc, g_cc))

        # Clustering coefficient — matching ours version
        p_cl = list(nx.clustering(pg, weight='weight').values())
        g_cl = list(nx.clustering(gg, weight='weight').values())
        mae_clust.append(mean_absolute_error(p_cl, g_cl))

    # Global metrics: vectorize post-processed predictions
    pred_vecs_pp = vectorize_batch(pred_matrices)
    pred_flat = pred_vecs_pp.flatten()
    gt_flat = gt_vectors.flatten()

    mae_val = mean_absolute_error(pred_flat, gt_flat)
    pcc_val = pearsonr(pred_flat, gt_flat)[0]
    jsd_val = jensenshannon(pred_flat + 1e-12, gt_flat + 1e-12)

    return {
        'MAE':       mae_val,
        'PCC':       pcc_val,
        'JSD':       jsd_val,
        'MAE_PC':    np.mean(mae_pc),
        'MAE_EC':    np.mean(mae_ec),
        'MAE_BC':    np.mean(mae_bc),
        'MAE_CC':    np.mean(mae_cc),
        'MAE_Clust': np.mean(mae_clust),
    }


# 3-Fold Cross-Validation
HIDDEN_DIM       = 32
NUM_HEADS        = 4
DROPOUT          = 0.2
EDGE_DIM         = 1
BETA             = False
MIN_MAX_SCALE    = True
DUAL_IN_DIM      = 1
DUAL_OUT_DIM     = 1
LR_RATE          = 0.001
BATCH_SIZE       = 16
MAX_EPOCHS       = 300
PATIENCE         = 7
WARM_UP_EPOCHS   = 30
NUM_SPLITS       = 3

METRIC_NAMES  = ['MAE', 'PCC', 'JSD', 'MAE_PC', 'MAE_EC', 'MAE_BC', 'MAE_CC', 'MAE_Clust']
METRIC_LABELS = ['MAE', 'PCC', 'JSD', 'MAE(PC)', 'MAE(EC)', 'MAE(BC)', 'MAE(CC)', 'MAE(Clust)']

print(f'\nHyperparameters:')
print(f'  hidden_dim={HIDDEN_DIM}, num_heads={NUM_HEADS}, dropout={DROPOUT}')
print(f'  lr={LR_RATE}, batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}')
print(f'  patience={PATIENCE}, warm_up_epochs={WARM_UP_EPOCHS}')
print()

kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=random_seed)

all_fold_metrics = []
all_fold_predictions = []
start_time_total = time.time()

for fold_num, (train_idx, test_idx) in enumerate(kf.split(source_pyg_all)):
    print(f'\n{"="*60}')
    print(f'FOLD {fold_num + 1}/{NUM_SPLITS}  (train: {len(train_idx)}, test: {len(test_idx)})')
    print(f'{"="*60}')
    fold_start = time.time()

    train_source = [source_pyg_all[i] for i in train_idx]
    train_target = [hr_tensors_train[i] for i in train_idx]
    test_source  = [source_pyg_all[i] for i in test_idx]
    test_gt_vecs = hr_train[test_idx]

    model = DualModel(
        n_source_nodes=N_LR, n_target_nodes=N_HR,
        num_heads=NUM_HEADS, edge_dim=EDGE_DIM,
        dropout=DROPOUT, beta=BETA,
        min_max_scale=MIN_MAX_SCALE,
        dual_node_in_dim=DUAL_IN_DIM,
        dual_node_out_dim=DUAL_OUT_DIM
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        epoch_loss = train_one_epoch(
            model, optimizer, train_source, train_target,
            dual_edge_index, N_HR, batch_size=BATCH_SIZE)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            if epoch >= WARM_UP_EPOCHS:
                patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{MAX_EPOCHS} | Loss: {epoch_loss:.6f} | '
                  f'Best: {best_loss:.6f} | Patience: {patience_counter}/{PATIENCE}')

        if patience_counter >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_matrices = predict_matrices(model, test_source, dual_edge_index, N_HR)
    pred_vectors  = vectorize_batch(pred_matrices)

    metrics = compute_all_metrics(pred_matrices, test_gt_vecs, N_HR)
    all_fold_metrics.append(metrics)

    fold_time = time.time() - fold_start
    print(f'\n  Fold {fold_num+1} Results ({fold_time/60:.1f} min):')
    for k, v in metrics.items():
        print(f'    {k}: {v:.6f}')

    all_fold_predictions.append((test_idx, pred_vectors))

    del model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_time = time.time() - start_time_total
ram_mb = psutil.Process().memory_info().rss / 1e6
print(f'\nTotal 3-fold CV time: {total_time:.1f}s ({total_time/60:.1f} min)')
print(f'RAM usage: {ram_mb:.0f} MB')


# Summary
print('\n' + '='*70)
print('DEFEND BASELINE — Average ± Std Across 3 Folds')
print('='*70)
for k, label in zip(METRIC_NAMES, METRIC_LABELS):
    vals = [m[k] for m in all_fold_metrics]
    print(f'  {label:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}')


# Save CSV predictions
for fold_num, (test_idx, pred_vecs) in enumerate(all_fold_predictions):
    melted = pred_vecs.flatten()
    df = pd.DataFrame({
        'ID': np.arange(1, len(melted) + 1),
        'Predicted': melted
    })
    filename = f'predictions_baseline_fold_{fold_num + 1}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename} — {len(melted)} entries')


# Bar plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
          '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for f_idx in range(NUM_SPLITS):
    ax = axes[f_idx]
    vals = [all_fold_metrics[f_idx][k] for k in METRIC_NAMES]
    bars = ax.bar(METRIC_LABELS, vals, color=colors)
    ax.set_title(f'Fold {f_idx+1}', fontweight='bold')
    ax.set_ylim(bottom=0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax.tick_params(axis='x', rotation=45)

ax = axes[NUM_SPLITS]
means = [np.mean([all_fold_metrics[f][k] for f in range(NUM_SPLITS)]) for k in METRIC_NAMES]
stds  = [np.std([all_fold_metrics[f][k] for f in range(NUM_SPLITS)]) for k in METRIC_NAMES]
bars = ax.bar(METRIC_LABELS, means, yerr=stds, capsize=4, color=colors)
ax.set_title('Avg. Across Folds', fontweight='bold')
ax.set_ylim(bottom=0)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{m:.4f}', ha='center', va='bottom', fontsize=7)
ax.tick_params(axis='x', rotation=45)

plt.suptitle('DEFEND Baseline — 3-Fold CV (v2-consistent metrics)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_barplots_baseline.png', dpi=150, bbox_inches='tight')
print('Saved cv_barplots_baseline.png')


# Write results to text file for easy reference
with open('baseline_results.txt', 'w') as f:
    f.write('DEFEND BASELINE — 3-Fold CV Results (v2-consistent metrics)\n')
    f.write(f'Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n')
    f.write(f'RAM: {ram_mb:.0f} MB\n')
    f.write(f'Params: 173,675\n\n')

    for fold_num in range(NUM_SPLITS):
        f.write(f'Fold {fold_num+1}:\n')
        for k in METRIC_NAMES:
            f.write(f'  {k}: {all_fold_metrics[fold_num][k]:.6f}\n')
        f.write('\n')

    f.write('Average ± Std:\n')
    for k, label in zip(METRIC_NAMES, METRIC_LABELS):
        vals = [m[k] for m in all_fold_metrics]
        f.write(f'  {label}: {np.mean(vals):.6f} ± {np.std(vals):.6f}\n')

print('Saved baseline_results.txt')
print('\nDone.')
