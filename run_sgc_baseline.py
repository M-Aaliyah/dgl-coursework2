#!/usr/bin/env python3
"""
SGC Naive Baseline: Brain Graph Super-Resolution
from: https://github.com/basiralab/DGL/tree/main/Tutorials/Tutorial-2

Approach:
  1. Precompute S^K · X  where S = D^{-1/2} A_tilde D^{-1/2}, X = adjacency rows
  2. Flatten into a vector
  3. Single linear layer maps to HR edge vector (35778 dims)
  4. Train with L1 loss
  5. 3-fold CV with identical evaluation to DEFEND-SR v2
"""

import os, sys, time, warnings, random, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

# Path configuration
DATA_DIR = os.environ.get('DATA_DIR', 'data')
sys.path.insert(0, os.environ.get('VECTORIZER_DIR', '.'))
from MatrixVectorizer import MatrixVectorizer

# Reproducibility
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

# Load data
lr_train = pd.read_csv(os.path.join(DATA_DIR, 'lr_train.csv')).values.astype(np.float32)
hr_train = pd.read_csv(os.path.join(DATA_DIR, 'hr_train.csv')).values.astype(np.float32)
lr_test  = pd.read_csv(os.path.join(DATA_DIR, 'lr_test.csv')).values.astype(np.float32)
print(f'LR Train : {lr_train.shape}')
print(f'HR Train : {hr_train.shape}')
print(f'LR Test  : {lr_test.shape}')

lr_train = np.nan_to_num(np.clip(lr_train, 0, None))
hr_train = np.nan_to_num(np.clip(hr_train, 0, None))
lr_test  = np.nan_to_num(np.clip(lr_test,  0, None))

# Constants
N_LR     = 160
N_HR     = 268
N_LR_VEC = 12720
N_HR_VEC = 35778


# Helpers
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



# SGC Precomputation — directly from Tutorial-2
def sgc_precompute(adj, K=2):
    """
    Precompute S^K · X where:
      S = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}   (Tutorial-2 Eq.)
      X = adjacency rows (node features)

    This is the same as Tutorial-2's sgc_precompute but on numpy matrices.
    """
    A = adj + np.eye(adj.shape[0])                       # A_tilde = A + I
    d = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(np.power(d + 1e-8, -0.5))      # D_tilde^{-1/2}
    S = D_inv_sqrt @ A @ D_inv_sqrt                      # Normalized adj

    # S^K · X  where X = adj (node features = adjacency rows)
    X = adj.copy()
    for _ in range(K):
        X = S @ X

    return X.astype(np.float32)



# SGC Model — directly from Tutorial-2 Section 2.1
class SGC(nn.Module):
    """
    Simple Graph Convolution: a single linear layer on precomputed features.
    From Tutorial-2: "the entire training reduces to logistic regression
    on the pre-processed features."

    For brain SR: input = flattened S^K·X, output = HR edge vector.
    """
    def __init__(self, input_dim, output_dim):
        super(SGC, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.W(x)



# Evaluation — identical to DEFEND-SR v2
def compute_all_metrics(pred_vecs, gt_vecs, n_hr=N_HR):
    num_samples = len(pred_vecs)
    pred_mats = anti_vectorize_batch(pred_vecs, n_hr)
    gt_mats   = anti_vectorize_batch(gt_vecs, n_hr)
    pred_mats = postprocess(pred_mats)

    mae_bc, mae_ec, mae_pc = [], [], []
    mae_cc, mae_clust = [], []

    for i in range(num_samples):
        pg = nx.from_numpy_array(pred_mats[i], edge_attr='weight')
        gg = nx.from_numpy_array(gt_mats[i],   edge_attr='weight')

        p_bc = list(nx.betweenness_centrality(pg, weight='weight').values())
        g_bc = list(nx.betweenness_centrality(gg, weight='weight').values())
        mae_bc.append(mean_absolute_error(p_bc, g_bc))

        try:
            p_ec = list(nx.eigenvector_centrality(pg, weight='weight', max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            p_ec = [0.0] * n_hr
        try:
            g_ec = list(nx.eigenvector_centrality(gg, weight='weight', max_iter=500).values())
        except nx.PowerIterationFailedConvergence:
            g_ec = [0.0] * n_hr
        mae_ec.append(mean_absolute_error(p_ec, g_ec))

        p_pc = list(nx.pagerank(pg, weight='weight').values())
        g_pc = list(nx.pagerank(gg, weight='weight').values())
        mae_pc.append(mean_absolute_error(p_pc, g_pc))

        # Closeness centrality — distance=None, matching ours
        p_cc = list(nx.closeness_centrality(pg, distance=None).values())
        g_cc = list(nx.closeness_centrality(gg, distance=None).values())
        mae_cc.append(mean_absolute_error(p_cc, g_cc))

        # Clustering coefficient — matching ours
        p_cl = list(nx.clustering(pg, weight='weight').values())
        g_cl = list(nx.clustering(gg, weight='weight').values())
        mae_clust.append(mean_absolute_error(p_cl, g_cl))

    pred_vecs_pp = vectorize_batch(pred_mats)
    pred_flat = pred_vecs_pp.flatten()
    gt_flat   = gt_vecs.flatten()

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
METRIC_NAMES  = ['MAE', 'PCC', 'JSD', 'MAE_PC', 'MAE_EC', 'MAE_BC', 'MAE_CC', 'MAE_Clust']
METRIC_LABELS = ['MAE', 'PCC', 'JSD', 'MAE(PC)', 'MAE(EC)', 'MAE(BC)', 'MAE(CC)', 'MAE(Clust)']

SGC_K      = 2
LR_RATE    = 1e-3
EPOCHS     = 300
PATIENCE   = 15
BATCH_SIZE = 16

print(f'\nSGC Baseline Config:')
print(f'  K={SGC_K}, lr={LR_RATE}, epochs={EPOCHS}, patience={PATIENCE}, batch_size={BATCH_SIZE}')
print(f'  Input dim: {N_LR * N_LR} (flattened S^K·X)')
print(f'  Output dim: {N_HR_VEC}')
print()

# Precompute SGC features for all training samples
print('Precomputing SGC features for all training samples...')
lr_matrices_all = anti_vectorize_batch(lr_train, N_LR)
sgc_features_all = np.stack([sgc_precompute(m, K=SGC_K) for m in lr_matrices_all])
# Flatten: (167, 160, 160) → (167, 25600)
sgc_flat_all = sgc_features_all.reshape(len(sgc_features_all), -1)
print(f'  SGC features shape: {sgc_flat_all.shape}')

kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)

all_fold_metrics = []
start_time_total = time.time()

for fold_num, (train_idx, test_idx) in enumerate(kf.split(sgc_flat_all)):
    print(f'\n{"="*60}')
    print(f'FOLD {fold_num + 1}/3  (train: {len(train_idx)}, test: {len(test_idx)})')
    print(f'{"="*60}')
    fold_start = time.time()

    X_train = torch.tensor(sgc_flat_all[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(hr_train[train_idx],     dtype=torch.float32, device=device)
    X_test  = torch.tensor(sgc_flat_all[test_idx],   dtype=torch.float32, device=device)
    Y_test  = hr_train[test_idx]  # keep as numpy for evaluation

    # Model: single linear layer, exactly as Tutorial-2 SGC
    model = SGC(input_dim=N_LR * N_LR, output_dim=N_HR_VEC).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    if fold_num == 0:
        print(f'  SGC params: {total_params:,}')

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        # Mini-batch training
        n = len(train_idx)
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()

            pred = model(X_train[batch_idx])
            # Clamp to [0, 1] to match data range
            pred = torch.clamp(pred, 0.0, 1.0)

            loss = F.l1_loss(pred, Y_train[batch_idx])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f} | '
                  f'Best: {best_loss:.6f} | Patience: {patience_counter}/{PATIENCE}')

        if patience_counter >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    # Predict
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_vecs = model(X_test)
        pred_vecs = torch.clamp(pred_vecs, 0.0, 1.0)
        pred_vecs = pred_vecs.cpu().numpy()

    # Evaluate
    metrics = compute_all_metrics(pred_vecs, Y_test, N_HR)
    all_fold_metrics.append(metrics)

    fold_time = time.time() - fold_start
    print(f'\n  Fold {fold_num+1} Results ({fold_time:.1f}s):')
    for k, v in metrics.items():
        print(f'    {k}: {v:.6f}')

    del model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

total_time = time.time() - start_time_total
ram_mb = psutil.Process().memory_info().rss / 1e6


# Summary
print(f'\nTotal 3-fold CV time: {total_time:.1f}s ({total_time/60:.1f} min)')
print(f'RAM usage: {ram_mb:.0f} MB')

print('\n' + '='*70)
print('SGC BASELINE — Average ± Std Across 3 Folds')
print('='*70)
for k, label in zip(METRIC_NAMES, METRIC_LABELS):
    vals = [m[k] for m in all_fold_metrics]
    print(f'  {label:12s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}')


# Bar plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
          '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for f_idx in range(3):
    ax = axes[f_idx]
    vals = [all_fold_metrics[f_idx][k] for k in METRIC_NAMES]
    bars = ax.bar(METRIC_LABELS, vals, color=colors)
    ax.set_title(f'Fold {f_idx+1}', fontweight='bold')
    ax.set_ylim(bottom=0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax.tick_params(axis='x', rotation=45)

ax = axes[3]
means = [np.mean([all_fold_metrics[f][k] for f in range(3)]) for k in METRIC_NAMES]
stds  = [np.std([all_fold_metrics[f][k] for f in range(3)]) for k in METRIC_NAMES]
bars = ax.bar(METRIC_LABELS, means, yerr=stds, capsize=4, color=colors)
ax.set_title('Avg. Across Folds', fontweight='bold')
ax.set_ylim(bottom=0)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{m:.4f}', ha='center', va='bottom', fontsize=7)
ax.tick_params(axis='x', rotation=45)

plt.suptitle('SGC Baseline — 3-Fold CV', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_barplots_sgc.png', dpi=150, bbox_inches='tight')
print('Saved cv_barplots_sgc.png')


# Write results to text file
with open('sgc_results.txt', 'w') as f:
    f.write('SGC BASELINE — 3-Fold CV Results\n')
    f.write(f'Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n')
    f.write(f'RAM: {ram_mb:.0f} MB\n')
    f.write(f'Params: {total_params:,}\n\n')

    for fold_num in range(3):
        f.write(f'Fold {fold_num+1}:\n')
        for k in METRIC_NAMES:
            f.write(f'  {k}: {all_fold_metrics[fold_num][k]:.6f}\n')
        f.write('\n')

    f.write('Average ± Std:\n')
    for k, label in zip(METRIC_NAMES, METRIC_LABELS):
        vals = [m[k] for m in all_fold_metrics]
        f.write(f'  {label}: {np.mean(vals):.6f} ± {np.std(vals):.6f}\n')

print('Saved sgc_results.txt')
print('\nDone.')
