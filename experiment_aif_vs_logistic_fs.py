# experiment_aif_vs_logistic_fs.py
"""
Comprehensive comparison: Original AIF vs AIF + L1 Logistic FS (10% label budget)
Generates CSV, bar charts, ROC curves, runtime & memory stats
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs_with_buffer import (
    AdaptiveIsolationForestWithLogisticFS
)

# ────────────────────────────────────────────────────────────────
# Resource monitoring
# ────────────────────────────────────────────────────────────────
process = psutil.Process()

def get_memory_mb():
    """Current memory usage in MB"""
    return process.memory_info().rss / (1024 * 1024)

# ────────────────────────────────────────────────────────────────
# NPZ Stream Reader (fixed version)
# ────────────────────────────────────────────────────────────────
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        self.X = data["X"].astype(np.float64)
        self.y = data["y"]

        self.le = LabelEncoder()
        self.y_idx = self.le.fit_transform(self.y).astype(np.int32)

        self.n = len(self.X)
        self.i = 0

        feature_names = [f"feature_{j}" for j in range(self.X.shape[1])]
        self.schema = Schema.from_custom(
            features=feature_names + ["class"],
            target="class",
            categories={"class": ["0", "1"]},
            name=os.path.basename(path)
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        x = self.X[self.i]
        label_idx = int(self.y_idx[self.i])
        inst = Instance.from_array(self.schema, np.append(x, label_idx))
        self.i += 1
        return inst, label_idx

# ────────────────────────────────────────────────────────────────
# Run one dataset
# ────────────────────────────────────────────────────────────────
def run_dataset(path: str):
    ds_name = os.path.basename(path)
    print(f"\n{'='*40}\n{ds_name}\n{'='*40}")

    stream = NPZStream(path)
    schema = stream.schema

    # Models
    aif = AdaptiveIsolationForest(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_l1 = AdaptiveIsolationForestWithLogisticFS(schema=schema, window_size=256, n_trees=50, seed=42)

    # Storage
    y_true = []
    scores_aif = []
    scores_l1 = []

    start_time = time.time()
    mem_start = get_memory_mb()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        y_true.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_l1 = aif_l1.score_instance(inst)

        scores_aif.append(s_aif)
        scores_l1.append(s_l1)

        aif.train(inst)
        aif_l1.train(inst, label=label_idx)

    runtime = time.time() - start_time
    mem_delta = get_memory_mb() - mem_start

    y_true = np.array(y_true)
    scores_aif = np.array(scores_aif)
    scores_l1 = np.array(scores_l1)

    # ── Metrics ──────────────────────────────────────────────────────
    auc_aif = roc_auc_score(y_true, scores_aif) if len(np.unique(y_true)) > 1 else np.nan
    auc_l1  = roc_auc_score(y_true, scores_l1)  if len(np.unique(y_true)) > 1 else np.nan

    ap_aif = average_precision_score(y_true, scores_aif)
    ap_l1  = average_precision_score(y_true, scores_l1)

    # Precision @ 1%, 2%, 5%
    k_ratios = [0.01, 0.02, 0.05]
    prec = {}
    for r in k_ratios:
        k = max(5, int(len(y_true) * r))
        topk_aif = np.argsort(-scores_aif)[:k]
        topk_l1  = np.argsort(-scores_l1)[:k]
        prec[f'Prec@{r*100:.0f}%_AIF'] = np.mean(y_true[topk_aif] == 1)
        prec[f'Prec@{r*100:.0f}%_L1']  = np.mean(y_true[topk_l1] == 1)

    anomaly_ratio = np.mean(y_true == 1) if np.any(y_true == 1) else 0.0

    return {
        'dataset': ds_name,
        'AUC_AIF': auc_aif,
        'AUC_L1': auc_l1,
        'AP_AIF': ap_aif,
        'AP_L1': ap_l1,
        'runtime_s': runtime,
        'memory_delta_MB': mem_delta,
        'anomaly_ratio': anomaly_ratio,
        **prec
    }, (y_true, scores_aif, scores_l1)

# ────────────────────────────────────────────────────────────────
# Main experiment loop
# ────────────────────────────────────────────────────────────────
DATA_DIR = "./semi_supervised_Datasets"
results = []
roc_data = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npz"):
        continue
    path = os.path.join(DATA_DIR, fname)

    res, roc_tuple = run_dataset(path)
    results.append(res)
    roc_data.append({
        'dataset': res['dataset'],
        'y_true': roc_tuple[0],
        'scores_aif': roc_tuple[1],
        'scores_l1': roc_tuple[2]
    })

# ────────────────────────────────────────────────────────────────
# Create DataFrame & save CSV
# ────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df['AUC_diff'] = df['AUC_L1'] - df['AUC_AIF']
df['AP_diff'] = df['AP_L1'] - df['AP_AIF']
df['Runtime_diff_s'] = df['runtime_s'] - df['runtime_s'].shift(1, fill_value=0)

print("\n" + "="*80)
print("             FINAL COMPARISON SUMMARY")
print("="*80)
print(df.round(4))

df.to_csv("aif_vs_l1_logistic_results.csv", index=False)
print("\nResults saved → aif_vs_l1_logistic_results.csv")

# ────────────────────────────────────────────────────────────────
# Visualization: Bar charts
# ────────────────────────────────────────────────────────────────
x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['AUC_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['AUC_L1'], width, label='AIF + L1 Logistic FS', color='#ff7f0e')
plt.xlabel('Dataset')
plt.ylabel('AUC')
plt.title('AUC Comparison: Original AIF vs AIF + L1 Logistic FS')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auc_comparison_bar.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['AP_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['AP_L1'], width, label='AIF + L1 Logistic FS', color='#ff7f0e')
plt.xlabel('Dataset')
plt.ylabel('Average Precision')
plt.title('Average Precision Comparison')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ap_comparison_bar.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['Prec@1%_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['Prec@1%_L1'], width, label='AIF + L1 Logistic FS', color='#ff7f0e')
plt.xlabel('Dataset')
plt.ylabel('Precision @ Top 1%')
plt.title('Precision @ Top 1% Comparison')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prec_at_1pct_bar.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['runtime_s'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['runtime_s'], width, label='AIF + L1 Logistic FS', color='#ff7f0e')
plt.xlabel('Dataset')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('runtime_bar.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 7))
plt.bar(x, df['memory_delta_MB'], color='#9467bd')
plt.xlabel('Dataset')
plt.ylabel('Memory Delta (MB)')
plt.title('Memory Usage Increase')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('memory_delta_bar.png', dpi=300)
plt.close()

print("\nSaved plots:")
print("  - auc_comparison_bar.png")
print("  - ap_comparison_bar.png")
print("  - prec_at_1pct_bar.png")
print("  - runtime_bar.png")
print("  - memory_delta_bar.png")

# ────────────────────────────────────────────────────────────────
# ROC Curves per dataset
# ────────────────────────────────────────────────────────────────
print("\nGenerating ROC curves...")
for roc in roc_data:
    y_true = roc['y_true']
    s_aif = roc['scores_aif']
    s_l1  = roc['scores_l1']
    ds = roc['dataset']

    fpr_a, tpr_a, _ = roc_curve(y_true, s_aif)
    roc_auc_a = auc(fpr_a, tpr_a)

    fpr_l, tpr_l, _ = roc_curve(y_true, s_l1)
    roc_auc_l = auc(fpr_l, tpr_l)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, color='#1f77b4', lw=2, label=f'AIF (AUC = {roc_auc_a:.4f})')
    plt.plot(fpr_l, tpr_l, color='#ff7f0e', lw=2, label=f'AIF+L1 (AUC = {roc_auc_l:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {ds}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = ds.replace('.npz', '').replace('-', '_')
    plt.savefig(f'roc_{safe_name}.png', dpi=300)
    plt.close()

print("All ROC curves saved as roc_*.png")

print("\nExperiment finished. All results and plots saved.")