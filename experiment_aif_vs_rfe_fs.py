# experiment_aif_vs_rfe_fs.py
"""
Compare Original AIF vs AIF + RFE (L1 Logistic) FS with 10% label budget
Full metrics, graphs, and detailed analysis
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.preprocessing import LabelEncoder

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest as OriginalAIF

# Your RFE version
from capymoa.anomaly.adaptive_isolation_forest_rfe_fs import AdaptiveIsolationForestWithRFEFS as AIFRFEFS

# --------------------------------------------------
# Resource monitor
# --------------------------------------------------
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# NPZ Stream
# --------------------------------------------------
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"]
        y = data["y"]

        self.df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.df["target"] = y

        self.le = LabelEncoder()
        self.df["target_idx"] = self.le.fit_transform(self.df["target"]).astype(np.int32)

        self.n = len(self.df)
        self.i = 0

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        target_name = "target"
        categories = {target_name: [str(c) for c in self.le.classes_]}

        self.schema = Schema.from_custom(
            features=feature_names + [target_name],
            target=target_name,
            categories=categories,
            name=path
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        row = self.df.iloc[self.i]
        x_features = row.drop(["target", "target_idx"]).values.astype(np.float64)
        label_idx = int(row["target_idx"])
        inst = Instance.from_array(self.schema, np.append(x_features, label_idx))
        self.i += 1
        return inst, label_idx

    def get_schema(self):
        return self.schema

# --------------------------------------------------
# Run experiment
# --------------------------------------------------
dataset_folder = "./semi_supervised_Datasets"
datasets = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".npz")])

results = []
roc_data = []

for path in datasets:
    ds_name = os.path.basename(path)
    print(f"\n================ {ds_name} ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    aif = OriginalAIF(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_rfe = AIFRFEFS(schema=schema, window_size=256, n_trees=50, seed=42, l1_c=0.01, label_budget=0.10)

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_rfe = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_rfe, labels = [], [], []

    start_time = time.time()
    start_mem = memory_mb()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        labels.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_rfe = aif_rfe.score_instance(inst)

        scores_aif.append(s_aif)
        scores_rfe.append(s_rfe)

        eval_aif.update(label_idx, s_aif)
        eval_rfe.update(label_idx, s_rfe)

        aif.train(inst)
        aif_rfe.train(inst, label=label_idx)  # Pass label for semi-supervised budget

    runtime = time.time() - start_time
    memory_delta = memory_mb() - start_mem

    labels_arr = np.array(labels)
    scores_aif_arr = np.array(scores_aif)
    scores_rfe_arr = np.array(scores_rfe)

    # ROC + AUC + store
    fpr_aif, tpr_aif, _ = roc_curve(labels_arr, scores_aif_arr)
    auc_aif_val = auc(fpr_aif, tpr_aif)

    fpr_rfe, tpr_rfe, _ = roc_curve(labels_arr, scores_rfe_arr)
    auc_rfe_val = auc(fpr_rfe, tpr_rfe)

    roc_data.append({
        'dataset': ds_name,
        'fpr_aif': fpr_aif,
        'tpr_aif': tpr_aif,
        'auc_aif': auc_aif_val,
        'fpr_rfe': fpr_rfe,
        'tpr_rfe': tpr_rfe,
        'auc_rfe': auc_rfe_val
    })

    # Average Precision
    ap_aif_val = average_precision_score(labels_arr, scores_aif_arr)
    ap_rfe_val = average_precision_score(labels_arr, scores_rfe_arr)

    # Precision @ top-K
    k_ratios = [0.01, 0.02, 0.05]
    prec_k = {}
    for k_ratio in k_ratios:
        k = max(5, int(len(labels_arr) * k_ratio))
        topk_aif = np.argsort(-scores_aif_arr)[:k]
        topk_rfe = np.argsort(-scores_rfe_arr)[:k]
        prec_k[f'Prec@{k_ratio*100:.0f}%_AIF'] = np.mean(labels_arr[topk_aif] == 1)
        prec_k[f'Prec@{k_ratio*100:.0f}%_RFE'] = np.mean(labels_arr[topk_rfe] == 1)

    anomaly_ratio = np.mean(labels_arr == 1) if np.any(labels_arr == 1) else 0.0
    selected_last = getattr(aif_rfe, 'last_selected_count', None)

    results.append({
        'dataset': ds_name,
        'AUC_AIF': auc_aif_val,
        'AUC_RFE': auc_rfe_val,
        'AP_AIF': ap_aif_val,
        'AP_RFE': ap_rfe_val,
        'runtime_s': runtime,
        'memory_MB': memory_delta,
        'anomaly_ratio': anomaly_ratio,
        'last_selected_features': selected_last,
        **prec_k
    })

    print(f"  AUC AIF:       {auc_aif_val:.4f}")
    print(f"  AUC RFE:       {auc_rfe_val:.4f}")
    print(f"  AP AIF:        {ap_aif_val:.4f}")
    print(f"  AP RFE:        {ap_rfe_val:.4f}")
    print(f"  Runtime:       {runtime:.2f} s")
    print(f"  Memory Δ:      {memory_delta:+.2f} MB")

# --------------------------------------------------
# Summary & Graphs
# --------------------------------------------------
df = pd.DataFrame(results)
df['AUC_diff'] = df['AUC_AIF'] - df['AUC_RFE']
df['AP_diff'] = df['AP_AIF'] - df['AP_RFE']

print("\n" + "="*80)
print("                  FINAL COMPARISON SUMMARY (RFE + 10% Label Budget)")
print("="*80)
print(df.round(4))

df.to_csv("aif_vs_rfe_fs_results.csv", index=False)
print("\nResults saved to: aif_vs_rfe_fs_results.csv")

# ── 1. AUC Bar Chart ────────────────────────────────────────────────────────
plt.figure(figsize=(14, 7))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df['AUC_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['AUC_RFE'], width, label='AIF + RFE (10% budget)', color='#ff7f0e')

plt.xlabel('Dataset')
plt.ylabel('AUC')
plt.title('AUC Comparison')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auc_comparison_rfe.png', dpi=300)
plt.close()
print("Saved: auc_comparison_rfe.png")

# ── 2. Precision@1% Bar Chart ───────────────────────────────────────────────
plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['Prec@1%_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['Prec@1%_RFE'], width, label='AIF + RFE (10% budget)', color='#ff7f0e')

plt.xlabel('Dataset')
plt.ylabel('Precision @ Top 1%')
plt.title('Precision @ Top 1% Comparison')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('precision_at_1pct_rfe.png', dpi=300)
plt.close()
print("Saved: precision_at_1pct_rfe.png")

# ── 3. Individual ROC Curves ────────────────────────────────────────────────
print("\nGenerating ROC curves...")
for roc in roc_data:
    plt.figure(figsize=(8, 6))
    plt.plot(roc['fpr_aif'], roc['tpr_aif'], label=f"Original AIF (AUC={roc['auc_aif']:.4f})", color='#1f77b4')
    plt.plot(roc['fpr_rfe'], roc['tpr_rfe'], label=f"AIF + RFE (AUC={roc['auc_rfe']:.4f})", color='#ff7f0e')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {roc["dataset"]}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = roc['dataset'].replace('.npz', '').replace('-', '_')
    plt.savefig(f'roc_{safe_name}_rfe.png', dpi=300)
    plt.close()

print("All ROC curves saved as roc_*_rfe.png")
print("\nExperiment complete!")