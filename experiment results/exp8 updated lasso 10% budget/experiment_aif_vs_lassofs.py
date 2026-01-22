# experiment_aif_vs_semisup_lasso_fs.py
"""
Semi-supervised comparison: Original AIF vs AIF with LassoFS (10% label budget)
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

# Your semi-supervised LassoFS version
from capymoa.anomaly.adaptive_isolation_forest_lasso_fs import AdaptiveIsolationForestWithLassoFS as AIFLassoFS

# --------------------------------------------------
# Resource monitor
# --------------------------------------------------
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# NPZ Stream (returns instance + label for semi-sup)
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
        return inst, label_idx  # Return label for semi-sup training

    def get_schema(self):
        return self.schema

# --------------------------------------------------
# Run experiment with 10% label budget
# --------------------------------------------------
dataset_folder = "./semi_supervised_Datasets"
datasets = sorted([os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".npz")])

results = []
roc_data = []

LABEL_BUDGET = 0.10  # 10% - change here if needed

for path in datasets:
    ds_name = os.path.basename(path)
    print(f"\n================ {ds_name} (10% label budget) ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    aif = OriginalAIF(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_lasso = AIFLassoFS(schema=schema, window_size=256, n_trees=50, seed=42, lasso_alpha=0.01, label_budget=LABEL_BUDGET)

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_lasso = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_lasso, labels = [], [], []

    start_time = time.time()
    start_mem = memory_mb()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        labels.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_lasso = aif_lasso.score_instance(inst)

        scores_aif.append(s_aif)
        scores_lasso.append(s_lasso)

        eval_aif.update(label_idx, s_aif)
        eval_lasso.update(label_idx, s_lasso)

        # Train: pass label to semi-supervised version
        aif.train(inst)
        aif_lasso.train(inst, label=label_idx)

    runtime = time.time() - start_time
    memory_delta = memory_mb() - start_mem

    labels_arr = np.array(labels)
    scores_aif_arr = np.array(scores_aif)
    scores_lasso_arr = np.array(scores_lasso)

    # ROC + AUC + store
    fpr_aif, tpr_aif, _ = roc_curve(labels_arr, scores_aif_arr)
    auc_aif_val = auc(fpr_aif, tpr_aif)

    fpr_lasso, tpr_lasso, _ = roc_curve(labels_arr, scores_lasso_arr)
    auc_lasso_val = auc(fpr_lasso, tpr_lasso)

    roc_data.append({
        'dataset': ds_name,
        'fpr_aif': fpr_aif,
        'tpr_aif': tpr_aif,
        'auc_aif': auc_aif_val,
        'fpr_lasso': fpr_lasso,
        'tpr_lasso': tpr_lasso,
        'auc_lasso': auc_lasso_val
    })

    # Average Precision
    ap_aif_val = average_precision_score(labels_arr, scores_aif_arr)
    ap_lasso_val = average_precision_score(labels_arr, scores_lasso_arr)

    # Precision @ top-K
    k_ratios = [0.01, 0.02, 0.05]
    prec_k = {}
    for k_ratio in k_ratios:
        k = max(5, int(len(labels_arr) * k_ratio))
        topk_aif = np.argsort(-scores_aif_arr)[:k]
        topk_lasso = np.argsort(-scores_lasso_arr)[:k]
        prec_k[f'Prec@{k_ratio*100:.0f}%_AIF'] = np.mean(labels_arr[topk_aif] == 1)
        prec_k[f'Prec@{k_ratio*100:.0f}%_LassoFS'] = np.mean(labels_arr[topk_lasso] == 1)

    anomaly_ratio = np.mean(labels_arr == 1) if np.any(labels_arr == 1) else 0.0
    selected_last = getattr(aif_lasso, 'last_selected_count', None)

    results.append({
        'dataset': ds_name,
        'AUC_AIF': auc_aif_val,
        'AUC_LassoFS': auc_lasso_val,
        'AP_AIF': ap_aif_val,
        'AP_LassoFS': ap_lasso_val,
        'runtime_s': runtime,
        'memory_MB': memory_delta,
        'anomaly_ratio': anomaly_ratio,
        'last_selected_features': selected_last,
        **prec_k
    })

    print(f"  AUC AIF:       {auc_aif_val:.4f}")
    print(f"  AUC LassoFS:   {auc_lasso_val:.4f}")
    print(f"  AP  AIF:       {ap_aif_val:.4f}")
    print(f"  AP  LassoFS:   {ap_lasso_val:.4f}")
    print(f"  Runtime:       {runtime:.2f} s")
    print(f"  Memory Δ:      {memory_delta:+.2f} MB")

# --------------------------------------------------
# Summary & Graphs
# --------------------------------------------------
df = pd.DataFrame(results)
df['AUC_diff'] = df['AUC_AIF'] - df['AUC_LassoFS']
df['AP_diff']  = df['AP_AIF']  - df['AP_LassoFS']

print("\n" + "="*80)
print("    FINAL COMPARISON SUMMARY (10% Label Budget)")
print("="*80)
print(df.round(4))

df.to_csv("aif_vs_semisup_lasso_fs_10pct.csv", index=False)
print("\nResults saved to: aif_vs_semisup_lasso_fs_10pct.csv")

# AUC Bar Chart
plt.figure(figsize=(14, 7))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df['AUC_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['AUC_LassoFS'], width, label='AIF + SemiSup LassoFS (10%)', color='#ff7f0e')

plt.xlabel('Dataset')
plt.ylabel('AUC')
plt.title('AUC Comparison (10% Label Budget)')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auc_comparison_10pct.png', dpi=300)
plt.close()
print("Saved: auc_comparison_10pct.png")

# Precision@1% Bar Chart
plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df['Prec@1%_AIF'], width, label='Original AIF', color='#1f77b4')
plt.bar(x + width/2, df['Prec@1%_LassoFS'], width, label='AIF + SemiSup LassoFS (10%)', color='#ff7f0e')

plt.xlabel('Dataset')
plt.ylabel('Precision @ Top 1%')
plt.title('Precision @ Top 1% Comparison (10% Label Budget)')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('precision_at_1pct_10pct.png', dpi=300)
plt.close()
print("Saved: precision_at_1pct_10pct.png")

# Individual ROC Curves
print("\nGenerating ROC curves...")
for roc in roc_data:
    plt.figure(figsize=(8, 6))
    plt.plot(roc['fpr_aif'], roc['tpr_aif'], label=f"Original AIF (AUC={roc['auc_aif']:.4f})", color='#1f77b4')
    plt.plot(roc['fpr_lasso'], roc['tpr_lasso'], label=f"AIF + SemiSup LassoFS (AUC={roc['auc_lasso']:.4f})", color='#ff7f0e')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {roc["dataset"]}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    safe_name = roc['dataset'].replace('.npz', '').replace('-', '_')
    plt.savefig(f'roc_{safe_name}_10pct.png', dpi=300)
    plt.close()

print("All ROC curves saved as roc_*_10pct.png")
print("\nExperiment complete! All files saved.")