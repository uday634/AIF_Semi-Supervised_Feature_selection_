# multi_experiment_npz_final_improved.py

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest
from capymoa.anomaly.adaptive_isolation_forest_fs import AdaptiveIsolationForestFS

# --------------------------------------------------
# Resource monitor
# --------------------------------------------------
process = psutil.Process()
def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# NPZ Stream (returns instance + label index)
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
# Find all datasets
# --------------------------------------------------
dataset_folder = "./semi_supervised_Datasets"
datasets = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".npz")]
datasets.sort()  # nicer order

# --------------------------------------------------
# Results storage
# --------------------------------------------------
results = []

# --------------------------------------------------
# Run experiments
# --------------------------------------------------
for npz_path in datasets:
    ds_name = os.path.basename(npz_path)
    print(f"\n================ {ds_name} ================")
    
    stream = NPZStream(npz_path)
    schema = stream.get_schema()
    
    aif = AdaptiveIsolationForest(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_fs = AdaptiveIsolationForestFS(schema=schema, window_size=256, n_trees=50, seed=42)
    
    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_fs = AnomalyDetectionEvaluator(schema)
    
    scores_aif, scores_fs, labels = [], [], []
    
    # Memory tracking for both models
    start_mem = memory_mb()
    start_time = time.time()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        labels.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_fs  = aif_fs.score_instance(inst)

        scores_aif.append(s_aif)
        scores_fs.append(s_fs)

        eval_aif.update(label_idx, s_aif)
        eval_fs.update(label_idx, s_fs)

        aif.train(inst)
        aif_fs.train(inst)

    runtime = time.time() - start_time
    final_mem = memory_mb()
    mem_delta = final_mem - start_mem

    # ── Conversions ────────────────────────────────────────────────
    labels     = np.array(labels)
    scores_aif = np.array(scores_aif)
    scores_fs  = np.array(scores_fs)

    # Basic statistics
    anomaly_ratio = np.mean(labels == 1) if 1 in labels else 0.0
    n_anomalies   = np.sum(labels == 1)
    n_normal      = len(labels) - n_anomalies

    # AUC & PR-AUC
    fpr_aif, tpr_aif, _ = roc_curve(labels, scores_aif)
    auc_aif = auc(fpr_aif, tpr_aif)

    fpr_fs, tpr_fs, _ = roc_curve(labels, scores_fs)
    auc_fs = auc(fpr_fs, tpr_fs)

    # Precision-Recall (more informative for imbalanced data)
    prec_aif, rec_aif, _ = precision_recall_curve(labels, scores_aif)
    ap_aif = average_precision_score(labels, scores_aif)

    prec_fs, rec_fs, _ = precision_recall_curve(labels, scores_fs)
    ap_fs = average_precision_score(labels, scores_fs)

    # Top-K metrics (e.g. top 1%, 2%, 5%)
    k_ratios = [0.01, 0.02, 0.05]
    topk_metrics = {}
    for k_ratio in k_ratios:
        k = max(10, int(len(labels) * k_ratio))
        topk_idx_aif = np.argsort(-scores_aif)[:k]
        topk_idx_fs  = np.argsort(-scores_fs)[:k]
        
        prec_at_k_aif = np.mean(labels[topk_idx_aif] == 1)
        prec_at_k_fs  = np.mean(labels[topk_idx_fs] == 1)
        
        topk_metrics[f'Precision@{k_ratio*100:.0f}%'] = {
            'AIF': prec_at_k_aif,
            'AIF_FS': prec_at_k_fs
        }

    # Store everything
    results.append({
        "dataset": ds_name,
        "n_instances": len(labels),
        "anomaly_ratio": anomaly_ratio,
        "n_anomalies": n_anomalies,
        "AUC_AIF": auc_aif,
        "AUC_AIF_FS": auc_fs,
        "AP_AIF": ap_aif,
        "AP_AIF_FS": ap_fs,
        "runtime_s": runtime,
        "memory_MB": mem_delta,
        **{f"{k}_AIF": v['AIF'] for k,v in topk_metrics.items()},
        **{f"{k}_AIF_FS": v['AIF_FS'] for k,v in topk_metrics.items()},
        # For later analysis
        "labels": labels,
        "scores_aif": scores_aif,
        "scores_fs": scores_fs,
    })

    print(f"  Instances:      {len(labels):,}")
    print(f"  Anomaly ratio:  {anomaly_ratio:.4f}  ({n_anomalies} anomalies)")
    print(f"  AUC     AIF:    {auc_aif:.4f}   AIF+FS: {auc_fs:.4f}")
    print(f"  AP      AIF:    {ap_aif:.4f}   AIF+FS: {ap_fs:.4f}")
    print(f"  Runtime:        {runtime:.2f}s")
    print(f"  Memory Δ:       {mem_delta:+.2f} MB")

# --------------------------------------------------
# Create enhanced summary
# --------------------------------------------------
df = pd.DataFrame(results)

# Reorder & select interesting columns
summary_cols = [
    "dataset", "n_instances", "anomaly_ratio", "n_anomalies",
    "AUC_AIF", "AUC_AIF_FS", "AP_AIF", "AP_AIF_FS",
    "Precision@1%_AIF", "Precision@1%_AIF_FS",
    "Precision@2%_AIF", "Precision@2%_AIF_FS",
    "Precision@5%_AIF", "Precision@5%_AIF_FS",
    "runtime_s", "memory_MB"
]

df_summary = df[summary_cols].copy()

# Add difference columns (positive = AIF better)
df_summary["AUC_diff"]   = df_summary["AUC_AIF"] - df_summary["AUC_AIF_FS"]
df_summary["AP_diff"]    = df_summary["AP_AIF"]  - df_summary["AP_AIF_FS"]
df_summary["Mem_Δ_MB"]   = df_summary["memory_MB"]

print("\n" + "="*70)
print("               FINAL SUMMARY & DIAGNOSTICS")
print("="*70)
print(df_summary.round(4).to_string(index=False))

# Save to CSV for later analysis
df_summary.to_csv("anomaly_detection_comparison_summary.csv", index=False)
print("\nSummary saved → anomaly_detection_comparison_summary.csv")

print("\nDone. Use the saved CSV + plots for deeper analysis.")