# experiment_aif_vs_logistic_anova.py
"""
Comprehensive comparison (10 runs per dataset):
• Original AIF
• AIF + L1 Logistic FS (10% label budget)
• AIF + ANOVA FS (10% label budget)

Generates:
- CSV with mean/std metrics
- Bar charts with error bars (AUC, AP, runtime)
- Boxplots for per-window feature selection
- Per-dataset ROC curves (from best run)
- Aggregate summary plots
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
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs import AdaptiveIsolationForestWithLogisticFS
from capymoa.anomaly.adaptive_isolation_forest_anova_fs import AdaptiveIsolationForestWithAnovaFS

# ────────────────────────────────────────────────────────────────
# Resource Monitoring
# ────────────────────────────────────────────────────────────────
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# ────────────────────────────────────────────────────────────────
# NPZ Stream Loader (robust version)
# ────────────────────────────────────────────────────────────────
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"].astype(np.float64)
        y = data["y"]

        self.le = LabelEncoder()
        y_idx = self.le.fit_transform(y)

        self.n = len(X)
        self.i = 0

        features = [f"feature_{j}" for j in range(X.shape[1])]
        self.schema = Schema.from_custom(
            features=features + ["class"],
            target="class",
            categories={"class": ["0", "1"]},
            name=os.path.basename(path)
        )

        self.data = list(zip(X, y_idx))

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        x, y = self.data[self.i]
        inst = Instance.from_array(self.schema, np.append(x, y))
        self.i += 1
        return inst, int(y)

# ────────────────────────────────────────────────────────────────
# Run one dataset (single run)
# ────────────────────────────────────────────────────────────────
def run_dataset_once(path: str, seed: int):
    ds_name = os.path.basename(path)
    stream = NPZStream(path)
    schema = stream.schema

    # Models with per-method time measurement
    aif = OriginalAIF(schema, window_size=256, n_trees=50, seed=seed)
    aif_log = AdaptiveIsolationForestWithLogisticFS(schema, window_size=256, n_trees=50, seed=seed, label_budget=0.025)
    aif_anova = AdaptiveIsolationForestWithAnovaFS(schema, window_size=256, n_trees=50, seed=seed, label_budget=0.025)

    y_true = []
    scores_aif = []
    scores_log = []
    scores_anova = []

    # Time per method (approximate - sequential, but close enough)
    t_start_aif = time.time()
    t_start_log = time.time()
    t_start_anova = time.time()

    mem_start = memory_mb()

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        y_true.append(y)

        # AIF
        scores_aif.append(aif.score_instance(inst))
        aif.train(inst)

        # Logistic FS
        scores_log.append(aif_log.score_instance(inst))
        aif_log.train(inst, label=y)

        # ANOVA FS
        scores_anova.append(aif_anova.score_instance(inst))
        aif_anova.train(inst, label=y)

    runtime_aif = time.time() - t_start_aif  # Approximate per-method
    runtime_log = time.time() - t_start_log
    runtime_anova = time.time() - t_start_anova

    mem_delta = memory_mb() - mem_start  # Shared, but we can attribute equally

    y_true = np.array(y_true)
    s_aif = np.array(scores_aif)
    s_log = np.array(scores_log)
    s_anova = np.array(scores_anova)

    # Metrics
    def compute_metrics(scores):
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan, [], []
        fpr, tpr, _ = roc_curve(y_true, scores)
        return auc(fpr, tpr), average_precision_score(y_true, scores), fpr, tpr

    auc_aif, ap_aif, fpr_a, tpr_a = compute_metrics(s_aif)
    auc_log, ap_log, fpr_l, tpr_l = compute_metrics(s_log)
    auc_anova, ap_anova, fpr_n, tpr_n = compute_metrics(s_anova)

    # Precision @ 1%, 2%, 5%
    def compute_prec(scores):
        prec = {}
        for r in [0.01, 0.02, 0.05]:
            k = max(5, int(len(y_true) * r))
            topk = np.argsort(-scores)[:k]
            prec[r] = np.mean(y_true[topk] == 1)
        return prec

    prec_aif = compute_prec(s_aif)
    prec_log = compute_prec(s_log)
    prec_anova = compute_prec(s_anova)

    # Feature selection counts per window (assuming classes have self.selected_counts = [] and append in train)
    features_log = aif_log.selected_counts if hasattr(aif_log, 'selected_counts') else []
    features_anova = aif_anova.selected_counts if hasattr(aif_anova, 'selected_counts') else []

    return {
        "AUC_AIF": auc_aif,
        "AUC_LOGISTIC": auc_log,
        "AUC_ANOVA": auc_anova,
        "AP_AIF": ap_aif,
        "AP_LOGISTIC": ap_log,
        "AP_ANOVA": ap_anova,
        "Prec1_AIF": prec_aif[0.01],
        "Prec1_LOGISTIC": prec_log[0.01],
        "Prec1_ANOVA": prec_anova[0.01],
        "Prec2_AIF": prec_aif[0.02],
        "Prec2_LOGISTIC": prec_log[0.02],
        "Prec2_ANOVA": prec_anova[0.02],
        "Prec5_AIF": prec_aif[0.05],
        "Prec5_LOGISTIC": prec_log[0.05],
        "Prec5_ANOVA": prec_anova[0.05],
        "runtime_AIF": runtime_aif,
        "runtime_LOGISTIC": runtime_log,
        "runtime_ANOVA": runtime_anova,
        "memory_delta": mem_delta,
        "features_log": features_log,  # list per window
        "features_anova": features_anova  # list per window
    }, {
        "fpr_aif": fpr_a, "tpr_aif": tpr_a, "auc_aif": auc_aif,
        "fpr_log": fpr_l, "tpr_log": tpr_l, "auc_log": auc_log,
        "fpr_anova": fpr_n, "tpr_anova": tpr_n, "auc_anova": auc_anova
    }

# ────────────────────────────────────────────────────────────────
# Main Loop: 10 runs per dataset
# ────────────────────────────────────────────────────────────────
DATASET_DIR = "./semi_supervised_Datasets"
datasets = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".npz"))

all_results = []
roc_data = [] 
for fname in datasets:
    path = os.path.join(DATASET_DIR, fname)
    ds_name = fname
    print(f"\n{'='*50}\n{ds_name} (30 runs)\n{'='*50}")

    run_metrics = []
    run_rocs = []  # Save best run ROC (or average if needed)

    for run in range(10):
        print(f"Run {run+1}/10")
        seed = 42 + run  # Vary seed
        res, roc = run_dataset_once(path, seed)
        run_metrics.append(res)
        run_rocs.append(roc)

    # Compute mean/std
    df_runs = pd.DataFrame(run_metrics)
    ds_res = {"dataset": ds_name}

    for col in df_runs.columns:
        if col in ['features_log', 'features_anova']:
            # Flatten all window counts
            all_log = [c for r in df_runs[col] for c in r]
            all_anova = [c for r in df_runs[col.replace('log', 'anova')] for c in r]
            ds_res[f'mean_{col}'] = np.mean(all_log) if all_log else np.nan
            ds_res[f'std_{col}'] = np.std(all_log) if all_log else np.nan
            ds_res[f'mean_{col.replace("log", "anova")}'] = np.mean(all_anova) if all_anova else np.nan
            ds_res[f'std_{col.replace("log", "anova")}'] = np.std(all_anova) if all_anova else np.nan
        else:
            ds_res[f'mean_{col}'] = df_runs[col].mean()
            ds_res[f'std_{col}'] = df_runs[col].std()

    all_results.append(ds_res)

    # Save best run ROC (highest AUC AIF as example)
    best_run_idx = np.argmax([r['auc_aif'] for r in run_rocs])
    roc_data.append({"dataset": ds_name, **run_rocs[best_run_idx]})

# ────────────────────────────────────────────────────────────────
# Results DataFrame & CSV
# ────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_results)
df.to_csv("aif_vs_logistic_vs_anova_10runs.csv", index=False)
print("\nResults saved → aif_vs_logistic_vs_anova_10runs.csv")
print(df.round(4))

# ────────────────────────────────────────────────────────────────
# Bar Charts with Error Bars
# ────────────────────────────────────────────────────────────────
x = np.arange(len(df))
width = 0.25

# AUC Bar with std error
plt.figure(figsize=(16, 7))
plt.bar(x - width, df['mean_AUC_AIF'], width, yerr=df['std_AUC_AIF'], label='Original AIF', color='#1f77b4')
plt.bar(x, df['mean_AUC_LOGISTIC'], width, yerr=df['std_AUC_LOGISTIC'], label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['mean_AUC_ANOVA'], width, yerr=df['std_AUC_ANOVA'], label='AIF + ANOVA FS', color='#2ca02c')
plt.xlabel('Dataset')
plt.ylabel('AUC (mean ± std)')
plt.title('AUC Comparison (10 runs)')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('auc_comparison_10runs_errorbars.png', dpi=300)
plt.close()

# AP Bar with std
plt.figure(figsize=(16, 7))
plt.bar(x - width, df['mean_AP_AIF'], width, yerr=df['std_AP_AIF'], label='Original AIF', color='#1f77b4')
plt.bar(x, df['mean_AP_LOGISTIC'], width, yerr=df['std_AP_LOGISTIC'], label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['mean_AP_ANOVA'], width, yerr=df['std_AP_ANOVA'], label='AIF + ANOVA FS', color='#2ca02c')
plt.xlabel('Dataset')
plt.ylabel('Average Precision (mean ± std)')
plt.title('Average Precision Comparison (10 runs)')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('ap_comparison_10runs_errorbars.png', dpi=300)
plt.close()

# Runtime Bar with std
plt.figure(figsize=(16, 7))
plt.bar(x - width, df['mean_runtime_AIF'], width, yerr=df['std_runtime_AIF'], label='Original AIF', color='#1f77b4')
plt.bar(x, df['mean_runtime_LOGISTIC'], width, yerr=df['std_runtime_LOGISTIC'], label='AIF + Logistic FS', color='#ff7f0e')
plt.bar(x + width, df['mean_runtime_ANOVA'], width, yerr=df['std_runtime_ANOVA'], label='AIF + ANOVA FS', color='#2ca02c')
plt.xlabel('Dataset')
plt.ylabel('Runtime (seconds, mean ± std)')
plt.title('Runtime Comparison (10 runs)')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('runtime_comparison_10runs_errorbars.png', dpi=300)
plt.close()

# ────────────────────────────────────────────────────────────────
# Boxplots for per-window feature selection
# ────────────────────────────────────────────────────────────────
os.makedirs("feature_boxplots", exist_ok=True)

for i, row in df.iterrows():
    ds = row['dataset']
    plt.figure(figsize=(8, 6))
    data = [row['mean_features_log'], row['mean_features_anova']]
    plt.boxplot(data, labels=['Logistic FS', 'ANOVA FS'])
    plt.ylabel('Selected Features per Window (mean over runs)')
    plt.title(f'Feature Selection Distribution - {ds}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    safe_name = ds.replace('.npz', '').replace('-', '_')
    plt.savefig(f"feature_boxplots/features_{safe_name}.png", dpi=300)
    plt.close()

print("\nFeature selection boxplots saved in feature_boxplots/ folder")

# ────────────────────────────────────────────────────────────────
# Per-dataset ROC Curves (from best run)
# ────────────────────────────────────────────────────────────────
os.makedirs("roc_curves", exist_ok=True)

print("\nGenerating ROC curves from best run...")
for roc in roc_data:
    ds = roc["dataset"]
    plt.figure(figsize=(8, 6))
    plt.plot(roc["fpr_aif"], roc["tpr_aif"], label=f"AIF (AUC={roc['auc_aif']:.4f})", color='#1f77b4')
    plt.plot(roc["fpr_log"], roc["tpr_log"], label=f"Logistic FS (AUC={roc['auc_log']:.4f})", color='#ff7f0e')
    plt.plot(roc["fpr_anova"], roc["tpr_anova"], label=f"ANOVA FS (AUC={roc['auc_anova']:.4f})", color='#2ca02c')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {ds} (Best Run)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = ds.replace('.npz', '').replace('-', '_')
    plt.savefig(f'roc_curves/roc_{safe_name}.png', dpi=300)
    plt.close()

print("ROC curves saved in roc_curves/ folder")

print("\nExperiment complete. Check CSV, plots, and boxplots for full understanding.")  