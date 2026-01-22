# experiment_aif_vs_variance_score_fs.py

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

# Your modified version
from capymoa.anomaly.adaptive_isolation_forest_variance_score_fs import AdaptiveIsolationForestWithVarianceScoreFS as AIFVarianceScoreFS

# --------------------------------------------------
# Resource monitor
# --------------------------------------------------
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# NPZ Stream (unchanged)
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

for path in datasets:
    ds_name = os.path.basename(path)
    print(f"\n================ {ds_name} ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    aif = OriginalAIF(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_fs = AIFVarianceScoreFS(schema=schema, window_size=256, n_trees=50, seed=42)

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_fs = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_fs, labels = [], [], []

    start_time = time.time()
    start_mem = memory_mb()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        labels.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_fs = aif_fs.score_instance(inst)

        scores_aif.append(s_aif)
        scores_fs.append(s_fs)

        eval_aif.update(label_idx, s_aif)
        eval_fs.update(label_idx, s_fs)

        aif.train(inst)
        aif_fs.train(inst)

    runtime = time.time() - start_time
    memory_delta = memory_mb() - start_mem

    labels = np.array(labels)
    scores_aif = np.array(scores_aif)
    scores_fs = np.array(scores_fs)

    # Metrics
    anomaly_ratio = np.mean(labels == 1) if np.any(labels == 1) else 0.0

    fpr_aif, tpr_aif, _ = roc_curve(labels, scores_aif)
    auc_aif = auc(fpr_aif, tpr_aif)

    fpr_fs, tpr_fs, _ = roc_curve(labels, scores_fs)
    auc_fs = auc(fpr_fs, tpr_fs)

    ap_aif = average_precision_score(labels, scores_aif)
    ap_fs = average_precision_score(labels, scores_fs)

    # Precision@K
    k_ratios = [0.01, 0.02, 0.05]
    prec_k = {}
    for k_ratio in k_ratios:
        k = max(1, int(len(labels) * k_ratio))
        topk_aif = np.argsort(-scores_aif)[:k]
        topk_fs = np.argsort(-scores_fs)[:k]
        prec_k[f'Prec_{k_ratio*100:.0f}%_AIF'] = np.mean(labels[topk_aif] == 1)
        prec_k[f'Prec_{k_ratio*100:.0f}%_FS'] = np.mean(labels[topk_fs] == 1)

    selected_last = getattr(aif_fs, 'last_selected_count', None)

    results.append({
        'dataset': ds_name,
        'AUC_AIF': auc_aif,
        'AUC_FS': auc_fs,
        'AP_AIF': ap_aif,
        'AP_FS': ap_fs,
        'runtime_s': runtime,
        'memory_MB': memory_delta,
        'anomaly_ratio': anomaly_ratio,
        'last_selected_features': selected_last
    })

    print(f"  AUC AIF: {auc_aif:.4f} | FS: {auc_fs:.4f}")
    print(f"  AP AIF: {ap_aif:.4f} | FS: {ap_fs:.4f}")
    print(f"  Runtime: {runtime:.2f}s | Memory Δ: {memory_delta:.2f} MB")

# --------------------------------------------------
# Summary & Graphs
# --------------------------------------------------
df = pd.DataFrame(results)
df['AUC_diff'] = df['AUC_AIF'] - df['AUC_FS']

print("\n" + "="*60)
print("              FINAL SUMMARY")
print("="*60)
print(df.round(4))

df.to_csv("aif_vs_variance_score_fs_results.csv", index=False)

# AUC Bar Graph
x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df['AUC_AIF'], width, label='Original AIF')
plt.bar(x + width/2, df['AUC_FS'], width, label='AIF + VarianceScore FS')
plt.xticks(x, df['dataset'], rotation=45, ha='right')
plt.ylabel('AUC')
plt.title('AUC Comparison')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('auc_comparison.png')
plt.close()

print("\nExperiment complete! Graphs saved as auc_comparison.png")