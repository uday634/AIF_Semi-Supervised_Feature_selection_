# multi_experiment_npz_final.py

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Avoid tkinter issues on Windows
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

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
# NPZ Stream using pandas
# --------------------------------------------------
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"]
        y = data["y"]

        # Convert features to DataFrame
        self.df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.df["target"] = y

        # Encode target externally
        self.le = LabelEncoder()
        self.df["target_idx"] = self.le.fit_transform(self.df["target"])

        self.n = len(self.df)
        self.i = 0

        # Schema (features + target)
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
        x_features = row.drop(["target", "target_idx"]).values.astype(float)
        inst = Instance.from_array(self.schema, np.append(x_features, row["target_idx"]))
        self.i += 1
        return inst, int(row["target_idx"])  # <-- cast to int

    def get_schema(self):
        return self.schema

# --------------------------------------------------
# Automatically find all NPZ files in a folder
# --------------------------------------------------
dataset_folder = "./semi_supervised_Datasets"
datasets = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".npz")]

# --------------------------------------------------
# Store results
# --------------------------------------------------
results = []

# --------------------------------------------------
# Run experiments
# --------------------------------------------------
for npz_path in datasets:
    print(f"\n================ Dataset: {npz_path} ================")
    
    stream = NPZStream(npz_path)
    schema = stream.get_schema()
    
    # Models
    aif = AdaptiveIsolationForest(schema=schema, window_size=256, n_trees=50, seed=42)
    aif_fs = AdaptiveIsolationForestFS(schema=schema, window_size=256, n_trees=50, seed=42)
    
    # Evaluators
    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_fs = AnomalyDetectionEvaluator(schema)
    
    # Storage
    scores_aif, scores_fs, labels = [], [], []

    # Timing & memory
    start_time_aif = time.time()
    start_mem_aif = memory_mb()
    
    # Prequential evaluation
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

    end_time_aif = time.time()
    end_mem_aif = memory_mb()

    # Convert to numpy
    scores_aif = np.array(scores_aif)
    scores_fs = np.array(scores_fs)
    labels = np.array(labels)

    # ROC curves and AUC
    fpr_aif, tpr_aif, _ = roc_curve(labels, scores_aif)
    fpr_fs, tpr_fs, _ = roc_curve(labels, scores_fs)
    roc_auc_aif = auc(fpr_aif, tpr_aif)
    roc_auc_fs = auc(fpr_fs, tpr_fs)

    # Save results
    results.append({
        "dataset": os.path.basename(npz_path),
        "AIF_AUC": roc_auc_aif,
        "AIF_FS_AUC": roc_auc_fs,
        "AIF_runtime_s": end_time_aif - start_time_aif,
        "AIF_FS_runtime_s": end_time_aif - start_time_aif,
        "AIF_memory_MB": end_mem_aif - start_mem_aif,
        "scores_aif": scores_aif,
        "scores_fs": scores_fs,
        "labels": labels,
        "fpr_aif": fpr_aif,
        "tpr_aif": tpr_aif,
        "fpr_fs": fpr_fs,
        "tpr_fs": tpr_fs
    })

    print(f"AIF AUC:       {roc_auc_aif:.4f}")
    print(f"AIF+FS AUC:    {roc_auc_fs:.4f}")
    print(f"Runtime AIF:   {end_time_aif - start_time_aif:.2f}s")
    print(f"Memory used:   {end_mem_aif - start_mem_aif:.2f} MB")

# --------------------------------------------------
# Summary Table
# --------------------------------------------------
df_summary = pd.DataFrame(results)
df_summary_table = df_summary[["dataset", "AIF_AUC", "AIF_FS_AUC", "AIF_runtime_s", "AIF_FS_runtime_s", "AIF_memory_MB"]]
print("\n================ SUMMARY =================")
print(df_summary_table.to_string(index=False))

# --------------------------------------------------
# Bar plot: AUC comparison
# --------------------------------------------------
x = np.arange(len(df_summary))
width = 0.35
plt.figure(figsize=(10,5))
plt.bar(x - width/2, df_summary["AIF_AUC"], width, label="AIF")
plt.bar(x + width/2, df_summary["AIF_FS_AUC"], width, label="AIF + FS")
plt.xticks(x, df_summary["dataset"], rotation=25)
plt.ylabel("AUC")
plt.title("AUC Comparison Across NPZ Datasets")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("auc_comparison.png")
plt.close()

# --------------------------------------------------
# Bar plot: Runtime comparison
# --------------------------------------------------
plt.figure(figsize=(10,5))
plt.bar(x - width/2, df_summary["AIF_runtime_s"], width, label="AIF Runtime (s)")
plt.bar(x + width/2, df_summary["AIF_FS_runtime_s"], width, label="AIF+FS Runtime (s)")
plt.xticks(x, df_summary["dataset"], rotation=25)
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison Across NPZ Datasets")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("runtime_comparison.png")
plt.close()

# --------------------------------------------------
# Optional: Plot ROC curves for all datasets
# --------------------------------------------------
for r in results:
    plt.figure(figsize=(7,6))
    plt.plot(r["fpr_aif"], r["tpr_aif"], label=f"AIF (AUC={r['AIF_AUC']:.4f})")
    plt.plot(r["fpr_fs"], r["tpr_fs"], label=f"AIF+FS (AUC={r['AIF_FS_AUC']:.4f})")
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {r['dataset']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_{r['dataset']}.png")
    plt.close()

print("\nAll plots saved as PNG files. Summary table displayed above.")
