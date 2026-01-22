"""
Comparison: Original AIF vs AIF with Semi-Supervised Mutual Information Feature Selection
10% label budget
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

from capymoa.anomaly.adaptive_isolation_forest_mi_fs import AdaptiveIsolationForestMIFS


# ----------------------------------------------------------------------
# Resource monitoring
# ----------------------------------------------------------------------
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)


# ----------------------------------------------------------------------
# NPZ Stream Loader
# ----------------------------------------------------------------------
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"]
        y = data["y"]

        self.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        self.df["target"] = y

        self.le = LabelEncoder()
        self.df["target_idx"] = self.le.fit_transform(self.df["target"]).astype(np.int32)

        self.i = 0
        self.n = len(self.df)

        feature_names = [f"f{i}" for i in range(X.shape[1])]
        target_name = "target"
        categories = {target_name: [str(c) for c in self.le.classes_]}

        self.schema = Schema.from_custom(
            features=feature_names + [target_name],
            target=target_name,
            categories=categories,
            name=os.path.basename(path),
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        row = self.df.iloc[self.i]
        x = row.drop(["target", "target_idx"]).values.astype(np.float64)
        label_idx = int(row["target_idx"])

        inst = Instance.from_array(self.schema, np.append(x, label_idx))
        self.i += 1
        return inst, label_idx

    def get_schema(self):
        return self.schema


# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
DATASET_DIR = "./semi_supervised_Datasets"
WINDOW_SIZE = 256
N_TREES = 50
LABEL_BUDGET = 0.10

datasets = sorted(
    os.path.join(DATASET_DIR, f)
    for f in os.listdir(DATASET_DIR)
    if f.endswith(".npz")
)

results = []
roc_data = []


# ----------------------------------------------------------------------
# Run experiment
# ----------------------------------------------------------------------
for path in datasets:
    name = os.path.basename(path)
    print(f"\n================ {name} ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    aif = OriginalAIF(schema, window_size=WINDOW_SIZE, n_trees=N_TREES, seed=42)
    aif_mi = AdaptiveIsolationForestMIFS(
        schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=42,
        label_budget=LABEL_BUDGET,
    )

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_mi = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_mi, labels = [], [], []

    start_time = time.time()
    start_mem = memory_mb()

    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()

        s1 = aif.score_instance(inst)
        s2 = aif_mi.score_instance(inst)

        scores_aif.append(s1)
        scores_mi.append(s2)
        labels.append(label_idx)

        eval_aif.update(label_idx, s1)
        eval_mi.update(label_idx, s2)

        aif.train(inst)
        aif_mi.train(inst, label=label_idx)

    runtime = time.time() - start_time
    mem_used = memory_mb() - start_mem

    labels = np.array(labels)
    scores_aif = np.array(scores_aif)
    scores_mi = np.array(scores_mi)

    # Metrics
    fpr_aif, tpr_aif, _ = roc_curve(labels, scores_aif)
    fpr_mi, tpr_mi, _ = roc_curve(labels, scores_mi)

    auc_aif = auc(fpr_aif, tpr_aif)
    auc_mi = auc(fpr_mi, tpr_mi)

    ap_aif = average_precision_score(labels, scores_aif)
    ap_mi = average_precision_score(labels, scores_mi)

    # Precision@K
    precs = {}
    for r in [0.01, 0.02, 0.05]:
        k = max(5, int(len(labels) * r))
        precs[f"P@{int(r*100)}%_AIF"] = np.mean(labels[np.argsort(-scores_aif)[:k]] == 1)
        precs[f"P@{int(r*100)}%_MI"] = np.mean(labels[np.argsort(-scores_mi)[:k]] == 1)

    results.append({
        "dataset": name,
        "AUC_AIF": auc_aif,
        "AUC_MI_FS": auc_mi,
        "AP_AIF": ap_aif,
        "AP_MI_FS": ap_mi,
        "runtime_s": runtime,
        "memory_MB": mem_used,
        "selected_features_last": aif_mi.last_selected_count,
        **precs,
    })

    roc_data.append({
        "dataset": name,
        "fpr_aif": fpr_aif,
        "tpr_aif": tpr_aif,
        "fpr_mi": fpr_mi,
        "tpr_mi": tpr_mi,
        "auc_aif": auc_aif,
        "auc_mi": auc_mi,
    })

    print(f"AUC  AIF: {auc_aif:.4f}")
    print(f"AUC  MI : {auc_mi:.4f}")
    print(f"AP   AIF: {ap_aif:.4f}")
    print(f"AP   MI : {ap_mi:.4f}")
    print(f"Time   : {runtime:.2f}s")
    print(f"Memory : {mem_used:+.2f} MB")


# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------
df = pd.DataFrame(results)
df["AUC_diff"] = df["AUC_MI_FS"] - df["AUC_AIF"]
df["AP_diff"] = df["AP_MI_FS"] - df["AP_AIF"]

df.to_csv("aif_vs_mi_fs_results.csv", index=False)
print("\nSaved results to aif_vs_mi_fs_results.csv")


# ----------------------------------------------------------------------
# Plot: AUC comparison
# ----------------------------------------------------------------------
plt.figure(figsize=(14, 6))
x = np.arange(len(df))
w = 0.35

plt.bar(x - w/2, df["AUC_AIF"], w, label="Original AIF")
plt.bar(x + w/2, df["AUC_MI_FS"], w, label="AIF + MI FS")

plt.xticks(x, df["dataset"], rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("AUC Comparison: AIF vs AIF + MI Feature Selection")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("auc_comparison_mi.png", dpi=300)
plt.close()


# ----------------------------------------------------------------------
# Plot ROC curves
# ----------------------------------------------------------------------
for roc in roc_data:
    plt.figure(figsize=(7, 6))
    plt.plot(roc["fpr_aif"], roc["tpr_aif"], label=f"AIF (AUC={roc['auc_aif']:.3f})")
    plt.plot(roc["fpr_mi"], roc["tpr_mi"], label=f"AIF+MI (AUC={roc['auc_mi']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {roc['dataset']}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    name = roc["dataset"].replace(".npz", "").replace("-", "_")
    plt.savefig(f"roc_{name}_mi.png", dpi=300)
    plt.close()

print("\nAll ROC plots saved.")
print("Experiment complete ✅")
