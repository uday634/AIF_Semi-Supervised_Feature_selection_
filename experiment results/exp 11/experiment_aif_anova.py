"""
AIF vs AIF + ANOVA Feature Selection (10% label budget)

- Same stream
- Same parameters
- Only difference: feature selection using ANOVA + 10% labels
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
from capymoa.anomaly._adaptive_isolation_forest import (
    AdaptiveIsolationForest as OriginalAIF
)

# 🔹 Your modified AIF with ANOVA FS
from capymoa.anomaly.adaptive_isolation_forest_anova_fs import (
    AdaptiveIsolationForestWithAnovaFS
)

# --------------------------------------------------
# Resource monitoring
# --------------------------------------------------
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# NPZ Stream Loader
# --------------------------------------------------
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"]
        y = data["y"]

        self.df = pd.DataFrame(
            X, columns=[f"f{i}" for i in range(X.shape[1])]
        )
        self.df["label"] = y

        le = LabelEncoder()
        self.df["label_idx"] = le.fit_transform(self.df["label"])

        self.n = len(self.df)
        self.i = 0

        feature_names = [f"f{i}" for i in range(X.shape[1])]
        target_name = "label"
        categories = {target_name: [str(c) for c in le.classes_]}

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
        x = row.drop(["label", "label_idx"]).values.astype(np.float64)
        y = int(row["label_idx"])
        inst = Instance.from_array(self.schema, np.append(x, y))
        self.i += 1
        return inst, y

    def get_schema(self):
        return self.schema

# --------------------------------------------------
# Experiment Configuration
# --------------------------------------------------
DATASET_DIR = "./semi_supervised_Datasets"
WINDOW_SIZE = 256
N_TREES = 50
SEED = 42
LABEL_BUDGET = 0.10

datasets = sorted(
    f for f in os.listdir(DATASET_DIR) if f.endswith(".npz")
)

results = []
roc_curves = []

# --------------------------------------------------
# Run Experiment
# --------------------------------------------------
for fname in datasets:
    path = os.path.join(DATASET_DIR, fname)
    print(f"\n================ {fname} ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    aif = OriginalAIF(
        schema=schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=SEED,
    )

    aif_anova = AdaptiveIsolationForestWithAnovaFS(
        schema=schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=SEED,
        label_budget=LABEL_BUDGET,
    )

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_anova = AnomalyDetectionEvaluator(schema)

    y_true = []
    scores_aif = []
    scores_anova = []

    start_time = time.time()
    start_mem = memory_mb()

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        y_true.append(y)

        s1 = aif.score_instance(inst)
        s2 = aif_anova.score_instance(inst)

        scores_aif.append(s1)
        scores_anova.append(s2)

        eval_aif.update(y, s1)
        eval_anova.update(y, s2)

        aif.train(inst)
        aif_anova.train(inst, label=y)

    runtime = time.time() - start_time
    mem_delta = memory_mb() - start_mem

    y_true = np.array(y_true)
    scores_aif = np.array(scores_aif)
    scores_anova = np.array(scores_anova)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    fpr_aif, tpr_aif, _ = roc_curve(y_true, scores_aif)
    fpr_anova, tpr_anova, _ = roc_curve(y_true, scores_anova)

    auc_aif = auc(fpr_aif, tpr_aif)
    auc_anova = auc(fpr_anova, tpr_anova)

    ap_aif = average_precision_score(y_true, scores_aif)
    ap_anova = average_precision_score(y_true, scores_anova)

    roc_curves.append(
        (fname, fpr_aif, tpr_aif, auc_aif, fpr_anova, tpr_anova, auc_anova)
    )

    results.append({
        "dataset": fname,
        "AUC_AIF": auc_aif,
        "AUC_AIF_ANOVA": auc_anova,
        "AP_AIF": ap_aif,
        "AP_AIF_ANOVA": ap_anova,
        "runtime_s": runtime,
        "memory_MB": mem_delta,
        "selected_features_last":
            getattr(aif_anova, "last_selected_count", None),
    })

    print(f"AUC  AIF        : {auc_aif:.4f}")
    print(f"AUC  AIF+ANOVA  : {auc_anova:.4f}")
    print(f"AP   AIF        : {ap_aif:.4f}")
    print(f"AP   AIF+ANOVA  : {ap_anova:.4f}")
    print(f"Runtime (s)     : {runtime:.2f}")
    print(f"Memory Δ (MB)   : {mem_delta:+.2f}")

# --------------------------------------------------
# Save Results
# --------------------------------------------------
df = pd.DataFrame(results)
df["AUC_gain"] = df["AUC_AIF_ANOVA"] - df["AUC_AIF"]

df.to_csv("aif_vs_aif_anova_results.csv", index=False)
print("\nSaved: aif_vs_aif_anova_results.csv")
print(df.round(4))

# --------------------------------------------------
# Plot ROC Curves
# --------------------------------------------------
for name, fpr1, tpr1, auc1, fpr2, tpr2, auc2 in roc_curves:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr1, tpr1, label=f"AIF (AUC={auc1:.3f})")
    plt.plot(fpr2, tpr2, label=f"AIF+ANOVA (AUC={auc2:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe = name.replace(".npz", "").replace("-", "_")
    plt.savefig(f"roc_{safe}.png", dpi=300)
    plt.close()

print("\nAll ROC curves saved.")
print("Experiment finished successfully.")
