"""
Baseline AIF vs AIF + Logistic FS vs AIF + ANOVA FS
--------------------------------------------------
• Same stream
• Same parameters
• 10% label budget
• Streaming evaluation
• Comprehensive plots + CSV output
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

from capymoa.anomaly.adaptive_isolation_forest_anova_fs import (
    AdaptiveIsolationForestWithAnovaFS
)

from capymoa.anomaly.adaptive_isolation_forest_logistic_fs import (
    AdaptiveIsolationForestWithLogisticFS
)

# ==================================================
# Resource Monitoring
# ==================================================
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# ==================================================
# NPZ Stream Loader
# ==================================================
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X, y = data["X"], data["y"]

        self.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        self.df["label"] = y

        le = LabelEncoder()
        self.df["label_idx"] = le.fit_transform(self.df["label"])

        self.i = 0
        self.n = len(self.df)

        features = [f"f{i}" for i in range(X.shape[1])]
        categories = {"label": [str(c) for c in le.classes_]}

        self.schema = Schema.from_custom(
            features=features + ["label"],
            target="label",
            categories=categories,
            name=os.path.basename(path),
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        row = self.df.iloc[self.i]
        x = row.drop(["label", "label_idx"]).values.astype(float)
        y = int(row["label_idx"])
        inst = Instance.from_array(self.schema, np.append(x, y))
        self.i += 1
        return inst, y

    def get_schema(self):
        return self.schema

# ==================================================
# Experiment Configuration
# ==================================================
DATASET_DIR = "./semi_supervised_Datasets"

WINDOW_SIZE = 256
N_TREES = 50
SEED = 42
LABEL_BUDGET = 0.10

datasets = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".npz"))

results = []
roc_data = []

# ==================================================
# Run Experiment
# ==================================================
for fname in datasets:
    print(f"\n================ {fname} ================")
    stream = NPZStream(os.path.join(DATASET_DIR, fname))
    schema = stream.get_schema()

    aif = OriginalAIF(schema, WINDOW_SIZE, N_TREES, seed=SEED)
    aif_log = AdaptiveIsolationForestWithLogisticFS(
        schema, WINDOW_SIZE, N_TREES, seed=SEED, label_budget=LABEL_BUDGET
    )
    aif_anova = AdaptiveIsolationForestWithAnovaFS(
        schema, WINDOW_SIZE, N_TREES, seed=SEED, label_budget=LABEL_BUDGET
    )

    y_true, s_aif, s_log, s_anova = [], [], [], []

    start_t = time.time()
    start_m = memory_mb()

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        y_true.append(y)

        s1 = aif.score_instance(inst)
        s2 = aif_log.score_instance(inst)
        s3 = aif_anova.score_instance(inst)

        s_aif.append(s1)
        s_log.append(s2)
        s_anova.append(s3)

        aif.train(inst)
        aif_log.train(inst, label=y)
        aif_anova.train(inst, label=y)

    runtime = time.time() - start_t
    mem = memory_mb() - start_m

    y_true = np.array(y_true)

    def compute_metrics(scores):
        fpr, tpr, _ = roc_curve(y_true, scores)
        return auc(fpr, tpr), average_precision_score(y_true, scores), fpr, tpr

    auc_aif, ap_aif, fpr1, tpr1 = compute_metrics(s_aif)
    auc_log, ap_log, fpr2, tpr2 = compute_metrics(s_log)
    auc_anova, ap_anova, fpr3, tpr3 = compute_metrics(s_anova)

    results.append({
        "dataset": fname,
        "AUC_AIF": auc_aif,
        "AUC_LOGISTIC": auc_log,
        "AUC_ANOVA": auc_anova,
        "AP_AIF": ap_aif,
        "AP_LOGISTIC": ap_log,
        "AP_ANOVA": ap_anova,
        "runtime_s": runtime,
        "memory_MB": mem,
        "features_logistic": aif_log.last_selected_count,
        "features_anova": aif_anova.last_selected_count,
    })

    roc_data.append(
        (fname,
         (fpr1, tpr1, auc_aif),
         (fpr2, tpr2, auc_log),
         (fpr3, tpr3, auc_anova))
    )

    print(f"AUC  AIF       : {auc_aif:.4f}")
    print(f"AUC  Logistic  : {auc_log:.4f}")
    print(f"AUC  ANOVA     : {auc_anova:.4f}")

# ==================================================
# Save CSV
# ==================================================
df = pd.DataFrame(results)
df.to_csv("aif_vs_logistic_vs_anova.csv", index=False)
print("\nSaved: aif_vs_logistic_vs_anova.csv")
print(df.round(4))

# ==================================================
# Plot ROC Curves
# ==================================================
os.makedirs("roc_curves", exist_ok=True)

for name, aif_d, log_d, anova_d in roc_data:
    plt.figure(figsize=(7, 6))
    plt.plot(*aif_d[:2], label=f"AIF (AUC={aif_d[2]:.3f})")
    plt.plot(*log_d[:2], label=f"Logistic FS (AUC={log_d[2]:.3f})")
    plt.plot(*anova_d[:2], label=f"ANOVA FS (AUC={anova_d[2]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title(name)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"roc_curves/{name.replace('.npz','')}.png", dpi=300)
    plt.close()

# ==================================================
# Aggregate Plots
# ==================================================
means = df.mean(numeric_only=True)

plt.figure(figsize=(7,5))
plt.bar(["AIF","Logistic FS","ANOVA FS"],
        [means["AUC_AIF"], means["AUC_LOGISTIC"], means["AUC_ANOVA"]])
plt.ylabel("Mean AUC")
plt.title("Average AUC Across Datasets")
plt.tight_layout()
plt.savefig("mean_auc.png", dpi=300)
plt.close()

plt.figure(figsize=(7,5))
plt.bar(["Logistic FS","ANOVA FS"],
        [means["features_logistic"], means["features_anova"]])
plt.ylabel("Avg Selected Features")
plt.title("Feature Reduction Comparison")
plt.tight_layout()
plt.savefig("feature_reduction.png", dpi=300)
plt.close()

print("\nAll plots saved.")
print("Experiment completed successfully.")
