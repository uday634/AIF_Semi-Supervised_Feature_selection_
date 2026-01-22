"""
Semi-supervised comparison with 10% label budget
Original AIF vs AIF + ExtraTrees Feature Selection

Metrics:
- ROC AUC
- Average Precision
- Precision@K
- Runtime
- Memory
- Feature count
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

# ── NEW MODEL ───────────────────────────────────────────────────────────────
from capymoa.anomaly.adaptive_isolation_forest_et_fs import AdaptiveIsolationForestETFS


# =============================================================================
# Resource monitor
# =============================================================================
process = psutil.Process()

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)


# =============================================================================
# NPZ Stream Loader
# =============================================================================
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
            name=os.path.basename(path),
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        row = self.df.iloc[self.i]
        x_features = row.drop(["target", "target_idx"]).values.astype(np.float64)
        label_idx = int(row["target_idx"])

        inst = Instance.from_array(
            self.schema, np.append(x_features, label_idx)
        )

        self.i += 1
        return inst, label_idx

    def get_schema(self):
        return self.schema


# =============================================================================
# Experiment setup
# =============================================================================
dataset_folder = "./semi_supervised_Datasets"
datasets = sorted(
    os.path.join(dataset_folder, f)
    for f in os.listdir(dataset_folder)
    if f.endswith(".npz")
)

results = []
roc_data = []

for path in datasets:
    ds_name = os.path.basename(path)
    print(f"\n================ {ds_name} ================")

    stream = NPZStream(path)
    schema = stream.get_schema()

    # ── Models ───────────────────────────────────────────────────────────────
    aif = OriginalAIF(
        schema=schema,
        window_size=256,
        n_trees=50,
        seed=42,
    )

    aif_et = AdaptiveIsolationForestETFS(
        schema=schema,
        window_size=256,
        n_trees=50,
        seed=42,
        label_budget=0.10,
    )

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_et = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_et, labels = [], [], []

    start_time = time.time()
    start_mem = memory_mb()

    # ── Streaming loop ───────────────────────────────────────────────────────
    while stream.has_more_instances():
        inst, label_idx = stream.next_instance()
        labels.append(label_idx)

        s_aif = aif.score_instance(inst)
        s_et = aif_et.score_instance(inst)

        scores_aif.append(s_aif)
        scores_et.append(s_et)

        eval_aif.update(label_idx, s_aif)
        eval_et.update(label_idx, s_et)

        aif.train(inst)
        aif_et.train(inst, label=label_idx)

    runtime = time.time() - start_time
    memory_delta = memory_mb() - start_mem

    labels_arr = np.array(labels)
    scores_aif_arr = np.array(scores_aif)
    scores_et_arr = np.array(scores_et)

    # ── ROC / AUC ─────────────────────────────────────────────────────────────
    fpr_aif, tpr_aif, _ = roc_curve(labels_arr, scores_aif_arr)
    auc_aif_val = auc(fpr_aif, tpr_aif)

    fpr_et, tpr_et, _ = roc_curve(labels_arr, scores_et_arr)
    auc_et_val = auc(fpr_et, tpr_et)

    roc_data.append({
        "dataset": ds_name,
        "fpr_aif": fpr_aif,
        "tpr_aif": tpr_aif,
        "auc_aif": auc_aif_val,
        "fpr_et": fpr_et,
        "tpr_et": tpr_et,
        "auc_et": auc_et_val,
    })

    # ── Average Precision ─────────────────────────────────────────────────────
    ap_aif = average_precision_score(labels_arr, scores_aif_arr)
    ap_et = average_precision_score(labels_arr, scores_et_arr)

    # ── Precision@K ───────────────────────────────────────────────────────────
    k_ratios = [0.01, 0.02, 0.05]
    prec_k = {}

    for r in k_ratios:
        k = max(5, int(len(labels_arr) * r))
        top_aif = np.argsort(-scores_aif_arr)[:k]
        top_et = np.argsort(-scores_et_arr)[:k]

        prec_k[f"Prec@{int(r*100)}%_AIF"] = np.mean(labels_arr[top_aif] == 1)
        prec_k[f"Prec@{int(r*100)}%_ETFS"] = np.mean(labels_arr[top_et] == 1)

    anomaly_ratio = np.mean(labels_arr == 1)
    selected_features = getattr(aif_et, "last_selected_count", None)

    result = {
        "dataset": ds_name,
        "AUC_AIF": auc_aif_val,
        "AUC_ETFS": auc_et_val,
        "AP_AIF": ap_aif,
        "AP_ETFS": ap_et,
        "runtime_s": runtime,
        "memory_MB": memory_delta,
        "anomaly_ratio": anomaly_ratio,
        "selected_features_last_window": selected_features,
        **prec_k,
    }

    results.append(result)

    print(f"  AUC AIF:   {auc_aif_val:.4f}")
    print(f"  AUC ETFS:  {auc_et_val:.4f}")
    print(f"  AP  AIF:   {ap_aif:.4f}")
    print(f"  AP  ETFS:  {ap_et:.4f}")
    print(f"  Runtime:   {runtime:.2f} s")
    print(f"  Memory Δ:  {memory_delta:+.2f} MB")


# =============================================================================
# Summary table
# =============================================================================
df = pd.DataFrame(results)
df["AUC_diff"] = df["AUC_ETFS"] - df["AUC_AIF"]
df["AP_diff"] = df["AP_ETFS"] - df["AP_AIF"]

print("\n" + "=" * 80)
print("FINAL COMPARISON SUMMARY")
print("=" * 80)
print(df.round(4))

df.to_csv("aif_vs_et_fs_results.csv", index=False)
print("\nSaved: aif_vs_et_fs_results.csv")


# =============================================================================
# Plots
# =============================================================================
x = np.arange(len(df))
width = 0.35

# ── AUC bar chart ────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df["AUC_AIF"], width, label="Original AIF")
plt.bar(x + width/2, df["AUC_ETFS"], width, label="AIF + ET-FS")
plt.xticks(x, df["dataset"], rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("AUC Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("auc_comparison_etfs.png", dpi=300)
plt.close()

# ── Precision@1% ─────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 7))
plt.bar(x - width/2, df["Prec@1%_AIF"], width, label="Original AIF")
plt.bar(x + width/2, df["Prec@1%_ETFS"], width, label="AIF + ET-FS")
plt.xticks(x, df["dataset"], rotation=45, ha="right")
plt.ylabel("Precision@1%")
plt.title("Precision@1% Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("precision_at_1pct_etfs.png", dpi=300)
plt.close()

# ── ROC curves ───────────────────────────────────────────────────────────────
for roc in roc_data:
    plt.figure(figsize=(7, 6))
    plt.plot(
        roc["fpr_aif"],
        roc["tpr_aif"],
        label=f"AIF (AUC={roc['auc_aif']:.4f})",
    )
    plt.plot(
        roc["fpr_et"],
        roc["tpr_et"],
        label=f"ET-FS (AUC={roc['auc_et']:.4f})",
    )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(roc["dataset"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_{roc['dataset'].replace('.npz','')}.png", dpi=300)
    plt.close()

print("\nExperiment completed successfully ✅")
