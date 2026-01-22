# multi_experiment.py

import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from capymoa.stream import CSVStream
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest
from capymoa.anomaly.adaptive_isolation_forest_fs import AdaptiveIsolationForestFS

# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------
datasets = [
     ("./dataset/backdoor.csv", "y", ["0", "1"]),
    ("./dataset/census.csv", "y", ["0", "1"]),
    ("./dataset/cover.csv", "y", ["0", "1"]),
    ("./dataset/fraud.csv", "y", ["0", "1"]),
    ("./dataset/http.csv", "y", ["0", "1"]),
    ("./dataset/Thursday-WorkingHours-Afternoon-Infilteration.csv", "y", ["0", "1"]),
    ("./dataset/Thursday-WorkingHours-Morning-WebAttacks.csv", "y", ["0", "1"]),
    ("./dataset/Tuesday-WorkingHours.csv", "y", ["0", "1"]),
    ("./dataset/Wednesday-workingHours.csv", "y", ["0", "1"]),
]

# --------------------------------------------------
# Resource monitor
# --------------------------------------------------
process = psutil.Process()
def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

# --------------------------------------------------
# Store all results
# --------------------------------------------------
results = []

# --------------------------------------------------
# Loop over datasets
# --------------------------------------------------
for csv_path, label_col, label_cats in datasets:
    print(f"\n========== Dataset: {csv_path} ==========")

    stream = CSVStream(
        file=csv_path,
        target=label_col,
        categories={label_col: label_cats}
    )
    schema = stream.get_schema()

    aif = AdaptiveIsolationForest(schema, window_size=256, n_trees=50, seed=42)
    aif_fs = AdaptiveIsolationForestFS(schema, window_size=256, n_trees=50, seed=42)

    eval_aif = AnomalyDetectionEvaluator(schema)
    eval_aif_fs = AnomalyDetectionEvaluator(schema)

    scores_aif, scores_fs, labels = [], [], []

    mem_start = memory_mb()

    time_aif = 0.0
    time_fs = 0.0

    # --------------------------------------------------
    # Stream loop
    # --------------------------------------------------
    while stream.has_more_instances():
        inst = stream.next_instance()
        labels.append(inst.y_index)

        # ---------- AIF ----------
        t0 = time.time()
        s1 = aif.score_instance(inst)
        aif.train(inst)
        time_aif += time.time() - t0

        # ---------- AIF + FS ----------
        t0 = time.time()
        s2 = aif_fs.score_instance(inst)
        aif_fs.train(inst)
        time_fs += time.time() - t0

        scores_aif.append(s1)
        scores_fs.append(s2)

        eval_aif.update(inst.y_index, s1)
        eval_aif_fs.update(inst.y_index, s2)

    mem_end = memory_mb()

    scores_aif = np.array(scores_aif)
    scores_fs = np.array(scores_fs)
    labels = np.array(labels)

    # ROC
    fpr_aif, tpr_aif, _ = roc_curve(labels, scores_aif)
    fpr_fs, tpr_fs, _ = roc_curve(labels, scores_fs)

    auc_aif = auc(fpr_aif, tpr_aif)
    auc_fs = auc(fpr_fs, tpr_fs)

    results.append({
        "dataset": csv_path,
        "auc_aif": auc_aif,
        "auc_fs": auc_fs,
        "fpr_aif": fpr_aif,
        "tpr_aif": tpr_aif,
        "fpr_fs": fpr_fs,
        "tpr_fs": tpr_fs,
        "time_aif": time_aif,
        "time_fs": time_fs,
        "memory": mem_end - mem_start
    })

    print(f"AIF AUC:      {auc_aif:.4f}")
    print(f"AIF+FS AUC:   {auc_fs:.4f}")
    print(f"AIF Time:     {time_aif:.2f}s")
    print(f"AIF+FS Time:  {time_fs:.2f}s")
    print(f"Memory Used:  {mem_end - mem_start:.2f} MB")

# ==================================================
# PLOTS AFTER ALL DATASETS FINISH
# ==================================================

# --------------------------------------------------
# Plot 1: ROC curves (all datasets)
# --------------------------------------------------
for r in results:
    plt.figure(figsize=(7,6))
    plt.plot(r["fpr_aif"], r["tpr_aif"], label=f"AIF (AUC={r['auc_aif']:.3f})")
    plt.plot(r["fpr_fs"], r["tpr_fs"], label=f"AIF+FS (AUC={r['auc_fs']:.3f})")
    plt.plot([0,1], [0,1], 'k--', alpha=0.4)
    plt.title(f"ROC – {r['dataset']}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

plt.show()

# --------------------------------------------------
# Plot 2: AUC comparison
# --------------------------------------------------
names = [r["dataset"] for r in results]
auc_aif = [r["auc_aif"] for r in results]
auc_fs = [r["auc_fs"] for r in results]

x = np.arange(len(names))
plt.figure(figsize=(8,5))
plt.bar(x - 0.15, auc_aif, width=0.3, label="AIF")
plt.bar(x + 0.15, auc_fs, width=0.3, label="AIF + FS")
plt.xticks(x, names, rotation=20)
plt.ylabel("AUC")
plt.title("AUC Comparison Across Datasets")
plt.legend()
plt.grid(axis="y")
plt.show()

# --------------------------------------------------
# Plot 3: Runtime comparison
# --------------------------------------------------
time_aif = [r["time_aif"] for r in results]
time_fs = [r["time_fs"] for r in results]

plt.figure(figsize=(8,5))
plt.bar(x - 0.15, time_aif, width=0.3, label="AIF Time")
plt.bar(x + 0.15, time_fs, width=0.3, label="AIF + FS Time")
plt.xticks(x, names, rotation=20)
plt.ylabel("Seconds")
plt.title("Runtime Comparison")
plt.legend()
plt.grid(axis="y")
plt.show()
