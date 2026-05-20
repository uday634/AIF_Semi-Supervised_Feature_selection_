# experiment_aif_vs_topn.py
# Compare:
# 1. Original AIF
# 2. Top-N Window AIF (active learning)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import multiprocessing as mp

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

from capymoa.instance import Instance
from capymoa.stream import Schema

# ================= ORIGINAL =================
from capymoa.anomaly._adaptive_isolation_forest import (
    AdaptiveIsolationForest as OriginalAIF
)

# ================= TOP-N MODEL =================
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs import (
    AdaptiveIsolationForestWithLogisticFS
)

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "./semi_supervised_Datasets"

N_RUNS = 10

WINDOW_SIZE = 256
LABEL_BUDGET = 0.025
L1_STRENGTH = 1.0

N_TREES = 100
M_TREES = 10

SUMMARY_CSV = "topn_window_summary.csv"
PER_RUN_CSV = "topn_window_per_run.csv"
PLOT_BAR = "topn_window_barplot.png"


# =========================================================
# STREAM
# =========================================================

class NPZStream:

    def __init__(self, path):

        data = np.load(path)

        X = data["X"].astype(np.float64)
        y = data["y"].ravel()

        le = LabelEncoder()
        y_idx = le.fit_transform(y)

        self.n = len(X)
        self.i = 0

        feat_names = [
            f"feature_{j}"
            for j in range(X.shape[1])
        ]

        self.schema = Schema.from_custom(
            features=feat_names + ["class"],
            target="class",
            categories={
                "class": [str(c) for c in le.classes_]
            },
            name=os.path.basename(path)
        )

        self.data = list(zip(X, y_idx))

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):

        x, y = self.data[self.i]

        inst = Instance.from_array(
            self.schema,
            np.append(x, [y])
        )

        self.i += 1

        return inst, y


# =========================================================
# AUC
# =========================================================

def safe_auc(y, scores):

    if len(np.unique(y)) > 1:

        fpr, tpr, _ = roc_curve(y, scores)

        return auc(fpr, tpr)

    return 0.5


# =========================================================
# WORKER
# =========================================================

def run_single(args):

    ds_name, run = args
    seed = 42 + run * 13

    stream = NPZStream(os.path.join(DATA_DIR, ds_name))

    # ================= ORIGINAL =================
    model_orig = OriginalAIF(
        schema=stream.schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=seed
    )

    # ================= TOP-N MODEL =================
    model_topn = AdaptiveIsolationForestWithLogisticFS(
        schema=stream.schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        m_trees=M_TREES,
        seed=seed,
        label_budget=LABEL_BUDGET,
        l1_strength=L1_STRENGTH
    )

    # ================= STORAGE =================
    y_true = []
    scores_orig = []
    scores_topn = []

    # ================= STREAM =================
    while stream.has_more_instances():

        inst, y = stream.next_instance()

        # ORIGINAL
        s_orig = model_orig.score_instance(inst)
        scores_orig.append(s_orig)

        # TOP-N
        s_topn = model_topn.score_instance(inst)
        scores_topn.append(s_topn)

        # TRAIN
        model_orig.train(inst)
        model_topn.train(inst, y)

        y_true.append(y)

    return {
        "dataset": ds_name,
        "run": run,
        "auc_orig": safe_auc(y_true, scores_orig),
        "auc_topn": safe_auc(y_true, scores_topn),
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    datasets = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".npz")
    ])

    tasks = [
        (ds, run)
        for ds in datasets
        for run in range(N_RUNS)
    ]

    print("=" * 60)
    print("AIF vs WRFS")
    print("=" * 60)

    results = []

    with mp.Pool(processes=mp.cpu_count()) as pool:

        for res in tqdm(
            pool.imap_unordered(run_single, tasks),
            total=len(tasks),
            desc="Running"
        ):
            results.append(res)

    # ================= SUMMARY =================
    summary = []

    for ds in datasets:

        ds_res = [r for r in results if r["dataset"] == ds]

        summary.append({
            "dataset": ds,

            "AUC_Original_mean":
                np.mean([r["auc_orig"] for r in ds_res]),

            "AUC_Original_std":
                np.std([r["auc_orig"] for r in ds_res]),

            "WRFS":
                np.mean([r["auc_topn"] for r in ds_res]),

            "WRFS_std":
                np.std([r["auc_topn"] for r in ds_res]),
        })

    df = pd.DataFrame(summary)
    df.to_csv(SUMMARY_CSV, index=False)

    # ================= PER RUN =================
    per_run = []

    for r in results:
        per_run.append({
            "dataset": r["dataset"],
            "run": r["run"],
            "AUC_Original": r["auc_orig"],
            "AUC_TopN": r["auc_topn"]
        })

    pd.DataFrame(per_run).to_csv(PER_RUN_CSV, index=False)

    # ================= PRINT =================
    print("\nFINAL SUMMARY\n")
    print(df.round(4))

    # ================= PLOT =================
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(14, 7))

    plt.bar(
        x - width/2,
        df["AUC_Original_mean"],
        width,
        yerr=df["AUC_Original_std"],
        label="Original AIF",
        capsize=5
    )

    plt.bar(
        x + width/2,
        df["AUC_TopN_mean"],
        width,
        yerr=df["AUC_TopN_std"],
        label="RANDOM WINDOW AIF",
        capsize=5
    )

    plt.xticks(
        x,
        [d.replace(".npz", "") for d in df["dataset"]],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Mean AUC")
    plt.title(f"Original vs Random Window AIF (budget={LABEL_BUDGET})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_BAR, dpi=300)
    plt.close()

    print("\nSaved:", PLOT_BAR)
    print("Done.")