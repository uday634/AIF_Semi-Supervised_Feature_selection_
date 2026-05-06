# experiment_aif_vs_adaptive.py
# Compares Original AIF vs AdaptiveIsolationForestWithLogisticFS (with m_trees)

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from tqdm import tqdm

from capymoa.instance import Instance
from capymoa.stream import Schema

# Original AIF
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest as OriginalAIF

# Your Adaptive version with m_trees
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs_with_activeLearning_Tournament import AdaptiveIsolationForestWithGlobalLRActiveTournament


# ========================= CONFIG =========================
DATA_DIR = "./semi_supervised_Datasets"

N_RUNS = 10
WINDOW_SIZE = 256
LABEL_BUDGET = 0.025
L1_STRENGTH = 1.0

N_TREES = 80        # Number of trees in ensemble
M_TREES = 10        # Number of candidate trees per window (for your adaptive model)

ROLLING_FREQ = 200

# Output files
SUMMARY_CSV = "aif_vs_adaptive_summary.csv"
PER_RUN_CSV = "aif_vs_adaptive_per_run_auc.csv"
PLOT_BAR = "aif_vs_adaptive_bar_TopN_Active.png"


# ========================= STREAM =========================
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"].astype(np.float64)
        y = data["y"].ravel()

        le = LabelEncoder()
        y_idx = le.fit_transform(y)

        self.n = len(X)
        self.i = 0

        feat_names = [f"feature_{j}" for j in range(X.shape[1])]
        self.schema = Schema.from_custom(
            features=feat_names + ["class"],
            target="class",
            categories={"class": [str(c) for c in le.classes_]},
            name=os.path.basename(path)
        )
        self.data = list(zip(X, y_idx))

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        x, y = self.data[self.i]
        inst = Instance.from_array(self.schema, np.append(x, [y]))
        self.i += 1
        return inst, y


def safe_auc(y, scores):
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, scores)
        return auc(fpr, tpr)
    return 0.5


# ========================= WORKER =========================
def run_single(args):
    ds_name, run = args
    seed = 42 + run * 13

    stream = NPZStream(os.path.join(DATA_DIR, ds_name))

    # Original AIF
    model_orig = OriginalAIF(
        schema=stream.schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=seed
    )

    # Your Adaptive model with m_trees
    model_adapt = AdaptiveIsolationForestWithGlobalLRActiveTournament(
        schema=stream.schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        m_trees=M_TREES,
        seed=seed,
        label_budget=LABEL_BUDGET,
        l1_strength=L1_STRENGTH
    )

    y_true = []
    scores_orig = []
    scores_adapt = []

    step = 0

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        step += 1

        s_o = model_orig.score_instance(inst)
        s_a = model_adapt.score_instance(inst)

        y_true.append(y)
        scores_orig.append(s_o)
        scores_adapt.append(s_a)

        model_orig.train(inst)
        model_adapt.train(inst, y)

    final_auc_orig = safe_auc(y_true, scores_orig)
    final_auc_adapt = safe_auc(y_true, scores_adapt)

    return {
        "dataset": ds_name,
        "run": run,
        "auc_orig": final_auc_orig,
        "auc_adapt": final_auc_adapt
    }


# ========================= MAIN =========================
if __name__ == "__main__":
    datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
    tasks = [(ds, run) for ds in datasets for run in range(N_RUNS)]

    print(f"Starting AIF vs AIF Tournament (N = 4) Experiment")
    print(f"Datasets: {len(datasets)} | Runs: {N_RUNS} | Total tasks: {len(tasks)}\n")

    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(run_single, tasks), total=len(tasks), desc="Running"):
            results.append(res)

    # ====================== Summary ======================
    summary = []
    for ds in datasets:
        ds_res = [r for r in results if r["dataset"] == ds]
        auc_orig = [r["auc_orig"] for r in ds_res]
        auc_adapt = [r["auc_adapt"] for r in ds_res]

        summary.append({
            "dataset": ds,
            "AUC_Original_mean": np.mean(auc_orig),
            "AUC_Original_std": np.std(auc_orig),
            "AUC_Adaptive_mean": np.mean(auc_adapt),
            "AUC_Adaptive_std": np.std(auc_adapt),
        })

    df = pd.DataFrame(summary)
    df.to_csv(SUMMARY_CSV, index=False)

    # Per-run final AUCs
    per_run = []
    for r in results:
        per_run.append({
            "dataset": r["dataset"],
            "run": r["run"],
            "AUC_Original": r["auc_orig"],
            "AUC_Adaptive": r["auc_adapt"]
        })
    pd.DataFrame(per_run).to_csv(PER_RUN_CSV, index=False)

    print("\nFinal AUC Summary:")
    print(df.round(4))

    # ====================== Bar Plot ======================
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(14, 7))
    plt.bar(x - width/2, df['AUC_Original_mean'], width, yerr=df['AUC_Original_std'],
            label='Original AIF', capsize=5)
    plt.bar(x + width/2, df['AUC_Adaptive_mean'], width, yerr=df['AUC_Adaptive_std'],
            label='AIF Tournament (N = 4)', capsize=5)

    plt.xticks(x, [d.replace('.npz', '') for d in df['dataset']], rotation=45, ha='right')
    plt.ylabel("Mean AUC")
    plt.title(f"Original AIF vs AIF (Top N) (label_budget={LABEL_BUDGET})")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_BAR, dpi=300)
    plt.close()

    print(f"\nBar plot saved: {PLOT_BAR}")
    print(f"Summary CSV   : {SUMMARY_CSV}")
    print(f"Per-run CSV   : {PER_RUN_CSV}")
    print("\n✅ Experiment completed successfully!")