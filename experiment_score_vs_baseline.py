# experiment_score_vs_baseline.py - FAST PARALLEL VERSION
import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from functools import partial

from capymoa.instance import Instance
from capymoa.stream import Schema

from capymoa.anomaly.adaptive_isolation_forest_logistic_fs import AdaptiveIsolationForestWithLogisticFS
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs_activeBuffer import ScoreBasedLogisticFS


process = psutil.Process()


class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"].astype(np.float64)
        y = data["y"]

        self.le = LabelEncoder()
        y_idx = self.le.fit_transform(y)

        self.n = len(X)
        self.i = 0

        feature_names = [f"feature_{j}" for j in range(X.shape[1])]
        self.schema = Schema.from_custom(
            features=feature_names + ["class"],
            target="class",
            categories={"class": [str(cls) for cls in self.le.classes_]},
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


def run_single_experiment(ds_name, label_budget=0.025, n_runs=10, window_size=256):
    """Run 10 runs for one dataset - this function will be parallelized"""
    path = os.path.join("./semi_supervised_Datasets", ds_name)
    stream_template = NPZStream(path)  # We recreate stream inside each run

    auc_score_runs = []
    auc_base_runs = []
    time_runs = []

    for run in range(n_runs):
        stream = NPZStream(path)   # fresh stream for each run
        seed = 42 + run * 13

        model_score = ScoreBasedLogisticFS(
            schema=stream.schema, window_size=window_size,
            label_budget=label_budget, seed=seed, n_trees=100
        )

        model_base = AdaptiveIsolationForestWithLogisticFS(
            schema=stream.schema, window_size=window_size,
            label_budget=label_budget, seed=seed, n_trees=100
        )

        y_true = []
        scores_score = []
        scores_base = []

        start_time = time.time()

        while stream.has_more_instances():
            inst, true_y = stream.next_instance()
            y_true.append(true_y)

            scores_score.append(model_score.score_instance(inst))
            scores_base.append(model_base.score_instance(inst))

            model_score.train(inst, true_y)
            model_base.train(inst, true_y)

        elapsed = time.time() - start_time

        auc_score = auc(*roc_curve(y_true, scores_score)[:2]) if len(np.unique(y_true)) > 1 else 0.5
        auc_base = auc(*roc_curve(y_true, scores_base)[:2]) if len(np.unique(y_true)) > 1 else 0.5

        auc_score_runs.append(auc_score)
        auc_base_runs.append(auc_base)
        time_runs.append(elapsed)

    return {
        'dataset': ds_name,
        'AUC_ScoreBased_mean': np.mean(auc_score_runs),
        'AUC_ScoreBased_std': np.std(auc_score_runs),
        'AUC_Baseline_mean': np.mean(auc_base_runs),
        'AUC_Baseline_std': np.std(auc_base_runs),
        'Time_mean_sec': np.mean(time_runs),
    }


# ========================= Main Parallel Execution =========================
if __name__ == "__main__":
    DATA_DIR = "./semi_supervised_Datasets"
    datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

    n_runs = 10
    window_size = 256
    label_budget = 0.025

    print(f"Starting PARALLEL experiment with {n_runs} runs per dataset...")
    print(f"Using up to {mp.cpu_count()} CPU cores\n")

    # Parallel execution
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(
            partial(run_single_experiment, 
                    label_budget=label_budget, 
                    n_runs=n_runs, 
                    window_size=window_size),
            datasets
        )

    df = pd.DataFrame(results)
    df.to_csv("score_vs_baseline_10runs_parallel_summary.csv", index=False)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED (Parallel Execution)")
    print("="*80)
    print(df.round(4))

    # Overall AUC Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['AUC_ScoreBased_mean'], width, 
            yerr=df['AUC_ScoreBased_std'], label='Score-based Top-N', capsize=5)
    plt.bar(x + width/2, df['AUC_Baseline_mean'], width, 
            yerr=df['AUC_Baseline_std'], label='Random Baseline', capsize=5)

    plt.xticks(x, df['dataset'], rotation=45, ha='right')
    plt.ylabel('AUC')
    plt.title('AUC Comparison (10 runs per dataset - Parallel)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('auc_comparison_10runs_parallel.png', dpi=300)
    plt.close()

    print("\nResults saved → score_vs_baseline_10runs_parallel_summary.csv")
    print("Plot saved → auc_comparison_10runs_parallel.png")
    print(f"\nCPU usage should now be much higher (up to {mp.cpu_count()} cores)")