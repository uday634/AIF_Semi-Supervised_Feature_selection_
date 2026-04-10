# experiment_score_vs_baseline.py - FINAL VERSION WITH TIME & MEMORY TRACKING
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
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs_tournament import TournamentBasedLogisticFS


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


def run_single_dataset(ds_name, label_budget=0.025, n_runs=10, window_size=256):
    """Run one dataset with 10 runs - returns summary including time and memory"""
    path = os.path.join("./semi_supervised_Datasets", ds_name)

    auc_score_runs = []
    auc_base_runs = []
    time_runs = []
    mem_runs = []

    rolling_auc_score_all = []
    rolling_auc_base_all = []

    for run in range(n_runs):
        stream = NPZStream(path)
        seed = 42 + run * 13

        model_score = TournamentBasedLogisticFS(
            schema=stream.schema,
            window_size=window_size,
            label_budget=label_budget,
            seed=seed,
            n_trees=100
        )

        model_base = AdaptiveIsolationForestWithLogisticFS(
            schema=stream.schema,
            window_size=window_size,
            label_budget=label_budget,
            seed=seed,
            n_trees=100
        )

        y_true = []
        scores_score = []
        scores_base = []

        rolling_y = []
        rolling_scores_score = []
        rolling_scores_base = []
        run_rolling_score = []
        run_rolling_base = []

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        while stream.has_more_instances():
            inst, true_y = stream.next_instance()
            y_true.append(true_y)
            rolling_y.append(true_y)

            s_score = model_score.score_instance(inst)
            s_base = model_base.score_instance(inst)

            scores_score.append(s_score)
            scores_base.append(s_base)
            rolling_scores_score.append(s_score)
            rolling_scores_base.append(s_base)

            model_score.train(inst, true_y)
            model_base.train(inst, true_y)

            # Rolling AUC every 50 instances
            if len(rolling_y) % 50 == 0 and len(np.unique(rolling_y)) > 1:
                auc_s = auc(*roc_curve(rolling_y, rolling_scores_score)[:2])
                auc_b = auc(*roc_curve(rolling_y, rolling_scores_base)[:2])
                run_rolling_score.append(auc_s)
                run_rolling_base.append(auc_b)

        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)

        elapsed = end_time - start_time
        mem_delta = end_mem - start_mem

        final_auc_score = auc(*roc_curve(y_true, scores_score)[:2]) if len(np.unique(y_true)) > 1 else 0.5
        final_auc_base = auc(*roc_curve(y_true, scores_base)[:2]) if len(np.unique(y_true)) > 1 else 0.5

        auc_score_runs.append(final_auc_score)
        auc_base_runs.append(final_auc_base)
        time_runs.append(elapsed)
        mem_runs.append(mem_delta)

        rolling_auc_score_all.append(run_rolling_score)
        rolling_auc_base_all.append(run_rolling_base)

    # Average rolling AUC
    max_len = max((len(lst) for lst in rolling_auc_score_all), default=0)
    avg_rolling_score = np.zeros(max_len)
    avg_rolling_base = np.zeros(max_len)
    count = np.zeros(max_len)

    for lst in rolling_auc_score_all:
        for i, val in enumerate(lst):
            avg_rolling_score[i] += val
            count[i] += 1
    for lst in rolling_auc_base_all:
        for i, val in enumerate(lst):
            avg_rolling_base[i] += val

    avg_rolling_score /= np.maximum(count, 1)
    avg_rolling_base /= np.maximum(count, 1)

    return {
        'dataset': ds_name,
        'AUC_ScoreBased_mean': np.mean(auc_score_runs),
        'AUC_ScoreBased_std': np.std(auc_score_runs),
        'AUC_Baseline_mean': np.mean(auc_base_runs),
        'AUC_Baseline_std': np.std(auc_base_runs),
        'Time_mean_sec': np.mean(time_runs),
        'Memory_delta_MB': np.mean(mem_runs),
        'rolling_auc_score': avg_rolling_score.tolist(),
        'rolling_auc_base': avg_rolling_base.tolist(),
        'rolling_windows': (np.arange(max_len) * 50).tolist()
    }


# ========================= Main Parallel Execution =========================
if __name__ == "__main__":
    DATA_DIR = "./semi_supervised_Datasets"
    datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

    n_runs = 10
    window_size = 256
    label_budget = 0.025

    print(f"Starting PARALLEL experiment with {n_runs} runs per dataset...")
    print(f"Using up to {mp.cpu_count()} CPU cores...\n")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(
            partial(run_single_dataset,
                    label_budget=label_budget,
                    n_runs=n_runs,
                    window_size=window_size),
            datasets
        )

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['rolling_auc_score', 'rolling_auc_base', 'rolling_windows']} for r in results])
    df.to_csv("score_vs_baseline_10runs_parallel_summary.csv", index=False)

    # ========================= Per-Window AUC Line Plots =========================
    os.makedirs("auc_plots", exist_ok=True)

    for res in results:
        ds_safe = res['dataset'].replace('.npz', '').replace('-', '_')
        windows = np.array(res['rolling_windows'])

        plt.figure(figsize=(12, 6))
        plt.plot(windows, res['rolling_auc_score'], label='Score-based Top-N', linewidth=2.5)
        plt.plot(windows, res['rolling_auc_base'], label='Random Baseline', linewidth=2.5)
        plt.xlabel('Number of Instances Processed')
        plt.ylabel('Rolling AUC')
        plt.title(f'Per-Window AUC Evolution (10-run average) - {res["dataset"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'auc_plots/per_window_auc_{ds_safe}.png', dpi=300)
        plt.close()

    # Overall Bar Chart
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['AUC_ScoreBased_mean'], width, 
            yerr=df['AUC_ScoreBased_std'], label='Score-based Top-N', capsize=5)
    plt.bar(x + width/2, df['AUC_Baseline_mean'], width, 
            yerr=df['AUC_Baseline_std'], label='Random Baseline', capsize=5)

    plt.xticks(x, df['dataset'], rotation=45, ha='right')
    plt.ylabel('AUC')
    plt.title('Overall AUC Comparison (10 runs per dataset - Parallel)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('auc_comparison_10runs_parallel.png', dpi=300)
    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(df.round(4))

    print("\nSaved Files:")
    print("• Summary CSV          : score_vs_baseline_10runs_parallel_summary.csv")
    print("• Overall AUC Plot     : auc_comparison_10runs_parallel.png")
    print("• Per-window AUC Plots : auc_plots/per_window_auc_*.png")
    print(f"CPU Cores Used         : {mp.cpu_count()}")