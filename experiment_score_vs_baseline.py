# experiment_aif_vs_globalLS.py
# UPDATED: Only compare Original AIF vs Global Logistic Selection (River)
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

# ====================== Import Models ======================
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest as OriginalAIF

# Global Logistic Selection (River-based - the global/online version)
from capymoa.anomaly.AdaptiveIsolationForestWithLogisticFSGlobal import AdaptiveIsolationForestWithLogisticFS


class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        X = data["X"].astype(np.float64)
        y = data["y"].ravel()

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
    """Run one dataset comparing Original AIF vs GlobalLS (River)"""
    path = os.path.join("./semi_supervised_Datasets", ds_name)

    auc_orig_runs = []
    auc_globalLS_runs = []
    time_runs = []
    mem_runs = []

    # Store per-run rolling AUCs for variance analysis
    all_rolling_orig = []
    all_rolling_globalLS = []

    for run in range(n_runs):
        stream = NPZStream(path)
        seed = 42 + run * 13

        # ====================== Initialize Models ======================
        model_orig = OriginalAIF(
            schema=stream.schema,
            window_size=window_size,
            n_trees=100,
            seed=seed
        )

        model_globalLS = AdaptiveIsolationForestWithLogisticFS(
            schema=stream.schema,
            window_size=window_size,
            n_trees=100,
            seed=seed,
            label_budget=label_budget,
            l1_strength=0.1          # River L1 strength (tunable)
        )

        y_true = []
        scores_orig = []
        scores_globalLS = []

        rolling_y = []
        rolling_orig = []
        rolling_globalLS = []

        run_rolling_orig = []
        run_rolling_globalLS = []

        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)

        while stream.has_more_instances():
            inst, true_y = stream.next_instance()

            y_true.append(true_y)
            rolling_y.append(true_y)

            # Score instances
            s_orig = model_orig.score_instance(inst)
            s_globalLS = model_globalLS.score_instance(inst)

            scores_orig.append(s_orig)
            scores_globalLS.append(s_globalLS)

            rolling_orig.append(s_orig)
            rolling_globalLS.append(s_globalLS)

            # Train models
            model_orig.train(inst)
            model_globalLS.train(inst, true_y)   # GlobalLS learns online on labeled data

            # Rolling AUC every 50 instances
            if len(rolling_y) % 50 == 0 and len(np.unique(rolling_y)) > 1:
                fpr, tpr, _ = roc_curve(rolling_y, rolling_orig)
                auc_o = auc(fpr, tpr)

                fpr, tpr, _ = roc_curve(rolling_y, rolling_globalLS)
                auc_g = auc(fpr, tpr)

                run_rolling_orig.append(auc_o)
                run_rolling_globalLS.append(auc_g)

        # Record time and memory
        elapsed = time.time() - start_time
        mem_delta = psutil.Process().memory_info().rss / (1024 * 1024) - start_mem

        # Final AUC
        def safe_auc(y_true, scores):
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, scores)
                return auc(fpr, tpr)
            return 0.5

        auc_orig_runs.append(safe_auc(y_true, scores_orig))
        auc_globalLS_runs.append(safe_auc(y_true, scores_globalLS))
        time_runs.append(elapsed)
        mem_runs.append(mem_delta)

        all_rolling_orig.append(run_rolling_orig)
        all_rolling_globalLS.append(run_rolling_globalLS)

    # ====================== Save Per-Run Rolling AUC to CSV ======================
    os.makedirs("rolling_auc_csv_aif_vs_globalLS", exist_ok=True)
    ds_safe = ds_name.replace('.npz', '').replace('-', '_').replace('.', '_')

    max_len = max((len(lst) for lst in all_rolling_globalLS), default=0)

    rolling_df = pd.DataFrame({'window': np.arange(max_len) * 50})

    for i in range(n_runs):
        rolling_df[f'original_run_{i}'] = [all_rolling_orig[i][j] if j < len(all_rolling_orig[i]) else np.nan 
                                           for j in range(max_len)]
        rolling_df[f'globalLS_run_{i}'] = [all_rolling_globalLS[i][j] if j < len(all_rolling_globalLS[i]) else np.nan 
                                           for j in range(max_len)]

    # Add mean and std
    rolling_df['original_mean'] = rolling_df[[f'original_run_{i}' for i in range(n_runs)]].mean(axis=1)
    rolling_df['original_std']  = rolling_df[[f'original_run_{i}' for i in range(n_runs)]].std(axis=1)
    rolling_df['globalLS_mean'] = rolling_df[[f'globalLS_run_{i}' for i in range(n_runs)]].mean(axis=1)
    rolling_df['globalLS_std']  = rolling_df[[f'globalLS_run_{i}' for i in range(n_runs)]].std(axis=1)

    rolling_df.to_csv(f"rolling_auc_csv_aif_vs_globalLS/rolling_auc_per_run_{ds_safe}.csv", index=False)

    # Return summary for main CSV
    return {
        'dataset': ds_name,
        'AUC_Original_mean': np.mean(auc_orig_runs),
        'AUC_Original_std': np.std(auc_orig_runs),
        'AUC_GlobalLS_mean': np.mean(auc_globalLS_runs),
        'AUC_GlobalLS_std': np.std(auc_globalLS_runs),
        'Time_mean_sec': np.mean(time_runs),
        'Memory_delta_MB': np.mean(mem_runs),
        'n_runs': n_runs
    }


# ========================= Main Execution =========================
if __name__ == "__main__":
    DATA_DIR = "./semi_supervised_Datasets"
    datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])

    n_runs = 10
    window_size = 256
    label_budget = 0.025

    print(f"Starting AIF vs GlobalLS Experiment ({n_runs} runs per dataset)...")
    print(f"Using up to {mp.cpu_count()} CPU cores...\n")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(
            partial(run_single_dataset,
                    label_budget=label_budget,
                    n_runs=n_runs,
                    window_size=window_size),
            datasets
        )

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv("aif_vs_globalLS_10runs_summary.csv", index=False)

    # Overall Bar Chart (only 2 methods)
    plt.figure(figsize=(16, 8))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['AUC_Original_mean'], width, 
            yerr=df['AUC_Original_std'], label='Original AIF', capsize=5)
    plt.bar(x + width/2, df['AUC_GlobalLS_mean'], width, 
            yerr=df['AUC_GlobalLS_std'], label='GlobalLS (River)', capsize=5)

    plt.xticks(x, df['dataset'], rotation=45, ha='right')
    plt.ylabel('Mean AUC')
    plt.title('Overall AUC Comparison (10 runs) - Original AIF vs GlobalLS (River)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('aif_vs_globalLS_comparison.png', dpi=300)
    plt.close()

    print("\n" + "="*90)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*90)
    print(df.round(4))

    print("\nSaved Files:")
    print("• Summary CSV                  : aif_vs_globalLS_10runs_summary.csv")
    print("• Per-run Rolling AUC CSVs     : rolling_auc_csv_aif_vs_globalLS/rolling_auc_per_run_*.csv")
    print("• Overall AUC Plot             : aif_vs_globalLS_comparison.png")