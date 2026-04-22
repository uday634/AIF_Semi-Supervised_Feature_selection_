# see_features_per_window.py
"""
Plots number of selected features PER WINDOW for every dataset
Shows Logistic FS vs ANOVA FS + total features reference line
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevents Tkinter crash
import matplotlib.pyplot as plt
import pandas as pd

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs_with_buffer import AdaptiveIsolationForestWithLogisticFS
from capymoa.anomaly.adaptive_isolation_forest_anova_fs import AdaptiveIsolationForestWithAnovaFS

# ────────────────────────────────────────────────────────────────
# NPZ Stream Loader (minimal - only for schema & data)
# ────────────────────────────────────────────────────────────────
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        self.X = data["X"].astype(float)
        self.y = data["y"].astype(int)
        self.i = 0

        features = [f"f{i}" for i in range(self.X.shape[1])]
        self.schema = Schema.from_custom(
            features=features + ["class"],
            target="class",
            categories={"class": ["0", "1"]},
            name=os.path.basename(path),
        )
        self.full_dim = self.schema.get_num_attributes()  # total original features

    def has_more_instances(self):
        return self.i < len(self.X)

    def next_instance(self):
        x = self.X[self.i]
        y = self.y[self.i]
        self.i += 1
        return Instance.from_array(self.schema, np.append(x, y)), y

# ────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────
DATASET_DIR = "./semi_supervised_Datasets"
WINDOW_SIZE = 256
N_TREES = 50          # doesn't matter much here
SEED = 42
LABEL_BUDGET = 0.10   # change to 0.01, 0.05, etc. if you want

os.makedirs("features_per_window_graphs", exist_ok=True)

summary_rows = []

# ────────────────────────────────────────────────────────────────
# Process each dataset
# ────────────────────────────────────────────────────────────────
for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".npz"):
        continue

    print(f"\n→ Processing: {fname}")

    path = os.path.join(DATASET_DIR, fname)
    stream = NPZStream(path)
    schema = stream.schema
    total_features = stream.full_dim

    # Models (only FS part matters)
    log_model = AdaptiveIsolationForestWithLogisticFS(
        schema, window_size=WINDOW_SIZE, n_trees=N_TREES, seed=SEED, label_budget=LABEL_BUDGET
    )
    anova_model = AdaptiveIsolationForestWithAnovaFS(
        schema, window_size=WINDOW_SIZE, n_trees=N_TREES, seed=SEED, label_budget=LABEL_BUDGET
    )

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        log_model.train(inst, label=y)
        anova_model.train(inst, label=y)

    # Get the lists we need (collected in train())
    log_per_window = log_model.selected_counts
    anova_per_window = anova_model.selected_counts

    if not log_per_window or not anova_per_window:
        print("  No windows completed or tracking not working")
        continue

    # Make sure same length
    n_windows = min(len(log_per_window), len(anova_per_window))
    log_per_window = log_per_window[:n_windows]
    anova_per_window = anova_per_window[:n_windows]

    print(f"  → Found {n_windows} complete windows")

    # ────────────────────────────────────────────────────────────
    # Plot for this dataset
    # ────────────────────────────────────────────────────────────
    windows = np.arange(1, n_windows + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(windows, log_per_window, label="Logistic FS", linewidth=2, color="#ff7f0e")
    plt.plot(windows, anova_per_window, label="ANOVA FS", linewidth=2, color="#2ca02c")
    plt.axhline(y=total_features, color="gray", linestyle="--", linewidth=1.5,
                label=f"Total features = {total_features}")

    plt.xlabel("Window number")
    plt.ylabel("Number of selected features")
    plt.title(f"Features selected per window – {fname}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = fname.replace(".npz", "").replace("-", "_")
    save_path = f"features_per_window_graphs/{safe_name}_per_window.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  Saved graph → {save_path}")

    # Add to summary
    summary_rows.append({
        "dataset": fname,
        "total_features": total_features,
        "windows": n_windows,
        "mean_logistic": np.mean(log_per_window) if log_per_window else np.nan,
        "mean_anova": np.mean(anova_per_window) if anova_per_window else np.nan,
    })

# ────────────────────────────────────────────────────────────────
# Summary CSV
# ────────────────────────────────────────────────────────────────
if summary_rows:
    df = pd.DataFrame(summary_rows)
    df.to_csv("features_per_window_summary.csv", index=False)
    print("\nSummary table saved → features_per_window_summary.csv")
    print(df.round(2))
else:
    print("\nNo data collected.")

print("\nDone. All graphs saved in folder: features_per_window_graphs/") 