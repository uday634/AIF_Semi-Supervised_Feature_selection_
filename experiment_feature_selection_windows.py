# ────────────────────────────────────────────────────────────────
# IMPORTANT: use non-GUI backend (prevents Tkinter + JVM crash)
# ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.anomaly._adaptive_isolation_forest import AdaptiveIsolationForest
from capymoa.anomaly.adaptive_isolation_forest_logistic_fs import (
    AdaptiveIsolationForestWithLogisticFS,
)
from capymoa.anomaly.adaptive_isolation_forest_anova_fs import (
    AdaptiveIsolationForestWithAnovaFS,
)

# Silence ANOVA numerical warnings (expected in semi-supervised streams)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in divide",
)

# ────────────────────────────────────────────────────────────────
# NPZ Stream Loader
# ────────────────────────────────────────────────────────────────
class NPZStream:
    def __init__(self, path):
        data = np.load(path)
        self.X = data["X"].astype(float)
        self.y = data["y"].astype(int)

        self.i = 0
        self.n = len(self.X)

        features = [f"f{i}" for i in range(self.X.shape[1])]
        self.schema = Schema.from_custom(
            features=features + ["class"],
            target="class",
            categories={"class": ["0", "1"]},
            name=os.path.basename(path),
        )

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        x = self.X[self.i]
        y = self.y[self.i]
        self.i += 1
        inst = Instance.from_array(self.schema, np.append(x, y))
        return inst, y


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
DATASET_DIR = "./semi_supervised_Datasets"
WINDOW_SIZE = 256
N_TREES = 50
SEED = 42
LABEL_BUDGET = 0.01


# ────────────────────────────────────────────────────────────────
# Run experiment
# ────────────────────────────────────────────────────────────────
for fname in os.listdir(DATASET_DIR):
    if not fname.endswith(".npz"):
        continue

    print(f"\nProcessing {fname}")

    stream = NPZStream(os.path.join(DATASET_DIR, fname))
    schema = stream.schema

    # Models
    aif_log = AdaptiveIsolationForestWithLogisticFS(
        schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=SEED,
        label_budget=LABEL_BUDGET,
    )

    aif_anova = AdaptiveIsolationForestWithAnovaFS(
        schema,
        window_size=WINDOW_SIZE,
        n_trees=N_TREES,
        seed=SEED,
        label_budget=LABEL_BUDGET,
    )

    logistic_counts = []
    anova_counts = []

    # ────────────────────────────────────────────────────────────
    # Stream processing
    # ────────────────────────────────────────────────────────────
    while stream.has_more_instances():
        inst, y = stream.next_instance()

        aif_log.train(inst, label=y)
        aif_anova.train(inst, label=y)

        # Append ONLY when a window completes
        if (
            hasattr(aif_log, "last_selected_count")
            and aif_log.last_selected_count is not None
            and len(aif_log.instances) == 0
        ):
            logistic_counts.append(aif_log.last_selected_count)

        if (
            hasattr(aif_anova, "last_selected_count")
            and aif_anova.last_selected_count is not None
            and len(aif_anova.instances) == 0
        ):
            anova_counts.append(aif_anova.last_selected_count)

    # ────────────────────────────────────────────────────────────
    # Align windows
    # ────────────────────────────────────────────────────────────
    min_len = min(len(logistic_counts), len(anova_counts))
    logistic_counts = logistic_counts[:min_len]
    anova_counts = anova_counts[:min_len]

    windows = np.arange(min_len)

    print(f"Total completed windows: {min_len}")

    # ────────────────────────────────────────────────────────────
    # Plot
    # ────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))

    plt.plot(
        windows,
        logistic_counts,
        label="Logistic FS",
        linewidth=2,
        marker="o",
        markersize=3,
    )

    plt.plot(
        windows,
        anova_counts,
        label="ANOVA FS",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    plt.xlabel("Window Index")
    plt.ylabel("Number of Selected Features")
    plt.title(f"Feature Selection per Window\n{fname}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_name = fname.replace(".npz", "_features_per_window.png")
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f"Saved plot → {out_name}")

print("\n✔ Feature selection per-window plotting complete.")
