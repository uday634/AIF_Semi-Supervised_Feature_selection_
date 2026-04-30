# experiment_active_globalLS_only.py
# Active GlobalLS — fixed all bugs, per-run progression plots + CSVs

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
from collections import deque
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from capymoa.instance import Instance
from capymoa.stream import Schema
from capymoa.anomaly.AdaptiveIsolationForestWithLogisticFSGlobal import AdaptiveIsolationForestWithLogisticFS


# ========================= CONFIG =========================
DATA_DIR       = "./semi_supervised_Datasets"
SUMMARY_CSV    = "active_globalLS_summary.csv"
PER_RUN_CSV    = "active_globalLS_per_run_auc.csv"
PROG_CSV_DIR   = "progression_csv"     # per-dataset progression CSVs
PROG_PLOT_DIR  = "progression_plots"   # per-dataset progression plots
PLOT_BAR       = "active_globalLS_final_auc.png"

N_RUNS         = 10
WINDOW_SIZE    = 256
LABEL_BUDGET   = 0.025
L1_STRENGTH    = 0.1
ROLLING_WINDOW = 500
ROLLING_FREQ   = 200


# ========================= STREAM =========================
class NPZStream:
    def __init__(self, path):
        data        = np.load(path, mmap_mode='r')
        X           = data["X"].astype(np.float64)
        y           = data["y"].ravel()
        le          = LabelEncoder()
        y_idx       = le.fit_transform(y)
        self.n      = len(X)
        self.i      = 0
        feat_names  = [f"feature_{j}" for j in range(X.shape[1])]
        self.schema = Schema.from_custom(
            features   = feat_names + ["class"],
            target     = "class",
            categories = {"class": [str(c) for c in le.classes_]},
            name       = os.path.basename(path)
        )
        self.data = list(zip(X, y_idx))

    def has_more_instances(self):
        return self.i < self.n

    def next_instance(self):
        x, y = self.data[self.i]
        inst = Instance.from_array(self.schema, np.append(x, [y]))
        self.i += 1
        return inst, y


# ========================= UTILS =========================
def safe_auc(y, scores):
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, scores)
        return auc(fpr, tpr)
    return 0.5


# ========================= WORKER =========================
def run_single(args):
    ds_name, run = args
    seed   = 42 + run * 13
    stream = NPZStream(os.path.join(DATA_DIR, ds_name))

    model = AdaptiveIsolationForestWithLogisticFS(
        schema       = stream.schema,
        window_size  = WINDOW_SIZE,
        n_trees      = 100,
        seed         = seed,
        label_budget = LABEL_BUDGET,
        l1_strength  = L1_STRENGTH
    )

    y_all, scores_all  = [], []
    win_y      = deque(maxlen=ROLLING_WINDOW)
    win_scores = deque(maxlen=ROLLING_WINDOW)

    # progression: list of (step, auc_value)
    progression = []
    step        = 0

    start_time = time.time()
    process    = psutil.Process()
    peak_mem   = process.memory_info().rss / (1024 * 1024)

    while stream.has_more_instances():
        inst, y = stream.next_instance()
        step   += 1

        s = model.score_instance(inst)
        y_all.append(y);        scores_all.append(s)
        win_y.append(y);        win_scores.append(s)

        # Budget enforced inside train() — only labels budget-selected instances
        model.train(inst, y)

        if step % ROLLING_FREQ == 0 and len(np.unique(win_y)) > 1:
            progression.append((step, safe_auc(win_y, win_scores)))

        # Limit psutil syscall frequency
        if step % 500 == 0:
            peak_mem = max(peak_mem, process.memory_info().rss / (1024 * 1024))

    return {
        "dataset"    : ds_name,
        "run"        : run,
        "final_auc"  : safe_auc(y_all, scores_all),
        "time"       : time.time() - start_time,
        "mem"        : peak_mem,
        "progression": progression,   # [(step, auc), ...]
    }


# ========================= SAVE CSVs =========================
def save_per_run_csv(results):
    """Flat CSV: dataset, run, AUC — sorted by dataset then run."""
    rows = [{"dataset": r["dataset"], "run": r["run"], "AUC": r["final_auc"]}
            for r in results]
    df = (pd.DataFrame(rows)
            .sort_values(["dataset", "run"])
            .reset_index(drop=True))
    df.to_csv(PER_RUN_CSV, index=False)
    print(f"• Per-run AUC CSV  : {PER_RUN_CSV}")


def save_progression_csvs(results, datasets):
    """
    One CSV per dataset.
    Columns: step, run_0, run_1, ..., run_N-1, mean, std
    Uses union of all checkpoint steps — missing entries filled with NaN.
    """
    os.makedirs(PROG_CSV_DIR, exist_ok=True)

    for ds in datasets:
        ds_res = sorted(
            [r for r in results if r["dataset"] == ds],
            key=lambda r: r["run"]
        )

        # Union of all steps across runs — handles unequal lengths safely
        all_steps = sorted(set(p[0] for r in ds_res for p in r["progression"]))
        if not all_steps:
            continue

        df = pd.DataFrame({"step": all_steps})
        run_cols = []

        for r in ds_res:
            col  = f"run_{r['run']}"
            amap = dict(r["progression"])
            df[col] = [amap.get(s, np.nan) for s in all_steps]
            run_cols.append(col)

        df["mean"] = df[run_cols].mean(axis=1)
        df["std"]  = df[run_cols].std(axis=1)

        safe = ds.replace(".npz", "").replace("-", "_")
        df.to_csv(f"{PROG_CSV_DIR}/{safe}_progression.csv", index=False)

    print(f"• Progression CSVs : {PROG_CSV_DIR}/")


# ========================= PLOTS =========================
def plot_bar(df):
    x      = np.arange(len(df))
    labels = [d.replace(".npz", "") for d in df["dataset"]]

    plt.figure(figsize=(max(10, len(df) * 1.2), 6))
    plt.bar(x, df["AUC_mean"], yerr=df["AUC_std"],
            capsize=5, color="orange", alpha=0.85)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean AUC-ROC")
    plt.ylim(0, 1.05)
    plt.title(f"Active GlobalLS — Final AUC  ({N_RUNS} runs, label_budget={LABEL_BUDGET})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_BAR, dpi=300)
    plt.close()
    print(f"• Final AUC bar    : {PLOT_BAR}")


def plot_progression(results, datasets):
    """
    One plot per dataset:
      - Faint coloured line per individual run  ← fixes the missing per-run lines bug
      - Bold black mean line
      - Grey ±std shaded band
    Uses union step axis + NaN alignment — fixes the ragged array crash bug.
    """
    os.makedirs(PROG_PLOT_DIR, exist_ok=True)
    cmap = plt.cm.get_cmap("tab10", N_RUNS)

    for ds in datasets:
        ds_res = sorted(
            [r for r in results if r["dataset"] == ds],
            key=lambda r: r["run"]
        )
        if not ds_res or not ds_res[0]["progression"]:
            continue

        # Union step axis — safe when runs have different checkpoint counts
        all_steps = sorted(set(p[0] for r in ds_res for p in r["progression"]))
        if not all_steps:
            continue

        # Align each run to common steps, NaN where checkpoint missing
        aligned = []
        for r in ds_res:
            amap = dict(r["progression"])
            aligned.append([amap.get(s, np.nan) for s in all_steps])

        auc_array = np.array(aligned, dtype=float)          # (n_runs, n_steps)
        mean_auc  = np.nanmean(auc_array, axis=0)
        std_auc   = np.nanstd(auc_array, axis=0)

        fig, ax = plt.subplots(figsize=(11, 5))

        # ---- Individual run lines (faint) ----
        for i, r in enumerate(ds_res):
            ax.plot(all_steps, aligned[i],
                    color=cmap(i), alpha=0.35, linewidth=1.0,
                    label=f"run {r['run']}")

        # ---- Mean ± std (bold) ----
        ax.plot(all_steps, mean_auc,
                color="black", linewidth=2.2, label="mean", zorder=5)
        ax.fill_between(all_steps,
                        mean_auc - std_auc,
                        mean_auc + std_auc,
                        color="black", alpha=0.12, zorder=4)

        ax.set_xlabel("Instances seen")
        ax.set_ylabel("Rolling AUC-ROC")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"AUC Progression — {ds.replace('.npz', '')}\n"
            f"(window={ROLLING_WINDOW}, every {ROLLING_FREQ} inst, "
            f"label_budget={LABEL_BUDGET}, {N_RUNS} runs)"
        )
        ax.legend(ncol=4, fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        safe = ds.replace(".npz", "").replace("-", "_")
        plt.savefig(f"{PROG_PLOT_DIR}/progression_{safe}.png", dpi=200)
        plt.close()

    print(f"• Progression plots: {PROG_PLOT_DIR}/")


# ========================= MAIN =========================
if __name__ == "__main__":
    datasets    = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
    tasks       = [(ds, run) for ds in datasets for run in range(N_RUNS)]
    n_workers   = mp.cpu_count()
    total_tasks = len(tasks)

    print("=" * 60)
    print("  Active GlobalLS Experiment")
    print("=" * 60)
    print(f"  Datasets     : {len(datasets)}")
    print(f"  Runs each    : {N_RUNS}")
    print(f"  Total tasks  : {total_tasks}")
    print(f"  Workers      : {n_workers}")
    print(f"  Label budget : {LABEL_BUDGET}  "
          f"(~{int(WINDOW_SIZE * LABEL_BUDGET)} labels/window)")
    print(f"  Roll window  : {ROLLING_WINDOW}  freq={ROLLING_FREQ}")
    print("=" * 60 + "\n")

    results   = []
    completed = 0
    t0        = time.time()

    with mp.Pool(processes=n_workers) as pool:
        it = pool.imap_unordered(run_single, tasks)

        if tqdm is not None:
            it = tqdm(it, total=total_tasks, desc="Tasks",
                      unit="task", dynamic_ncols=True)

        for res in it:
            results.append(res)
            completed += 1
            elapsed = time.time() - t0
            eta     = (elapsed / completed) * (total_tasks - completed)
            msg = (f"  ✔ {completed}/{total_tasks}"
                   f"  {res['dataset']}  run {res['run']}"
                   f"  AUC={res['final_auc']:.4f}"
                   f"  ETA={eta/60:.1f}m")
            # tqdm.write keeps the progress bar intact
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    print(f"\nAll done in {(time.time() - t0)/60:.1f} min\n")

    # ---- Summary ----
    summary = []
    for ds in datasets:
        ds_res = [r for r in results if r["dataset"] == ds]
        summary.append({
            "dataset"    : ds,
            "AUC_mean"   : np.mean([r["final_auc"] for r in ds_res]),
            "AUC_std"    : np.std( [r["final_auc"] for r in ds_res]),
            "time_mean"  : np.mean([r["time"]       for r in ds_res]),
            "mem_peak_MB": np.mean([r["mem"]         for r in ds_res]),
            "n_runs"     : len(ds_res),
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"• Summary CSV      : {SUMMARY_CSV}\n")
    print(df_summary.round(4).to_string(index=False))
    print()

    # ---- All outputs ----
    save_per_run_csv(results)
    save_progression_csvs(results, datasets)
    plot_bar(df_summary)
    plot_progression(results, datasets)

    print("\n✅ EXPERIMENT COMPLETE")
    print(f"   Summary CSV      : {SUMMARY_CSV}")
    print(f"   Per-run CSV      : {PER_RUN_CSV}")
    print(f"   Progression CSVs : {PROG_CSV_DIR}/")
    print(f"   Progression plots: {PROG_PLOT_DIR}/")
    print(f"   Final AUC bar    : {PLOT_BAR}")