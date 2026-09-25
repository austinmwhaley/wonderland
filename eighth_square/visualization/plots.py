import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ema(values, window):
    if window <= 1 or len(values) <= 1:
        return np.asarray(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_run(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(metrics_path):
        return None
    rows = []
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    meta = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            meta = json.load(f)
    return {"dir": run_dir, "name": os.path.basename(run_dir), "meta": meta, "rows": rows}


def load_all_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        r = load_run(d)
        if r:
            runs.append(r)
    return runs


def metric_series(run, key):
    xs, ys = [], []
    for row in run["rows"]:
        if key in row:
            xs.append(row.get("timestep", len(xs)))
            ys.append(row[key])
    return np.asarray(xs), np.asarray(ys)


def filter_runs(runs, filters):
    out = []
    for r in runs:
        meta = r["meta"]
        if all(meta.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def plot_runs(
    runs,
    metric="return",
    ema_window=20,
    out_path=None,
    show_loss=False,
    title=None,
    xlabel="timestep",
    ylabel=None,
):
    if not runs:
        raise ValueError("no runs matched the filters")
    fig, ax = plt.subplots(1, 2 if show_loss else 1, figsize=(12, 4.5))
    axes = ax if show_loss else [ax]
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for run in runs:
        xs, ys = metric_series(run, metric)
        if len(ys) == 0:
            continue
        if ema_window and len(ys) > ema_window:
            xs, ys = xs[ema_window - 1 :], ema(ys, ema_window)
        label = run["name"]
        if run["meta"].get("tag"):
            label = f"{run['meta']['algo']}_{run['meta']['env']}_{run['meta']['tag']}"
        axes[0].plot(xs, ys, label=label, linewidth=1.2)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel or metric)
    axes[0].set_title(title or f"{metric} over time")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    if show_loss:
        for run in runs:
            xs, ys = metric_series(run, "loss")
            if len(ys) == 0:
                continue
            if ema_window and len(ys) > ema_window:
                xs, ys = xs[ema_window - 1 :], ema(ys, ema_window)
            axes[1].plot(xs, ys, linewidth=1.0)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("loss")
        axes[1].set_title("loss")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path
    return fig
