import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run import RESULTS_DIR
from visualization.plots import filter_runs, load_all_runs, plot_runs


def main():
    parser = argparse.ArgumentParser(description="Visualize logged experiment runs.")
    parser.add_argument("--filter", action="append", default=[],
                        help="key=value filter on run config, repeatable (e.g. --filter algo=dqn --filter env=cartpole)")
    parser.add_argument("--metric", default="return", help="metric to plot (return, eval_return, regret, loss)")
    parser.add_argument("--ema", type=int, default=20, help="exponential moving average window (0 = raw)")
    parser.add_argument("--loss", action="store_true", help="add a loss subplot")
    parser.add_argument("--out", default=None, help="output png path (default: results/plots/<metric>.png)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    filters = {}
    for f in args.filter:
        k, _, v = f.partition("=")
        filters[k] = int(v) if v.isdigit() else v

    runs = filter_runs(load_all_runs(RESULTS_DIR), filters)
    if not runs:
        sys.exit("no runs found. Train something first, e.g. "
                 "python run_experiment.py --algo dqn --env cartpole --steps 30000")
    out = args.out or os.path.join(RESULTS_DIR, "plots", f"{args.metric}_comparison.png")
    plot_runs(runs, metric=args.metric, ema_window=args.ema, out_path=out,
              show_loss=args.loss, title=args.title)
    print(f"saved {len(runs)} run(s) -> {out}")


if __name__ == "__main__":
    main()