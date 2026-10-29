"""Plot per-split ACC and LOSS curves (Plan A: one figure per split, all models merged).

Usage:
    python plot_curves.py [--out-dir outputs]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = {"vgg16": "#1f77b4", "googlenet": "#ff7f0e", "resnet34": "#2ca02c",
          "resnet50": "#d62728", "mobilenet_v2": "#9467bd"}
MARKERS = {"vgg16": "o", "googlenet": "s", "resnet34": "^",
           "resnet50": "D", "mobilenet_v2": "v"}


def plot_all(out_dir: Path):
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for h in sorted(out_dir.glob("*/history.json")):
        histories[h.parent.name] = json.loads(h.read_text(encoding="utf-8"))
    if not histories:
        print("no history.json found, run train.py first")
        return

    for metric in ["acc", "loss"]:
        for split in ["train", "val", "test"]:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            for name, hist in histories.items():
                epochs = [e["epoch"] for e in hist["epochs"]]
                values = [e[f"{split}_{metric}"] for e in hist["epochs"]]
                ax.plot(epochs, values, label=name, linewidth=1.6,
                        color=COLORS.get(name), marker=MARKERS.get(name),
                        markersize=3, markevery=max(1, len(epochs) // 10))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy" if metric == "acc" else "Loss")
            ax.set_title(f"{split.capitalize()} {metric.capitalize()}")
            if metric == "acc":
                ax.set_ylim(0, 1.02)
                ax.grid(True, linestyle="--", alpha=0.4)
            else:
                ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", framealpha=0.9)
            fig.tight_layout()
            path = curves_dir / f"{metric}_{split}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()
    plot_all(ROOT / args.out_dir)
