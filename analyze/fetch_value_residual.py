"""
Fetch value residual experiment data from wandb and generate blog-ready plots.

Produces:
1. 4-way val loss comparison (baseline, block_attnres, value_residual_only, both)
2. Lambda-by-layer plot showing how much each layer pulls from token embeddings
3. Summary table for the blog

Usage:
    uv run python analyze/fetch_value_residual.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import wandb


OUT_DIR = Path("analyze/wandb_data/value_residual")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 4 runs we care about for the blog comparison
TARGET_RUNS = {
    "baseline": "baseline",
    "block-attnres-n4": "block_attnres",
    "value-residual-only": "value_residual_only",
    "value-residual-block-attnres": "value_residual_both",
}

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "block_attnres": "Block AttnRes",
    "value_residual_only": "Value Residual Only",
    "value_residual_both": "Block AttnRes + Value Residual",
}

COLORS = {
    "baseline": "#888888",
    "block_attnres": "#2196F3",
    "value_residual_only": "#FF9800",
    "value_residual_both": "#4CAF50",
}


def fetch_runs():
    """Fetch all relevant runs from wandb."""
    api = wandb.Api()
    all_runs = api.runs("attnres")

    matched = {}
    for r in all_runs:
        if r.name in TARGET_RUNS and r.state == "finished":
            key = TARGET_RUNS[r.name]
            # Keep the best run if duplicates
            if key not in matched or (r.summary.get("val/loss", 999) < matched[key].summary.get("val/loss", 999)):
                matched[key] = r

    # Also check for still-running "both" run
    for r in all_runs:
        if r.name in TARGET_RUNS and r.state == "running":
            key = TARGET_RUNS[r.name]
            if key not in matched:
                matched[key] = r
                print(f"  Note: '{r.name}' is still running (iter={r.summary.get('iter', '?')})")

    print(f"Matched {len(matched)} runs: {list(matched.keys())}")
    return matched


def fetch_loss_curves(runs: dict) -> dict[str, list[dict]]:
    """Fetch val/train loss history for each run."""
    curves = {}
    for key, run in runs.items():
        print(f"  Fetching loss curve for '{key}'...")
        history = list(run.scan_history(
            keys=["iter", "train/loss", "val/loss"],
            page_size=10000,
        ))
        curves[key] = [
            {"iter": row["iter"], "train_loss": row.get("train/loss"), "val_loss": row.get("val/loss")}
            for row in history
            if row.get("val/loss") is not None
        ]
    return curves


def fetch_lambda_history(runs: dict) -> dict[str, list[dict]]:
    """Fetch lambda-by-layer history for value residual runs."""
    lambda_data = {}
    for key, run in runs.items():
        if "value_residual" not in key:
            continue
        print(f"  Fetching lambda history for '{key}'...")
        history = list(run.scan_history(page_size=10000))

        # Find lambda keys from last row
        lambda_keys = sorted([k for k in history[-1].keys() if "lambda" in k])
        if not lambda_keys:
            print(f"    No lambda keys found, skipping")
            continue

        rows = []
        for row in history:
            if any(k in row and row[k] is not None for k in lambda_keys):
                entry = {"iter": row.get("iter")}
                for k in lambda_keys:
                    entry[k] = row.get(k)
                rows.append(entry)
        lambda_data[key] = {"keys": lambda_keys, "rows": rows}
    return lambda_data


def save_summary_table(runs: dict):
    """Save a summary comparison table."""
    rows = []
    for key, run in runs.items():
        rows.append({
            "variant": key,
            "display_name": DISPLAY_NAMES.get(key, key),
            "val_loss": run.summary.get("val/loss"),
            "train_loss": run.summary.get("train/loss"),
            "iter": run.summary.get("iter"),
            "state": run.state,
        })
    rows.sort(key=lambda x: x["val_loss"] or 999)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Print markdown table for the blog
    print("\n### Results Table (markdown)\n")
    print("| Variant | Val Loss | Train Loss |")
    print("|---------|----------|------------|")
    for r in rows:
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] else "running..."
        tl = f"{r['train_loss']:.4f}" if r["train_loss"] else "—"
        print(f"| {r['display_name']} | {vl} | {tl} |")

    return rows


def plot_loss_comparison(curves: dict):
    """Plot 4-way val loss comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for key in ["baseline", "block_attnres", "value_residual_only", "value_residual_both"]:
        if key not in curves:
            continue
        data = curves[key]
        iters = [d["iter"] for d in data]
        vals = [d["val_loss"] for d in data]
        ax.plot(iters, vals,
                label=DISPLAY_NAMES[key],
                color=COLORS[key],
                linewidth=2,
                marker="o", markersize=3)

    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Does Depth Attention Make Value Residuals Redundant?", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_comparison_4way.png", dpi=150)
    plt.close()
    print(f"  Saved loss_comparison_4way.png")


def plot_lambda_by_layer(lambda_data: dict):
    """Plot lambda values by layer at end of training.
    Shows how much each layer pulls from the original token embeddings."""

    for key, data in lambda_data.items():
        if not data["rows"]:
            continue

        lambda_keys = data["keys"]
        last_row = data["rows"][-1]

        # Parse layer indices from key names like "value_residual/lambda_layer_1"
        layer_lambdas = []
        for k in lambda_keys:
            parts = k.split("_")
            layer_idx = int(parts[-1])
            val = last_row.get(k)
            if val is not None:
                layer_lambdas.append((layer_idx, val))

        layer_lambdas.sort()
        layers = [l[0] for l in layer_lambdas]
        lambdas = [l[1] for l in layer_lambdas]

        # Bar chart: lambda by layer
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(layers, lambdas, color="#FF9800", edgecolor="#E65100", linewidth=0.5)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("λ (mixing weight for V₀)", fontsize=12)
        ax.set_title(f"Value Residual λ by Layer — {DISPLAY_NAMES.get(key, key)}", fontsize=13)
        ax.set_xticks(layers)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="init (0.5)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"lambda_by_layer_{key}.png", dpi=150)
        plt.close()
        print(f"  Saved lambda_by_layer_{key}.png")

        # Lambda evolution over training
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for k in lambda_keys:
            layer_idx = k.split("_")[-1]
            iters = [r["iter"] for r in data["rows"] if r.get(k) is not None]
            vals = [r[k] for r in data["rows"] if r.get(k) is not None]
            ax.plot(iters, vals, label=f"Layer {layer_idx}", linewidth=1.5)

        ax.set_xlabel("Training Iteration", fontsize=12)
        ax.set_ylabel("λ (mixing weight for V₀)", fontsize=12)
        ax.set_title(f"Value Residual λ Evolution — {DISPLAY_NAMES.get(key, key)}", fontsize=13)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"lambda_evolution_{key}.png", dpi=150)
        plt.close()
        print(f"  Saved lambda_evolution_{key}.png")

    # Save raw lambda data as CSV
    for key, data in lambda_data.items():
        if data["rows"]:
            with open(OUT_DIR / f"lambda_history_{key}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iter"] + data["keys"])
                writer.writeheader()
                writer.writerows(data["rows"])
            print(f"  Saved lambda_history_{key}.csv")


def main():
    print("Fetching runs from wandb...")
    runs = fetch_runs()

    print("\nSummary:")
    save_summary_table(runs)

    print("\nFetching loss curves...")
    curves = fetch_loss_curves(runs)

    print("\nFetching lambda history...")
    lambda_data = fetch_lambda_history(runs)

    print("\nGenerating plots...")
    plot_loss_comparison(curves)
    plot_lambda_by_layer(lambda_data)

    print(f"\nDone! All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
