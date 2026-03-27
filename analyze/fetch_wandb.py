"""
Fetch all training data from wandb for the attnres project.

Usage:
    uv run python analyze/fetch_wandb.py
    uv run python analyze/fetch_wandb.py --project attnres --out analyze/wandb_data

Outputs:
    - wandb_data/runs_summary.json  — summary stats per run
    - wandb_data/loss_curves.csv    — train/val loss over time for all runs
    - wandb_data/boundary_gates.csv — adaptive boundary gate values (if available)
    - wandb_data/plots/             — loss curve plots
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import wandb


def fetch_runs(project: str, entity: str | None = None):
    """Fetch all finished runs from the project."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": "finished"})
    print(f"Found {len(runs)} finished runs in '{path}'")
    return runs


def fetch_loss_curves(runs) -> pd.DataFrame:
    """Pull train/val loss history for all runs."""
    all_rows = []
    for run in runs:
        print(f"  Fetching history for '{run.name}' ({run.id})...")
        history = run.scan_history(
            keys=["iter", "train/loss", "val/loss", "lr", "mfu"],
            page_size=10000,
        )
        for row in history:
            if "val/loss" in row and row["val/loss"] is not None:
                all_rows.append({
                    "run_name": run.name,
                    "run_id": run.id,
                    "iter": row.get("iter"),
                    "train_loss": row.get("train/loss"),
                    "val_loss": row.get("val/loss"),
                    "lr": row.get("lr"),
                    "mfu": row.get("mfu"),
                })
    return pd.DataFrame(all_rows)


def fetch_boundary_data(runs) -> pd.DataFrame:
    """Pull adaptive boundary gate values if logged."""
    all_rows = []
    for run in runs:
        if "adaptive" not in run.name.lower() and "boundary" not in run.name.lower():
            continue
        print(f"  Fetching boundary data for '{run.name}'...")
        # Get all boundary/* keys
        history = run.scan_history(page_size=10000)
        for row in history:
            gate_keys = [k for k in row.keys() if k.startswith("boundary/")]
            if gate_keys:
                entry = {
                    "run_name": run.name,
                    "run_id": run.id,
                    "iter": row.get("iter"),
                }
                for k in gate_keys:
                    entry[k] = row[k]
                all_rows.append(entry)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def fetch_run_summaries(runs) -> list[dict]:
    """Get summary stats for each run."""
    summaries = []
    for run in runs:
        summaries.append({
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": run.created_at,
            "runtime_seconds": run.summary.get("_runtime"),
            "best_val_loss": run.summary.get("val/loss"),
            "final_train_loss": run.summary.get("train/loss"),
            "final_iter": run.summary.get("iter"),
            "config": dict(run.config),
        })
    return summaries


def plot_loss_curves(df: pd.DataFrame, out_dir: Path):
    """Plot val loss curves for all runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Val loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in df.groupby("run_name"):
        group = group.sort_values("iter")
        ax.plot(group["iter"], group["val_loss"], label=name, marker=".", markersize=3)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("AttnRes Experiments — Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "val_loss_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved val_loss_comparison.png")

    # Train loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in df.groupby("run_name"):
        group = group.sort_values("iter")
        ax.plot(group["iter"], group["train_loss"], label=name, marker=".", markersize=3)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Training Loss")
    ax.set_title("AttnRes Experiments — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved train_loss_comparison.png")


def plot_boundary_gates(df: pd.DataFrame, out_dir: Path):
    """Plot boundary gate evolution over training."""
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gate_cols = [c for c in df.columns if c.startswith("boundary/gate_")]
    if not gate_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gate values over training
    ax = axes[0]
    for col in sorted(gate_cols):
        layer_idx = col.split("_")[-1]
        ax.plot(df["iter"], df[col], label=f"layer {layer_idx}", alpha=0.7)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Gate Value (sigmoid)")
    ax.set_title("Boundary Gates Over Training")
    ax.legend(fontsize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Effective number of boundaries
    ax = axes[1]
    if "boundary/n_effective" in df.columns:
        ax.plot(df["iter"], df["boundary/n_effective"], color="purple")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Effective # Boundaries")
        ax.set_title("Effective Block Count Over Training")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "boundary_gates.png", dpi=150)
    plt.close()
    print(f"  Saved boundary_gates.png")


def main():
    parser = argparse.ArgumentParser(description="Fetch wandb data for attnres experiments")
    parser.add_argument("--project", default="attnres")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--out", default="analyze/wandb_data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching runs...")
    runs = fetch_runs(args.project, args.entity)

    print("\nFetching run summaries...")
    summaries = fetch_run_summaries(runs)
    with open(out_dir / "runs_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"  Saved runs_summary.json ({len(summaries)} runs)")

    # Print quick comparison table
    print("\n=== Results Summary ===")
    print(f"{'Run':<30} {'Val Loss':>10} {'Train Loss':>12} {'Iters':>8}")
    print("-" * 65)
    for s in sorted(summaries, key=lambda x: x.get("best_val_loss") or 999):
        print(f"{s['name']:<30} {s.get('best_val_loss', '?'):>10} "
              f"{s.get('final_train_loss', '?'):>12} {s.get('final_iter', '?'):>8}")

    print("\nFetching loss curves...")
    loss_df = fetch_loss_curves(runs)
    loss_df.to_csv(out_dir / "loss_curves.csv", index=False)
    print(f"  Saved loss_curves.csv ({len(loss_df)} rows)")

    print("\nFetching boundary data...")
    boundary_df = fetch_boundary_data(runs)
    if not boundary_df.empty:
        boundary_df.to_csv(out_dir / "boundary_gates.csv", index=False)
        print(f"  Saved boundary_gates.csv ({len(boundary_df)} rows)")
    else:
        print("  No boundary data found")

    print("\nGenerating plots...")
    plot_loss_curves(loss_df, out_dir)
    plot_boundary_gates(boundary_df, out_dir)

    print(f"\nDone! All data saved to {out_dir}/")


if __name__ == "__main__":
    main()
