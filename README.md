# Experiments with Attention Residuals

Original paper: https://arxiv.org/pdf/2603.15031

## Current findings

- **Baseline reproduction**: adding AttnRes (full + block variants) to nanoGPT. trained on OpenWebText at GPT-2 124M scale. main finding so far: smaller gradient magnitudes across layers even at 12-layer scale. loss improvement is marginal (expected, the paper's gains come from deeper models)

## Working on

- **Adaptive block boundaries**: learn where to place block boundaries rather than using fixed block sizes. some layers might benefit from more frequent depth attention than others.
- **Gated block outputs**: add a sigmoid gate to each block output before depth attention. lets the model learn which layers are important within each block, rather than treating all block outputs equally.
- **MoE + AttnRes**: see if AttnRes provides better routing signal for MoE models. the depth attention weights could inform expert selection.

## Setup

```bash
uv sync
uv run python -m pytest tests/ -v
```

## Training

```bash
# launch on Modal (detached, runs on A100)
modal run --detach modal_train.py --variant baseline
modal run --detach modal_train.py --variant full_attnres
modal run --detach modal_train.py --variant block_attnres
```

## Analysis

```bash
# extract + plot training dynamics
uv run python analyze/extract_dynamics.py --ckpt checkpoints/baseline/ckpt.pt --output figures/dynamics_baseline.npz
uv run python analyze/plot_dynamics.py --dynamics-files baseline:figures/dynamics_baseline.npz full_attnres:figures/dynamics_full.npz --output figures/dynamics.png

# depth attention heatmaps
uv run python analyze/extract_depth_attention.py --ckpt checkpoints/full_attnres/ckpt.pt --output figures/alphas_full.npz
uv run python analyze/plot_depth_heatmaps.py --input figures/alphas_full.npz --output figures/depth_heatmap_full.png

# per-token routing analysis
uv run python analyze/plot_token_routing.py --ckpt checkpoints/full_attnres/ckpt.pt --output-dir figures
```
