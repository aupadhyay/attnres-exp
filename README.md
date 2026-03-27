# Attention Residuals + Value Residuals

Playing around with Attention Residuals on NanoGPT. Implemented a few variants, including adaptive block boundaries and implementing value residual learning.

* [Attention Residuals paper](https://arxiv.org/abs/2603.15031)
* [Value Residuals paper](https://arxiv.org/abs/2410.17897)
* [Writeup](https://axu.sh/post/attention-residuals)

## Setup

```bash
uv sync
uv run python -m pytest tests/ -v
```

## Training

```bash
# launch on Modal (detached, runs on A100)
modal run --detach modal_train.py --variant baseline # baseline (standard PreNorm residuals)
modal run --detach modal_train.py --variant full_attnres # full AttnRes
modal run --detach modal_train.py --variant block_attnres # block AttnRes
modal run --detach modal_train.py --variant adaptive_attnres # adaptive block boundaries
modal run --detach modal_train.py --variant value_residual # block AttnRes + value residual learning
modal run --detach modal_train.py --variant value_residual_only # value residual learning only
```

## Visuals

The `analyze/` directory has scripts to generate:
* Depth attention heatmaps
* Training dynamics (gradient norms, activation magnitudes)
* Loss curve comparisons
* Adaptive boundary gate plots
* Value residual lambda-by-layer charts
* Query vector PCA/cosine similarity plots
* Per-token routing visualizations (these weren't very useful)
