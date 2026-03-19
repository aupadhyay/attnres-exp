# Baseline (standard PreNorm residuals) for AttnRes comparison
# Single A100, ~2-3 hours

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'baseline'

out_dir = 'out-attnres-baseline'

# batch: 12 * 1024 * 40 = 491,520 tokens/iter (matching nanoGPT default effective batch)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# 100K steps * ~491K tokens/step ~ 49B tokens
max_iters = 100000
lr_decay_iters = 100000

eval_interval = 500
eval_iters = 200
log_interval = 10

# save checkpoints at regular intervals for dynamics analysis
always_save_checkpoint = True  # save at every eval for dynamics analysis

# model: GPT-2 small
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AttnRes: disabled (baseline)
residual_mode = 'baseline'

# optimizer
learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
