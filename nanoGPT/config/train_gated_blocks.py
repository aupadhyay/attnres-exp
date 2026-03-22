# Block AttnRes + Gated Block Outputs
# Per-layer gates suppress noisy contributions within blocks

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'gated-blocks'

out_dir = 'out-gated-blocks'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

max_iters = 10000
lr_decay_iters = 10000

eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AttnRes: block with gated outputs
residual_mode = 'block_attnres'
attnres_n_blocks = 4
use_gated_blocks = True
gate_init_bias = 2.0

compile = False

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
