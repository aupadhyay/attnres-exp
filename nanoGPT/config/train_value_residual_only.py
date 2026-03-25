# Value Residual Learning ONLY (no AttnRes) -- ablation
# Tests whether value residuals help on their own

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'value-residual-only'

out_dir = 'out-value-residual-only'

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

# AttnRes: disabled (baseline residuals)
residual_mode = 'baseline'

# Value residual: enabled
use_value_residual = True
value_residual_mode = 'learnable_per_layer'
value_residual_lambda_init = 0.0

compile = False

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
