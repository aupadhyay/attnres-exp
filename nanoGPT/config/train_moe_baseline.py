# MoE baseline (no AttnRes) for comparison
# Single A100, ~4-5 hours (more params from experts)

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'moe-baseline'

out_dir = 'out-moe-baseline'

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

# AttnRes: disabled
residual_mode = 'baseline'

# MoE: enabled
use_moe = True
num_experts = 8
moe_top_k = 2
moe_aux_loss_coeff = 0.01

compile = False  # MoE dynamic routing may cause compile issues

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
