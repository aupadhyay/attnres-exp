# MoE + Block AttnRes (for Agent 4 comparison)

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'moe-block-attnres'

out_dir = 'out-moe-attnres'

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

# AttnRes: block mode
residual_mode = 'block_attnres'
attnres_n_blocks = 4

# MoE: enabled
use_moe = True
num_experts = 8
moe_top_k = 2
moe_aux_loss_coeff = 0.01

compile = False

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
