# Adaptive Block Boundaries AttnRes
# Learns optimal block partitioning during training

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'adaptive-boundaries'

out_dir = 'out-adaptive-boundaries'

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

# AttnRes: adaptive boundaries
residual_mode = 'adaptive_attnres'
boundary_tau_start = 5.0
boundary_tau_end = 0.1
boundary_anneal_start_frac = 0.2
boundary_anneal_end_frac = 0.7
boundary_n_target = 4
boundary_reg_lambda = 0.1

compile = False  # dynamic block lists cause compile issues

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
