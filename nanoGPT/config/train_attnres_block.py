# Block AttnRes (N=4) -- 12 layers / 4 blocks = 3 layers per block
# Single A100, ~2-3 hours

wandb_log = True
wandb_project = 'attnres'
wandb_run_name = 'block-attnres-n4'

out_dir = 'out-attnres-block'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

max_iters = 10000
lr_decay_iters = 10000

eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True  # save at every eval for dynamics analysis

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AttnRes: block with 4 blocks
residual_mode = 'block_attnres'
attnres_n_blocks = 4
compile = False  # dynamic layer_outputs list causes torch.compile issues

learning_rate = 6e-4
weight_decay = 1e-1
warmup_iters = 2000
min_lr = 6e-5
