"""
Modal wrapper for running AttnRes training on remote A100.

Usage:
    modal run modal_train.py --variant baseline
    modal run modal_train.py --variant full_attnres
    modal run modal_train.py --variant block_attnres
"""

import modal
import os

app = modal.App("attnres-experiments")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv")
    .run_commands(
        "uv pip install --system "
        "torch==2.2.0 numpy tiktoken datasets transformers wandb tqdm requests"
    )
)

vol = modal.Volume.from_name("attnres-data", create_if_missing=True)

VARIANT_TO_CONFIG = {
    "baseline": "config/train_attnres_baseline.py",
    "full_attnres": "config/train_attnres_full.py",
    "block_attnres": "config/train_attnres_block.py",
}

VARIANT_TO_OUTDIR = {
    "baseline": "out-attnres-baseline",
    "full_attnres": "out-attnres-full",
    "block_attnres": "out-attnres-block",
}


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(variant: str = "baseline"):
    import subprocess
    import shutil

    # Copy nanoGPT code to working directory
    work_dir = "/root/nanoGPT"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree("/data/nanoGPT", work_dir)

    # Ensure data is available — check volume cache first, then prepare if needed
    data_dir = os.path.join(work_dir, "data", "openwebtext")
    if os.path.exists("/data/openwebtext/train.bin"):
        print("Using cached data from volume")
        shutil.copy2("/data/openwebtext/train.bin", os.path.join(data_dir, "train.bin"))
        shutil.copy2("/data/openwebtext/val.bin", os.path.join(data_dir, "val.bin"))
    elif not os.path.exists(os.path.join(data_dir, "train.bin")):
        print("Preparing OpenWebText data...")
        subprocess.run(
            ["python", "prepare.py"],
            cwd=data_dir,
            check=True,
        )
        vol.commit()

    # Point output dir directly at the persistent volume so checkpoints
    # survive even if the container dies mid-training
    dest_dir = f"/data/checkpoints/{variant}"
    os.makedirs(dest_dir, exist_ok=True)

    # Auto-resume if a checkpoint exists from a previous run
    has_checkpoint = os.path.exists(os.path.join(dest_dir, "ckpt.pt"))
    init_from = "resume" if has_checkpoint else "scratch"
    if has_checkpoint:
        print(f"Found existing checkpoint in {dest_dir}, resuming training")

    config = VARIANT_TO_CONFIG[variant]
    print(f"Starting training: {variant} ({config}), init_from={init_from}")

    result = subprocess.run(
        ["python", "train.py", config,
         f"--out_dir={dest_dir}", f"--init_from={init_from}"],
        cwd=work_dir,
        check=True,
    )

    vol.commit()
    print(f"Training complete. Checkpoints at {dest_dir}")


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/data": vol},
)
def prepare_data():
    """One-time data preparation."""
    import subprocess
    import shutil

    work_dir = "/root/nanoGPT"
    shutil.copytree("/data/nanoGPT", work_dir)

    data_dir = os.path.join(work_dir, "data", "openwebtext")
    subprocess.run(["python", "prepare.py"], cwd=data_dir, check=True)

    # Save back to volume
    shutil.copytree(
        os.path.join(data_dir),
        "/data/openwebtext",
        dirs_exist_ok=True,
    )
    vol.commit()
    print("Data prepared and saved to volume.")


@app.local_entrypoint()
def main(variant: str = "baseline"):
    train.remote(variant=variant)
