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

    # Ensure data is prepared
    data_dir = os.path.join(work_dir, "data", "openwebtext")
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print("Preparing OpenWebText data...")
        subprocess.run(
            ["python", "prepare.py"],
            cwd=data_dir,
            check=True,
        )
        # Cache prepared data back to volume
        vol.commit()

    config = VARIANT_TO_CONFIG[variant]
    print(f"Starting training: {variant} ({config})")

    result = subprocess.run(
        ["python", "train.py", config],
        cwd=work_dir,
        check=True,
    )

    # Copy checkpoints to persistent volume
    out_dir = os.path.join(work_dir, VARIANT_TO_OUTDIR[variant])
    dest_dir = f"/data/checkpoints/{variant}"
    if os.path.exists(out_dir):
        shutil.copytree(out_dir, dest_dir, dirs_exist_ok=True)
        vol.commit()
        print(f"Checkpoints saved to {dest_dir}")


@app.function(
    image=image,
    gpu="A100",
    timeout=30 * 60,
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
