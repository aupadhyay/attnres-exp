"""Export per-token routing data for the Remotion animation."""
import json, sys, os, torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPT, GPTConfig

PROMPT = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]"
)

ckpt = torch.load("data/v2/block_attnres/ckpt.pt", map_location="cpu")
config = GPTConfig(**ckpt["model_args"])
model = GPT(config)
sd = ckpt["model"]
for k in list(sd.keys()):
    if k.startswith("_orig_mod."): sd[k[10:]] = sd.pop(k)
model.load_state_dict(sd)
model.eval()
print("Model loaded")

enc = tiktoken.get_encoding("gpt2")
token_ids = enc.encode(PROMPT)
tokens = [enc.decode([t]) for t in token_ids]
T = len(token_ids)
print(f"{T} tokens: {tokens}")

with torch.no_grad():
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    pos = torch.arange(0, T)
    x = model.transformer.drop(model.transformer.wte(input_ids) + model.transformer.wpe(pos))
    emb_per_block = []
    block_outputs = [x]
    for bi in range(model.config.attnres_n_blocks):
        for layer in model.transformer.h[bi*model.layers_per_block:(bi+1)*model.layers_per_block]:
            x = layer(x)
        block_outputs.append(x)
        alpha = model.depth_attn[bi].get_alpha(block_outputs)
        emb_per_block.append(alpha[0, 0, :].tolist())
        x = model.depth_attn[bi](block_outputs)
    print("Forward pass done")

# Split flat token list into lines, preserving original indices.
# \n tokens are appended as {"tok": "\n", "idx": i} at end of their line
# so the React component can use their emb value for the line-end highlight.
lines = []
current = []
for i, tok in enumerate(tokens):
    if '\n' in tok:
        before, after = tok.split('\n', 1)
        if before:
            current.append({"tok": before, "idx": i})
        current.append({"tok": "\n", "idx": i})  # keep \n at line end
        lines.append(current)
        current = [{"tok": after, "idx": i}] if after else []
    else:
        current.append({"tok": tok, "idx": i})
if current:
    lines.append(current)

print(f"\nLines:")
for li, line in enumerate(lines):
    print(f"  {li}: {[t['tok'] for t in line]}")

print(f"\nB4 emb at line starts:")
b4 = emb_per_block[3]
for li, line in enumerate(lines):
    if line:
        print(f"  '{line[0]['tok']}' idx={line[0]['idx']} emb={b4[line[0]['idx']]:.3f}")

os.makedirs("viz/src", exist_ok=True)
json.dump({"tokens": tokens, "lines": lines, "emb_per_block": emb_per_block, "n_blocks": 4},
          open("viz/src/routing_data.json", "w"), indent=2)
print("\nSaved viz/src/routing_data.json")
