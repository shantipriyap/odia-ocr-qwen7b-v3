"""
Patch modeling_internvl_chat.py on the server to fix:
  1. image_flags.squeeze(-1) collapses shape [1] → [] (0-D scalar) when batch=1
  2. In except fallback, selected.sum() crashes on a 0-D bool tensor / Python bool
"""
import sys

CACHE_DIR = (
    "/root/.cache/huggingface/modules/transformers_modules/"
    "OpenGVLab/InternVL2_hyphen_8B/"
    "6fb9ad6924f69424e57fab2ab061d707688f0296"
)
fp = f"{CACHE_DIR}/modeling_internvl_chat.py"

txt = open(fp).read()
original = txt

# ── Fix 1: squeeze → view(-1) so the tensor stays 1-D regardless of length ────
old1 = "        image_flags = image_flags.squeeze(-1)"
new1 = "        image_flags = image_flags.view(-1)  # keep 1-D even when len==1"
if old1 in txt:
    txt = txt.replace(old1, new1, 1)
    print("✓ Fix 1 applied: squeeze → view(-1)")
else:
    print("✗ Fix 1 not applied: pattern not found")
    print("  Looking for:", repr(old1))

# ── Fix 2: guard n_token = selected.sum() in the except block ──────────────────
old2 = "            n_token = selected.sum()"
new2 = (
    "            # Guard: selected may be a 0-D tensor or Python bool on small batches\n"
    "            n_token = int(torch.as_tensor(selected, dtype=torch.long).sum().item())"
)
if old2 in txt:
    txt = txt.replace(old2, new2, 1)
    print("✓ Fix 2 applied: n_token guard")
else:
    print("✗ Fix 2 not applied: pattern not found")
    print("  Looking for:", repr(old2))

if txt == original:
    print("\nERROR: No changes made — check patterns above")
    sys.exit(1)

open(fp, "w").write(txt)
print("\n✓ File written. Clearing .pyc cache …")

import os, glob
for pyc in glob.glob(f"{CACHE_DIR}/**/*.pyc", recursive=True):
    os.remove(pyc)
    print(f"  removed {pyc}")

print("Done.")
