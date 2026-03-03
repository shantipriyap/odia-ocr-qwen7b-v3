"""
Definitive patch for modeling_internvl_chat.py:
1. Use integer indexing instead of boolean indexing for vit_embeds (avoids bool return bug)
2. Replace the fragile try/except with a robust assignment using nonzero positions
3. Remove all debug prints
"""
import re, sys

FP = (
    "/root/.cache/huggingface/modules/transformers_modules/"
    "OpenGVLab/InternVL2_hyphen_8B/"
    "6fb9ad6924f69424e57fab2ab061d707688f0296/modeling_internvl_chat.py"
)

txt = open(FP).read()

# ── Step 0: strip ALL previously added debug lines ──────────────────────────
txt = re.sub(r"        # DEBUG\n.*?", "", txt)
txt = re.sub(r"        import sys\n", "", txt)
txt = re.sub(r'        print\(f"\[DBG.*?\n', "", txt)
txt = re.sub(r'        print\(f"\[DBG.*?\n', "", txt)  # run twice for multi-line stubs
print("Step 0: debug prints removed")

# ── Step 1: ensure image_flags is always 1D (was squeeze → view) ─────────────
# Find and standardize this line (might already be view or might be squeeze)
txt = re.sub(
    r"        image_flags = image_flags\.(squeeze\(-1\)|view\(-1\).*)  # .*",
    "        image_flags = image_flags.view(-1)  # ensure 1D",
    txt,
)
print("Step 1: image_flags.view(-1) standardized")

# ── Step 2: replace boolean vit_embeds filter with integer indexing ──────────
old2 = "        vit_embeds = vit_embeds[image_flags == 1]"
new2 = (
    "        # Use integer indexing to avoid bool-return bug with boolean masks\n"
    "        _real_idx = (image_flags == 1).nonzero(as_tuple=False).view(-1)\n"
    "        if _real_idx.numel() > 0:\n"
    "            vit_embeds = vit_embeds[_real_idx]  # [n_real_tiles, n_patches, C_vit]\n"
    "        # else: no real tiles — vit_embeds stays as-is (handled below)"
)
if old2 in txt:
    txt = txt.replace(old2, new2, 1)
    print("Step 2: vit_embeds filter → integer indexing")
else:
    print("Step 2: pattern not found, checking existing code...")
    idx = txt.find("extract_feature")
    print(repr(txt[idx : idx + 300]))

# ── Step 3: replace try/except assignment with robust nonzero-based fill ─────
# Find the try-except block regardless of exact whitespace history
pattern3 = re.compile(
    r"        try:\n"
    r"            input_embeds\[selected\] = input_embeds\[selected\] \* 0\.0 \+ vit_embeds\.reshape\(-1, C\)\n"
    r"        except Exception.*?input_embeds\[selected\] = input_embeds\[selected\] \* 0\.0 \+ vit_embeds\[:n_token\]",
    re.DOTALL,
)
new3 = (
    "        # Robust vit→text embedding injection using nonzero positions\n"
    "        _vit_flat = vit_embeds.reshape(-1, C)          # [n_real_patches, C]\n"
    "        _sel_pos = selected.nonzero(as_tuple=False).view(-1)  # [n_img_ctx]\n"
    "        _n_sel = _sel_pos.shape[0]\n"
    "        _n_vit = _vit_flat.shape[0]\n"
    "        if _n_sel > 0 and _n_vit > 0:\n"
    "            _use = min(_n_sel, _n_vit)\n"
    "            input_embeds[_sel_pos[:_use]] = (\n"
    "                input_embeds[_sel_pos[:_use]] * 0.0 + _vit_flat[:_use]\n"
    "            )"
)

match3 = pattern3.search(txt)
if match3:
    txt = txt[: match3.start()] + new3 + txt[match3.end() :]
    print("Step 3: try/except replaced with robust fill")
else:
    print("Step 3: try/except pattern not found exactly — trying looser match")
    # Looser: find try block and until the except/else closes
    idx_try = txt.find("        try:\n            input_embeds[selected]")
    if idx_try >= 0:
        # find next 'input_embeds = input_embeds.reshape(B, N, C)' after try
        idx_end = txt.find("        input_embeds = input_embeds.reshape(B, N, C)", idx_try)
        if idx_end >= 0:
            old_block = txt[idx_try:idx_end]
            print("Replacing block:")
            print(repr(old_block[:200]))
            txt = txt[:idx_try] + new3 + "\n" + txt[idx_end:]
            print("Step 3: done via loose match")
        else:
            print("Step 3: could not find end of except block")
    else:
        print("Step 3: try block not found at all")

open(FP, "w").write(txt)
print("\n✓ File written")

# Clear __pycache__
import os, glob
base = (
    "/root/.cache/huggingface/modules/transformers_modules/"
    "OpenGVLab/InternVL2_hyphen_8B/"
    "6fb9ad6924f69424e57fab2ab061d707688f0296"
)
removed = 0
for pyc in glob.glob(f"{base}/**/*.pyc", recursive=True):
    os.remove(pyc)
    removed += 1
pycache_dir = f"{base}/__pycache__"
if os.path.isdir(pycache_dir):
    for f in os.listdir(pycache_dir):
        fp = os.path.join(pycache_dir, f)
        os.remove(fp)
        removed += 1
print(f"✓ {removed} .pyc files cleared")

# Quick sanity check — parse the file
import ast
try:
    ast.parse(txt)
    print("✓ Syntax OK")
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR: {e}")
    sys.exit(1)
