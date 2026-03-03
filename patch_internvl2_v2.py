fp = (
    "/root/.cache/huggingface/modules/transformers_modules/"
    "OpenGVLab/InternVL2_hyphen_8B/"
    "6fb9ad6924f69424e57fab2ab061d707688f0296/modeling_internvl_chat.py"
)
txt = open(fp).read()

# Remove all existing debug prints so we start clean
import re
# Remove DBG lines
clean = re.sub(r'        # DEBUG\n        import sys\n.*?file=sys\.stderr\)\n', '', txt, flags=re.DOTALL)
clean = re.sub(r'        print\(f"\[DBG.*?flush=True\)\n', '', clean)

# Now add the definitive fix: replace the problematic image_flags filtering block
# The approach: extract_feature → filter flags → replace img context tokens robustly

old = (
    '        vit_embeds = self.extract_feature(pixel_values)\n'
    '        vit_embeds = vit_embeds[image_flags == 1]'
)
new = (
    '        vit_embeds = self.extract_feature(pixel_values)\n'
    '        # image_flags may arrive as various shapes; ensure 1D bool mask matches vit_embeds dim-0\n'
    '        _flags = image_flags.view(-1)  # ensure 1D [total_tiles]\n'
    '        _mask = (_flags == 1)  # 1D bool tensor [total_tiles]\n'
    '        if _mask.shape[0] == vit_embeds.shape[0]:\n'
    '            vit_embeds = vit_embeds[_mask]  # [n_real_tiles, n_patches, C_vit]\n'
    '        # else: skip filter (sizes already match or no padding tiles)'
)

if old in clean:
    clean = clean.replace(old, new, 1)
    print("✓ vit_embeds filter patched")
else:
    print("Pattern not found — showing nearby lines:")
    idx = clean.find("extract_feature")
    for i, ln in enumerate(clean[max(0,idx-50):idx+400].split('\n')):
        print(f"  {repr(ln)}")

# Also robustify the try-except block inside the same forward
old2 = (
    '        try:\n'
    '            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)\n'
    '        except Exception as e:\n'
    '            vit_embeds = vit_embeds.reshape(-1, C)\n'
    '            print(f\'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, \'\n'
    '                  f\'vit_embeds.shape={vit_embeds.shape}\')\n'
    '            # Guard: selected may be a 0-D tensor or Python bool on small batches\n'
    '            n_token = int(torch.as_tensor(selected, dtype=torch.long).sum().item())\n'
    '            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]'
)
new2 = (
    '        # Robustly assign vit patches to img_context positions\n'
    '        _vit_flat = vit_embeds.reshape(-1, C)          # [n_real_tiles * n_patches, C]\n'
    '        _n_sel = int(selected.sum().item())             # number of <IMG_CONTEXT> positions\n'
    '        _n_vit = _vit_flat.shape[0]\n'
    '        if _n_sel == _n_vit:\n'
    '            input_embeds[selected] = input_embeds[selected] * 0.0 + _vit_flat\n'
    '        elif _n_sel > 0 and _n_vit > 0:\n'
    '            # Mismatch: truncate or pad to fit\n'
    '            _use = min(_n_sel, _n_vit)\n'
    '            _sel_idx = selected.nonzero(as_tuple=True)[0][:_use]\n'
    '            input_embeds[_sel_idx] = input_embeds[_sel_idx] * 0.0 + _vit_flat[:_use]'
)

if old2 in clean:
    clean = clean.replace(old2, new2, 1)
    print("✓ try-except block replaced with robust assignment")
else:
    print("try-except pattern not found — showing section:")
    idx2 = clean.find('        try:')
    print(repr(clean[idx2:idx2+600]))

open(fp, 'w').write(clean)
print("✓ File written")

import os, glob
base = (
    "/root/.cache/huggingface/modules/transformers_modules/"
    "OpenGVLab/InternVL2_hyphen_8B/"
    "6fb9ad6924f69424e57fab2ab061d707688f0296"
)
for pyc in glob.glob(f"{base}/**/*.pyc", recursive=True) + glob.glob(f"{base}/__pycache__/*.pyc"):
    try:
        os.remove(pyc)
        print(f"  removed {pyc}")
    except Exception:
        pass
print("✓ .pyc cleared")
