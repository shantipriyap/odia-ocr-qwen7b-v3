"""
scp this to /tmp on the server, then run:
  source /root/venv/bin/activate && python3 /tmp/monkey_patch_internvl2.py

It writes a monkey-patch block into train_internvl2.py right after
  model = get_peft_model(model, lora_cfg)
that replaces InternVLChatModel.forward with a version that:
  1. Does not rely on boolean tensor indexing (uses nonzero → integer idx)
  2. Is compatible with transformers 5.x + PEFT 2.x argument passing
"""
import re

SRC = "/root/phase3_paragraph/train_internvl2.py"
txt = open(SRC).read()

# Idempotency: remove any previous monkey-patch block
txt = re.sub(
    r"\n    # ─+ MONKEY-PATCH.*?# ─+ END MONKEY-PATCH.*?\n",
    "\n",
    txt,
    flags=re.DOTALL,
)

PATCH = '''
    # ─────────────────────────────────────────────────────────────────────────
    # MONKEY-PATCH: replace InternVLChatModel.forward with a robust version
    # that avoids bool-tensor indexing issues with transformers 5.x + PEFT 2.x
    # ─────────────────────────────────────────────────────────────────────────
    import types
    from torch.nn import CrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def _robust_forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        inputs_embeds=None,   # ignored; required for PEFT compat
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        # ── 1. ViT feature extraction ─────────────────────────────────────────
        if pixel_values is not None and image_flags is not None:
            vit_embeds = self.extract_feature(pixel_values)   # [total, patches, C_vit]
            # Integer indexing — avoids bool-return edge-case in PyTorch+CUDA
            flags_1d = image_flags.view(-1).long()
            real_indices = (flags_1d == 1).nonzero(as_tuple=False).view(-1)
            if real_indices.numel() > 0:
                vit_embeds = vit_embeds[real_indices]         # [n_real, patches, C_vit]
        else:
            vit_embeds = None

        # ── 2. Text embeddings ────────────────────────────────────────────────
        em = self.language_model.get_input_embeddings()
        input_embeds = em(input_ids).clone()                  # [B, N, C]
        B, N, C = input_embeds.shape
        input_embeds_2d = input_embeds.reshape(B * N, C)      # [B*N, C]

        # ── 3. Inject ViT patches at <IMG_CONTEXT> positions ─────────────────
        if vit_embeds is not None:
            ids_1d = input_ids.reshape(B * N)                 # [B*N]
            ctx_id = self.img_context_token_id
            # Use torch.eq + nonzero for guaranteed tensor output
            sel_pos = torch.eq(ids_1d, ctx_id).nonzero(as_tuple=False).view(-1)
            n_sel = sel_pos.shape[0]
            vit_flat = vit_embeds.reshape(-1, C)              # [n_real_patches, C]
            n_vit = vit_flat.shape[0]
            if n_sel > 0 and n_vit > 0:
                n_use = min(n_sel, n_vit)
                input_embeds_2d[sel_pos[:n_use]] = (
                    input_embeds_2d[sel_pos[:n_use]] * 0.0
                    + vit_flat[:n_use].to(input_embeds_2d.dtype)
                )

        input_embeds = input_embeds_2d.reshape(B, N, C)       # [B, N, C]

        # ── 4. Language model forward ─────────────────────────────────────────
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        # ── 5. Loss ───────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Bind to the base model (unwrapped from PEFT)
    _base = model.base_model.model if hasattr(model, "base_model") else model
    # Ensure img_context_token_id is set (tokenizer is in outer scope of main())
    if not getattr(_base, "img_context_token_id", None):
        _base.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        print(f"[PATCH] img_context_token_id set to {_base.img_context_token_id}")
    _base.forward = types.MethodType(_robust_forward, _base)
    print("[PATCH] Robust forward injected into InternVLChatModel")
    # ─────────────────────────────────────────────────────────────────────────
    # END MONKEY-PATCH
    # ─────────────────────────────────────────────────────────────────────────
'''

# Insert after: model.enable_input_require_grads()
anchor = "    model.enable_input_require_grads()\n    model.print_trainable_parameters()"
if anchor in txt:
    txt = txt.replace(anchor, anchor + PATCH, 1)
    print("✓ Monkey-patch inserted after enable_input_require_grads()")
else:
    print("Anchor not found, trying fallback anchor...")
    anchor2 = "    model.print_trainable_parameters()"
    if anchor2 in txt:
        txt = txt.replace(anchor2, anchor2 + PATCH, 1)
        print("✓ Monkey-patch inserted after print_trainable_parameters()")
    else:
        print("ERROR: Could not find insertion point")
        import sys; sys.exit(1)

open(SRC, "w").write(txt)
print("✓ train_internvl2.py updated")

# Syntax check
import ast
try:
    ast.parse(txt)
    print("✓ Syntax OK")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    import sys; sys.exit(1)
