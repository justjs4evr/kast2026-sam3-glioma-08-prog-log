"""Clean SAM3 wrapper for text-promptable semantic segmentation of BraTS sub-regions.

Written from scratch (replaces the student-notebook-derived KastSAM3). Key properties:

  1. NO monkeypatches. SAM3 internal stages (_encode_prompt -> _run_encoder ->
     _run_decoder -> _run_segmentation_heads) are called directly from `_ground()`.
     (This MedSAM3 fork already passes use_reentrant=False at every checkpoint
     call site, so the notebook's global torch.utils.checkpoint patch was never
     needed; activation checkpointing stays ON.)
  2. NO backbone duplication. SAM3's `_get_img_feats` natively indexes vision
     features with `find_input.img_ids`, so the ViT runs once per image (batch B)
     while grounding runs per (image, text) pair (N = B*K).
  3. Correct mask routing. Uses the dedicated `semantic_seg` head (1x1 conv on the
     pixel decoder output, one map per (image, text) pair) — NOT `pred_masks[:, 0]`,
     which is an arbitrary one of 200 DETR instance queries.
  4. 4-channel MRI input: patch_embed expanded (RGB weights kept, 4th channel =
     mean of RGB), absolute pos-embed bicubic-interpolated, RoPE tables rebuilt
     for the new token grid.
  5. PEFT strategies: 'lora_enc' (LoRA on ViT trunk), 'seg_head_ft' (unfreeze the
     segmentation_head module: pixel decoder + semantic/instance heads — the only
     detector-side params on the semantic_seg gradient path), 'lora_enc+seg_head'.
     The new 4ch patch_embed is always trainable.
  6. Text prompts are cached but re-encodable via `set_prompts()` for eval-time
     cross-family evaluation.
"""
import os
import sys

if os.path.isdir("/workspace/MedSAM3") and "/workspace/MedSAM3" not in sys.path:
    sys.path.insert(0, "/workspace/MedSAM3")

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.data_misc import FindStage

SEG_HEAD_TRAINABLE_PATTERNS = ("segmentation_head",)


def rope_freqs_2d(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> torch.Tensor:
    """2D axial RoPE table [end_x*end_y, dim//2] complex (matches SAM3 vitdet layout)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    fx = torch.outer(torch.arange(end_x).float(), freqs)
    fy = torch.outer(torch.arange(end_y).float(), freqs)
    cx = torch.polar(torch.ones_like(fx), fx).unsqueeze(1).expand(end_x, end_y, -1)
    cy = torch.polar(torch.ones_like(fy), fy).unsqueeze(0).expand(end_x, end_y, -1)
    return torch.cat([cx, cy], dim=-1).reshape(end_x * end_y, -1)


class Sam3SemSeg(nn.Module):
    def __init__(self, checkpoint_path: str, prompts: list[str], image_size: int = 512,
                 in_channels: int = 4, pixel_mean=None, pixel_std=None,
                 medsam3_lora_path: str | None = None,
                 peft: str = "lora_enc", lora_r: int = 12, lora_alpha: int = 24):
        super().__init__()
        assert peft in ("lora_enc", "seg_head_ft", "lora_enc+seg_head", "none")
        self.image_size = image_size
        self.peft = peft

        self.sam = build_sam3_image_model(checkpoint_path=None, load_from_HF=False,
                                          eval_mode=False, enable_segmentation=True,
                                          enable_inst_interactivity=False)
        self._load_checkpoint(checkpoint_path)
        self._adapt_patch_embed(in_channels)
        self._interp_pos_embed()
        self._rebuild_rope()

        if medsam3_lora_path and os.path.exists(medsam3_lora_path):
            self._merge_medsam3_lora(medsam3_lora_path)

        mean = pixel_mean if pixel_mean is not None else [0.0] * in_channels
        std = pixel_std if pixel_std is not None else [1.0] * in_channels
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, in_channels, 1, 1))
        self.register_buffer("pixel_std",
                             torch.clamp(torch.tensor(std).view(1, in_channels, 1, 1), min=1e-6))

        self._apply_peft(lora_r, lora_alpha)

        self.prompts: list[str] | None = None
        self._text_cache: dict[str, torch.Tensor] = {}
        self._geo_cache: dict[tuple, Prompt] = {}
        self.set_prompts(prompts)

    # ------------------------------------------------------------------ setup
    def _load_checkpoint(self, path: str):
        sd = torch.load(path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        if any(k.startswith("detector.") for k in sd):
            sd = {k.removeprefix("detector."): v for k, v in sd.items() if k.startswith("detector.")}
        missing, unexpected = self.sam.load_state_dict(sd, strict=False)
        print(f"[sam3-ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    def _trunk(self):
        trunk = self.sam.backbone.vision_backbone.trunk
        return trunk.base_model.model if hasattr(trunk, "base_model") else trunk

    def _adapt_patch_embed(self, in_channels: int):
        trunk = self._trunk()
        old = trunk.patch_embed.proj
        k = old.kernel_size[0]
        st = old.stride[0]
        p = old.padding[0] if isinstance(old.padding, tuple) else old.padding
        self.token_grid = (self.image_size + 2 * p - k) // st + 1
        if old.in_channels == in_channels:
            print(f"[patch-embed] native {in_channels}ch, grid={self.token_grid}")
            return
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            for c in range(3, in_channels):
                new.weight[:, c] = old.weight.mean(dim=1)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        trunk.patch_embed.proj = new
        print(f"[patch-embed] {old.in_channels}->{in_channels}ch, grid={self.token_grid}")

    def _interp_pos_embed(self):
        trunk = self._trunk()
        pe = getattr(trunk, "pos_embed", None)
        if pe is None:
            return
        g = self.token_grid
        n_tok = pe.shape[1] if pe.dim() == 3 else None
        if pe.dim() == 3:  # [1, L, C] (optionally with cls token)
            side = int(n_tok ** 0.5)
            cls_tok, grid_pe = (None, pe) if side * side == n_tok else (pe[:, :1], pe[:, 1:])
            side = int(grid_pe.shape[1] ** 0.5)
            dim = grid_pe.shape[-1]
            x = grid_pe.reshape(1, side, side, dim).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(g, g), mode="bicubic", align_corners=False)
            x = x.permute(0, 2, 3, 1).reshape(1, g * g, dim)
            trunk.pos_embed = nn.Parameter(torch.cat([cls_tok, x], 1) if cls_tok is not None else x)
        elif pe.dim() == 4:  # [1, H, W, C]
            x = pe.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(g, g), mode="bicubic", align_corners=False)
            trunk.pos_embed = nn.Parameter(x.permute(0, 2, 3, 1))
        print(f"[pos-embed] interpolated to {g}x{g}")

    def _rebuild_rope(self):
        trunk = self._trunk()
        if not hasattr(trunk, "blocks"):
            return
        n = 0
        for blk in trunk.blocks:
            attn = getattr(blk, "attn", None)
            if attn is None or getattr(attn, "freqs_cis", None) is None:
                continue
            gs = blk.window_size if getattr(blk, "window_size", 0) else self.token_grid
            head_dim = attn.freqs_cis.shape[-1] * 2
            attn.freqs_cis = rope_freqs_2d(head_dim, gs, gs).to(
                attn.freqs_cis.device, attn.freqs_cis.dtype)
            n += 1
        print(f"[rope] rebuilt tables on {n} blocks")

    def _merge_medsam3_lora(self, path: str):
        """Load MedSAM3-v1 vision-encoder LoRA and fold deltas into base weights."""
        from lora_layers import LoRAConfig, apply_lora_to_model, LoRALinear
        cfg = LoRAConfig(rank=16, alpha=32, dropout=0.1,
                         target_modules=["qkv", "proj", "fc1", "fc2"],
                         apply_to_vision_encoder=True, apply_to_text_encoder=False,
                         apply_to_geometry_encoder=False, apply_to_detr_encoder=False,
                         apply_to_detr_decoder=False, apply_to_mask_decoder=False)
        self.sam = apply_lora_to_model(self.sam, cfg)
        st = torch.load(path, map_location="cpu")
        if "model" in st:
            st = st["model"]
        missing, unexpected = self.sam.load_state_dict(st, strict=False)
        n_loaded = sum(1 for k in st if k not in unexpected)
        print(f"[medsam3-lora] loaded={n_loaded}/{len(st)}")
        assert n_loaded > 0, "MedSAM3 LoRA loaded zero keys"
        with torch.no_grad():
            for _, mod in self.sam.named_modules():
                lora = getattr(mod, "lora", None)
                if lora is None or not hasattr(mod, "original_layer"):
                    continue
                a, b = lora.lora_A, lora.lora_B
                scale = lora.alpha / lora.rank
                w = mod.original_layer.weight
                if a.dim() == b.dim() == 2:
                    if a.shape[1] == b.shape[0]:
                        delta = (a @ b).T * scale
                    elif a.shape[0] == b.shape[1]:
                        delta = (b @ a) * scale
                    else:
                        continue
                    if delta.shape == w.shape:
                        w.data += delta.to(w.device, w.dtype)
                        mod.lora = None
        restored = []
        for name, mod in list(self.sam.named_modules()):
            if isinstance(mod, LoRALinear):
                *parents, attr = name.split(".")
                obj = self.sam
                for p in parents:
                    obj = getattr(obj, p)
                restored.append((obj, attr, mod.original_layer))
        for obj, attr, orig in restored:
            setattr(obj, attr, orig)
        print(f"[medsam3-lora] merged + restored {len(restored)} linears")

    def _apply_peft(self, lora_r: int, lora_alpha: int):
        for p in self.sam.parameters():
            p.requires_grad = False
        if "lora_enc" in self.peft:
            from peft import LoraConfig, get_peft_model
            trunk = self.sam.backbone.vision_backbone.trunk
            self.sam.backbone.vision_backbone.trunk = get_peft_model(
                trunk, LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                  target_modules=["qkv", "proj", "fc1", "fc2"],
                                  lora_dropout=0.05, bias="none"))
            self._rebuild_rope()  # peft wrapping must not stale the tables
        if "seg_head" in self.peft:
            n = 0
            for name, p in self.sam.named_parameters():
                if any(t in name for t in SEG_HEAD_TRAINABLE_PATTERNS):
                    p.requires_grad = True
                    n += p.numel()
            assert n > 0, "seg_head_ft matched zero parameters"
            print(f"[peft] segmentation-head params unfrozen: {n/1e6:.2f}M")
        for name, p in self.sam.named_parameters():
            if "patch_embed" in name or "lora_" in name:
                p.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[peft={self.peft}] trainable: {total/1e6:.2f}M params")

    # ------------------------------------------------------------------ prompts
    @torch.no_grad()
    def set_prompts(self, prompts: list[str]):
        """(Re-)encode text prompts. Cheap to call repeatedly with the same list."""
        if prompts == self.prompts and self._text_cache:
            return
        dev = next(self.sam.parameters()).device
        was_training = self.training
        self.eval()
        enc = self.sam.backbone.forward_text(prompts)
        self._text_cache = {k: v.to(dev) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        self.prompts = list(prompts)
        self.num_texts = len(prompts)
        if was_training:
            self.train()

    def _empty_geo_prompt(self, n: int, device) -> Prompt:
        key = (n, str(device))
        if key not in self._geo_cache:
            self._geo_cache[key] = Prompt(
                box_embeddings=torch.zeros(0, n, 4, device=device),
                point_embeddings=torch.zeros(0, n, 2, device=device),
                mask_embeddings=torch.zeros(0, n, 1, 256, 256, device=device),
                box_mask=torch.zeros(n, 0, dtype=torch.bool, device=device),
                point_mask=torch.zeros(n, 0, dtype=torch.bool, device=device),
                mask_mask=torch.zeros(n, 0, dtype=torch.bool, device=device),
                box_labels=torch.zeros(0, n, dtype=torch.long, device=device),
                point_labels=torch.zeros(0, n, dtype=torch.long, device=device))
        return self._geo_cache[key]

    # ------------------------------------------------------------------ forward
    def _ground(self, backbone_out: dict, find_input: FindStage) -> dict:
        """Text-conditioned grounding: SAM3's forward_grounding stages, called
        explicitly (no find_target / no Hungarian matching step)."""
        n = find_input.img_ids.shape[0]
        geo = self._empty_geo_prompt(n, find_input.img_ids.device)
        prompt, prompt_mask, backbone_out = self.sam._encode_prompt(backbone_out, find_input, geo)
        backbone_out, enc, _ = self.sam._run_encoder(backbone_out, find_input, prompt, prompt_mask)
        out = {"encoder_hidden_states": enc["encoder_hidden_states"]}
        out, hs = self.sam._run_decoder(memory=out["encoder_hidden_states"],
                                        pos_embed=enc["pos_embed"],
                                        src_mask=enc["padding_mask"],
                                        out=out, prompt=prompt, prompt_mask=prompt_mask,
                                        encoder_out=enc)
        self.sam._run_segmentation_heads(out=out, backbone_out=backbone_out,
                                         img_ids=find_input.img_ids,
                                         vis_feat_sizes=enc["vis_feat_sizes"],
                                         encoder_hidden_states=out["encoder_hidden_states"],
                                         prompt=prompt, prompt_mask=prompt_mask, hs=hs)
        return out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] in [0,1]. Returns logits [B, K, H, W] (K = #prompts)."""
        B, _, H, W = images.shape
        K = self.num_texts
        dev = images.device

        x = (images - self.pixel_mean) / self.pixel_std
        backbone_out = dict(self.sam.backbone.forward_image(x))  # vision feats at batch B
        backbone_out.update(self._text_cache)                    # text feats for K prompts

        find_input = FindStage(
            img_ids=torch.arange(B, device=dev).repeat_interleave(K),  # [0,0,0,1,1,1,...]
            text_ids=torch.arange(K, device=dev).repeat(B),            # [0,1,2,0,1,2,...]
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)

        out = self._ground(backbone_out, find_input)
        sem = out["semantic_seg"][:, 0]                # [N, h, w] — dedicated semantic head
        logits = sem.reshape(B, K, *sem.shape[-2:])
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------ ckpt io
    TRAINABLE_KEY_HINTS = ("lora_", "patch_embed", "pixel_mean", "pixel_std")

    def trainable_state_dict(self) -> dict:
        hints = list(self.TRAINABLE_KEY_HINTS)
        if "seg_head" in self.peft:
            hints += list(SEG_HEAD_TRAINABLE_PATTERNS)
        return {k: v for k, v in self.state_dict().items() if any(h in k for h in hints)}
