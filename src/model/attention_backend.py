"""Attention backend selection for CUDA and ROCm.

FlashAttention is NVIDIA-only. On ROCm (and whenever flash-attn is missing),
fall back to PyTorch scaled_dot_product_attention and transformers' SDPA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def transformers_attn_implementation() -> str:
    """Prefer FlashAttention 2 on NVIDIA when available; otherwise SDPA."""
    if is_rocm():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _sdpa_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """SDPA drop-in matching flash_attn_func layout (B, L, H, D)."""
    # flash-attn expects (batch, seqlen, nheads, headdim)
    # SDPA expects (batch, nheads, seqlen, headdim)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if q_t.shape[1] != k_t.shape[1]:
        repeat_factor = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(repeat_factor, dim=1)
        v_t = v_t.repeat_interleave(repeat_factor, dim=1)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.transpose(1, 2), None


def get_flash_attn_func():
    """Return flash_attn_func, or an SDPA-compatible fallback."""
    if not is_rocm():
        try:
            from flash_attn_interface import flash_attn_func  # type: ignore
            return flash_attn_func
        except Exception:
            pass
        try:
            from flash_attn.flash_attn_interface import (  # type: ignore
                flash_attn_func as _flash_attn_func,
            )

            def flash_attn_func(*args, **kwargs):
                res = _flash_attn_func(*args, **kwargs)
                return res, None

            return flash_attn_func
        except Exception:
            pass
    return _sdpa_flash_attn_func


flash_attn_func = get_flash_attn_func()


def amp_device_type(device: Optional[torch.device] = None) -> str:
    """Autocast device type. ROCm still uses the 'cuda' namespace in PyTorch."""
    if device is not None:
        if device.type in ("cuda", "hip"):
            return "cuda"
        return device.type
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
