"""Pure-PyTorch replacement for flash-attn's `flash_attn_func`.

Used automatically when flash-attn is not installed, e.g. on AMD GPUs (ROCm), on CPU, or on NVIDIA setups where
flash-attn cannot be built. Same layout as flash-attn: q, k, v are (B, L, H, D); returns (out, None) like the
flash-attn 3 interface used in this repo.
"""
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    dtype = q.dtype
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
    if k.shape[1] != q.shape[1]:  # grouped-query attention
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=softmax_scale, is_causal=causal)
    return out.transpose(1, 2).to(dtype), None
