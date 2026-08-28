import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from torch import nn

from ..attention_backend import flash_attn_func

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Tripo2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Tripo2DiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that tripo2 split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        hidden_states = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        if type(hidden_states) == tuple:
            hidden_states = hidden_states[0]
        #   hidden_states = F.scaled_dot_product_attention(
        #       query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        #   )
           #hidden_states = 
        hidden_states = hidden_states.reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FusedTripo2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0) with fused
    projection layers. This is used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on
    query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedTripo2AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # NOTE that tripo2 split heads first, then split qkv
        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
