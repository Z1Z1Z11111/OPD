# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apply monkey-patch function to text CausalLM models."""

import importlib.util
import sys
from typing import Optional

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward, _prepare_from_posids
from transformers.modeling_utils import PreTrainedModel

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)

_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None


def _varlen_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    query_length: int = 0,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    from torch.nn.attention.varlen import varlen_attn

    q, k, v, (cu_q, cu_k), (max_q, max_k) = _prepare_from_posids(
        query_states, key_states, value_states, position_ids
    )
    left_window = -1 if sliding_window is None else int(sliding_window)
    window = (left_window, 0) if is_causal else (left_window, left_window)
    out = varlen_attn(
        q, k, v, cu_q, cu_k, int(max_q), int(max_k), scale=softmax_scale, window_size=window
    )
    return out.view(query_states.size(0), -1, out.size(-2), out.size(-1))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    if _HAS_FLASH_ATTN:
        attn_output = _flash_attention_forward(query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs)
    else:
        attn_output = _varlen_flash_attention_forward(query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs)

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
):
    """Replace _flash_attention_forward with _ulysses_flash_attention_forward."""
    module = sys.modules[model.__module__]
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    assert num_attention_heads % ulysses_sp_size == 0
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0

    if use_fused_kernels:
        raise NotImplementedError("use_fused_kernels is not enabled in this minimal GRPO build.")

    if use_remove_padding or ulysses_sp_size > 1:
        from transformers.integrations import flash_attention

        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = _ulysses_flash_attention_forward
        if not _HAS_FLASH_ATTN:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            ALL_ATTENTION_FUNCTIONS["sdpa"] = flash_attention.flash_attention_forward
            ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]
