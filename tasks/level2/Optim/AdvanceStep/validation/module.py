from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    """AdvanceStep 算子的 PyTorch 参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor,
                input_positions: torch.Tensor, seq_lens: torch.Tensor,
                slot_mapping: torch.Tensor, block_tables: torch.Tensor,
                num_seqs: int, num_queries: int, block_size: int) -> List[torch.Tensor]:
        # Clone tensors that will be modified in-place
        input_tokens = input_tokens.clone()
        input_positions = input_positions.clone()
        seq_lens = seq_lens.clone()
        slot_mapping = slot_mapping.clone()

        n_pad = num_seqs - num_queries
        # Match kernel stride pattern: only core 0 handles padding with stride = total_core_num
        # total_core_num = aivNum (number of AIV cores on the platform)
        total_core_num = 48  # ascend910b has 48 AIV cores
        if n_pad > 0:
            for i in range(0, n_pad, total_core_num):
                input_tokens[num_queries + i] = 0
                input_positions[num_queries + i] = 0
                slot_mapping[num_queries + i] = -1

        for index in range(num_queries):
            # Update input_tokens
            input_tokens[index] = sampled_token_ids[index]

            seq_len = seq_lens[index].item()
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1

            # Update seq_lens and input_positions
            seq_lens[index] = next_seq_len
            input_positions[index] = next_input_pos

            # Calculate slot mapping
            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size

            block_tables_flat = block_tables.flatten()
            slot_num = (block_tables_flat[block_index].item() + block_tables.shape[1] * index) * block_size + block_offset
            slot_mapping[index] = slot_num

        return [input_tokens, input_positions, seq_lens, slot_mapping]


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor,
                input_positions: torch.Tensor, seq_lens: torch.Tensor,
                slot_mapping: torch.Tensor, block_tables: torch.Tensor,
                num_seqs: int, num_queries: int, block_size: int) -> List[torch.Tensor]:
        import kernel_gen_ops

        # Create copies for in-place modification by kernel
        input_tokens = input_tokens.clone()
        input_positions = input_positions.clone()
        seq_lens = seq_lens.clone()
        slot_mapping = slot_mapping.clone()

        return kernel_gen_ops.advance_step(input_tokens, sampled_token_ids, input_positions,
                                            seq_lens, slot_mapping, block_tables,
                                            num_seqs, num_queries, block_size)
