import torch


def get_inputs(param, device=None):
    """根据 test_cases.csv 行生成输入张量。返回与 Model.forward() 签名匹配的元组。"""
    num_seqs = int(param['num_seqs'])
    num_queries = int(param['num_queries'])
    block_size = int(param['block_size'])
    max_blocks_per_seq = int(param.get('max_blocks_per_seq', '8'))

    # Generate input data
    # input_tokens: [num_seqs], int64
    input_tokens = torch.randint(0, 1000, (num_seqs,), dtype=torch.int64)

    # sampled_token_ids: [num_queries, 1], int64
    sampled_token_ids = torch.randint(0, 1000, (num_queries, 1), dtype=torch.int64)

    # input_positions: [num_seqs], int64 - current positions in sequences
    input_positions = torch.randint(0, max_blocks_per_seq * block_size, (num_seqs,), dtype=torch.int64)

    # seq_lens: [num_seqs], int64 - current sequence lengths
    seq_lens = torch.randint(1, max_blocks_per_seq * block_size, (num_seqs,), dtype=torch.int64)

    # slot_mapping: [num_seqs], int64 - placeholder values
    slot_mapping = torch.full((num_seqs,), -1, dtype=torch.int64)

    # block_tables: [num_seqs, max_blocks_per_seq], int64
    block_tables = torch.randint(0, 1024, (num_seqs, max_blocks_per_seq), dtype=torch.int64)

    if device:
        input_tokens = input_tokens.to(device)
        sampled_token_ids = sampled_token_ids.to(device)
        input_positions = input_positions.to(device)
        seq_lens = seq_lens.to(device)
        slot_mapping = slot_mapping.to(device)
        block_tables = block_tables.to(device)

    return (input_tokens, sampled_token_ids, input_positions, seq_lens,
            slot_mapping, block_tables, num_seqs, num_queries, block_size)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    abs_diffs = []
    rel_diffs = []
    is_pass = True

    for gold, kernel in zip(outputs, outputs_new):
        diff = torch.abs(gold.to(torch.float64) - kernel.to(torch.float64))
        abs_diffs.append(diff.flatten())
        if not torch.equal(gold, kernel):
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.tensor([0.0])

    return is_pass, all_abs_diff, all_rel_diff
