# Dispatch Layout Examples

This directory contains test examples for the dispatch_layout operator.

## Operators

1. **dispatch_layout**: Original MC2 operator for computing dispatch layout
   - Input: `topkIdx`, `numTokensPerExpert`, `groupEp`, `rank`, `numRanks`
   - Output: `recvExpandPerExpert`, `sendRecvMap`, `expertTokensMap`, `sendTokenIdx`, `recvTokenIdx`

2. **get_dispatch_layout**: Simplified torch_extension wrapper
   - Input: `topkIdx`, `numExperts`, `numRanks`
   - Output: `numTokensPerExpert`, `sendTokenIdxSmall`

## Running Tests

```bash
# Test both operators
python dispatch_layout_sample.py --num-processes 16

# Test only dispatch_layout
python dispatch_layout_sample.py --num-processes 16 --test-type dispatch_layout

# Test only get_dispatch_layout
python dispatch_layout_sample.py --num-processes 16 --test-type get_dispatch_layout
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-processes` | 16 | Number of processes to spawn |
| `--num-tokens` | 1024 | Number of tokens |
| `--num-topk` | 8 | Number of top-k experts |
| `--num-experts` | 16 | Number of experts |
| `--test-type` | both | Which operator to test |

## Dependencies

- torch
- torch_npu
- torch.distributed (hccl backend)
- npu_ops_transformer (torch_extension)
- utils from common/examples