# dispatch_layout Operator

## Overview

The `dispatch_layout` operator computes dispatch layout information for MoE (Mixture of Experts) distributed dispatch operations. It analyzes top-k expert indices and generates token distribution statistics across ranks and experts.

## Functionality

Given the top-k expert selection for each token, this operator computes:
- Number of tokens to be sent to each rank (`numTokensPerRank`)
- Number of tokens assigned to each expert (`numTokensPerExpert`)
- Token-to-rank mapping matrix (`isTokenInRank`)
- Send token indices for dispatch (`sendTokenIdxSmall`)

## Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| topkIdx | int64 | [num_tokens, num_topk] | Top-k expert indices for each token |

## Attributes

| Name | Type | Description |
|------|------|-------------|
| num_tokens | int | Number of tokens |
| num_ranks | int | Number of ranks in distributed group |
| num_experts | int | Total number of experts |
| num_topk | int | Number of top-k experts selected per token |
| local_ranksize | int | Local rank size for multi-machine scenario |

## Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| numTokensPerRank | int32 | [num_ranks] | Token count per rank |
| numTokensPerExpert | int32 | [num_experts] | Token count per expert |
| isTokenInRank | int32 | [num_tokens, num_ranks] | Token-to-rank mapping (1 if token assigned to rank) |
| notifySendData | int32 | [num_tokens, num_experts] | Temporary buffer for notification |
| sendTokenIdxSmall | int32 | [num_tokens, num_topk] | Token indices for sending |

## Supported Platforms

- Ascend910B (Atlas A2)
- Ascend910_93 (Atlas A3)
- Ascend950

## Usage

### Torch Extension

```python
import torch
from npu_ops_transformer.ops import dispatch_layout

topkIdx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.long).npu()

numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData, sendTokenIdxSmall = dispatch_layout(
    topkIdx=topkIdx,
    num_tokens=num_tokens,
    num_ranks=num_ranks,
    num_experts=num_experts,
    num_topk=num_topk,
    local_ranksize=local_ranksize
)
```

## Constraints

- `num_ranks` must be in range [1, 384]
- `num_experts` must be in range [1, 512]
- `num_experts` must be divisible by `num_ranks`
- `num_topk` must be in range [1, 16]
- For A2 multi-machine: `num_tokens` must be in range [1, 4096]
- For other scenarios: `num_tokens` must be in range [1, 8000]
- `topkIdx` must be 2D tensor with dtype int64

## License

CANN Open Software License Agreement Version 2.0