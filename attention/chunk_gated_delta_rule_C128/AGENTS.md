# AGENTS.md - chunk_gated_delta_rule_C128 Operator

Chunk Gated Delta Rule C128 attention operator for recurrent-style LLM inference on Ascend NPU. Implements gated delta rule with chunk-based processing for efficient sequential attention.

## Structure

```
chunk_gated_delta_rule_C128/
├── op_host/              # Tiling + OpDef + InferShape + aclnn API
│   ├── *_def.cpp         # Operator prototype (FP16 only)
│   ├── *_tiling.cpp      # Tiling strategy
│   └── op_api/           # aclnn C API wrapper
├── op_kernel/            # Ascend C Kernel (3-stage pipeline)
│   ├── *_stage1.h        # Stage 1: q' computation
│   ├── *_stage2.h        # Stage 2: state update
│   ├── *_stage3.h        # Stage 3: attention output
│   └── *_utils.h         # Shared utilities
├── op_graph/             # Graph mode proto
└── tests/
    ├── pytest/           # PyTorch API tests (torch_npu)
    └── ut/               # C++ unit tests
```

## Operator Signature

**Inputs (FP16):**
- `query`, `key`, `value` - Attention tensors
- `beta` - Decay/gating parameter
- `initial_state` - Recurrent state (Nv × Dv × Dk)
- `actual_seq_lengths` - Sequence lengths (INT32)
- `g` - Optional gating tensor (FLOAT)

**Outputs (FP16):**
- `out` - Attention output
- `final_state` - Updated recurrent state

**Attribute:** `scale_value` (default 1.0)

## Supported Platforms

- `ascend910b` (Atlas A2) - Full dynamic support
- `ascend910_93` (Atlas A3)

## Constraints

- Batch_size > 0
- NV must be multiple of NK (GQA support)
- DK, DV ≤ 128
- Data type: FP16 only

## Kernel Architecture

3-stage pipeline with matmul fusion:
1. **Stage1**: Compute q' = q - β·k·cum_decay, prepare attention inputs
2. **Stage2**: Update recurrent state via matmul
3. **Stage3**: Compute attention output with chunk masking

Uses `lib/matmul_intf.h` for matmul operations. Each stage requires `SyncAll<false>()` between stages.

## Testing

```bash
# pytest (PyTorch API)
cd tests/pytest && bash test_run.sh single  # Configure paramset first

# Build UT
bash build.sh --ophost_test --noexec
bash build.sh --opkernel_test --noexec
```

## Key Files

| Purpose | File |
|---------|------|
| Operator definition | `op_host/chunk_gated_delta_rule_C128_def.cpp` |
| Tiling strategy | `op_host/chunk_gated_delta_rule_C128_tiling.cpp` |
| Kernel main | `op_kernel/chunk_gated_delta_rule_C128.h` |
| Stage 1 logic | `op_kernel/chunk_gated_delta_rule_C128_stage1.h` |
| Golden (CPU) | `tests/pytest/chunk_gated_delta_rule_C128_golden.py` |
| NPU wrapper | `tests/pytest/chunk_gated_delta_rule_C128_operator_single.py` |

## Build

```bash
# From ops-transformer root
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_C128 -j16
```

## Notes

- Uses AIC (AI Core) for matmul + AIV (AI Vector) for masking
- `CopyCast` utility for FP16↔FP32 conversion
- Chunk size and max_group_length determined by tiling