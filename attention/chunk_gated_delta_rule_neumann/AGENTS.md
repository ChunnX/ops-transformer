# AGENTS.md - chunk_gated_delta_rule_neumann Operator

Chunk Gated Delta Rule Neumann attention operator for recurrent-style LLM inference on Ascend NPU. Implements gated delta rule with chunk-based processing and Neumann series inverse for efficient sequential attention.

## Skills Directory

Operator-specific development and debugging skills are organized in `SKILL/`:

```
SKILL/
├── aclnn-single-op-test/     # aclnn 单算子测试开发指南
├── operator-clone/            # 算子克隆与重命名流程
├── precision-debugging/       # 精度调试经验（数据类型匹配问题）
```

## Migration Plans

Migration and implementation tracking in `docs/`:

```
docs/
├── CHUNK16_NEUMANN_MIGRATION_PLAN.md  # CHUNK=16 + Neumann 求逆详细修改计划
└── TODOLIST.md                        # 实施进度跟踪（每一步状态）
```

## Structure

```
chunk_gated_delta_rule/
├── op_host/              # Tiling + OpDef + InferShape + aclnn API
│   ├── *_def.cpp         # Operator prototype (BF16 only)
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

**Inputs (BF16):**
- `query`, `key`, `value` - Attention tensors
- `beta` - Decay/gating parameter
- `initial_state` - Recurrent state (Nv × Dv × Dk)
- `actual_seq_lengths` - Sequence lengths (INT32)
- `g` - Optional gating tensor (FLOAT)

**Outputs (BF16):**
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
- Data type: BF16 only

## CHUNK=16 Migration Status (Active)

### Confirmed Decisions

| 决策项 | 值 | 原因 |
|--------|-----|------|
| **chunkSize** | **16** (统一) | 与 FlashKDA 对齐；decay bf16 安全范围更大 |
| **求逆方法** | **Neumann 级数 3步** | 6次 Cube MMA，减少 Vector 循环 |
| **精度策略** | **FP16 MMA + FP16 Add** | 与 FlashKDA 完全对齐 |
| **Stage 影响范围** | Stage1/2/3 全部修改 | 统一 chunkSize，避免数据流重构 |

### Key Modification Locations

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `op_host/*_tiling.cpp` | 113 | `c = 64 → 16` |
| `op_kernel/*_stage1.h` | 32 | `INVERSE_SHAPE = 32 → 16` |
| `op_kernel/*_stage1.h` | 33 | `INVERSE_COUNT` 重计算 |
| `tests/pytest/*_golden.py` | - | `chunk_size = 64 → 16` |

### Progress Tracking

- **详细计划**: `docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md`
- **进度跟踪**: `docs/TODOLIST.md`
- **执行日志**: `docs/PROJECT_EXECUTION_LOG.md` (每一步更新)

### Neumann Inverse Algorithm

```
(I+L)^{-1} = Σ_{k=0}^{15} (-L)^k  (L 幂零，L^16=0)

倍增法 3 步:
  初始: INV₀ = I - L           [16×16 FP16]
  步1:  L²  = L × L,   INV₁ = INV₀ + INV₀ × L²
  步2:  L⁴  = L² × L², INV₂ = INV₁ + INV₁ × L⁴
  步3:  L⁸  = L⁴ × L⁴, INV₃ = INV₂ + INV₂ × L⁸ (= 精确解)

总计: 6 次 16×16 FP16 MMA + 3 次 FP16 Add
```

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
| Operator definition | `op_host/chunk_gated_delta_rule_def.cpp` |
| Tiling strategy | `op_host/chunk_gated_delta_rule_tiling.cpp` |
| Kernel main | `op_kernel/chunk_gated_delta_rule.h` |
| Stage 1 logic | `op_kernel/chunk_gated_delta_rule_stage1.h` |
| Golden (CPU) | `tests/pytest/chunk_gated_delta_rule_golden.py` |
| NPU wrapper | `tests/pytest/chunk_gated_delta_rule_operator_single.py` |

## Build

```bash
# From ops-transformer root
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule -j16
```

## Notes

- Uses AIC (AI Core) for matmul + AIV (AI Vector) for masking
- `CopyCast` utility for BF16↔FP32 conversion
- Chunk size and max_group_length determined by tiling