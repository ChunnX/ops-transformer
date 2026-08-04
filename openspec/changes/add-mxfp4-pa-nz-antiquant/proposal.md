## Why

MXFP4 (`FLOAT4_E2M1`) mode 6 已具备非 PageAttention 的 per-token-group 对称反量化基础，但 PA_NZ 场景没有完整的 host、tiling、参数页寻址和非对称 offset 路由。当前实现若仅打开 PA 路由，会仍按逻辑序列而不是 `blockTable` 取 NZ 参数，无法正确反量化物理页中的 KV cache。

本变更以已有 FP8 PA 的页映射模型和已有 MXFP4 NZ group-plane 消费模型为参考，为 `FusedInferAttentionScore` 定义并实现 MXFP4 PA_NZ 契约。当前树中的 standalone `IncreFlashAttention` 没有 `FLOAT4_E2M1` / mode 6 的 kernel 模板或等价 Arch35 后端；它不是 FIA 的运行时后端，必须另立需求实现，不能由本变更隐式覆盖。

## What Changes

- 为 `FLOAT4_E2M1`、mode 6、PA_NZ 增加对称和非对称反量化支持。
- 固定 groupSize 为 32 个逻辑 head-dimension 元素；每个 `(physicalPage, kvHead, tokenInPage)` 有 `headDim / 32` 个 scale，非对称时有同数量的 offset。
- 固定参数生产链路：逻辑 `[B,N,S,G]` → PageAttention `[P,N,Bs,G]` → PA_NZ storage `[P,N,G,Bs]`，其中 `P=totalBlockNum`、`Bs=blockSize`、`G=headDim/32`。
- 在 FIA 中通过 `blockTable` 对 scale 和非空 offset 做物理页寻址，并按 NZ group-plane 搬入 UB。
- 在 VF 中实现 `(decode(FP4) + optionalOffset) * E8M0Scale`；offset 为空时保留对称计算。
- 为本变更目标组合以外的 PA layout 建立明确的拒绝边界：本 change 不新增 BNBD、BBH 或非 PA mode 6 的能力。

## Capabilities

### New Capabilities

- `mxfp4-pa-nz-antiquant`: MXFP4 mode 6 在 PA_NZ 下的 FIA 参数契约、页寻址与对称/非对称反量化。

### Modified Capabilities

- None.

## Impact

- FIA host checker、tiling/template-key、Arch35 antiquant processor、FP4 VF 和相关测试。
- PA cache 与量化参数生产端必须按照本变更的最终 PA_NZ 参数 storage 传入 scale 和非空 offset。
- 不涉及性能调优、BNBD/BBH、非 PA mode 6 的接口或数值语义变更。
