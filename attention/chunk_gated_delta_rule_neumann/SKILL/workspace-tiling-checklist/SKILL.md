# Workspace and Tiling Checklist Skill

用于修改 `chunkSize`、workspace、stageWorkspace、mask、Neumann 临时矩阵时做一致性检查。触发场景：用户要求切 C16/C64、清理 kernel/workspace、删除 tmp、修改 tiling、修复 workspace 越界或挂起。

## 必查文件

- `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`
- `op_kernel/chunk_gated_delta_rule_neumann_tiling_data.h`
- `op_kernel/chunk_gated_delta_rule_neumann.h`
- 当前 selector 指向的 `op_kernel/chunk_gated_delta_rule_neumann_stage1_*.h`
- `tests/pytest/chunk_gated_delta_rule_neumann_golden.py`

## 修改 chunkSize 检查

| 场景 | host `chunkSize` | golden chunk | 说明 |
|---|---:|---:|---|
| `C16_1616_Neumann` | 16 | 16 | 主 chunk 就是 16。 |
| `C64_1616_NeumannCompose` | 64 | 64 | 1616 是内部子块，不是主 chunk。 |
| `C64_3232_NeumannCompose` | 64 | 64 | 3232 是内部子块，不是主 chunk。 |

规则：host tiling、kernel `chunkSize_`、golden chunk 必须同口径，否则性能和精度比较无效。

## Workspace 计算检查

host 侧常见项：

- `interWorkspaceSz`: `gCumExp/kCumDecay/vInner/qPrime/attnInter/kg/qkt/highState/mask`
- `stageWorkspaceSz`: stage1 per-core 临时空间
- Neumann extra workspace: `L_pow + tmp + INV`

修改规则：

- 删除 kernel 中某个 GM workspace 前，必须同步修改 host `stageWorkspaceSz`。
- 增加 `c*c` 矩阵空间时，要按 `sizeHigh * aiCoreNum * c * c * count` 计算。
- `TASK_RATIO=2` 时 AIV 通常只写半块，不能误写完整矩阵导致 subBlock 冲突。

## Kernel GM Offset 检查

在 Stage1 `SetGlobalTensors()` 中每新增/删除一个 workspace：

1. 检查 `SetGlobalBuffer` 地址。
2. 检查后续 `workSpaceOffset += ...` 是否一致。
3. 检查 host `stageWorkspaceSz` 是否覆盖所有 offset。
4. 检查 AIC/AIV 是否使用同一套 core index 语义。

## 同步检查

常见 cross-core flags：

- AIV 连续化 Q/K 后通知 AIC。
- AIC `K@K` 后通知 AIV 生成 `L/I-L`。
- AIV 生成 `L/I-L` 后通知 AIC 做 Neumann。
- AIV 生成 `GBK/VBeta` 后通知 AIC 做最终 matmul。

规则：删除或移动一段计算后，同步 flag 必须一起重审；不要保留死等待或重复 set/wait。

## 清理 Checklist

删除多余 kernel/workspace 前：

- `rg "tmpWsGm_|NeumannAddCompute|L_powWsGm_|INVWsGm_" op_kernel`
- 确认被删对象没有被当前 selector 引用。
- 构建目标包并安装。
- 跑三方精度：`CGDR_GOLDEN_CHUNK_SIZE=<16|64> bash test_run.sh compare`。
- 跑至少一个 msprof smoke，确认没有 hang 或 op_summary 缺失。

## 常见错误

- C64 子块方案误把 host `chunkSize` 切成 16/32。
- kernel 删除了 workspace，但 host `stageWorkspaceSz` 未同步缩小或 offset 未重排。
- golden chunk 未随 host tiling 切换，导致 CPU/NPU 比较口径错。
- `stage1_C16_FP16_CVInverse.h` 名字是 FP16，但实现仍可能是 FP32；不要只看文件名判断数据类型。
