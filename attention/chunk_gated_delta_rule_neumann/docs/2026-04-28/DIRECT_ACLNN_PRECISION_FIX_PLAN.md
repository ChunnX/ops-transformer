# 单算子直调精度修复方案 - 2026-04-28

**日期**: 2026-04-28  
**目标**: 按用户要求不走 `torch_npu`，使用 aclnn 单算子直调验证 `chunk_gated_delta_rule_neumann` 精度并修复执行问题。

---

## 问题背景

- 直接使用 `ascend910b` 构建包在当前设备上 `GetWorkspaceSize` 报 `561103`，plog 显示 `ParseDynamicKernelConfig fail`。
- `npu-smi info` 显示当前硬件为 `Ascend910`，因此改用 `--soc=ascend910_93` 构建。
- `ascend910_93` 包安装后 `GetWorkspaceSize` 通过，但首次 kernel 执行报 `507015`。
- plog 明确报错：`The UB address accessed by the VEC instruction is not aligned`。
- 修复 UB 对齐后 kernel 出现同步挂起，直调停在 `aclnnChunkGatedDeltaRuleNeumann()` 后的 stream 同步阶段。

---

## 修复方案

### 1. 使用匹配 SoC 构建

- 当前设备按 `Ascend910` / A3 路径构建，使用：
  - `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16`
- 避免继续使用 `ascend910b`，否则动态 kernel 配置解析失败。

### 2. 修复 AIV UB 对齐

- `chunkSize=16`、`TASK_RATIO=2` 时，`betaUbBfloat16_` 长度为 `8 * sizeof(bfloat16_t) = 16B`。
- 后续 `float` UB buffer 若从 16B 偏移开始，会触发 AIV vector 指令非 32B 对齐访问。
- 在 `betaUbBfloat16_` 后将 `buffOffset` 对齐到 `BLOCK_SIZE`。

### 3. 修复 AIC workspace 映射

- 误将 AIC 分支 `coreIdx_ /= TASK_RATIO`，导致 AIC 和 AIV 对同一逻辑任务使用不同 workspace 分片。
- 结果是 AIC 等待/读取的 `L_powWsGm_`、`INVWsGm_` 与 AIV 写入不一致，触发同步挂起。
- 恢复原版映射：仅 AIV 根据 `TASK_RATIO` 折算 `coreIdx_`，AIC 使用原始 block id。

### 4. 拆分 Round3 同步 flag

- Neumann Round1 与 Round3 原先复用 `0x1/0x2`，存在跨轮残留或消费顺序不清的问题。
- Round3 改用 `0xa/0xb`，让三轮 `tmp ready / add done` flag 成对独立。

### 5. 整理 workspace 统计语义

- `L_pow/tmp/INV` 实际从 `stageWsAddr_` 内切分，应计入 `stageWorkspaceSz`。
- 将 Neumann 三个 `16×16` float workspace 从 `interWorkspaceSz` 挪到 `stageWorkspaceSz`，总 workspace 大小不变，语义更一致。

---

## 验证结果

验证方式均为 aclnn 单算子直调，不使用 `torch_npu` pytest：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11
cd attention/chunk_gated_delta_rule_neumann
./tests/test_neumann_direct
```

| 用例 | 参数 | 结果 |
|---|---|---|
| default | `B=2,T=128,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000001` |
| var_seqlens | `seqlens=17,65,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000000` |
| short_tail | `B=1,T=5,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000000`，FinalState max_abs_err `0.000031` |
| GQA spot check | `B=2,T=128,nk=2,nv=4,dk=64,dv=64` | PASS |
| dims_32 spot check | `B=2,T=128,nk=4,nv=4,dk=32,dv=32` | PASS |

---

## 当前结论

- `ascend910_93` 构建 + 单算子直调链路已可稳定执行并通过当前阈值。
- 原 `507015` 的直接原因是 AIV UB 非 32B 对齐。
- 后续挂起的直接原因是 AIC `coreIdx_` 错误折算造成 AIC/AIV workspace 不一致。
- 当前实现仍是 AIV 拼完整 `L` / `INV` 后由 AIC 做 Neumann matmul，尚未落地“纯 AIC 生成初始矩阵”的长期方案。
