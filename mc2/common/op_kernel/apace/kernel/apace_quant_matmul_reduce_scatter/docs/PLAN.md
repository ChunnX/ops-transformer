# apace_quant_matmul_reduce_scatter_v3 算子开发计划

> 本文档在开发流程中持续更新。设计依据：`docs/DESIGN.md`（v3 = v2 + 取消 workspace 轮转回压）。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | apace_quant_matmul_reduce_scatter_v3 |
| 数学公式 | C_r = Σ_{i=0}^{rankSize-1} dequant(A_i[r·M/R:(r+1)·M/R, :]) × dequant(B_i)，R=4（与 v2 完全相同） |
| 输入 | A: [M, K/4] fp8_e4m3fn_t/fp8_e5m2_t；ScaleA: fp8_e8m0_t；B: [K, N] fp8；ScaleB: fp8_e8m0_t；workspace: GM（**新大小公式**） |
| 输出 | C: [M/4, N] bfloat16_t |
| 算子类别 | MC2 通算融合（Quant MatMul + AllToAll PUT + ReduceSum 3 级流水） |
| 需求类型 | 通用 |
| v3 变更核心 | 取消 workspace 轮转回压：槽位寻址去取模（每 tile 独占）、删除通道 3 回压 flag（3 处）、workspace = maxTilesPerCore×coreNum×baseM×baseN×2B、新增 maxTilesPerCore ≤ 15 host 校验 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/apace_quant_matmul_reduce_scatter_v3/quant_matmul_mx_kernel.h` — 复制自 v2 + 变更点 C1/C2/C3/C10 | ⬜ |
| `kernel/apace_quant_matmul_reduce_scatter_v3/apace_quant_matmul_reduce_scatter_v3_udma_impl.h` — 复制自 v2 + 变更点 C4/C5/C9 | ⬜ |
| `kernel/apace_quant_matmul_reduce_scatter_v3/apace_quant_matmul_reduce_scatter_v3_tiling_data.h` — 复制自 v2 + 变更点 C6（pipelineNum→maxTilesPerCore） | ⬜ |
| `st/apace_quant_matmul_reduce_scatter_v3/src/main.cpp` — C7 workspace 新公式 + C8 校验 | ⬜ |
| `st/apace_quant_matmul_reduce_scatter_v3/src/kernel_launcher.h` — 4 dtype 入口更名 _V3 | ⬜ |
| `st/apace_quant_matmul_reduce_scatter_v3/cases.csv` — v2 10 用例 + 新增 T11-T14 | ⬜ |
| `st/apace_quant_matmul_reduce_scatter_v3/{CMakeLists.txt,run.sh,scripts/}` — 复制更名 | ⬜ |
| 共享层 `block/`、`tiling/`、`basic/`、`utils/`、`third_party/` — [REUSE] 禁止修改 | ⬜ |

---

## 3. 测试计划

精度标准：rtol=atol=1e-2（继承 v2，数值路径零变更）。

**Golden 计算**：复用 v2 `scripts/gen_data.py` + `scripts/verify_result.py`（CPU 参考实现不变）。

**用例**（可追溯至 DESIGN.md §8）：

| 编号 | 用例 | 需求追溯 | 输入 (M,K,N) | 预期 |
|-----|------|---------|--------------|------|
| T1-T10 | v2 全量回归（4 dtype × shape 系列） | §0.2 #5（其余全部保留） | 2048,4096,2048 / 512,1024,512 / 1120,2048,1024 等 | 精度 PASS，与 v2 结果一致 |
| T11 | 大 shape 边界 | §0.2 #4（~18MB 可接受） | 16384,4096,2048 | 精度 PASS，workspace ~19-21MB 分配成功 |
| T12 | **tiles/core>4 回压消除收益** | §0.2 #1 #2（核心优化目标） | 16384,4096,4096（T=8） | 精度 PASS + 性能优于 v2 同 shape |
| T13 | maxTilesPerCore>15 校验 | §0.2 #2 + DESIGN §4.3 | 构造 T>15 shape（如 32768,…,8192） | host 拒绝 launch，报错清晰 |
| T14 | 尾块拆分路径 | §0.2 #7（UpdateTailTile 上界） | 尾块 shape（如 2252 系列） | 精度 PASS（证明 +1 上界正确） |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| T0 前置验证 | v2 工程打印 T11/T12 tiling 日志确认 usedCoreNum | ✅（2026-08-12，usedCoreNum=28，T11 T=5 / T12 T=10 均 ≤15 放行，v2 两 shape 精度 PASS） |
| 框架搭建 | v3 工程从 v2 复制 + 更名（kernel/st/类名/常量 _V3）+ 编译通过 | ✅（2026-08-12，BUILD_RC=0 一次通过，qrun z00949060-v3-build1） |
| Kernel 改造 | C1-C6/C10 变更点落地（删回压 3 处 + 寻址去取模 + tiling 字段） + 编译通过 | ✅（diff 自审与 DESIGN §1.1 逐项一致） |
| Host 改造 | C7 workspace 新公式 + C8 maxTilesPerCore≤15 校验 + cases.csv 新增用例 | ✅（13 用例 csv：v2 10 回归 + T11 + T12 + T14=2252,2048,1024） |
| 精度验证 | T1-T11、T14 全部 PASS（rtol=atol=1e-2） | ✅（2026-08-12，qrun z00949060-v3-precision，13/13 PASS，rel_err max=6.06e-3 ≪ 1e-2；含 T12 tiles/core=10>4 回压消除场景与 T14=2252 尾块拆分/早退分支） |
| 校验验证 | T13 host 正确拒绝 | ✅（2026-08-12，qrun z00949060-v3-t13：T13-a M=32768,N=8192 T=37>15 正确拒绝 RC=255 报错清晰；T13-b M=10240,N=10752 T=15 恰好放行 4 卡全 PASS） |
| 性能验收 | T12 对比 v2 基线（57.6us @主用例），msprof 采集归档 | ✅（2026-08-12，qrun z00949060-v3-perf，docs/perf/round_001/；T12 263.6us vs v2 291.4us = -9.6%，主用例 57.6us 持平，判定达标） |

### T0 前置用例结果（2026-08-12，qrun 任务 z00949060-v3-t0-tiling）

v2 二进制（--skip-build）实测 SWAT tiling 输出：

| Shape (M,K,N) | baseM×baseN | usedCoreNum | blockNums | maxTilesPerCore=CeilDiv(blockNums,28) | C8 校验 | v2 精度 |
|---------------|-------------|-------------|-----------|---------------------------------------|---------|---------|
| T11: 16384,4096,2048 | 256×256 | **28** | 16×8=128 | **5** | ≤15 ✅ 放行 | PASS |
| T12: 16384,4096,4096 | 256×256 | **28** | 16×16=256 | **10** | ≤15 ✅ 放行（余量 5 档） | PASS |

**结论**：
- 本机 950PR 实际 usedCoreNum=**28**（非 DESIGN §3.2 假设的 32），WALKTHROUGH 问题 1 的反例（usedCoreNum ≤ 17 导致 T12 误拒）**不成立**，T12 重点用例可行；
- T12 对应 v3 workspace = 10×28×256×256×2B = **35.0MB**（28 核下 T=10，比 DESIGN 按 32 核估算的 32.0MB 略大 9%，同属用户已接受的 32MB 量级遗留风险范畴，如实记录）；
- T11 对应 v3 workspace = 5×28×256×256×2B = **17.5MB**（符合 ~18MB 口径）；
- v2 在两个大 shape 上精度均 PASS（T12 在 v2 下 T=10>4 回压触发，为 v3 回压消除收益对比提供同 shape 基线）。

原始日志：`/tmp/opencode/t0_t11.log`、`/tmp/opencode/t0_t12.log`（开发机临时目录）。

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-08-12 | maxTilesPerCore 上界公式 | `CeilDiv(blockDimM×blockDimN, coreNum) + 1`；依据 Blaze `block_scheduler_qbmm.h`：roundNums_ 初始 ≤ CeilDiv(blockNums,coreNums)（L63-66），UpdateTailTile 至多 +1（L92-94），GetTileIdx 迭代数 = roundNums_（L190-235），零形状早退只减不增（L168-170）。见 DESIGN §2 |
| 2026-08-12 | flagId=5 计数器风险 | 删除回压后 AIC 可领先 AIV 全部 T 个 tile，flagId=5 计数器（0-15）可能溢出 → 新增 host 校验 maxTilesPerCore ≤ 15。官方依据：CrossCoreWaitFlag_ISASI.md「计数范围为 0-15，超出则异常报错」。见 DESIGN §4.3 |
| 2026-08-12 | CommFlagIdV2 简化 | 可选优化，默认保持 v2 映射不变（避开 3 是更强约束，天然满足新约束）；仅未来需要第 11 个通信 tile 时切换。见 DESIGN §4.4 |
| 2026-08-12 | local 后置 / fragmentTensor / buffer 复用 | 用户约束：本期不考虑（后续版本） |
| 2026-08-12 | 【遗留风险】T12 workspace 35.0MB | 用户已接受 32.0MB 口径（WALKTHROUGH 仲裁）；T0 实测 usedCoreNum=28（非 32），T=CeilDiv(256,28)=10 → 实际 35.0MB，同量级遗留，后续版本（local 后置/fragmentTensor）再评估 workspace 复用 |
| 2026-08-12 | T13 用例构造 | 拒绝用例 M=32768,K=4096,N=8192（blockNums=1024，T=37>15，对 baseM/baseN 取值不敏感，robust）；cases.csv 仅含预期 PASS 用例，T13 用 CLI 单独验证报错 |

---

## 6. 测试结果

### 6.1 精度通路

**状态**: ✅ 13/13 PASS | **脚本**: run.sh + scripts/verify_result.py（4 卡，source /home/l00947526/Ascend806/cann/set_env.sh）

| 编号 | 用例 | 结果 | Max Rel Diff |
|-----|------|------|--------------|
| T1-T10 | v2 回归（4 dtype × 2048/512/1120 系列） | ✅ 10/10 PASS | ≤6.1e-3 |
| T11 | 大 shape 边界 16384,4096,2048 | ✅ PASS | 0（全等） |
| T12 | tiles/core>4 16384,4096,4096 | ✅ PASS | 0（全等） |
| T13-a | 上界校验拒绝 32768,4096,8192（T=37>15） | ✅ 正确拒绝（RC=255，报错含 maxTilesPerCore/blockNums/coreNum 全参数，提示回退 v2） | — |
| T13-b | 边界恰好放行 10240,4096,10752（T=15，workspace 52.5MB） | ✅ 放行且 4 卡全 PASS | ≤6.1e-3 |
| T14 | 尾块拆分 2252,2048,1024 | ✅ PASS | ≤6.1e-3 |

> 注：v3 数值路径与 v2 完全相同（仅改同步与槽位寻址），rel_err 与 v2 同水平；T11/T12 输出与 CPU golden 全等（count(|abs_err|>0.01)=0）。

### 6.2 产物 & 执行状态

- [x] v3 kernel/st 编译通过（`--npu-arch dav-3510`，BUILD_RC=0）
- [x] 4 卡 ST 全用例运行完成（13 精度用例 + T13 两校验用例）

---

## 7. 性能验收

**状态**: ✅ | **数据**: docs/perf/round_001/（summary.txt + parse_v3/v2.txt + 原始日志）

| 指标 | v3 值 | v2 基线 | 判定 |
|------|-------|---------|------|
| Task Duration（主用例 2048,4096,2048，T=1） | 57.559 us | 57.626 us（同日复测）/ 57.6 us（round_001） | ✅ 持平（-0.1%，T=1 回压本不触发） |
| Task Duration（T11 16384,4096,2048，T=5） | 198.444 us | 199.223 us | ✅ 持平微优（-0.4%） |
| Task Duration（T12 大 shape 16384,4096,4096，T=10>4） | **263.561 us** | **291.446 us** | ✅ **-9.6%（-27.9us）回压消除收益显著** |
| 主导流水 | cube 95.5-95.7%（T12）/ 86.5%（主用例），mm 计算主导 | 同 v2 | ✅ 符合 DESIGN §5 预期 |

**达标判定**: ✅ | **理由**: T12 重点用例 v3 显著优于 v2（-9.6%），主用例与 T11 无回归，符合 DESIGN §8 预期；精度在 perf 采集中同步 PASS。详见 docs/perf/round_001/summary.txt。

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| 精度 | 13（T1-T10 回归 + T11 + T12 + T14） | 13 | 0 | ✅ |
| 校验 | 2（T13-a 拒绝 + T13-b 边界放行） | 2 | 0 | ✅ |
| 性能 | 3（主用例 + T11 + T12，v2 同 shape 基线对比） | 3 | 0 | ✅ |

---

## 9. 开发自检（对照 DESIGN.md，提交审查前门禁）

- [x] C1-C10 全部变更点已在代码中实现（v2→v3 diff 逐项核对：kernel.h C1/C2/C3/C10、udma_impl.h C4/C5/C9（保持 v2 映射）/类名 _V3、tiling_data.h C6、main.cpp C7/C8、kernel_launcher.h 4 入口 _V3）
- [x] 设计框架未偏离：Blaze 组件/通信组件/reduceSum 搬运/通道 1/通道 2 与 v2 完全一致；共享层 block/tiling/basic/utils/third_party diff 为空（[REUSE] 未改）
- [x] 前置验证步骤已执行：T0（usedCoreNum=28 实测）在开发早期完成；C8 校验顺序（tiling 后、aclrtMalloc 前）与取值来源（tileQbmmTilingData 最终值）按串讲问题 7 决议实现
- [x] 偏离说明：无偏离。补充记录——T13-b 边界用例（M=10240,N=10752,T=15）workspace 52.5MB 超出已接受的 32-35MB 口径，但该用例仅用于一次性校验边界验证，非常规目标 shape，如实记录
