---
name: ascendc-operator-performance-optim
description: Use when developing, reviewing, or optimizing ops-transformer Ascend C AI Core operator performance, especially 910B/B3 arch22 tiling, DataCopy, memory, pipeline, package/load gates, NPU profiling, or performance regression work.
---

# AscendC 算子性能优化闭环

本 skill 用于 `ops-transformer` 仓库中的 AscendC AI Core 算子性能优化。默认事实：

- 主线目录：`attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`。
- 当前目标：Ascend 910B B3，`--soc=ascend910b`，架构特化目录 `arch22`。
- custom package vendor：`custom_transformer`。
- 当前本地 ST runner 只优先覆盖 `attention/lightning_indexer` 与 `attention/recurrent_gated_delta_rule`，不能外推为全仓库 ST coverage。
- `agent_ops/<op>/` 和 `scripts/agent_context` 不是当前仓库必备基建；若未来引入，只记录摘要、manifest 和 hash，不长期保存 raw profiling dump。

## 强制边界

- 不为了让精度通过而修改 golden、阈值、case JSON 或 runner 语义。
- 不为了制造性能提升而删减必要计算、跳过写回、改变 dtype/format 支持范围或只测有利 case。
- `unsupported`、跳过、runner gate 不算精度通过。
- `build.sh --opkernel` 只证明 binary 编译成功；涉及 tiling/opdef/op_api/metadata 时默认重新 `--pkg` 并安装。
- 只有精度正常、目标 case 提升、代表性矩阵无稳定回退、profiling 证明加载新产物时，才能保留优化。

## 读取顺序

先读本仓库方法和策略：

1. `agent/performance-optimization-method.md`
2. `agent/policies/accuracy-integrity.md`
3. `agent/policies/performance-integrity.md`
4. `agent/runbooks/build.md`
5. `agent/runbooks/package-deploy.md`
6. `agent/runbooks/performance.md`

排查具体代码时，按需读取本 skill 的 references：

| 阶段 | Reference |
| --- | --- |
| Tiling | `references/tiling-prof.md` |
| DataCopy | `references/data-copy-prof.md` |
| API usage | `references/api-usage-prof.md` |
| Memory | `references/memory-prof.md` |
| Pipeline | `references/pipeline-prof.md` |

只加载当前阶段需要的 reference，避免一次性加载所有资料。

## Phase 1: 候选和路径确认

确认目标算子：

- 位于主线目录，存在 `op_kernel/` 和 AscendC AI Core 入口。
- 优先检查 `<domain>/<op>/op_kernel/arch22/`，没有 arch22 时再看通用实现和条件编译。
- 检查 `<domain>/<op>/op_host/*tiling*`、`op_host/config/`、`op_host_aclnn*`、`op_graph/`、`examples/`、`tests/`。
- 对 MC2 类算子，先确认多卡、rank、HCCL 和拓扑；单卡 smoke 或 simulator 不代表真实性能。

输出候选摘要：

```text
operator:
domain:
api:
target shape/dtype/format/attrs:
source files:
accuracy entry:
profiling entry:
package/load risk:
```

## Phase 2: Baseline Gate

建立 baseline 前必须完成：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

记录 package/load evidence：

```bash
sha256sum build_out/cann-ops-transformer-custom_linux-aarch64.run 2>/dev/null || true
sha256sum /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so 2>/dev/null || true
find /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer -path '*ascend910b*' -type f | rg '<op_name>|kernel|config'
```

精度验证按可用入口选择：

```bash
bash build.sh -u --ophost_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opapi_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opkernel_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
```

两个本地 ST 已支持算子可用：

```bash
python3 scripts/st_accuracy/run_st_accuracy.py run --ops lightning_indexer,recurrent_gated_delta_rule --soc ascend910b --out build/st_accuracy/<tag>
```

## Phase 3: Profiling Baseline

默认通过 eager example profiling：

```bash
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
cd build
msprof op --application=./test_aclnn_<op_name> --output=../artifacts/<op_name>/msprof/<tag>
```

矩阵必须覆盖目标 shape、小 shape、边界 shape、常见中等 shape 和相关 dtype/attrs。每个 case 至少采集 3 次，默认用 median。

如果一个 aclnn 调用触发多个 runtime task，同时记录目标 kernel time 和 total device time。batch runner 的 `--launch-count` 必须覆盖真实 device task 数；混合 kernel suffix 或 tiling key 时必须重建 case 映射。

## Phase 4: 排查和修改

先写优化假设：

```text
hypothesis:
suspected bound:
expected effect:
changed files:
risk:
rollback condition:
```

按顺序排查：

1. Tiling：blockDim、核间切分、UB tile、tail、dtype/shape tiling key。
2. DataCopy：对齐、单次搬运量、stride copy、小 DMA 描述数。
3. API usage：TPipe/TQue/TBuf 生命周期、counter mask、vector API 选择。
4. Memory：UB/L1/L0/BT/FP buffer 驻留复用、bank conflict。
5. Pipeline：CopyIn/Compute/CopyOut overlap、double buffer、同步粒度。

修改后按影响面选择构建：

- 只改 kernel 且能显式同步 runtime binary/config：可用 `--opkernel` 快速验证，并记录同步路径。
- 改 `op_host`、tiling、opdef、op_api、metadata、workspace、dtype/format 或 tiling key：必须重新 `--pkg` 并安装。

## Phase 5: 精度和性能复验

每轮 candidate：

1. 重新构建/安装或同步新产物。
2. 记录 package/load evidence。
3. 复跑精度。
4. 用 baseline 的同一矩阵 profiling。
5. 至少 3 次采样并计算 median。

判定：

| 结果 | 处理 |
| --- | --- |
| 目标 case 提升且代表性矩阵无稳定回退 | 可保留，继续证据归档 |
| 只在目标大 shape 提升但小/边界/中等 shape 回退 | 拒绝或继续调整 |
| 收益低于约 1% 且无同窗口 A/B 证明 | 默认视为噪声 |
| 精度失败、case 映射不可信、加载旧产物 | 数据无效，修复后重采 |

## Phase 6: 收尾

记录：

```text
operator:
changed files:
build/install:
accuracy:
profiling:
matrix:
package evidence:
decision:
retention action:
```

按 `agent/policies/evidence-retention.md` 处理 raw evidence。形成可复用经验时更新：

- `agent/lessons/performance-optimization.md`
- `.agents/skills/ascendc-performance-best-practices/`
- `.agents/skills/ops-profiling/`

从 `ops-math` 迁移的成功或失败案例只能作为假设来源；任何保留结论都必须来自本仓库复验。
