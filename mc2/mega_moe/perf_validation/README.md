# mega_moe 性能优化

> **环境**：Atlas A3（ascend910_93 / arch22），单节点 16 卡，20 AIC + 40 AIV
> **对比**：`master` @ `0b0856d56` vs `megamoe-optimize`
> **覆盖**：BF16 非量化 + W8A8 量化，88 个用例（真实模型 shape + 边界 + 功能验证），**精度全 PASS**

---

## 1. 性能收益

本轮优化在 **rank 数较低的部署形态**下收益最为显著。dispatch/combine 的数据搬运原先按 rank 维
串行展开，rank 数远小于芯片可用计算核数时大量核闲置；优化后闲置核全部参与搬运，rank 数越少、
可回收的闲置核越多，收益越大。

下表为主流开源模型真实 shape 在低 rank 部署下的实测数据（W8A8 量化，同时段背靠背采集，
模型参数取自各模型官方 `config.json`）：

| 模型 | rank 数 | 专家数(总/每rank) | topk | hidden×intermediate | batch | master(us) | optimize(us) | 提升 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-235B-A22B | 4 | 128 / 32 | 8 | 4096×1536 | 2048 | 5,041.8 | 2,499.6 | **+50.4%** |
| Qwen3-235B-A22B | 4 | 128 / 32 | 8 | 4096×1536 | 1024 | 2,678.4 | 1,393.2 | **+48.0%** |
| Qwen2-57B-A14B | 4 | 64 / 16 | 8 | 3584×2560 | 2048 | 5,000.4 | 2,774.7 | **+44.5%** |
| GLM-4.5 | 4 | 160 / 40 | 8 | 5120×1536 | 2048 | 5,225.0 | 3,074.3 | **+41.2%** |
| Qwen2-57B-A14B | 4 | 64 / 16 | 8 | 3584×2560 | 1024 | 2,563.6 | 1,524.3 | **+40.5%** |
| GLM-4.5 | 4 | 160 / 40 | 8 | 5120×1536 | 1024 | 2,848.5 | 1,708.8 | **+40.0%** |
| DeepSeek-V4-Flash | 4 | 256 / 64 | 6 | 4096×2048 | 2048 | 4,122.6 | 2,636.5 | **+36.0%** |
| Llama-4 Maverick | 4 | 128 / 32 | 8 | 5120×2048 | 2048 | 5,313.9 | 3,489.9 | **+34.3%** |
| Llama-4 Maverick | 4 | 128 / 32 | 8 | 5120×2048 | 1024 | 2,826.4 | 1,953.4 | **+30.9%** |
| DeepSeek-V4-Flash | 4 | 256 / 64 | 6 | 4096×2048 | 1024 | 2,327.7 | 1,752.7 | **+24.7%** |
| Qwen3-Coder-480B | 4 | 160 / 40 | 8 | 6144×2560 | 1024 | 3,453.3 | 2,791.5 | +19.2% |
| DeepSeek-V4-Flash | 8 | 256 / 32 | 6 | 4096×2048 | 1024 | 1,713.7 | 1,417.3 | +17.3% |
| Qwen3-235B-A22B | 8 | 128 / 16 | 8 | 4096×1536 | 1024 | 1,722.0 | 1,452.2 | +15.7% |
| Qwen2-57B-A14B | 8 | 64 / 8 | 8 | 3584×2560 | 1024 | 1,837.5 | 1,559.4 | +15.1% |
| GLM-4.5 | 8 | 160 / 20 | 8 | 5120×1536 | 1024 | 1,987.5 | 1,783.7 | +10.3% |
| Llama-4 Maverick | 8 | 128 / 16 | 8 | 5120×2048 | 1024 | 2,106.6 | 1,928.0 | +8.5% |
| DeepSeek-V4-Flash | 16 | 256 / 16 | 6 | 4096×2048 | 1024 | 1,331.8 | 1,295.3 | +2.7% |
| DeepSeek-V4-Pro | 16 | 384 / 24 | 6 | 7168×3072 | 1024 | 2,734.8 | 2,801.6 | -2.4% |

**18 用例平均 +26.5%，其中 10 例 ≥20%，最高 +50.4%，精度全部 PASS（cos > 0.9999）。**

### 收益规律

| 维度 | 规律 | 实测对照 |
|---|---|---|
| **rank 数** | rank 越少，可回收的闲置核越多，收益越大 | Qwen3-235B batch1024：4 rank **+48.0%** → 8 rank +15.7% → 16 rank ≈0 |
| **batch** | batch 增大使每 rank 数据量上升，闲置核被填满后收益见顶 | Qwen3-235B 4 rank：batch1024 +48.0% → batch2048 **+50.4%**（增幅收窄） |
| **专家数** | 专家越少，每 rank 分到的数据越集中，越易填满闲置核 | Qwen2-57B(64 专家) 4 rank **+40.5%**；DeepSeek-V4-Flash(256 专家) 同档 +24.7% |

> **16 rank 场景收益有限**：此时闲置核本就很少，可优化空间小。专家数多且 hidden 大的模型
> （如 GLM-5.2、Kimi-K2.5）因权重体积限制必须 16 卡部署，落在此区间；而 DeepSeek-V4-Flash
> 虽同为 256 专家，但 hidden 较小可压到 4/8 卡部署，因而能取得 20% 以上收益。

---

## 2. 优化方案

全部为**只削减开销、不改变计算语义**的改动，每项均经全量回归验证。

| # | 方案 | 说明 |
|---|---|---|
| 1 | 开启硬件 unit flag | 让 GMM 的结果搬运与矩阵乘在硬件层面流水重叠，`aic_fixpipe_ratio` 提升 5~10 倍。 |
| 2 | 编译期关闭阶段插桩 | kernel 内 11 处调试打点改为编译期短路，消除每次执行的时钟读取与 GM 写开销，固定开销 −9.5%。 |
| 3 | 前缀和倍增化 | 路由统计的专家前缀和由 O(EP) 串行链改为倍增扫描，依赖链降至 O(log₂EP)，该段耗时 4.81 → 0.80us。 |
| 4 | token 统计分核 | 每专家 token 数的统计从单核承担改为按专家区间分摊到全部核，各核切片互不重叠、无需归约。 |
| 5 | **搬运任务二维化** | dispatch/combine 的任务划分从「一核一 rank」扩展为「rank × 数据块」两个维度，让 rank 数少时闲置的核全部参与搬运——**这是低 rank 场景大幅收益的主要来源**。 |
| 6 | 自适应分块 | 按核数、rank 数、块大小下限动态决定分块粒度，避免把大块搬运切碎反而拖低带宽效率。 |

---

## 3. 验证方法

### 3.1 编译与安装

```bash
cd ops-transformer
source <CANN>/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe -j$(($(nproc)+1))
# 产物须约 20MB；22KB 说明 --soc 写错（必须小写）导致空包
ls -la build_out/cann-ops-transformer-custom_linux-aarch64.run
yes | ./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=<VENDOR_DIR>
```

对比测试时，master 与优化分支各自安装到独立的 `<VENDOR_DIR>`，避免互相覆盖。

### 3.2 运行环境

```bash
source <CANN>/ascend-toolkit/set_env.sh
source <VENDOR_DIR>/vendors/custom_transformer/bin/set_env.bash
export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<独立目录>/.torch_extensions
export HCCL_BUFFSIZE=512   # 4 rank 且 batch>=1024 时必需，默认 200MB 不足
```

### 3.3 精度与性能验证

`golden_megamoe.py` 是**单脚本合一工具**（无外部依赖）：rank0 自动 profiler 抓取每个用例的
kernel Duration，同时与 NPU 小算子级联参考实现比对精度，**一次跑出性能 + 精度**。

```bash
python3 golden_megamoe.py --list                        # 列出全部用例
python3 golden_megamoe.py --case qwen3_235b_ep4_b1024_w8a8   # 单用例（输出 PASS + Duration）
python3 golden_megamoe.py --all                         # 全量 88 用例，约 67 分钟
```

- **通过标准**：每 rank `cos > 0.99` 且 `etn_ok=True`（expert_token_nums 严格相等）。
- **最新全量结果**：88/88 用例通过，0 失败。
- 全量回归脚本：`scripts/run_regression.sh <vendor>`；5 用例快速抽样：`scripts/smoke5.sh <vendor>`。
