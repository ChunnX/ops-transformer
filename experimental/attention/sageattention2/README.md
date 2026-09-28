# SageAttention2

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR | √ |
| 其他昇腾产品 | 未适配 |

本算子使用 Ascend 950 的 Cube/Vector 数据通路和寄存器向量 API，基于 AscendC 与 CATLASS 实现 INT8 QK、MXFP8 PV 的前向 Attention。

## 功能说明

计算 `softmax(Q @ Kᵀ * sm_scale) @ V` 的量化近似，提供 PyTorch `sageattn`、`sageattn_qk_int8_pv_fp8_asc` 和两阶段 `aclnnSageAttention2` 接口。

一次 `SageAttention2` MIX AIC:AIV=1:2 kernel launch 内依次完成三个阶段：

1. **Stats**：开启 smooth K 时计算 K 均值；对齐长序列使用 Stream-K 分配，跨核片段通过 FP32 atomic add 汇合。
2. **Quant**：按 token 将 Q/K 量化为 INT8，生成吸收 `sm_scale` 的固定平衡 FP16 scale；V 沿 sequence 每 32 个 token、每个 channel 独立量化为 E4M3，生成 E8M0 scale。
3. **Core**：INT8 QK、online softmax、MXFP8 PV 与 FP32 O 更新流水执行。PV 由硬件应用 MX scale，末尾仅归一化和输出转换。

V 量化采用 `scale = 2^ceil(log2(max(abs(V_group))/448))`，全零分组使用有限的正 scale。FP16/BF16 的绝对值最大值直接在 16 位中归约，只将分组最大值转换为 FP32 以生成 scale。V 不再做全序列 per-channel 量化或减均值处理。

Stats 的均值清零→累加和 Stats→Quant 使用 `SyncAll<true>()` 同步所有 AIV；Quant→Core 使用 `SyncAll<false>()` 同步所有 AIC/AIV，确保量化数据写入 GM 后才开始 Attention。预处理按 `2 * cores` 分配 AIV 任务，空闲核也参与屏障。满足核数、KV 头数为偶数及 Sq 对齐条件时，Stats、Quant 和 Attention 按同一 KV head 分组到两组核，改善双 die 的 L2 局部性；实际收益取决于设备的逻辑核映射，正确性不依赖物理映射。Stats 先清零均值并同步，完整 head 独占写回，跨核 head 原子累加。FP32 原子加的顺序可能变化，不保证重复调用逐位一致。Host 使用 `SetScheduleMode(1)` 的 batch 调度，保证全核同步所需的同时驻留。

dtype 和 wide 通过 `ASCENDC_TPL_ARGS_DECL` 声明为模板参数，Host 用 `GET_TPL_TILING_KEY(input_dtype, wide)` 选择入口。FP16/BF16 与普通/wide 共四种模板组合，tiling key 分别为 1/27/257/283（dtype 占低 8 位，wide 占第 8 位）。每次调用仍只启动一个融合 kernel。`WIDE` 同时传给 Quant 的 V 排列和 Attention 的 N256/N512 路径，融合设备代码不再按 dtype 或 shape 执行这两处分发；公共 `attention.h` 只提供 `Attention<T, WIDE>`。

中间结果仅保存在算子 workspace：K 均值、E8M0 V scale、Q8/K8/V8、FP16 Q/K scale 。各区域按 512 字节对齐，tiling 检查大小溢出并加上 CANN 系统 workspace 前缀；每次调用独立分配，无跨调用全局状态。ACLNN 只注册和启动一个融合算子，接口签名不变。

QK 在 Cube 中累加为 INT32，经固定 `2^-11` DEQF16 转为 FP16，再乘预处理的 Q/K scale；score 与 max 使用 FP16，ExpSub 输出 FP32。每个块的 max 在 FP16 中减去 `FP16(ln(448))`，与旧的平移后 max 取最大值；概率、online alpha 和 LSE 共用这一状态，因此省去概率逐元素乘 448，LSE 使用 `shifted_max + log(denom)`。V 使用 E4M3 数据与 E8M0 分组 scale；P 的 MX scale 固定为 1。PV、持续累加的 O 和 softmax 分母全部使用 FP32，除法采用默认 AscendC Div。

连续对齐输入的 Q/V 原始读取使用 bypass L2 的 cache hint，保留 K 与量化输出的默认缓存策略，减少一次性读取对后续数据的挤占。连续对齐输入按 64 行双缓冲量化，Q/K 复用每 token 的 FP32 scale 倒数；非对齐输入保留原有 Q/K 舍入路径。两类输入都使用同一个 V MX 分组量化实现，不足 32 个 token 的尾组补零。V scale 在 workspace 中按 `[B,Hkv,Kp/64,Dp,2]` 保存，每个 E8M0 scale 占一字节。

PV 通过 CATLASS 的 MX scale 布局和搬运接口加载 L0B scale，使用硬件 MXFP8 MMA。P 的单位 scale 在 L0A 初始化一次。V scale 按 KV head 缓存在 L1，容量为 176 KiB；超过容量时按 PV tile 搬运，仍为同一 MXFP8 算法。

通用路径采用 M128/N256 分块、四个 score 槽和两个 P/PV 共享槽。D padding 为 128、Sq 为 128 的倍数且 Sk 为 512 的倍数并不少于 4096 时，使用 N512 路径：两个 N256 Cube 片段共享一次 softmax，流水跨 query 连续执行。O 常驻 UB，P 经 UB→L1 交给 Cube，QK/PV 经 L0C→UB 交给 Vector。wide 路径的 K/V 各使用两个 L1 缓冲区预取；Cube 按 N256 片段推进，Vector 按 N512 分组推进。

wide 路径将 FP8 P 原位写入已经消费的 FP16 score 条带，再通过带源间隔的 UB→L1 搬运保持既有 L1 布局，使两个 PV 槽能保留待合并的结果。同一 query 的相邻两组输出配对更新，在两次 PV 等待之间先计算下一组 softmax 并供 P；读取完成后才释放两个 PV 槽，保持两次 FP32 FMA 的舍入顺序。alpha 和延迟输出记录各用三个槽，跨 query 无法配对及奇数组尾部退回单组更新。P_READY 后的 Cube 计算依赖保护 score 槽复用，UB 高水位为 246 KiB。

## 参数说明

输入 Q/K/V 均为四维、同设备、同 dtype 的 FP16 或 BF16 NPU Tensor，最后一维连续。PyTorch 接口要求基础 ND/NCHW 存储，不支持私有存储格式。输出 shape/dtype 与 Q 相同。

| 参数 | 类型/默认值 | 说明 |
|---|---|---|
| `q` | Tensor | HND 为 `[B,Hq,Sq,D]`，NHD 为 `[B,Sq,Hq,D]` |
| `k`、`v` | Tensor | HND 为 `[B,Hkv,Sk,D]`，NHD 为 `[B,Sk,Hkv,D]` |
| `tensor_layout` | str，`HND` | 支持 `HND`、`NHD` |
| `is_causal` | bool，`False` | 开启时要求 Sq=Sk |
| `qk_quant_gran` | str，`per_thread` | Q/K INT8 量化粒度，支持 `per_thread`（默认）、`per_warp`、`per_token`；非法值报错 ValueError |
| `sm_scale` | float/None，`None` | 默认 `1/sqrt(D)`，须可表示为有限 FP32 |
| `pv_accum_dtype` | str，`fp32+fp16` | PV 累加精度，支持 `fp32+fp16`（默认）、`fp32+fp32`、`fp32`；非法值报错 ValueError |
| `smooth_k` | bool，`True` | 量化前减去 K 沿序列维的均值 |
| `smooth_v` | 兼容参数，`False` | 传 `True` 且 `pv_accum_dtype` 为 `fp32+fp16` 或 `fp32+fp32` 时 warning 并忽略（对齐开源实现） |
| `return_lse` | bool，`False` | 返回 `(out, lse)`；LSE 为 FP32、shape `[B,Hq,Sq]` |

本实现按 Ascend 950 的计算和执行模型提供接口：Q/K 量化支持 `per_thread`（默认）、`per_warp` 与 `per_token`；PV 累加支持 `fp32+fp16`（默认）、`fp32+fp32` 与兼容别名 `fp32`。上述标准参数均原生静默支持，不触发回退告警；传入非法选项时明确报错 ValueError。当 `smooth_v=True` 且 `pv_accum_dtype` 为 `fp32+fp16` 或 `fp32+fp32` 时，按开源实现规范发出 warning 并忽略。最终 O 仍按输入 dtype 输出。

**不再支持 smooth_v 计算。** V 改用 MXFP8 细粒度量化，每个 channel 沿序列每 32 个 token 独立生成 scale，能够适应局部数值范围，降低整条序列共用 scale 时离群值对其他元素的影响。因此当前路径无需再做 V 均值统计、减均值及输出补偿，省去相应计算与访存。`smooth_v` 仅作为兼容参数保留，传入 `True` 时告警并忽略。这是当前量化方案的取舍，不表示任意输入分布下平滑都不可能改善误差。

支持 GQA（Hq 必须能被 Hkv 整除）、1≤D≤128、D padding、序列尾块、末维连续的 stride 输入。非因果模式支持 Sq≠Sk。LSE 包含 K smoothing 的行常数补偿。

不支持任意 attention mask、dropout、varlen 和 autograd。

```python
import torch
import torch_npu
from sageattention2 import sageattn

q = torch.randn(1, 4, 4096, 128, dtype=torch.float16, device="npu")
k = torch.randn(1, 2, 4096, 128, dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
out, lse = sageattn(q, k, v, return_lse=True)
```

ACLNN 声明见 [aclnn_sage_attention2.h](op_api/aclnn_sage_attention2.h)。先调用 `aclnnSageAttention2GetWorkspaceSize` 获得 workspace 大小和 executor，再调用 `aclnnSageAttention2`。可运行的调用示例见 [benchmark_aclnn.cpp](benchmarks/benchmark_aclnn.cpp)。

CUDA 默认接口、两级累加精度与量化粒度的详细对照见 [与 CUDA 默认接口的差异](docs/CUDA接口差异说明.md)。

## 编译与安装

已验证环境：CANN Toolkit/950 ops **9.2.0-beta.1**、PyTorch **2.9.0**、torch_npu **2.9.0.post8**，CATLASS commit `337cc89254d46f5d6041d117aca64edf7197027a`。构建和测试依赖见 [requirements.txt](requirements.txt)。

本目录 `experimental/attention/sageattention2` 由 ops-transformer 顶层工程构建（`--ops=sageattention2`），包含唯一的 `SageAttention2` 注册入口，无额外的同级阶段算子目录。

在本目录执行；CANN 和 CATLASS 路径按实际环境设置：

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export CATLASS_ROOT=/path/to/catlass
export ASCEND_RT_VISIBLE_DEVICES=0

bash install_opp.sh
source build/opp/vendors/custom_transformer/bin/set_env.bash
python -m pip install --no-deps .
# 源码开发也可使用：export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

`install_opp.sh` 调用仓库完整构建流程并安装到本目录的 `build/opp`。PyTorch 使用本目录独立的 C++ 扩展，通过 `torch.ops.sageattention2.forward` 和 torch_npu 的 `EXEC_NPU_CMD_EXT` 调用 ACLNN。首次 NPU 调用仅编译本算子的 C++ bridge，后续使用 PyTorch 编译缓存；需要 C++17 编译器和 Ninja。自定义 OPP 由 `ASCEND_CUSTOM_OPP_PATH` 指定，须在导入 torch_npu 前设置。

测试使用已安装的 ACLNN 融合算子：

```bash
bash benchmarks/build_aclnn_benchmark.sh
SAGEATTN_TEST_STANDALONE=1 \
    python -m pytest -q tests -o junit_family=xunit1 --junitxml=build/pytest-results.xml
```

修改 kernel 后须重新完整构建、安装，再执行 NPU 测试。

接入结构、安装与图模式范围见 [PyTorch 接入说明](docs/PyTorch接入说明.md)。

## 测试说明

CPU 参考采用 per-token Q/K 与 32-token MXFP8 V 量化，使用上述 FP16 平移 max 定义及 FP8 饱和转换；独立 SDPA 标准未改变。

测试全部通过公开 ACLNN 路径运行，并对照独立 per-token QK / MXFP8 PV CPU 算法、FP32 SDPA 或可解析的常数输入结果。覆盖 dtype/layout/stride/GQA/causal、smooth K、smooth_v warning/ignore、LSE、长序列、模板选择边界、重复调用、并发流和独立 C++ ACLNN 入口。

PyTorch 接入测试另覆盖 Meta/FakeTensor、dispatcher schema、`torch.compile(backend="eager", fullgraph=True)` 和私有存储格式检查；图模式范围见 [PyTorch 接入说明](docs/PyTorch接入说明.md)。

[测试矩阵](tests/task_cases.py) 由 Python 参数组合直接生成并去重，包含 204 条功能、16 条精度及 5 条非法参数用例。正向用例统一使用标准 `per_thread`、`fp32+fp16` 和 `smooth_v=False`，出现 fallback 告警即判失败；兼容别名、回退告警和忽略参数由 `tests/test_mxfp8.py` 专门验证。性能用例统一定义在 [benchmarks/cases.py](benchmarks/cases.py)。

MXFP8 PV 用例采用 CPU 独立量化的端到端对照，使用下述统一标准，并检查重复调用数值一致性及独立 SDPA 误差。wide 的 LSE 对照包含 FP16 scale/score 舍入的量化参考。

数值参考是本目录的 per-token QK / MXFP8 PV CPU 算法及独立 FP32 SDPA。CPU 同算法检查采用 FP16 rtol/atol=2^-9、BF16=2^-6、至少 99% 匹配比例；最大绝对误差通常分别限于 0.1/1。SDPA 的 max/mean/RMSE 误差比例门限为 2/1.2/1.2。

MXFP8 回归统一检查最大绝对误差上限：FP16 为 0.1，BF16 为 1.0。分组 scale、非对齐尾块和 L1 缓存容量边界另有独立可解析输入测试。

`tests/test_wide_sequence.py` 补充 4K/8K/16K 完整序列的 wide 精度覆盖：12 项双 dtype、causal/non-causal、GQA 和 HND/NHD 用例逐元素对照 CPU 同算法（`block_k=512`），同时检查可选 LSE。另以 P-01～P-07 的完整 B/H/S 尺寸运行两条公开接口，检查全部输出和 LSE 的一致性，并对首尾 query head 的全部 batch/序列做 CPU 对照；这七项 CPU 对照是 head 抽样，不代表全 head 的独立精度验证。

FP16 平移的舍入会改变 P 的实际缩放，因此不保证与原先 `448 * exp(score - max)` 逐位或按旧量化参考一致。零 score、V 全 1 时，FP16 输出为约 0.999512；较大且相等的 score 还可能使 P 超过 448 后饱和。独立 SDPA 检查保留原有标准。

上述结果不等同于原始 GPU per-thread/per-warp L-A/L-B 或正式 AscendOpTest 验收；这些对照尚未执行。

## 性能测试

验收计时与任务书配套 `test_perf.py::_time_ms` 一致：每个实现默认先执行 50 次预热（包含首次编译/初始化），预热后同步；用一对 Device Event 包住连续 100 次完整算子调用，循环内不主动同步；全部提交后同步，以总耗时除以 100 作为平均耗时。先测 FIA，再测 Sage；每个实现测量完成后间隔 10 秒。统计范围包含 Stats、Quant 和 Attention，不预先缓存量化结果。

PyTorch 默认预热 50 次，参数为 `--warmup` 或 `SAGEATTN_PERF_WARMUP`；测量次数使用 `--repeats` 或 `SAGEATTN_PERF_ITERS`，默认 100 次。预热次数不按运行时间自动调整。ACLNN 使用等价的批量 ACL Event 计时，每次正常调用 GetWorkspaceSize 和执行接口，workspace/output 复用；两阶段调用均位于批量计时区间内。报告保存总耗时、调用次数和平均耗时，加速比使用 FIA 平均耗时除以 Sage 平均耗时。

```bash
python benchmarks/run_validation_performance.py --output build/performance_batch
bash benchmarks/build_aclnn_benchmark.sh
mkdir -p build/performance_batch/aclnn
./build/benchmarks/benchmark_aclnn 4096 1 32 0 0 50 100 build/performance_batch/aclnn/P-01.json
```

### msprof op

采集单个完整算子；应用只调用一次目标算子，预热和重放由 msprof op 默认配置处理。不设置 `--warm-up`、`--aic-metrics` 或 `--launch-count`，保留默认指标。

```bash
msprof op --kernel-name=SageAttention2 --output=build/msprof/P-01/sage \
    --application="python benchmarks/profile_single_op.py --case P-01 --operator sage"
msprof op --kernel-name=FusedInferAttentionScore --output=build/msprof/P-01/fia \
    --application="python benchmarks/profile_single_op.py --case P-01 --operator fia"
```

`benchmark_aclnn` 参数依次为 `SEQ BATCH KV_HEADS BF16 CAUSAL WARMUP_CALLS REPEATS OUTPUT_JSON`。性能结果随自测材料提供。

## 目录说明

| 目录 | 内容 |
|---|---|
| `op_kernel` | Stats、Quant、Cube/Vector 分块、online softmax 和 O 更新 |
| `op_host`、`op_api` | 参数校验、注册、tiling 和两阶段 ACLNN |
| `python` | PyTorch ACLNN 适配层和 CPU 参考 |
| `tests` | 公开接口功能、精度和多流回归 |
| `benchmarks` | 性能测试脚本和独立 ACLNN 调用示例 |
