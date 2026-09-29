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

1. **Stats**：开启 smooth K 时计算 K 均值；每个 KV head 独占单个 AIV 按行分批 FP32 累加，无跨核原子加，重复调用逐位一致。
2. **Quant**：按 token 将 Q/K 量化为 INT8，生成吸收 `sm_scale` 的固定平衡 FP16 scale；V 沿 sequence 每 32 个 token、每个 channel 独立量化为 E4M3，生成 E8M0 scale。
3. **Core**：INT8 QK、online softmax、MXFP8 PV 与 FP32 O 更新流水执行。PV 由硬件应用 MX scale，末尾仅归一化和输出转换。

V 量化采用 `scale = 2^ceil(log2(max(abs(V_group))/448))`，全零分组使用有限的正 scale。FP16/BF16 的绝对值最大值直接在 16 位中归约，只将分组最大值转换为 FP32 以生成 scale。V 不再做全序列 per-channel 量化或减均值处理。

Stats 的均值清零→累加和 Stats→Quant 使用 `SyncAll<true>()` 同步所有 AIV；Quant→Core 使用 `SyncAll<false>()` 同步所有 AIC/AIV，确保量化数据写入 GM 后才开始 Attention。预处理按 `2 * cores` 分配 AIV 任务，空闲核也参与屏障。满足核数、KV 头数为偶数及 Sq 对齐条件时，Stats、Quant 和 Attention 按同一 KV head 分组到两组核，改善双 die 的 L2 局部性；实际收益取决于设备的逻辑核映射，正确性不依赖物理映射。Stats 先清零均值并同步，每个 KV head 由单个 AIV 独占累加写回，无跨核原子加，输出与调用次数无关、逐位确定。Host 使用 `SetScheduleMode(1)` 的 batch 调度，保证全核同步所需的同时驻留。

dtype 和 wide 通过 `ASCENDC_TPL_ARGS_DECL` 声明为模板参数，Host 用 `GET_TPL_TILING_KEY(input_dtype, wide)` 选择入口。FP16/BF16 与普通/wide 共四种模板组合，tiling key 分别为 1/27/257/283（dtype 占低 8 位，wide 占第 8 位）。每次调用仍只启动一个融合 kernel。`WIDE` 同时传给 Quant 的 V 排列和 Attention 的 N256/N512 路径，融合设备代码不再按 dtype 或 shape 执行这两处分发；公共 `attention.h` 只提供 `Attention<T, WIDE>`。

中间结果仅保存在算子 workspace：K 均值、E8M0 V scale、Q8/K8/V8、FP16 Q/K scale 。各区域按 512 字节对齐，tiling 检查大小溢出并加上 CANN 系统 workspace 前缀；每次调用独立分配，无跨调用全局状态。ACLNN 只注册和启动一个融合算子，接口签名不变。

QK 在 Cube 中累加为 INT32，经固定 `2^-11` DEQF16 转为 FP16，再乘预处理的 Q/K scale；score 与 max 使用 FP16，ExpSub 输出 FP32。每个块的 max 在 FP16 中减去 `FP16(ln(448))`，与旧的平移后 max 取最大值；概率、online alpha 和 LSE 共用这一状态，因此省去概率逐元素乘 448。kernel 输出的 LSE 为 raw log2 域值 `(shifted_max + log(denom)) * log2(e)`，不含平滑校正；PyTorch 层按开源契约换算 `lse / log2(e) + sm_scale * <q, mean_k>`（`smooth_k=False` 时仅换算），与开源 `core.py` 返回值逐项对齐。V 使用 E4M3 数据与 E8M0 分组 scale；P 的 MX scale 固定为 1。PV、持续累加的 O 和 softmax 分母全部使用 FP32，除法采用默认 AscendC Div。

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
| `qk_quant_gran` | str，`per_thread` | Q/K INT8 量化粒度，支持 `per_thread`（默认）、`per_warp`；非法值报错 ValueError |
| `sm_scale` | float/None，`None` | 默认 `1/sqrt(D)`，须可表示为有限 FP32 |
| `pv_accum_dtype` | str，`fp32+fp16` | PV 累加精度，支持 `fp32+fp16`（默认）、`fp32+fp32`、`fp32`；非法值报错 ValueError |
| `smooth_k` | bool，`True` | 量化前减去 K 沿序列维的均值 |
| `smooth_v` | bool，`False` | 量化前减去 V 沿序列维的均值，输出端按 softmax 权重和为 1 补偿；仅 `pv_accum_dtype="fp32"` 时生效，其余取值 warning 并忽略 |
| `return_lse` | bool，`False` | 返回 `(out, lse)`；LSE 为 FP32、shape `[B,Hq,Sq]` |

本实现按 Ascend 950 的计算和执行模型提供接口：Q/K 量化支持 `per_thread`（默认）与 `per_warp`；PV 累加支持 `fp32+fp16`（默认）、`fp32+fp32` 与兼容别名 `fp32`。传入非法选项时明确报错 ValueError。

**粒度与累加配置的 NPU 映射**（对齐任务书允许的"将开源粒度映射到 NPU Tile/Block"口径）：

- `qk_quant_gran`：Q/K 对称 INT8 scale 按连续 token 组共享，对齐开源 CUDA thread/warp 映射——`per_thread` 为每 4 个 Q token / 16 个 K token 一个 scale，`per_warp` 为其 8 倍（32/64）；组边界按序列维对齐。V 侧 MXFP8 的细粒度分组（每 channel × 每 32 token）进一步抑制截断误差。
- `pv_accum_dtype`：参数经算子属性直通内核（`SagePvAccum`），选择块内 PV 部分和折入 FP32 运行状态前的存储精度。`fp32+fp16`（默认验收配置）把部分和舍入到 FP16 再折入；`fp32+fp32` 与 `fp32` 折入时不舍入，二者在 950PR 上共用同一数据通路，`fp32` 额外允许 `smooth_v` 生效。950PR 的 MXFP8 MMA 在 L0C 原生以 FP32 累加，PV、O 与 softmax 分母全程 FP32。

最终 O 仍按输入 dtype 输出。

**V 平滑（`smooth_v`）。** V 采用 MXFP8 细粒度量化，每个 channel 沿序列每 32 个 token 独立生成 scale，能够适应局部数值范围，降低整条序列共用 scale 时离群值对其他元素的影响。`pv_accum_dtype="fp32"` 时额外提供均值居中：先统计每个 KV head 沿序列维的 V 通道均值，量化前从 V 中减去，归一化输出上再补回，与开源实现的补偿方式一致。居中后的 V 由 FP32 运行状态累加，补回项不受部分和舍入影响，因此 `fp32+fp16` 与 `fp32+fp32` 不接受该开关，传 `True` 时按开源实现规范发出 warning 并忽略。

`sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`（默认 `pv_accum_dtype="fp32+fp16"`）；可通过 kwargs 覆盖下沉参数，例如 `sageattn(q, k, v, pv_accum_dtype="fp32+fp32")`。

支持 GQA（Hq 必须能被 Hkv 整除）、1≤D≤128、D padding、序列尾块、末维连续的 stride 输入。非因果模式支持 Sq≠Sk。LSE 的 K smoothing 行常数补偿由 PyTorch 层按开源契约添加（见上文 LSE 说明）。

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

ACLNN 声明见 [aclnn_sage_attention2.h](op_api/aclnn_sage_attention2.h)。先调用 `aclnnSageAttention2GetWorkspaceSize` 获得 workspace 大小和 executor，再调用 `aclnnSageAttention2`。两阶段调用的计时口径见"性能测试"一节。

## 编译与安装

已验证环境：CANN Toolkit/950 ops **9.2.0-beta.1**、PyTorch **2.9.0**、torch_npu **2.9.0.post8**，CATLASS commit `337cc89254d46f5d6041d117aca64edf7197027a`。

本目录 `experimental/attention/sageattention2` 由 ops-transformer 顶层工程构建（`--ops=sageattention2`），包含唯一的 `SageAttention2` 注册入口，无额外的同级阶段算子目录。

在仓库顶层执行；CANN 和 CATLASS 路径按实际环境设置：

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export CATLASS_ROOT=/path/to/catlass

bash build.sh --pkg --soc=ascend950 --ops=sageattention2 --build-type=Release -O3
source build/opp/vendors/custom_transformer/bin/set_env.bash
# 源码方式使用 PyTorch 适配层：
export PYTHONPATH="$PWD/experimental/attention/sageattention2/python${PYTHONPATH:+:$PYTHONPATH}"
```

PyTorch 使用本目录独立的 C++ 扩展，通过 `torch.ops.sageattention2.forward` 和 torch_npu 的 `EXEC_NPU_CMD_EXT` 调用 ACLNN。首次 NPU 调用仅编译本算子的 C++ bridge，后续使用 PyTorch 编译缓存；需要 C++17 编译器和 Ninja。自定义 OPP 由 `ASCEND_CUSTOM_OPP_PATH` 指定，须在导入 torch_npu 前设置。

测试使用已安装的 ACLNN 融合算子：

```bash
SAGEATTN_TEST_STANDALONE=1 \
    python -m pytest -q python/test_python_api.py -o junit_family=xunit1 --junitxml=build/pytest-results.xml
```

修改 kernel 后须重新完整构建、安装，再执行 NPU 测试。

内核数值规格另有独立的可移植 C++ 误差一致性测试（无 AscendC 依赖，宿主机或容器均可编译运行），固化各数值构件的浮点误差界（13 大类 29 项断言）：FastReciprocal（最差 binade 包络 2^-14）、per-token INT8 往返（<= scale/2）、E4M3 饱和舍入（<= 2^-4）、E8M0 分组 scale 覆盖性与紧致性、DEQF16(2^-11)×FP16 平衡 scale 的 2048 抵消恒等式（<= 2^-9）、ln(448) 平移的满量程/饱和裕量、LSE 契约（raw log2 内核值 + 开源平滑校正与 GQA 广播，双精度参考 <= 1e-5）、smooth-K softmax 整行不变性（FP32 内核算术精度 <= 1e-4）、`qk_quant_gran` 分组共享 scale（BroadcastGroupMaxima）、`fp32+fp16` 部分和 FP16 舍入往返界、smooth_v 居中补偿不展宽 MXFP8 误差包络、片上 Streaming Q 量化 bit-exact 等价：

```bash
g++ -O2 -std=c++17 tests/numeric_error_test.cpp -o build/numeric_error_test && ./build/numeric_error_test
```

## 测试说明

CPU 参考采用 per-token Q/K 与 32-token MXFP8 V 量化，使用上述 FP16 平移 max 定义及 FP8 饱和转换；独立 SDPA 标准未改变。

测试全部通过公开 ACLNN 路径运行（`python/test_python_api.py`），并对照独立 QK INT8 / MXFP8 PV CPU 算法（`python/sageattention2/reference.py`，per_thread=Q4/K16、per_warp=Q32/K64 与内核同配置）、FP32 SDPA 或可解析的常数输入结果。覆盖 dtype/layout/stride、GQA、causal（含组粒度下的严格因果独立性）、smooth_k、smooth_v（`fp32` 生效、其余告警忽略）、`sageattn` 固定分发 FP8 PV（NPU 上拦截下沉实参）、LSE shape/数值、head_dim padding（D=48→64、D=96→128）、非法参数报错与 SDPA 替换冒烟。

L-B 层在宿主机套件内闭环（`test_tc04b_lb_precision_gates`）：NPU 数值通路（CPU golden）与独立的开源 sm89 FP8 路径逐配置（per_thread/per_warp 各自同配置）对比，两者相对 FP32 SDPA 的误差比须满足 max<=2、mean<=1.2、RMSE<=1.2，LSE 相对偏差 <= 5e-2。

FP16 平移的舍入会改变 P 的实际缩放，因此不保证与原先 `448 * exp(score - max)` 逐位或按旧量化参考一致。零 score、V 全 1 时，FP16 输出为约 0.999512；较大且相等的 score 还可能使 P 超过 448 后饱和。独立 SDPA 检查保留原有标准。

上述结果不等同于原始 GPU per-thread/per-warp L-A/L-B 或正式 AscendOpTest 验收；这些对照尚未执行。

## 性能测试

验收计时与任务书配套 `test_perf.py::_time_ms` 一致：每个实现默认先执行 50 次预热（包含首次编译/初始化），预热后同步；用一对 Device Event 包住连续 100 次完整算子调用，循环内不主动同步；全部提交后同步，以总耗时除以 100 作为平均耗时。先测 FIA，再测 Sage；每个实现测量完成后间隔 10 秒。统计范围包含 Stats、Quant 和 Attention，不预先缓存量化结果。

PyTorch 默认预热 50 次，参数为 `--warmup` 或 `SAGEATTN_PERF_WARMUP`；测量次数使用 `--repeats` 或 `SAGEATTN_PERF_ITERS`，默认 100 次。预热次数不按运行时间自动调整。ACLNN 使用等价的批量 ACL Event 计时，每次正常调用 GetWorkspaceSize 和执行接口，workspace/output 复用；两阶段调用均位于批量计时区间内。报告保存总耗时、调用次数和平均耗时，加速比使用 FIA 平均耗时除以 Sage 平均耗时。

PyTorch 路径以 `python/test_python_api.py` 同源的公开接口调用测量；aclnn 路径每次正常调用 GetWorkspaceSize 与执行接口，两阶段调用均位于批量计时区间内。性能结果（P-01～P-07，PyTorch 与 aclnn 双路径）随自测报告提供，msprof op 采集时应用只调用一次目标算子，预热和重放由 msprof op 默认配置处理。

### aclnn C++ 基准（tests/benchmark_aclnn.cpp）

双路径口径对标 `aclnnFusedInferAttentionScoreV5` vs `aclnnSageAttention2`（P-01～P-07，固定 per_thread + fp32+fp16），两阶段调用均位于批量 ACL Event 计时区间内；默认 warmup=50、repeats=100（`--warmup`/`--repeats` 可覆盖），用例间 500ms 热冷却，`--output` 落盘 JSON：

```bash
g++ -O2 -std=c++17 -I$ASCEND_HOME_PATH/aarch64-linux/include     -I$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include     tests/benchmark_aclnn.cpp -o build/benchmark_aclnn     -L$ASCEND_HOME_PATH/aarch64-linux/lib64 -lascendcl -lopapi
./build/benchmark_aclnn --warmup 50 --repeats 100 --output build/aclnn_bench.json
```

## 目录说明

| 目录 | 内容 |
|---|---|
| `op_kernel` | Stats、Quant、Cube/Vector 分块、online softmax 和 O 更新 |
| `op_host`、`op_api` | 参数校验、注册、tiling 和两阶段 ACLNN |
| `python` | PyTorch ACLNN 适配层和 CPU 参考 |
| `tests` | 内核数值规格一致性测试（可移植 C++） |
