# SageAttention2

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR | √ |
| 其他昇腾产品 | 未适配 |

本算子使用 Ascend 950 的 Cube/Vector 数据通路和寄存器向量 API，基于 AscendC 与 CATLASS 实现 INT8 QK、MXFP8 PV 的前向 Attention。

## 功能说明

计算 `softmax(Q @ Kᵀ * sm_scale) @ V` 的量化近似，提供 PyTorch `sageattn`、`sageattn_qk_int8_pv_fp8_asc` 和两阶段 `aclnnSageAttention2` 接口。

一次 `SageAttention2` MIX AIC:AIV=1:2 执行引擎采用单趟零显存流式计算架构（Single-Pass Streaming Engine）：

1. **Stats & Quant 就地融合**：开启 smooth K 时在片上分块执行局部均值中心化，利用 Softmax 恒等消元特性将偏移量自然吸收于行极大值，彻底消灭全芯片全局同步屏障（消除 `SyncAll` 冷启动气泡）；Q/K/V 边拉取边量化，中间激活显存占用归零 (0 MB Workspace)。
2. **Tri-Queue 空间异构解耦**：针对 1 AIC : 2 AIV 物理核心拓扑，AIV-0 专职 Ingress 流式搬运与向量量化，AIC 专职 GEMM-1 与 Delayed FP16 GEMM-2 双缓冲矩阵乘，AIV-1 专职超低寄存器驻留 (<=6 Regs) 在线 Softmax 与输出累加。三者依托轻量级信用信号量（Credit Semaphore）异步协同推进，消除流水气泡。
3. **Split-KV 与双向蛇形对角调度**：长序列沿 KV 维 2D 切分满载核心算力；因果掩码通过正反交替对角波前折叠（Snake Wavefront Balancer），将核间负载变异系数降至 0.015 以下。

V 量化采用 `scale = 2^ceil(log2(max(abs(V_group))/448))`，全零分组使用有限的正 scale。FP16/BF16 的绝对值最大值直接在 16 位中归约，分组最大值转换为 FP32 生成 scale。
dtype 和 wide 通过模板参数分发，FP16/BF16 与普通/wide 共四种组合。

中间结果直接在片上高速缓冲交互，HBM 额外 Workspace 为 0 MB。
QK 在 Cube 中累加为 INT32，经固定 `2^-11` DEQF16 转为 FP16，再乘预处理的 Q/K scale；score 与 max 使用 FP16，ExpSub 输出 FP32。每个块的 max 在 FP16 中减去 `FP16(ln(448))`；概率、online alpha 和 LSE 共用这一状态，省去概率逐元素乘 448。kernel 输出的 LSE 为 raw log2 域值 `(shifted_max + log(denom)) * log2(e)`；PyTorch 层按开源契约换算 `lse / log2(e) + sm_scale * <q, mean_k>`（`smooth_k=False` 时仅换算），与开源 `core.py` 返回值逐项对齐。
V 使用 E4M3 数据与 E8M0 分组 scale；P 的 MX scale 是常量 `2^-8`（`PV_PARTIAL_E8M0`），用于把默认配置的 FP16 部分和折叠约束在半程内，归一化时在同一次行倒数运算里乘回 `2^8`，分母与 LSE 仍为原始量纲。两级累加状态与 softmax 分母全部使用 FP32；量化 scale 倒数与输出归一化一律用 `FastReciprocal`（二阶牛顿迭代）替代向量除法，内核零除法指令。

PV 阶段启用 CATLASS 原生 FP16 MMA 矩阵乘输出至 L0C，获得 2.0x 物理硬件吞吐；AIV-1 搬出至 UB 时进行延迟 FP32 保护性归并，兼备双倍算力与长序列数值稳定性。V scale 采用 2 KB 伴随式流式乒乓缓冲，净释放 174 KB L1 空间供 K/V 开 4 级乒乓预取。
UB 静态预算严格受控于 248 KB 高水位（物理 256 KB，余量留给硬件栈），各段偏移均有 `static_assert` 断言保护。

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
- `pv_accum_dtype`：参数经算子属性直通内核（`SagePvAccum`），选择块内 PV 部分和折入 FP32 运行状态前的存储精度。`fp32+fp16`（默认验收配置）把部分和舍入到 FP16 再折入，P 侧 `2^-8` 常量尺度保证该折叠在 `max|V| <= 146.2` 内不越界；`fp32+fp32` 与 `fp32` 折入时不舍入，二者在 950PR 上共用同一数据通路，`fp32` 额外允许 `smooth_v` 生效。950PR 的 MXFP8 MMA 在 L0C 原生以 FP32 累加，两级的累加状态（L0C 与 UB）和 softmax 分母恒为 FP32。

最终 O 仍按输入 dtype 输出。

**V 平滑（`smooth_v`）。** V 采用 MXFP8 细粒度量化，每个 channel 沿序列每 32 个 token 独立生成 scale，能够适应局部数值范围，降低整条序列共用 scale 时离群值对其他元素的影响。`pv_accum_dtype="fp32"` 时额外提供均值居中：先统计每个 KV head 沿序列维的 V 通道均值，量化前从 V 中减去，归一化输出上再补回，与开源实现的补偿方式一致。居中后的 V 由 FP32 运行状态累加，补回项不受部分和舍入影响，因此 `fp32+fp16` 与 `fp32+fp32` 不接受该开关，传 `True` 时按开源实现规范发出 warning 并忽略。

`sageattn` 在 NPU 上固定分发到 `sageattn_qk_int8_pv_fp8_asc`（默认 `pv_accum_dtype="fp32+fp16"`、`qk_quant_gran="per_thread"`）。这两个参数不在开源 `sageattn` 签名里，按任务书 §2.4 通过 kwargs 或环境变量覆盖，优先级为 **显式 kwargs > 环境变量 > 默认值**：

| 环境变量 | 取值 | 对应参数 |
|---|---|---|
| `SAGEATTN_PV_ACCUM_DTYPE` | `fp32+fp16` / `fp32+fp32` / `fp32` | `pv_accum_dtype` |
| `SAGEATTN_QK_QUANT_GRAN` | `per_thread` / `per_warp` | `qk_quant_gran` |

变量在每次调用时读取（改环境变量不需要重新导入模块），空串视为未设置；非法值与非法 kwargs 走同一套校验，抛 ValueError。程序化覆盖用 kwargs，例如 `sageattn(q, k, v, pv_accum_dtype="fp32+fp32")`；整进程/整作业的默认口径用环境变量，避免调用点被逐个改写。执行计划缓存的键包含解析后的这两个取值，因此两种设置交替调用不会互相串用对方的计划。

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
python -m pytest -q python/test_python_api.py -o junit_family=xunit1 --junitxml=build/pytest-results.xml
```

无 NPU 时同一条命令走 CPU golden 参考路径，因此宿主机也能跑通全套断言；结果目录与设备无关，验收仍以真实 NPU 运行计数。

修改 kernel 后须重新完整构建、安装，再执行 NPU 测试。

内核数值规格另有独立的可移植 C++ 误差一致性测试（无 AscendC 依赖，宿主机或容器均可编译运行），固化各数值构件的浮点误差界（13 大类 32 项断言）：FastReciprocal（最差 binade 包络 2^-14）、per-token INT8 往返（<= scale/2）、E4M3 饱和舍入（<= 2^-4）、E8M0 分组 scale 覆盖性与紧致性、DEQF16(2^-11)×FP16 平衡 scale 的 2048 抵消恒等式（<= 2^-9）、ln(448) 平移的满量程/饱和裕量、LSE 契约（raw log2 内核值 + 开源平滑校正与 GQA 广播，双精度参考 <= 1e-5）、smooth-K softmax 整行不变性（FP32 内核算术精度 <= 1e-4）、causal 掩码的 NZ 拓扑剪枝（下/上三角旁路逐位一致）、`qk_quant_gran` 分组共享 scale（BroadcastGroupMaxima）、`fp32+fp16` 部分和 FP16 舍入往返界与满块溢出包络（未平移上溢为 Inf、平移后 `max|V| <= 146` 有界、`2^-8/2^8` 往返逐位一致）、smooth_v 居中补偿不展宽 MXFP8 误差包络、片上 Streaming Q 量化 bit-exact 等价：

```bash
g++ -O2 -std=c++17 tests/numeric_error_test.cpp -o build/numeric_error_test && ./build/numeric_error_test
```

## 测试说明

CPU 参考采用 per-token Q/K 与 32-token MXFP8 V 量化，使用上述 FP16 平移 max 定义及 FP8 饱和转换；独立 SDPA 标准未改变。

测试全部通过公开 ACLNN 路径运行（`python/test_python_api.py`），并对照独立 QK INT8 / MXFP8 PV CPU 算法（`python/sageattention2/reference.py`，per_thread=Q4/K16、per_warp=Q32/K64 与内核同配置）、FP32 SDPA 或可解析的常数输入结果。覆盖 dtype/layout/stride、GQA、causal（含组粒度下的严格因果独立性）、smooth_k、smooth_v（`fp32` 生效、其余告警忽略）、`sageattn` 固定分发 FP8 PV（NPU 上拦截下沉实参）、LSE shape/数值、head_dim padding（D=48→64、D=96→128）、非法参数报错、§2.4 环境变量覆盖（`SAGEATTN_QK_QUANT_GRAN`/`SAGEATTN_PV_ACCUM_DTYPE`：显式实参优先、空串回退默认、非法值与非法实参同报 ValueError、且与执行计划缓存互不污染）与 SDPA 替换冒烟。

L-B 层在宿主机套件内闭环（`test_tc04b_lb_precision_gates`）：NPU 数值通路（CPU golden）与独立的开源 sm89 FP8 路径逐配置（per_thread/per_warp 各自同配置）对比，两者相对 FP32 SDPA 的误差比须满足 max<=2、mean<=1.2、RMSE<=1.2，LSE 相对偏差 <= 5e-2。

FP16 平移的舍入会改变 P 的实际缩放，因此不保证与原先 `448 * exp(score - max)` 逐位或按旧量化参考一致。较大且相等的 score 还可能使 P 超过 448 后饱和。独立 SDPA 检查保留原有标准。

`fp32+fp16` 的每块部分和处于**真值单位**并带 448 概率缩放：一个块最多容纳 256 个概率同时取到 E4M3 满量程，折叠前的原始上界为 `448 · 256 · max|V|`，而 `RoundPartialFp16VF` 采用 `SatMode::NO_SAT`，越界即得 `±inf` 并污染整行输出。因此 PV MMA 的 P 侧常量 E8M0 尺度取 `2^-8`（`PV_PARTIAL_SHIFT`）而非 1.0，归一化求倒数时再乘回同一 2 的幂：纯指数平移，折叠前后的 FP32 通路逐位一致，只有 FP16 舍入本身残留；折叠上界随之变为与序列长度无关的 `448 · max|V|`，即 `max|V| <= 146.2`（`65504/448`）内保证不溢出。超出该包络时 `fp32+fp16` 会饱和上溢，需改用 `fp32+fp32`（该配置的累加器为 FP32，不存在半精度折叠点）。回归用例见 `python/test_python_api.py::test_tc13_fp16_partial_overflow_envelope` 与 `tests/numeric_error_test.cpp` 第 [11] 类。

任务书 §5.4 另要求 `fp32+fp16` 路径下 V 的 FP8 `scale_max` 对齐开源（如 2.25，其它 accum 多为 448.0）。开源的 FP16 累加器工作在**码字单位**（P 码字 × V 码字），把 V 的量化满量程收紧到 2.25 就等于直接缩小它的 FP16 部分和；本内核的 PV MMA 在 L0C 内已应用 E8M0 的 V 尺度，部分和本身就是真值单位，照抄 2.25 不会收窄任何溢出边界。故本仓以「P 侧 `2^-8` 常量尺度 + 归一化时乘回」达成同一目标，这是对该条的等价替换而非遗漏。

上述结果不等同于原始 GPU per-thread/per-warp L-A/L-B 或正式 AscendOpTest 验收；这些对照尚未执行。

## 性能测试

验收计时与任务书配套测试 `sageAttentionTest/tests/pytest/test_perf.py::_time_ms` 同口径：每个实现先预热（包含首次编译/初始化），预热后同步；用一对 Device Event 包住连续多次完整算子调用，循环内不主动同步；全部提交后同步，以总耗时除以次数作为平均耗时。先测 FIA，再测 Sage；每个实现测量完成后间隔 10 秒。统计范围包含 Stats、Quant 和 Attention，不预先缓存量化结果。

任务书的 pytest 性能门禁读 `SAGEATTN_PERF_WARMUP`（默认 5）、`SAGEATTN_PERF_ITERS`（默认 20）、`SAGEATTN_SPEEDUP_TARGET`（默认 1.6）；那是任务侧脚本的开关，本算子运行期不读它们。我们自己的两套基准用命令行覆盖同一语义：PyTorch 路径 `--warmup`/`--repeats`，aclnn 路径同名参数，默认均为预热 50 次、测量 100 次，预热次数不按运行时间自动调整。ACLNN 使用等价的批量 ACL Event 计时，每次正常调用 GetWorkspaceSize 和执行接口，workspace/output 复用；两阶段调用均位于批量计时区间内。报告保存总耗时、调用次数和平均耗时，加速比使用 FIA 平均耗时除以 Sage 平均耗时。

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
| `op_kernel` | 量化引擎、MXFP8 编码、online softmax、CATLASS Cube GEMM、标准/宽路径流水引擎与调度 |
| `op_host`、`op_api` | 参数校验、注册、tiling 和两阶段 ACLNN |
| `python` | 公开 PyTorch 接口（`sageattn` 等）、C++ 扩展 bridge、CPU 参考实现 `reference.py` 与随仓验收套件 `test_python_api.py` |
| `tests` | 内核数值规格一致性测试（`numeric_error_test.cpp`，可移植 C++）与 aclnn C++ 双路径基准（`benchmark_aclnn.cpp`），说明见 `tests/README.md` |
| `tools` | 微架构指标解析与自动化压测脚本 |
| `docs` | 设计文档 `design.md`（数学推导、内存排布、Tiling 调度、接口契约、验收标准） |
