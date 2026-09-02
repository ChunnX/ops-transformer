# Ascend 950 IFA 头开销测试结果

## 1. 测试目标

将 Ascend 910C 上的 `incre_flash_attention` 头开销测量方法迁移到 Ascend 950，
分别采集 STARS 派发、AICore kick、AIV kernel 入口和首次 MTE2 前四个时间点，
拆分调度、硬件初始化和软件初始化开销。

## 2. 环境

- 容器：`ChenxiDocker2`
- 设备：device 2，`Ascend950PR_9579`
- 可用核数：28 AIC / 56 AIV（架构上限 32/64）
- CANN：9.0.0
- torch：2.7.1
- 编译架构：`dav-3510`
- 采样：Mode 1、Mode 2 各 250 组，剔除前 50 组
- SYS_CNT：1000 MHz

当前 950PR SKU 的可用核数不是架构上限。测试通过
`torch.npu.get_device_properties()` 读取实际核数，并使用 tiling 产生的
`blockDim` 启动 kernel，因此本次满核范围为 28/56。

## 3. 代码迁移

从 `ascend910_93` 复制以下实现：

- `csrc/incre_flash_attention/ascend950`
- `csrc/incre_flash_attention_meta/ascend950`

950 适配内容：

1. CMake 编译参数由 `dav-2201` 改为 `dav-3510`。
2. Host launch 从固定 24 个 AIC 改为使用 tiling 返回的实际 `blockDim`。
3. 移除 3510 编译器不支持的 `__sk__/SK_BIND` 路径；本测试使用普通
   `__global__` kernel 和 NPUGraph，不依赖 SK。
4. 移除当前 CANN 9.0 不提供的 host profiler 元数据接口。
5. metadata 输入在当前调用链固定为 `nullptr`，950 版本不实例化无效的
   A3 metadata GM 到 UB 搬运分支。
6. 测试脚本动态读取 AIC/AIV 数量，并校验不超过 950 的 32/64 上限。

## 4. 测量方法

时间点定义沿用 `HEAD_OVERHEAD_TEST_GUIDE.md`：

- T0：STARS task dispatch
- TKick：FFTS 最早 PMU start
- T1：AIV kernel 入口
- T2：第一条 MTE2 前

测试使用两个独立编译产物：

- Mode 1：`IFA_HEAD_TIMING_MODE=1`，入口记录 T1 后立即返回。
- Mode 2：`IFA_HEAD_TIMING_MODE=2`，执行初始化并在首次 MTE2 前记录 T2。

正式统计：

- STARS 调度：Mode 1 的 `TKick-T0`
- 硬件初始化：Mode 1 的 `T1-TKick`
- 调度与硬件初始化：Mode 1 的 `T1-T0`
- 总头开销：Mode 2 的 `T2-T0`
- 软件初始化：`median(Mode 2: T2-T0) - median(Mode 1: T1-T0)`

Mode 1 和 Mode 2 是两个独立 run，绝对时间戳不能跨 run 相减。Mode 2 内部的
`T2-T1` 仅用于诊断，不作为正式的软件初始化结果。

### 4.1 950 profiler 数据解析

950 与 A3 的原始 profiler 格式存在差异：

- STARS 记录为 32 字节，按 `<IIQIIII` 解析；通过 task ID 与
  `ai_core_op_summary.db` 关联，`sys_cycle` 作为 T0。
- FFTS 记录仍为 128 字节，但 MIX kernel 使用 `func_type=42`。
- 同一 task 会产生 AIC/AIV 两份 PMU start，二者可能相差数个 cycle；
  取最早值作为 TKick。
- 950 SYS_CNT 为 1000 MHz，即 1 cycle = 1 ns。

分析脚本保留了原有 A3 格式解析，并增加上述 A5/950 fallback。

## 5. 执行方法

Mode 1：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /data00/huawei/ChenxiDir/MyTransformerBranch2/ops-transformer/examples/fast_kernel_launch_example

NPU_SOC_VERSION=ascend950 IFA_HEAD_TIMING_MODE=1 \
python3 setup.py cmake_build

NPU_DEVICE=2 HEAD_TIMING_POINT=entry \
HEAD_TIMING_LAUNCHES=250 HEAD_TIMING_WARMUP=50 \
HEAD_TIMING_SYSCNT_MHZ=1000 \
bash tests/incre_flash_attention/run_head_overhead.sh \
tests/incre_flash_attention/head_overhead_results_950/mode1_entry
```

Mode 2：

```bash
NPU_SOC_VERSION=ascend950 IFA_HEAD_TIMING_MODE=2 \
python3 setup.py cmake_build

NPU_DEVICE=2 HEAD_TIMING_POINT=mte2 \
HEAD_TIMING_LAUNCHES=250 HEAD_TIMING_WARMUP=50 \
HEAD_TIMING_SYSCNT_MHZ=1000 \
bash tests/incre_flash_attention/run_head_overhead.sh \
tests/incre_flash_attention/head_overhead_results_950/mode2_mte2
```

## 6. 测试结果

| 区间 | 中位数/cycles | 中位数/us |
|---|---:|---:|
| T0 -> TKick | 152 | 0.152 |
| TKick -> T1 | 2316 | 2.316 |
| T0 -> T1 | 2468 | 2.468 |
| T0 -> T2 | 3040 | 3.040 |
| 软件初始化（跨 run） | 572 | 0.572 |
| T2 -> T1（Mode 2内部诊断） | 1124 | 1.124 |

稳态分布：

| 指标 | 样本数 | 最小/cycles | 平均/cycles | 中位数/cycles | 最大/cycles |
|---|---:|---:|---:|---:|---:|
| Mode 1: TKick-T0 | 200 tasks | 151 | 152.37 | 152 | 153 |
| Mode 1: T1-TKick | 11200 cores | 451 | 2359.17 | 2316 | 6385 |
| Mode 1: T1-T0 | 11200 cores | 604 | 2511.54 | 2468 | 6538 |
| Mode 2: T2-T1（诊断） | 11200 cores | 433 | 1224.41 | 1124 | 4200 |
| Mode 2: T2-T0 | 11200 cores | 1150 | 3149.27 | 3040 | 6484 |

## 7. 完整性校验

- 每个 mode：250 个 launch、14000 条记录
- 每个 launch：56 个 AIV block
- 有效 AIV block：0-55
- 有效 AI Core：0-27
- `T0 <= TKick <= T1` 错误数：0
- `T2 < T1` 错误数：0

本地原始结果目录：

- Mode 1：`head_overhead_results_950/mode1_entry/run_20260902_111550`
- Mode 2：`head_overhead_results_950/mode2_mte2/run_20260902_111713`

原始 profiler 数据未纳入 Git，报告保留稳定统计和复现命令。

## 8. 结论

在当前 28 AIC / 56 AIV 的 Ascend950PR_9579 上，FA 从 STARS 派发到首次
MTE2 前的总头开销中位数为 3040 cycles（3.040 us）。其中正式统计的调度开销为
0.152 us、硬件初始化为 2.316 us、软件初始化为 0.572 us。250 组采样剔除前
50 组后数据完整，未发现时间点逆序。
