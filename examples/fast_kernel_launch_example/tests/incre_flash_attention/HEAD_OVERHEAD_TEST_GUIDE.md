# IncreFlashAttention 头开销测试与验证文档

## 1. 需求流程管控点

### 1.1 需求 RoadMap 记录

（由用户补充）

### 1.2 沟通记录

（由用户补充）

### 1.3 测量目标

本测试用于测量 Ascend A3 上 `incre_flash_attention` 的 kernel 头开销。测试使用两个
独立编译产物，Mode 1测量到T1的累计时间，Mode 2测量到T2的累计时间：

| 时间点 | 来源 | 含义 |
|---|---|---|
| T0 | STARS `func_type=0` 记录的 `sys_cycle` | STARS 开始派发 IFA task |
| TKick | FFTS `pmu_start_cnt` | AICore task 被调度到硬件开始执行 |
| T1 | Mode 1 在 AIV kernel 入口处读取 | 读取后直接写回并 return，不执行 `TPipe/Init/Process` |
| T2 | Mode 2 主循环的 `GetActualSeqLen()` 前读取 | 第一条MTE2数据搬运前的历史探针位置 |

正式结果按以下方式计算：

- STARS 调度开销：Mode 1 同次 launch 的 `TKick - T0`
- 硬件初始化开销：Mode 1 同次 launch 的 `T1 - TKick`
- 调度与硬件初始化累计值：Mode 1 的 `T1 - T0`
- 总头开销：Mode 2 的 `T2 - T0`
- 软件初始化开销：`median(Mode 2: T2-T0) - median(Mode 1: T1-T0)`

Mode 1与Mode 2是两次不同的执行，分别拥有自己的T0和TKick，不能跨run相减绝对时间戳。
软件初始化只对两个run内已经计算出的稳态持续时间中位数做差。Mode 2虽然也记录T1，
但其 `T1-TKick` 和 `T2-T1` 只用于诊断，不作为正式分段结果。

### 1.4 测量边界

- 仅统计 AIV。AIC 在调试路径中直接退出，不参与 T1/T2 统计。
- `IFA_HEAD_TIMING_MODE=1` 为严格入口模式；`IFA_HEAD_TIMING_MODE=2` 为原始 Mode2 探针模式。
- A3 为 24 个 AI Core、48 个 AIV block，每个 `ai_core` 对应两个 AIV block。
- 当前满核用例设置 `batch_size=24`、`kv_head_num=1`，因此
   `usedCoreNum=min(B*N2, 24)=24`，48 个 AIV block 均可到达 T2。
- 调试路径不执行正式计算，不验证 attention 输出精度。
- 为避免 early return 后 pipeline event 残留造成后续 launch 死锁，调试路径跳过两条
   `V_MTE2 SetFlag` 和两条 `MTE3_V SetFlag`。因此 `T2-T1` 不包含这四条事件初始化指令，
   但包含此前的参数解析、tiling 读取、buffer 初始化和 AIV 初始化。
- 当前 `FLASH_DECODE=0` 路径在T2前的反向有效batch检查中调用了
   `GlobalTensor::GetValue()`。该接口使用Scalar流水标量访存，不属于MTE2；因此T2严格位于
   第一条MTE2数据搬运之前，但不位于所有类型GM标量读取之前。

## 2. 测试接口使用说明

### 2.1 文件与入口

| 文件 | 作用 |
|---|---|
| `test.py` | 构造 batch=24 的 IFA 输入，捕获含101个 IFA 节点的 NPUGraph，回读每次、每个 AIV 的 T1/T2 |
| `run_head_overhead.sh` | 创建结果目录，调用 msprof，运行测试，调用分析脚本并修正结果目录读取权限 |
| `analyze_head_overhead.py` | 解析 STARS和FFTS记录，按task关联T0/TKick/T1/T2并生成统计 |
| `incre_flash_attention_arch22.h` | Mode 1 在 SK kernel 入口写回 T1并立即退出 |
| `incre_flash_attention_preload_dd.h` | Mode 2 在历史探针位置写回T1/T2并退出 |

测试主入口为 `run_head_overhead.sh`。脚本接受一个可选的结果根目录参数；未指定时，结果写入
`tests/incre_flash_attention/head_overhead_results/` 下的时间戳目录。

### 2.2 参数说明

| # | 参数 | 数据类型 | Shape | 必选/可选 | 约束 | 说明（重点写上约束） |
|---|---|---|---|:---:|---|---|
| 1 | `NPU_DEVICE` | 环境变量，整数 | 标量 | 可选 | 必须是空闲且可见的逻辑设备 ID | 默认值为0；与其他 msprof 任务冲突时应切换设备 |
| 2 | `HEAD_TIMING_LAUNCHES` | 环境变量，整数 | 标量 | 可选 | 必须大于0 | 完整采集固定为101 |
| 3 | `HEAD_TIMING_OUTPUT` | 环境变量，路径 | - | 可选 | 父目录必须可写 | T1/T2 原始 CSV 输出路径 |
| 4 | `batch_size` | Python 整数 | B | 必选 | 满核测试必须为24 | 小于24时只有部分 AI Core 能到达 T2 |
| 5 | `q_head_num` | Python 整数 | N1 | 必选 | 当前用例固定64 | Query head 数 |
| 6 | `kv_head_num` | Python 整数 | N2 | 必选 | 当前用例固定1 | 与 B 一起决定 BN 并影响 `usedCoreNum` |
| 7 | `kv_seq_length` | Python 整数 | S2 | 必选 | 当前用例固定8192 | KV Cache 序列长度 |
| 8 | `block_size` | Python 整数 | Bs | 必选 | 当前用例固定128 | PagedAttention page 大小 |
| 9 | `syscnt_mhz` | Python 浮点数 | 标量 | 可选 | A3 当前环境固定50.0 MHz | 用于 cycles 到微秒换算 |
| 10 | `warmup` | Python 整数 | 标量 | 可选 | 小于 launch 数 | 默认跳过前2个 launch |
| 11 | `IFA_HEAD_TIMING_MODE` | 编译环境变量，整数 | 标量 | 必选 | 仅允许1或2 | 1为入口立即返回，2为首次MTE2前返回 |
| 12 | `HEAD_TIMING_POINT` | 运行环境变量，字符串 | 标量 | 必选 | 与编译模式一致 | Mode 1使用`entry`，Mode 2使用`mte2` |

> **Shape 缩写说明**：
>
> - `B` = `batch_size`
> - `N1` = Query head 数
> - `N2` = KV head 数
> - `S2` = KV Cache 序列长度
> - `Bs` = PagedAttention block size
> - `D` = head dimension

### 2.3 输出文件

每次执行产生以下关键文件：

| 文件 | 内容 |
|---|---|
| `ifa_head_timing_raw.csv` | 每个 launch、每个 AIV block 的原始 T1/T2 |
| `PROF_*/device_*/data/stars_soc.data.*` | STARS 原始二进制记录 |
| `PROF_*/device_*/data/ffts_profile.data.*` | A3 FFTS PMU记录，提供TKick |
| `PROF_*/device_*/sqlite/ai_core_op_summary.db` | task、stream 与 op name 的对应关系 |
| `ifa_head_overhead.csv` | 关联后的T0/TKick/T1/T2和各段差值 |
| `ifa_head_summary.csv` | 每个指标的样本数及min/mean/median/max |

`ifa_head_overhead.csv` 的一行对应一个 launch 中的一个 AIV block。主要字段如下：

| 字段 | 含义 |
|---|---|
| `launch_index` | NPUGraph 中 IFA 节点序号，范围0至100 |
| `stream_id`、`task_id` | 从 profiler 数据库取得的 IFA task 标识 |
| `vector_block` | AIV block ID，范围0至47 |
| `ai_core` | 归属 AI Core ID，范围0至23 |
| `stars_dispatch_cycle` | T0 |
| `aicore_kick_cycle` | TKick |
| `kernel_entry_cycle` | T1 |
| `before_first_mte2_cycle` | T2 |
| `stars_to_kick_cycles` | `TKick-T0` |
| `kick_to_kernel_entry_cycles` | `T1-TKick` |
| `software_init_cycles` | Mode 2内部的`T2-T1`诊断值，不作为正式软件初始化结果 |
| `t2_from_stars_cycles` | `T2-T0` |

### 2.4 使用场景

该测试适用于 A3 `incre_flash_attention` 的 kernel launch 头开销拆分、不同 batch 或 tiling
策略的横向比较，以及 STARS 调度时间与设备侧时间戳的同域校验。

### 2.5 Modules 接口

__不涉及__。本测试直接调用 `torch.ops.ascend_ops.npu_fused_infer_attention_score`，
不经过独立的 Python Modules 封装层。

## 3. 算子实现设计

### 3.1 测量与分析流程

```text
┌──────────────────────────────────────────────────────────────────┐
│ Host                                                             │
│ Mode 1构建 → 101节点NPUGraph → msprof → T0 + TKick + T1         │
│ Mode 2构建 → 101节点NPUGraph → msprof → T0 + TKick + T1 + T2    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Device AIV                                                       │
│ Mode 1: T0 → TKick → T1(kernel第一条语句) → 写回 → return        │
│ Mode 2: T0 → TKick → T1 → 参数、tiling、buffer初始化             │
│              → 反向GetActualSeqLen → T2(历史探针) → return       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Offline Analysis                                                 │
│ Mode 1: TKick-T0、T1-TKick、T1-T0                               │
│ Mode 2: T2-T0；Mode 2内部T1相关差值仅作诊断                     │
│ 软件初始化 = median(Mode 2:T2-T0) - median(Mode 1:T1-T0)       │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 T1/T2 设备侧记录

Mode 1 的 T1 在 `incre_flash_attention_sk` 函数第一条语句处读取。随后只读取
`args->attentionOut`、获取 AIV block ID并写回记录，然后直接 return。该路径不展开其他
kernel 参数，不创建 `TPipe`，不构造算子对象，也不执行 `Init/InitQuant/Process`。

Mode 2 同样在入口读取 T1，但不在入口退出。按用户要求，T2恢复到原始版本的位置：
`FLASH_DECODE=0` 的反向有效batch检查执行完之后、主循环的 `GetActualSeqLen()` 之前。
取得T2后才写回T1/T2。反向检查中的 `GetActualSeqLen()` 使用Scalar流水标量访存，
不属于MTE2，因此当前T2仍位于第一条MTE2数据搬运之前。Mode 2记录的T1只用于数据诊断；
正式T1及其累计时间以入口立即返回的Mode 1为准。

每个 AIV 使用8个 `uint64_t` slot：

| Slot | 内容 |
|---:|---|
| 0 | magic：`0x1FA20268` |
| 1 | `vector_block` |
| 2 | core type，AIV固定为1 |
| 3 | T1 |
| 4 | T2；未到达时为0 |
| 5 | commit 标识，最后写入 |
| 6 | `ai_core` |
| 7 | 保留 |

### 3.3 STARS 解析设计

A3 的 `stars_soc.data` 按64字节步长遍历，每条记录的前32字节按小端格式
`<IHHQIIII` 解析。使用字段如下：

| 字段 | 解析规则 | 用途 |
|---|---|---|
| `header` | 32位无符号整数 | 提取事件类型 |
| `func_type` | `header & 0x3F` | 仅选择0，即 STARS dispatch |
| `task_id_raw` | 16位无符号整数 | 保留原始 task ID |
| `task_id` | `task_id_raw & 0x7FFF` | 与 profiler 数据库 task ID 对齐 |
| `stream_id` | 16位无符号整数 | 与 task ID 共同唯一标识 IFA task |
| `sys_cycle` | 64位无符号整数 | 作为 T0 |

分析前先查询 `ai_core_op_summary.db` 的 `ge_summary` 表，筛选 op name 包含
`incre_flash_attention` 的 `(stream_id, task_id)`。随后仅保留匹配这些键且
`func_type=0` 的 STARS 记录。

过滤后的 STARS dispatch 按 `sys_cycle` 升序排列；T1/T2 按 `launch_index` 排列。
两侧 launch 数必须一致，否则分析直接失败，不输出错位结果。

A3的 `ffts_profile.data` 按128字节记录解析，关键字段为：

- `fields[0] & 0x3F`：40为task PMU，41为block PMU。
- `fields[2] & 0x7FFF`：task ID。
- `fields[3]`：stream ID。
- `fields[19]`：`pmu_start_cnt`，作为TKick。

同一个IFA task的48条AIV block记录和1条task记录具有相同TKick。分析器按
`(stream_id, task_id)` 与STARS和kernel记录直接关联，并校验每个task只有一个TKick。
这比PR中面向无可靠核ID数据的时间邻近配对更适合当前A3采集格式。

### 3.4 满核 tiling 关系

当前 BSH 用例满足：

- `B=24`
- `N2=1`
- `gOuter=1`
- `bn=B*N2*gOuter=24`
- `coreNum=24`
- `usedCoreNum=min(bn, coreNum)=24`

Host launch 配置固定为24个 AIC、48个 AIV。Kernel 中
`aiCoreIdx=vector_block/2`，因此48个 AIV block 映射到24个 AI Core。

## 4. 算子测试结果

### 4.1 代码提交路径

（由用户补充）

### 4.2 测试环境

| 项目 | 版本或配置 |
|---|---|
| 容器 | `ChenxiDocker` |
| NPU | `Ascend910_9392`，A3 |
| Driver / npu-smi | `26.0.rc2` |
| CANN | `9.1.0` |
| Python | `3.11.9` |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post2.dev20260210` |
| CPU 架构 | `aarch64` |
| 实测逻辑设备 | device 4 |

### 4.3 首次准备步骤

1. 确认 `ChenxiDocker` 正在运行，并确认目标 NPU 设备没有其他 profiling 任务。
2. 在容器中加载 CANN 9.1.0 环境，使 `msprof`、Ascend C runtime 和 torch_npu 可见。
3. 检查 `NPU_SOC_VERSION` 为 `ascend910_93`。
4. Mode 1 设置 `IFA_HEAD_TIMING_MODE=1` 后重新构建扩展。
5. Mode 2 设置 `IFA_HEAD_TIMING_MODE=2` 后重新构建扩展。
6. 若只修改 `test.py` 的 batch、shape 或采样次数，不需要重新构建扩展。
7. 每个 mode 均用少量 launch 冒烟，确认记录数和设备同步正常。

### 4.4 每次完整采集步骤

1. 确认 `test.py` 中 `batch_size=24`、`kv_head_num=1`、`kv_seq_length=8192`。
2. 选择空闲逻辑设备，通过 `NPU_DEVICE` 传给测试脚本。
3. 构建 Mode 1，确认编译日志包含 `IFA_HEAD_TIMING_MODE=1`。
4. 使用 `HEAD_TIMING_POINT=entry` 执行 `run_head_overhead.sh`，保存入口 profile。
5. 验证每轮48条入口记录且T2记录数为0。
6. 构建 Mode 2，确认编译日志包含 `IFA_HEAD_TIMING_MODE=2`。
7. 使用 `HEAD_TIMING_POINT=mte2` 执行 `run_head_overhead.sh`，保存MTE2 profile。
8. 验证每轮48条入口记录和48条T2记录。
9. 两次采集均由 `test.py` 捕获101个 IFA 节点并整图 replay 一次。
10. `analyze_head_overhead.py` 分别查询 `ge_summary`，筛选各自101个 IFA task。
11. 分析脚本解析 STARS和FFTS数据，按task提取T0和TKick。
12. Mode 1计算 `TKick-T0`、`T1-TKick` 和 `T1-T0`；Mode 2计算 `T2-T0`。
13. 用Mode 2的稳态 `T2-T0` 中位数减去Mode 1的稳态 `T1-T0` 中位数，得到软件初始化。
14. Mode 2内部的 `T1-TKick` 和 `T2-T1` 仅保留为诊断数据，不写入正式分段结论。

### 4.5 验收检查

完整满核采集必须同时满足：

- 每个 mode 的 IFA task 数均为101。
- 每个 mode 的原始和合并 CSV 均为4848行数据，即 `101*48`，不含表头。
- 每个 launch 恰好有48个 `vector_block`。
- Mode 1每个 launch 的有效 T1数为48、T2数为0。
- Mode 2每个 launch 的有效 T1和T2数均为48。
- 每个 launch 的有效 `ai_core` 集合严格为0至23。
- Mode 1满足`T0<=TKick<=T1`，不存在负的调度和硬件初始化差值。
- Mode 2满足`T2>=T0`，不存在负的总头开销。
- STARS、FFTS和timing task数不一致时必须判定失败。

### 4.6 本次满核测试结果

结果目录：

- 入口立即返回：`tkick_head_overhead_results/mode1_entry/run_20260901_123226/`
- 历史Mode2探针：`tkick_head_overhead_results/mode2_legacy/run_20260901_123051/`
- 四点汇总：`tkick_head_overhead_results/ifa_four_point_summary.csv`

数据完整性：

| 检查项 | 结果 |
|---|---:|
| 每个 mode 的 launch 数 | 101 |
| 每个 mode 的 IFA task 数 | 101 |
| 每个 mode 的有效TKick数 | 101 |
| 每个 launch 的 AIV 记录数 | 48 |
| Mode 1每个 launch 的有效 T1数 | 48 |
| Mode 1每个 launch 的有效 T2数 | 0 |
| Mode 2每个 launch 的有效 T2数 | 48 |
| 有效 AI Core | 0至23，共24个 |
| 每个 mode 的原始记录总数 | 4848 |
| `T0/TKick/T1`顺序错误 | 0 |
| 负 `T2-T1` | 0 |

稳态统计跳过 launch 0和1，共统计99个 launch、4752条 AIV 记录。下表只列正式采用的
直接测量结果：

| 区间 | 样本数 | 最小值/cycles | 中位数/cycles | 平均值/cycles | 最大值/cycles | 中位数/us |
|---|---:|---:|---:|---:|---:|---:|
| Mode 1，T0→TKick：STARS调度 | 99 | 0 | 0 | 0.34 | 1 | 0.000 |
| Mode 1，TKick→T1：严格入口硬件初始化 | 4752 | 95 | 128 | 125.66 | 153 | 2.560 |
| Mode 1，T0→T1：调度与硬件初始化累计 | 4752 | 96 | 128 | 126.00 | 154 | 2.560 |
| Mode 2，T0→T2：稳态总头开销 | 4752 | 293 | 441 | 438.96 | 568 | 8.820 |
| Mode 2，T0→T2：含冷启动全部样本 | 4848 | 293 | 441 | 440.62 | 778 | 8.820 |

> A3 当前环境的 SYS_CNT 频率按50 MHz换算，即1 cycle为0.02 us。

正式分段结果：

| 阶段 | 计算方式 | 中位数/cycles | 中位数/us |
|---|---|---:|---:|
| STARS调度 | Mode 1：`TKick-T0` | 0 | 0.000 |
| 硬件初始化 | Mode 1：`T1-TKick` | 128 | 2.560 |
| 软件初始化 | `median(Mode 2:T2-T0) - median(Mode 1:T1-T0)` | 313 | 6.260 |
| 总头开销 | Mode 2：`T2-T0` | 441 | 8.820 |

分段中位数满足 `0 + 128 + 313 = 441 cycles`，即
`0.000 + 2.560 + 6.260 = 8.820 us`。Mode 2全部4848个样本中的
`T2-T0` 最大值为778 cycles，即15.560 us。

Mode 2 CSV中的 `T1-TKick=4.660 us` 和 `T2-T1=4.160 us` 是同run诊断值。
由于Mode 1和Mode 2是不同编译产物，这两个诊断值不纳入正式分段结论。

### 4.7 单算子 UT 测试结果

**不涉及**。当前构建为头开销专用 early-return kernel，不执行完整 attention 计算，不能用于数值精度 UT。

### 4.8 接入 GE+SK 测试结果

已通过 SK kernel + PyTorch Extension + NPUGraph 路径执行。101个连续 IFA task 均完成，
未出现 CV 同步或跨 launch event 死锁。

### 4.9 接入 AclGraph 测试结果

使用 torch_npu NPUGraph 完成等价图模式验证；原始结果见本次结果目录。

### 4.10 MonkeyTest 与 mssanitizer

（由用户补充）

### 4.11 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| capture 后立即读取不到记录 | NPUGraph capture 只建图，kernel 在 replay 时执行 | replay 并同步后读取 |
| 只有18个有效 AI Core | `batch_size=18` 导致 `usedCoreNum=18` | 将 batch 改为24并重新采集 |
| 有48条 T1但只有36条 T2 | 24个核已启动，但仅18个核有工作；每核对应两个 AIV | 检查 `usedCoreNum` 和 batch |
| 第二个 launch 开始卡死 | early return 前创建的 pipeline event 未被正常完成 | 调试路径不得保留这些 `SetFlag` |
| STARS 解析出现一半全零记录 | 错误地按32字节步长解析 A3 数据 | 按64字节记录步长、32字节有效载荷解析 |
| task 对不上 | 将 task ID 和 stream ID 字段顺序解析错误，或未过滤 op name | 使用 `task_id_raw & 0x7FFF`，并与 `ge_summary` 联合筛选 |
| 看起来有两个Mode 2或两个TKick | 同一个Mode 2 run输出了多个区间；每个launch实际只有一个TKick | 正式结果仅使用Mode 2的`T2-T0`，其T1相关区间只作诊断 |
