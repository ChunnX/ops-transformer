# URMA P2P Benchmark

URMA 点对点通信基准测试算子，用于测量昇腾 NPU 上 URMA P2P 通信的性能指标（时延分解与带宽）。

## 双模式设计

算子支持两种通信算法，通过 `comm_alg` 属性切换：

| 模式 | 值 | 通信上下文 | 核心流程 |
|------|------|------|------|
| SQE 级 | `sqe` | `HcclCombinOpParam`（框架自动创建） | PutCommNotifySQE → DoorBell → PollCQ → UpdatePICI |
| Hcomm 级 | `hcomm` | `P2pHcommContext`（Host 端 `Mc2Context` 创建，含 `hcommHandle_`） | WriteNbi → Drain → P2pSendNotify |

- **SQE 模式**：手动组装 SQE、敲 DoorBell、轮询 CQ、管理 PICI，适用于底层 URMA 行为的细粒度测量。
- **Hcomm 模式**：使用 `Hcomm<COMM_PROTOCOL_UBC_CTP>` 封装层，流程更简洁（WriteNbi + Drain），适用于高层通信 API 的性能评估。

注意：SQE 和 Hcomm 使用不同的 opName（上下文标签），避免 SQE 先运行后缓存不含 `hcommHandle_` 的 Mc2Context 被 Hcomm 复用导致死锁。

## 测试模式

通过 `mode` 参数选择：

| mode | 名称 | 说明 |
|------|------|------|
| 0 | Overhead+Latency | 仅发送 1 个 token，测量每个操作阶段的精确时延，数据大小通过 `h`（hidden size）控制 |
| 1 | Bandwidth | 批量发送所有 token，统一 Drain 最大化带宽利用率 |

### SQE 模式阶段分解（mode=0）

Kernel 内部通过 `GetSystemCycle()` 在 9 个 slot 打点，输出为 `performance_info`（int64, shape `[12]`）：

| Slot | 标记点 | 含义 |
|------|------|------|
| [0] | PROCESS_START | Process() 开始 |
| [1] | DUPLICATE_END | Duplicate(dstRankIds) + PipeBarrier 完成 |
| [2] | INIT_SQ_CQ_LOAD_END | InitRankComm 步骤 1-6 完成（SQ/CQ 加载 + PICI + CQE 初始化 + CQ 轮询） |
| [3] | SQE_TEMPLATE_END | InitRankComm 步骤 7-8 完成（SQE 模板生成 + 元数据更新） |
| [4] | PROCESS_TOKEN_END | ProcessToken 完成（数据写入本地窗口） |
| [5] | P2P_SEND_END | P2pSend 完成（SQE 填写 + 写入环形缓冲区） |
| [6] | DOORBELL_END | TriggerDoorBell 完成 |
| [7] | POLL_CQ_END | PollCommCQUpdateSQCI 完成 |
| [8] | UPDATE_PICI_END | UpdatePICI 完成 = TOTAL_END |

### Hcomm 模式阶段分解（mode=0）

| Slot | 标记点 | 含义 |
|------|------|------|
| [0] | HCOMM_PROCESS_START | Process() 开始 |
| [1] | HCOMM_DATA_COPY_END | 数据从 GM 写入本地窗口完成 |
| [2] | HCOMM_WRITE_NBI_END | WriteNbi 发送完成 |
| [3] | HCOMM_DRAIN_END | Drain 等待完成 |
| [4] | HCOMM_NOTIFY_END | 状态通知 WriteNbi 完成 |

时延统一以 us 为单位：`(cycle差) / 1000 = us`（AIV 系统时钟 50MHz）。

## 算子接口

### 输入

| 名称 | 类型 | Shape | 说明 |
|------|------|------|------|
| context | int32 | ND | Mc2MoeContext 通信上下文（Hcomm 模式需要，SQE 模式可为空） |
| x | float16/bfloat16 | `[bs, h]` | 发送的 token 数据 |
| dst_rank_id | int32 | `[bs]` | 每个 token 的目标 rank（可选输入） |

### 输出

| 名称 | 类型 | Shape | 说明 |
|------|------|------|------|
| y | float16/bfloat16 | `[max_bs * world_size, h]` | 接收的 token 数据 |
| receive_cnt | int32 | `[world_size]` | 从各 rank 接收的 token 数 |
| performance_info | int64 | `[12]` | Kernel 内部各阶段 cycle 计数 |

### 属性

| 名称 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|------|------|
| group | String | Required | - | HCCL 通信域名称 |
| world_size | Int | Required | - | 通信域 rank 总数 |
| max_bs | Int | Required | - | 最大 batch size |
| mode | Int | Required | - | 测试模式：0=overhead+latency, 1=bandwidth |
| ccl_buffer_size | Int | Required | - | HCCL 缓冲区大小（MB） |
| comm_alg | String | Optional | `"sqe"` | 通信算法：`sqe` 或 `hcomm` |
| dst_rank_id | Int | Optional | `-1` | P2P 目标 rank ID，-1 表示全部 rank |
| aiv_num | Int | Optional | `-1` | AIV 核数，-1 使用平台默认值 |

## Python 测试脚本

[urma_p2p_benchmark.py](urma_p2p_benchmark.py) 是完整的基准测试入口脚本，通过 `npu_ops_transformer.ops.npu_urma_p2p_benchmark` 调用算子。

### 命令行参数

```
--test_type       latency / bandwidth / all（默认 all）
--data_sizes      数据大小列表(字节)，默认 256B~2MB
--iterations      迭代次数（默认 100）
--warmup          预热次数（默认 10）
--dst_rank        P2P 目标 rank，-1=自动 (rank+1)%world_size
--world_size      通信域 rank 总数（默认 8）
--rank_per_dev    每设备 rank 数（默认 8）
--server_num      服务器数量（默认 1）
--master_ip       HCCL 初始化主节点 IP（默认 127.0.0.1）
--graph_type      图模式：0=单算子, 3=aclgraph（默认 0）
--output_csv      CSV 输出文件路径
--output_dir      报告输出目录
--comm_alg        通信算法：sqe / hcomm（默认 sqe）
--dtype           数据类型：fp16 / bf16（默认 fp16）
```

### 输出

- **终端打印**：每个数据大小的阶段分解表（mean/min/max/std/p50/p90/p99），抖动分析
- **CSV 文件**：包含所有测试配置的统计数据
- **文本报告**：`urma_p2p_benchmark_report.txt`，包含环境信息、阶段分解、带宽趋势

### 运行示例

```bash
# 全量测试（SQE 模式，fp16）
python urma_p2p_benchmark.py --test_type all --comm_alg sqe --dtype fp16

# 仅带宽测试（Hcomm 模式）
python urma_p2p_benchmark.py --test_type bandwidth --comm_alg hcomm

# 仅时延测试，指定数据大小
python urma_p2p_benchmark.py --test_type latency --data_sizes 256 512 1024 4096
```

## 目录结构

```
urma_p2p_benchmark/
├── urma_p2p_benchmark.py        # Python 基准测试脚本
├── op_api/                       # Host 端 aclnn 算子 API
│   ├── aclnn_urma_p2p_benchmark.h   # aclnn 两段式接口声明
│   ├── aclnn_urma_p2p_benchmark.cpp # aclnn 实现（双模式上下文创建）
│   ├── p2p_hcomm_context.h          # P2pHcommContext 结构体 + 5 参数版 GetP2pHcommContextTensor
│   └── p2p_hcomm_context.cpp        # Hcomm 上下文创建实现（动态加载 HCCL 符号）
├── op_graph/                     # GE 图模式相关
│   ├── urma_p2p_benchmark_proto.h   # 算子 proto 定义（REG_OP）
│   ├── urma_p2p_benchmark_gen_task.cpp # gen_task 实现（Mc2MoeGenTaskCallbackV2）
│   ├── fallback_urma_p2p_benchmark.cpp # Fallback 实现（调用 aclnn API）
│   └── CMakeLists.txt
├── op_host/                      # Host 端算子定义 + Tiling
│   ├── urma_p2p_benchmark_def.cpp    # OpDef 定义（输入/输出/属性/MC2 声明）
│   ├── op_tiling/
│   │   └── urma_p2p_benchmark_tiling.cpp # Tiling 实现（参数校验/窗口检查/通信配置）
│   ├── config/ascend950/
│   │   ├── urma_p2p_benchmark_binary.json    # 二进制编译配置
│   │   └── urma_p2p_benchmark_simplified_key.ini # Simplified key 配置
│   └── CMakeLists.txt
├── op_kernel/                    # Kernel 端实现
│   ├── urma_p2p_benchmark.h          # UrmaP2pBenchmark Kernel 类（双模式 Init/Process）
│   ├── urma_p2p_benchmark.cpp        # Kernel 入口函数
│   ├── urma_p2p_benchmark_tiling.h   # Tiling 数据结构定义
│   ├── urma_p2p_benchmark_tiling_key.h # Tiling Key 定义（URMA+A5）
│   ├── p2p_urma_comm.h               # SQE 级 P2P 通信类（P2pUrmaSqeComm）
│   ├── p2p_hcomm_comm.h              # Hcomm 级 P2P 通信类（P2pHcommComm）
│   └── p2p_hcomm_context.h           # Kernel 端 P2pHcommContext 结构体定义
└── CMakeLists.txt               # 顶层构建配置
```

## 通信上下文

### SQE 模式（P2pUrmaSqeComm）

使用 `HcclCombinOpParam` 上下文，通过 `GetHcclContext<HCCL_GROUP_ID_0>()` 获取。SQE 级 API 流程：

1. `InitRankComm`：加载 SQ/CQ 信息、获取 PICI、准备 SQE 模板
2. `P2pSend`：提交 WriteWithNotify SQE（4 参数简化版或 6 参数完整版）
3. `TriggerDoorBell`：敲 JFS DoorBell 通知硬件
4. `DrainWaitCompletion`：轮询 CQ 等待完成 + 更新 PICI

### Hcomm 模式（P2pHcommComm）

使用 `P2pHcommContext` 上下文（含 `hcommHandle_` 字段），通过 Host 端 `P2pHcommContextHelper::GetP2pHcommContextTensor` 以 `COMM_PROTOCOL_UBC_CTP` 协议创建。Hcomm API 流程：

1. `P2pSend`：`Hcomm.WriteNbi` 非阻塞写入
2. `Drain`：`Hcomm.Drain` 等待完成
3. `P2pSendNotify`：`Hcomm.WriteNbi` 推送 32 字节状态通知

地址模型：
- 本地窗口：`workspace + rankId * totalWinSize + 1MB数据区偏移`
- 远端 URMA：`context->epHcclBuffer_[rankId] + 1MB数据区偏移`
- 状态区：`epHcclBuffer_[rankId]`（0 号 slot 基地址）

## HCCL_BUFFSIZE 计算

脚本自动计算所需 HCCL 缓冲区大小：

```
token_actual_len = ((h * 2 + 31) // 32) * 32 + 44
token_need_size = ((token_actual_len + 511) // 512) * 512
actual_size = (bs * token_need_size * world_size) * 2
hccl_buffsize = ceil(actual_size / 1024 / 1024 / 2) + 5  (MB)
```

## 支持平台

- 昇腾 950（Ascend950），A5 架构
- AIV 引擎通信（`CommEngine::COMM_ENGINE_AIV`）
- 仅支持 URMA 通信模式（`TILINGKEY_TPL_URAMA`）

## 数据类型

- fp16（float16）- 默认
- bf16（bfloat16）

## 图模式支持

- `graph_type=0`：单算子模式（推荐，支持 kernel timing）
- `graph_type=3`：aclgraph 入图模式（暂不支持 kernel 打点，使用 msprof profiling）
