# URMA P2P Benchmark — <<<>>> 直调

URMA 点对点通信基准测试工具，使用 `<<<>>>` kernel 直调方式，**不依赖 mc2 算子框架**，单进程多线程，编译运行轻量。

## 三种通信模式

| 模式 | opcode | SQE 大小 | WQEBB 占用 | 特点 |
|------|--------|----------|-----------|------|
| `--sqe_mode 0` (Write) | `0x3` | 64B | 1 个 (64B) | 纯数据搬运，无通知，comm_samples 风格 |
| `--sqe_mode 1` (WriteWithNotify) | `0x5` | 96B | 2 个 (128B) | 数据搬运 + 远端状态通知，mc2 风格 |
| `--sqe_mode 2` (Hcomm) | N/A | N/A | N/A | 高层 API: `WriteNbi` + `Drain`，无需手动组装 SQE |

## 目录结构

```
urma_p2p_benchmark_direct/
├── kernel/
│   ├── common/                              # 共享文件
│   │   ├── aiv_urma_channel_decode.h        #   ChannelEntity 解码 (SQ/CQ/buffer/token)
│   │   └── urma_p2p_bench_tiling.h          #   Tiling 结构体
│   ├── sqe_write/                           # 模式0: Write (64B, opcode 0x3)
│   │   ├── aiv_urma_sqe.h                   #   SQE 原语 (GenerateCommWriteSQE/PutSQEToRing/...)
│   │   ├── sqe_write_kernel.h               #   Kernel 类
│   │   ├── sqe_write_launch.cpp             #   __global__ 入口 + <<<>>> launch
│   │   └── sqe_write_launch.h               #   launch 声明
│   ├── sqe_write_with_notify/               # 模式1: WriteWithNotify (96B, opcode 0x5)
│   │   ├── aiv_urma_sqe_mc2.h               #   SQE 原语 (GenerateCommWriteWithNotifySQE/...)
│   │   ├── sqe_write_with_notify_kernel.h   #   Kernel 类
│   │   ├── sqe_write_with_notify_launch.cpp #   __global__ 入口 + <<<>>> launch
│   │   └── sqe_write_with_notify_launch.h   #   launch 声明
│   └── hcomm/                               # 模式2: Hcomm (WriteNbi + Drain)
│       ├── hcomm_kernel.h                   #   Kernel 类
│       ├── hcomm_launch.cpp                 #   __global__ 入口 + <<<>>> launch
│       └── hcomm_launch.h                   #   launch 声明
├── host/
│   └── test_urma_p2p_bench.cpp              # C++ main: 单进程多线程 HCCL + channel setup + <<<>>>
├── CMakeLists.txt
└── run.sh
```

## 编译

### 前置条件

- CANN Toolkit (>= 9.0), `ASCEND_HOME_PATH` 已设置
- 支持的芯片: `ascend950pr_9599` (默认), 可通过 `-v` 修改
- 至少 2 张 NPU 卡 (P2P 通信需要)

### 编译命令

```bash
bash run.sh -c ON -v ascend950pr_9599
```

编译产物：`out/bin/test_urma_p2p_bench`

## 运行

### 基本用法

单进程多线程模式，参考 HCCL `01_send_recv` 样例：`HcclGetRootInfo` + `HcclCommInitRootInfo`，每个设备一个线程，共享同一份 `rootInfo`。**不需要 rank table 文件**。

```bash
# 编译后直接运行 (默认卡0和卡1, WriteWithNotify模式)
bash run.sh -r ON -m 1 -n 16 -s 65536 -l 100 -w 10
```

### 指定通信的卡

通过 `--dev_ids` (`-i`) 指定参与通信的物理设备ID列表：

```bash
# 卡0和卡3通信
bash run.sh -r ON -i 0,3 -m 1 -n 16 -s 65536 -l 100 -w 10

# 卡2和卡5通信
bash run.sh -r ON -i 2,5 -m 2 -n 32 -s 131072 -l 200 -w 20

# 3张卡
bash run.sh -r ON -i 0,2,4 -m 1 -n 16 -s 65536 -l 100 -w 10
```

`dev_ids` 列表第 N 个 = rank N 的物理设备号。不传默认 `0,1,...,devCount-1`。

### 指定目标 rank

通过 `--peer` (`-p`) 指定目标 rank（不填则自动 `(rank+1)%devCount`）：

```bash
# rank 0 发给 rank 1 (3卡场景下指定 peer)
bash run.sh -r ON -i 0,2,4 -p 1 -m 1 -n 16 -s 65536
```

### 命令行参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--dev_ids` | `-i` | `0,1` | 参与通信的物理设备ID列表 (逗号分隔) |
| `--dev_count` | `-d` | 2 | 设备数 (与 dev_ids 一起使用, dev_ids 优先) |
| `--peer` | `-p` | 自动 | 目标 rank (rank号, 不是设备号) |
| `--sqe_mode` | `-m` | 1 | 0=Write(samples), 1=WriteWithNotify(mc2), 2=Hcomm(WriteNbi+Drain) |
| `--num_sqe` | `-n` | 16 | 数据 SQE 数量 (不含 trailing completion SQE) |
| `--sqe_bytes` | `-s` | 65536 | 每个 SQE 传输的数据量 (字节) |
| `--buf_bytes` | `-b` | 104857600 (100MB) | 注册 workspace 大小 |
| `--db_interval` | `-k` | 1 | 每 K 个 SQE 敲一次 DoorBell |
| `--loop_count` | `-l` | 100 | 正式测试迭代次数 |
| `--warmup` | `-w` | 10 | 预热迭代次数 |
| `--soc-version` | `-v` | ascend950pr_9599 | 芯片版本 |
| `--cmake-rebuild` | `-c` | OFF | ON=重新编译 |
| `--run-test` | `-r` | OFF | ON=运行测试 |

### 运行示例

```bash
# 基本测试: mc2 WriteWithNotify, 16 SQE × 64KB
bash run.sh -r ON -i 0,1 -m 1 -n 16 -s 65536 -l 100 -w 10

# Hcomm 模式
bash run.sh -r ON -i 0,1 -m 2 -n 16 -s 65536 -l 100 -w 10

# 大包带宽测试: 64 SQE × 256KB
bash run.sh -r ON -i 0,1 -m 1 -n 64 -s 262144 -l 200 -w 20

# 小包延迟测试: 1 SQE × 256B
bash run.sh -r ON -i 0,1 -m 0 -n 1 -s 256 -l 1000 -w 100

# DoorBell 间隔测试: 每8个SQE敲一次 (仅SQE模式)
bash run.sh -r ON -i 0,1 -m 1 -n 64 -s 65536 -k 8 -l 100 -w 10

# 三种模式对比
bash run.sh -r ON -i 0,1 -m 0 -n 16 -s 65536 -l 100   # Write
bash run.sh -r ON -i 0,1 -m 1 -n 16 -s 65536 -l 100   # WriteWithNotify
bash run.sh -r ON -i 0,1 -m 2 -n 16 -s 65536 -l 100   # Hcomm
```

### 输出格式

每次迭代输出一行：

```
[rank 0 (dev 0)][42] assemble=12.34us issue=15.67us wait=8.90us total=24.57us bw=42.71GB/s
```

| 字段 | 含义 |
|------|------|
| `rank` | 逻辑 rank 号 |
| `dev` | 物理设备号 |
| `assemble` | n 个数据 SQE 组装 + ring 写入 + doorbell 时间 |
| `issue` | assemble + completion SQE + 最终 doorbell |
| `wait` | 从 doorbell 到 CQ 完成事件落地 (Hcomm 模式 ≈0) |
| `total` | 端到端总时间 |
| `bw` | 带宽 = numSqe × sqeBytes / total |

## 工作原理

### Host 侧流程 (单进程多线程)

```
main()
  ├─ aclInit()
  ├─ HcclGetRootInfo(rootInfo)          # 主线程生成 rootInfo
  ├─ std::thread per device:            # 每个设备一个线程
  │   ├─ aclrtSetDevice(physicalDev)    #   设置物理设备
  │   ├─ HcclCommInitRootInfo(devCount, rootInfo, rank, &comm)  # 初始化通信域
  │   ├─ aclrtMalloc workspace          #   申请 URMA 目标缓冲区
  │   ├─ HcclCommMemReg                 #   注册 workspace
  │   ├─ HcclRankGraphGetLayers/GetLinks #  枚举 UBC_CTP/UBC_TP 链路
  │   ├─ HcclChannelAcquire             #   获取 ChannelHandle (跨 rank 握手)
  │   ├─ HcommChannelGetPtrByHandle     #   handle → device VA (dlsym 兼容新旧 ABI)
  │   ├─ barrier(devCount)              #   等所有 rank 就绪
  │   └─ for iter: <<<>>> launch + synchronize + 读打点
  └─ join all threads
```

### Kernel 侧流程

三种模式共享 `aiv_urma_channel_decode.h`（从 ChannelEntity 提取 SQ/CQ/buffer/token），区别在 SQE 组装方式：

**SQE Write / WriteWithNotify:**
```
DecodePeerChannel → GenerateCommWriteSQE → for(n): SetCommWriteSQE + PutSQEToRing + DoorBell
→ trailing cqe=1 SQE → DoorBell → Poll CQ
```

**Hcomm:**
```
DecodePeerChannel → Hcomm.Init → for(n): WriteNbi → Drain
```

### Timing Slots

| Slot | 位置 | SQE 模式含义 | Hcomm 模式含义 |
|------|------|------------|---------------|
| `[0]` | 循环开始前 | 起始 | 起始 |
| `[1]` | n 个数据 SQE 完毕 | 组装+ring写入+doorbell | n × WriteNbi 调用 |
| `[2]` | completion SQE + doorbell | issue 完成 | Drain 完毕 |
| `[3]` | CQ poll 收到 completion | 等待完成 | ≈[2] (Drain 内部已等待) |

## SQE 模式选择指南

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 测纯 URMA 写入性能 | `-m 0` | 无 notify 开销 |
| 模拟 mc2 MoE dispatch | `-m 1` | 与 mc2 算子 SQE 类型一致 |
| 对比底层 vs 高层 API | `-m 0/1` vs `-m 2` | SQE 模式 vs Hcomm 封装开销 |
| 小包延迟 | `-m 0` | notify 开销在小包占比大 |
| 大包带宽 | 三者均可 | notify 开销在大包可忽略 |

## 与 mc2 算子版的对比

| 维度 | mc2 算子版 | 本直调版 |
|------|-----------|---------|
| 调用方式 | GE 图框架调度 aclnn | `<<<>>>` 直调 |
| HCCL 初始化 | `HcclCommInitClusterInfoConfig` + rank table | `HcclGetRootInfo` + `HcclCommInitRootInfo` (单进程多线程) |
| 通道建立 | `MC2().HcclGroup()` 框架自动 | `HcclCommMemReg` + `HcclChannelAcquire` |
| SQE 模式 | 仅 WriteWithNotify | 运行时切换 Write / WriteWithNotify / Hcomm |
| 编译依赖 | mc2 全量代码 | 仅 CANN headers |
| rank table | 需要 | 不需要 |

## 限制

- 单核执行 (blockDim=1, aivId==0)
- `sqeBytes` 需 ≤ `bufBytes`，建议 32B 对齐
- WriteWithNotify 每个 SQE 占 2 个 WQEBB，`numSqe` 需满足 `numSqe * 2 < sqDepth` (通常 256)
- 需要物理 UBC_CTP 或 UBC_TP 链路 (同机 NPU 间)
- Hcomm 模式 `dbInterval` 参数无效 (Hcomm 每次 WriteNbi 默认 commit=true)
