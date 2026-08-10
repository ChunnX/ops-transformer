# URMA P2P Benchmark — <<<>>> 直调

URMA 点对点通信基准测试工具，使用 `<<<>>>` kernel 直调方式，**不依赖 mc2 算子框架**，编译运行轻量。

支持三种 SQE 模式运行时切换：

| 模式 | opcode | SQE 大小 | WQEBB 占用 | 特点 |
|------|--------|----------|-----------|------|
| `--sqe_mode 0` (Write) | `0x3` | 64B | 1 个 (64B) | 纯数据搬运，无通知，comm_samples 风格 |
| `--sqe_mode 1` (WriteWithNotify) | `0x5` | 96B | 2 个 (128B) | 数据搬运 + 远端状态通知，mc2 风格 |
| `--sqe_mode 2` (Hcomm) | N/A | N/A | N/A | 高层 API: `WriteNbi` + `Drain`，无需手动组装 SQE |

## 目录结构

```
urma_p2p_benchmark_direct/
├── kernel/
│   ├── aiv_urma_sqe.h                # SQE 原语 - Write 版 (64B, opcode 0x3)
│   ├── aiv_urma_sqe_mc2.h            # SQE 原语 - WriteWithNotify 版 (96B, opcode 0x5)
│   ├── aiv_urma_channel_decode.h      # ChannelEntity 解码 (SQ/CQ/buffer/token 字段提取)
│   ├── urma_p2p_bench_tiling.h        # Tiling 结构体 (含 sqeMode 运行时选择字段)
│   ├── urma_p2p_bench_kernel.h        # Kernel 类: decode → build SQE → issue → doorbell → poll CQ → 打点
│   ├── urma_p2p_bench.cpp             # __global__ __aicore__ 入口 + <<<>>> launch 封装
│   └── urma_p2p_bench.h               # launch 函数声明 (host 端引用)
├── host/
│   └── test_urma_p2p_bench.cpp        # C++ main: ACL init → HCCL init → channel setup → <<<>>> → 读打点
├── CMakeLists.txt
└── run.sh
```

## 工作原理

### 整体数据流

```
Host (test_urma_p2p_bench.cpp)
  │
  ├─ aclInit + aclrtSetDevice + aclrtCreateStream
  │
  ├─ HcclCommInitClusterInfoConfig (rank table)
  │     └→ HcclComm comm  (HCCL 通信域句柄)
  │
  ├─ setup_ctx(comm, rank, world, bufBytes)
  │     ├─ aclrtMalloc workspace           (申请 URMA 目标缓冲区)
  │     ├─ HcclCommMemReg                  (注册 workspace 给 HCCL)
  │     ├─ HcclRankGraphGetLayers/GetLinks (枚举 UBC_CTP/UBC_TP 链路)
  │     ├─ HcclChannelDesc 填充            (每条链路一个 desc, memHandles=workspace)
  │     ├─ HcclChannelAcquire              (获取 ChannelHandle, 跨 rank 交换 buffer 信息)
  │     ├─ HcommChannelGetPtrByHandle      (handle → device VA, 兼容新旧 hcomm ABI)
  │     └→ channelPtrTable / stateGm       (拷到 device 供 kernel 读取)
  │
  ├─ urma_p2p_bench_launch<<<1, nullptr, stream>>>(...)
  │
  └─ aclrtMemcpy timingGm → host → 打印统计
          │
          ▼
Kernel (urma_p2p_bench_kernel.h)
  │
  ├─ channelPtrTable_.GetValue(slot)     读取 ChannelEntity 的 device VA
  ├─ DecodePeerChannel(channelEntityVA)  从 GM 加载 256+128+128+64+64=640B 到 UB, 提取:
  │     ├─ sqVa, sqDbVa, wqeSize, sqDepth  (SQ 环形缓冲区信息)
  │     ├─ cqVa, cqeSize, cqDepth          (CQ 环形缓冲区信息)
  │     ├─ localBufAddr, localTokenId      (本端注册 buffer)
  │     ├─ remoteBufAddr, remoteTokenId    (远端 buffer = URMA 写入目标)
  │     └─ tpID, remoteEID[128], remoteTokenValue  (通信元数据)
  │
  ├─ InvalidateCqRing (首次: CQ 全填 0xff, 防止 poll 误判)
  │
  ├─ 生成 SQE 模板 (取决于 sqeMode):
  │     ├─ sqeMode=0: GenerateCommWriteSQE      → opcode=0x3, 64B
  │     └─ sqeMode=1: GenerateCommWriteWithNotifySQE → opcode=0x5, 96B
  │
  ├─ 填充 channel 驱动字段 (tp_id, eid, token 等, 同一 peer 只填一次)
  │
  ├─ 循环 issue N 个数据 SQE (cqe=0):
  │     ├─ 设置 localAddr/remoteAddr/length/(notifyAddr/notifyData)
  │     ├─ DataCopy SQE → HBM SQ ring
  │     └─ 每 dbInterval 个 SQE 敲一次 DoorBell (RingJFSDoorbell)
  │
  ├─ issue 1 个 completion SQE (cqe=1, 小数据)
  ├─ RingJFSDoorbell (最终 doorbell, 释放 trailing SQE)
  │
  ├─ Poll CQ (spin 直到 cqe=1 的完成事件落地)
  │     └─ SQ 有序 → completion 落地 = 所有数据 SQE 已 drain
  │
  └─ 持久化 sqPiLinear / cqCiLinear → stateGm (供下一轮复用)
```

### 两种 SQE 模式的差异

#### Write 模式 (`aiv_urma_sqe.h`, opcode `0x3`)

```
64 字节 SQE, 占用 1 个 WQEBB (64B ring slot)

  Byte 0-1   : header (reserved)
  Byte 2     : odr=0b001, cqe=0/1
  Byte 3     : owner=1, rmt_jetty_type=0b01, token_en=1
  Byte 5     : opcode=0x3
  Byte 11    : sge_num=1
  Byte 8-11  : tp_id (24bit) | sge_num (8bit)
  Byte 12-15 : rmt_jetty_id
  Byte 16-31 : rmt_eid[128]
  Byte 32-35 : rmt_token_value
  Byte 40-47 : rmt_addr (remote dest)
  Byte 48-51 : length
  Byte 52-55 : token_id (source)
  Byte 56-63 : data_addr (local source)
```

#### WriteWithNotify 模式 (`aiv_urma_sqe_mc2.h`, opcode `0x5`)

```
96 字节 SQE, 占用 2 个 WQEBB (128B ring slot)

  Byte 0-1   : header (reserved)
  Byte 2     : odr=0b001, cqe=0/1
  Byte 3     : owner=1, rmt_jetty_type=0b01, token_en=1
  Byte 5     : opcode=0x5
  Byte 11    : sge_num=1
  Byte 8-11  : tp_id (24bit) | sge_num (8bit)
  Byte 12-15 : rmt_jetty_id
  Byte 16-31 : rmt_eid[128]
  Byte 32-35 : rmt_token_value
  Byte 40-47 : rmt_addr (remote dest)
  Byte 48-51 : notify_token_id          ← Write 模式无此字段
  Byte 52-55 : notify_token_value       ← Write 模式无此字段
  Byte 56-63 : notify_addr              ← Write 模式无此字段
  Byte 64-71 : notify_data              ← Write 模式无此字段
  Byte 80-83 : length                   ← Write 模式在 byte 48
  Byte 84-87 : token_id (rmt_token_id)
  Byte 88-95 : data_addr (local source) ← Write 模式在 byte 56
```

关键区别：WriteWithNotify 在写数据的同时向远端 `notifyAddr` 写入 `notifyData`（默认 `0x3F800000` = 1.0f 标志值），远端可通过轮询该地址感知数据到达。mc2 的 MoE dispatch 通信依赖此机制通知接收方。

#### Hcomm 模式 (`sqeMode=2`)

不手动组装 SQE，而是使用 CANN 提供的 `Hcomm<COMM_PROTOCOL_UBC_CTP>` 高层封装：

```cpp
AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm;
hcomm.Init(hcommBuf, 512);  // 512B UB buffer

// 循环 issue N 个非阻塞写入
for (i = 0..N-1)
    hcomm.WriteNbi(channelHandle, remoteBufAddr + off, localBufAddr + off, sqeBytes);

// 统一等待完成 (Hcomm 内部: doorbell + poll CQ)
hcomm.Drain(channelHandle);
```

`Hcomm::WriteNbi` 内部自动完成 SQE 组装、ring 写入、doorbell 敲击；`Drain` 内部自动完成 CQ 轮询。与 SQE 模式的区别：

| 维度 | SQE 模式 (0/1) | Hcomm 模式 (2) |
|------|----------------|-----------------|
| SQE 组装 | 手动 (GenerateCommWriteSQE + SetCommWriteSQE) | Hcomm 内部自动 |
| Ring 写入 | 手动 (PutSQEToRing / DataCopy) | Hcomm 内部自动 |
| DoorBell | 手动 (RingJFSDoorbell + dbInterval) | Hcomm 内部自动 (commit=true 每次) |
| CQ 轮询 | 手动 (CqeSlotEmpty spin) | Hcomm 内部自动 (Drain) |
| UB 开销 | sqe(64/96B) + cqe(64B) + decodeScratch(640B) | 额外 512B Hcomm init buffer |
| timing[1] (assemble) | N 个 SQE 组装+ring写入+doorbell | N 个 WriteNbi 调用 |
| timing[2] (issue) | trailing SQE + 最终 doorbell | ≈ timing[1] (无 trailing) |
| timing[3] (wait) | CQ poll 等待 | Drain 内部等待 |

> Hcomm 模式下 `dbInterval` 参数无效（Hcomm 每次 WriteNbi 默认 commit=true，即每次都敲 doorbell）。`timing[2]` 和 `timing[3]` 在 Hcomm 模式下几乎相同（Drain 紧跟 WriteNbi），与 SQE 模式的 issue+wait 语义不同。

### Host 侧 HCCL 通道建立

```
HcclCommInitClusterInfoConfig(rankTable, rankId, config, &comm)
    │
    │  comm = HCCL 通信域句柄
    ▼
aclrtMalloc + HcclCommMemReg(comm, workspace)
    │
    │  workspace 被注册为 URMA 可达的 device 内存
    │  返回 memHandle
    ▼
HcclRankGraphGetLayers + HcclRankGraphGetLinks(comm, layer, self, peer)
    │
    │  枚举 self→peer 的所有物理链路
    │  过滤 protocol == UBC_CTP(4) 或 UBC_TP(5)
    ▼
HcclChannelDescInit + 填充 (remoteRank, protocol, endpoints, memHandles=workspace)
    │
    │  每个 peer 一条 desc, 附带 workspace 的 memHandle
    ▼
HcclChannelAcquire(comm, COMM_ENGINE_AIV, descs[], &channels[])
    │
    │  HCCL 跨 rank 交换 buffer 注册信息
    │  每个 peer 的 ChannelEntity.localBufferAddr ← 对端的 workspace 地址
    │  返回 ChannelHandle[]
    ▼
HcommChannelGetPtrByHandle (dlsym, 兼容新旧 ABI)
    │
    │  old ABI: handle → device VA (需要翻译)
    │  new ABI (9.1.0): handle 就是 device VA (直接用)
    ▼
channelPtrsHost[] → aclrtMemcpy → channelPtrTableDev (GM, 供 kernel 读取)
```

## 编译

### 前置条件

- CANN Toolkit (>= 9.0), `ASCEND_HOME_PATH` 已设置
- 支持的芯片: `ascend950pr_9599` (默认), 可通过 `-v` 修改
- 至少 2 张 NPU 卡 (P2P 通信需要)

### 编译命令

```bash
# 方式1: 通过 run.sh
bash run.sh -c ON -v ascend950pr_9599

# 方式2: 手动 cmake
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend950pr_9599 -DCMAKE_BUILD_TYPE=Release \
         -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH
make -j
cmake --install .  # 安装到 out/bin/
```

编译产物：
- `out/bin/test_urma_p2p_bench` — 可执行文件
- `libascendc_kernels_urma_p2p.so` — kernel 动态库

## 运行

### 准备 rank table 文件

2 卡 rank table 示例 (`rank_table_2p.json`)：

```json
{
    "status": "completed",
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.0.0.1",
            "device": [
                {"device_id": "0", "device_ip": "192.168.1.1", "rank_id": "0"},
                {"device_id": "1", "device_ip": "192.168.1.2", "rank_id": "1"}
            ]
        }
    ]
}
```

### 启动测试

需要**两个终端**分别运行 rank 0 和 rank 1（或用脚本并行启动）：

```bash
# 终端1: rank 0
export RANK_TABLE_FILE=/path/to/rank_table_2p.json
bash run.sh -r ON -a 0 -d 2

# 终端2: rank 1
export RANK_TABLE_FILE=/path/to/rank_table_2p.json
bash run.sh -r ON -a 1 -d 2
```

### 命令行参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--rank_id` | `-a` | 0 | 当前 rank ID |
| `--ranksize` | `-d` | 2 | 通信域 rank 总数 |
| `--num_sqe` | `-n` | 16 | 数据 SQE 数量 (不含 trailing completion SQE) |
| `--sqe_bytes` | `-s` | 65536 | 每个 SQE 传输的数据量 (字节) |
| `--buf_bytes` | `-b` | 104857600 (100MB) | 注册 workspace 大小 |
| `--db_interval` | `-k` | 1 | 每 K 个 SQE 敲一次 DoorBell |
| `--loop_count` | `-l` | 100 | 正式测试迭代次数 (不含 warmup) |
| `--warmup` | `-w` | 10 | 预热迭代次数 (不纳入统计) |
| `--sqe_mode` | `-m` | 0 | 0=Write(samples), 1=WriteWithNotify(mc2), 2=Hcomm(WriteNbi+Drain) |
| `--soc-version` | `-v` | ascend950pr_9599 | 芯片版本 |
| `--build-type` | `-b` | Release | Debug / Release |
| `--cmake-rebuild` | `-c` | OFF | ON=重新编译 |
| `--run-test` | `-r` | OFF | ON=运行测试 |

### 运行示例

```bash
# 基本测试: mc2 WriteWithNotify 模式, 16 SQE × 64KB
bash run.sh -r ON -a 0 -d 2 -m 1 -n 16 -s 65536 -l 100 -w 10

# Hcomm 模式: WriteNbi+Drain, 16 SQE × 64KB
bash run.sh -r ON -a 0 -d 2 -m 2 -n 16 -s 65536 -l 100 -w 10

# 大包带宽测试: mc2 模式, 64 SQE × 256KB
bash run.sh -r ON -a 0 -d 2 -m 1 -n 64 -s 262144 -l 200 -w 20

# 小包延迟测试: samples Write 模式, 1 SQE × 256B
bash run.sh -r ON -a 0 -d 2 -m 0 -n 1 -s 256 -l 1000 -w 100

# DoorBell 间隔测试: 每 8 个 SQE 敲一次 (仅 SQE 模式, Hcomm 模式忽略)
bash run.sh -r ON -a 0 -d 2 -m 1 -n 64 -s 65536 -k 8 -l 100 -w 10

# 三种模式对比 (分别运行)
bash run.sh -r ON -a 0 -d 2 -m 0 -n 16 -s 65536 -l 100   # Write
bash run.sh -r ON -a 0 -d 2 -m 1 -n 16 -s 65536 -l 100   # WriteWithNotify
bash run.sh -r ON -a 0 -d 2 -m 2 -n 16 -s 65536 -l 100   # Hcomm
```

### 输出格式

每次迭代输出一行：

```
[iter] assemble=X.XXus issue=X.XXus wait=X.XXus total=X.XXus bw=XX.XXGB/s
```

| 字段 | 含义 | 计算方式 |
|------|------|---------|
| `assemble` | N 个数据 SQE 组装 + 写入 ring + doorbell 时间 | `timing[1] - timing[0]` |
| `issue` | 组装 + trailing completion SQE + 最终 doorbell 时间 | `timing[2] - timing[0]` |
| `wait` | DoorBell 到 CQ 完成事件落地的等待时间 | `timing[3] - timing[2]` |
| `total` | 端到端总时间 | `timing[3] - timing[0]` |
| `bw` | 带宽 = numSqe × sqeBytes / total | `totalData / totalNs` |

时间单位为 us，时钟频率 1000MHz (1 cycle = 1ns)。

## Timing Slots

Kernel 内部通过 `GetSystemCycle()` 在 4 个关键节点打点，写入 GM：

| Slot | 打点位置 | 含义 |
|------|---------|------|
| `[0]` | SQE 模板生成后, 数据 SQE 循环开始前 | 起始点 |
| `[1]` | N 个数据 SQE 全部 issue 完毕 | 组装完成 |
| `[2]` | trailing completion SQE issue + 最终 doorbell 敲完 | 全部 issue 完成 |
| `[3]` | CQ poll 收到 completion | 等待完成 |

## SQE 模式选择指南

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 测纯 URMA 写入性能 | `-m 0` (Write) | 无 notify 开销，测量纯搬运能力 |
| 模拟 mc2 MoE dispatch 通信 | `-m 1` (WriteWithNotify) | 与 mc2 算子的 SQE 类型一致，含 notify 开销 |
| 对比两种 SQE 开销差异 | 分别运行 `-m 0` 和 `-m 1` | 同参数下 assemble 时间差 ≈ notify 字段填充开销 |
| 对比底层 vs 高层 API 开销 | 分别运行 `-m 0/1` 和 `-m 2` | SQE 模式 vs Hcomm WriteNbi+Drain 的封装开销 |
| 小包延迟测试 | `-m 0` | notify 固定开销在小包占比大，会掩盖真实延迟 |
| 大包带宽测试 | 两者均可 | notify 开销在大包中占比可忽略 |

## 与 mc2 算子版的对比

| 维度 | mc2 算子版 (`urma_p2p_benchmark/`) | 直调版 (本工具) |
|------|-------------------------------------|-----------------|
| 调用方式 | GE 图框架调度 aclnn | `<<<>>>` 直调 |
| HCCL 上下文 | `MC2().HcclGroup()` 框架自动创建 | `HcclCommInitClusterInfoConfig` 手动创建 |
| 通信通道 | `GetHcclContext` → `HcclCombinOpParam` | `HcclCommMemReg` + `HcclChannelAcquire` |
| Channel 解码 | `GetURMASqInfoTensor` / `GetPICI` (框架内) | `DecodePeerChannel` (直接读 ChannelEntity) |
| SQE 组装 | `moe_distribute_base.h` 高层封装 | 裸操作: `PutSQEToRing` / `RingJFSDoorbell` / `CqeSlotEmpty` |
| Notify 机制 | `PutCommNotifySQE` (含 PICI 管理) | `SetCommWriteWithNotifySQE` (直接填 notifyAddr/Data) |
| SQE 模式 | 仅 WriteWithNotify | 运行时切换 Write / WriteWithNotify |
| 编译依赖 | mc2 全量代码 | 仅 CANN headers, 无 mc2 依赖 |
| 编译方式 | `bash build.sh --pkg --ops=urma_p2p_benchmark` | `cmake + make` |
| 运行依赖 | 算子包安装 + npu_ops_transformer | 可执行文件 + RANK_TABLE_FILE |

## 限制

- 仅单核执行 (blockDim=1, aivId==0)
- P2P 点对点: 每个 rank 向 `(rankId+1) % ranksize` 发送
- `sqeBytes` 需 ≤ `bufBytes`，且建议 32B 对齐
- WriteWithNotify 模式下 SQE 占 2 个 WQEBB，`numSqe` 需满足 `numSqe * 2 < sqDepth` (通常 sqDepth=256)
- 需要物理 UBC_CTP 或 UBC_TP 链路 (同机内的 NPU 间)
- `HcommChannelGetPtrByHandle` 通过 `dlsym` 动态解析，兼容新旧 hcomm ABI
