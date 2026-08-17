# URMA P2P Benchmark — <<<>>> 直调

URMA 点对点通信基准测试工具，使用 `<<<>>>` kernel 直调方式，**不依赖 mc2 算子框架**，单进程多线程，编译运行轻量。

## 四种通信模式

| 模式 | opcode | SQE 大小 | SQE 接口来源 | 特点 |
|------|--------|----------|-------------|------|
| `--sqe_mode 0` (Write) | `0x3` | 64B | 自写裸操作 | 纯数据搬运，无通知 |
| `--sqe_mode 1` (WriteWithNotify) | `0x5` | 96B | 自写裸操作 | 数据搬运 + 远端通知 |
| `--sqe_mode 2` (hcomm) | N/A | N/A | `Hcomm::WriteNbi` | 高层 API，commit 模板参数控制是否敲 DoorBell |
| `--sqe_mode 3` (mc2_native) | `0x5` | 96B | `moe_distribute_base_native.h` | **与 mc2 算子版逻辑完全一致**，含 PutCommNotifySQE/PollCommCQUpdateSQCI/GetPICI/UpdatePICI |

`-m 2` (hcomm) 和 `-m 3` (mc2_native) 已对齐为统一的 Write/Commit/Drain 三段式 API，适合对比高层 API 与裸操作的性能差异。
`-n` (num_sqe) 和 `-k` (db_interval) 仅 `-m 0/1` 使用。

## mc2_native / hcomm 的两种 benchMode

`-m 2` (hcomm) 和 `-m 3` (mc2_native) 均支持两种 benchMode，API 完全对齐:

| benchMode | 说明 | `-l` 含义 | timing 计算 |
|-----------|------|----------|------------|
| `--bench_mode 0` (batch_write) | `-l` 次 Write → Commit → Drain | Write 次数 | slot[0]=assemble(总量), slot[1]=commit(1次), slot[2]=drain(1次), slot[3]=total; avg 除以 loopCount |
| `--bench_mode 1` (write_drain_loop) | 循环 `-l` 次 { WriteWithCommit → Drain } | Write+Drain 组数 | slot[0-3]=累计, slot[4-7]=最后一轮; avg 除以 loopCount |

### API 对齐

| 操作 | hcomm | mc2_native |
|------|-------|------------|
| 不 commit 写 | `WriteNbi<false>` | `Write()` |
| commit 写 | `WriteNbi<true>` | `WriteWithCommit()` |
| 敲 DoorBell | `Commit()` | `Commit()` |
| poll CQ | `Drain()` | `Drain()` |

## 目录结构

```
urma_p2p_benchmark_direct/
├── kernel/
│   ├── common/                              # 共享文件
│   │   ├── aiv_urma_channel_decode.h        #   ChannelEntity 解码 (仅 -m 0/1/2 用)
│   │   └── urma_p2p_bench_tiling.h          #   Tiling 结构体 + benchMode 定义
│   ├── sqe_write/                           # -m 0: Write (64B, opcode 0x3)
│   │   ├── aiv_urma_sqe.h                   #   SQE 原语
│   │   ├── sqe_write_kernel.h               #   Kernel 类
│   │   ├── sqe_write_launch.cpp             #   __global__ 入口 + <<<>>> launch
│   │   └── sqe_write_launch.h
│   ├── sqe_write_with_notify/               # -m 1: WriteWithNotify (96B, opcode 0x5)
│   │   ├── aiv_urma_sqe_mc2.h               #   SQE 原语
│   │   ├── sqe_write_with_notify_kernel.h   #   Kernel 类
│   │   ├── sqe_write_with_notify_launch.cpp
│   │   └── sqe_write_with_notify_launch.h
│   ├── hcomm/                               # -m 2: Hcomm (WriteNbi + Commit + Drain)
│   │   ├── hcomm_kernel.h
│   │   ├── hcomm_launch.cpp
│   │   └── hcomm_launch.h
│   └── mc2_native/                          # -m 3: mc2_native (moe_distribute_base 完整接口)
│       ├── moe_distribute_base_native.h     #   无依赖版 moe_distribute_base.h (保留全部接口含PICI)
│       ├── mc2_native_kernel.h              #   Kernel 类: InitRankComm→Write/Commit/Drain
│       ├── mc2_native_launch.cpp
│       └── mc2_native_launch.h
├── host/
│   └── test_urma_p2p_bench.cpp              # C++ main: 单进程多线程 HCCL + channel setup + <<<>>>
├── cases/
│   ├── input_cases.csv                      # 批跑用例输入 (32条: mode×benchMode×sqeBytes×loop)
│   └── output_results.csv                   # 批跑结果输出 (自动生成)
├── benchmark_results.md                     # 数据整理表格模板
├── CMakeLists.txt
├── run.sh                                   # 单跑脚本
└── batch_run.sh                             # 批跑脚本
```

## 编译

### 前置条件

- CANN Toolkit (>= 9.0), `ASCEND_HOME_PATH` 已设置
- 支持的芯片: `ascend950pr_9599` (默认), 可通过 `-v` 修改
- 至少 2 张 NPU 卡 (P2P 通信需要)
- 端口冲突时: `export HCCL_IF_BASE_PORT=64000`

### 编译命令

```bash
bash run.sh -c ON -v ascend950pr_9599
```

编译产物：`out/bin/test_urma_p2p_bench`

## 运行

### 基本用法

单进程多线程模式，参考 HCCL `01_send_recv` 样例：`HcclGetRootInfo` + `HcclCommInitRootInfoConfig`，每个设备一个线程。**不需要 rank table 文件**。

```bash
# mc2_native batch_write: 攒一批 Write 后 Commit + Drain
bash run.sh -r ON -d 2 -m 3 -o 0 -s 65536 -l 100

# mc2_native write_drain_loop: 每次 Write+Commit 后立即 Drain, 看平均单次延迟
bash run.sh -r ON -d 2 -m 3 -o 1 -s 4096 -l 100

# hcomm batch_write (对比 mc2_native)
bash run.sh -r ON -d 2 -m 2 -o 0 -s 65536 -l 100

# 带正确性验证: sender 填 pattern, receiver 比对 (-m 2, 3)
bash run.sh -r ON -d 2 -m 3 -o 0 -s 65536 -l 100 -g ON

# 其他模式 (-m 0/1 仍用 -n)
bash run.sh -r ON -d 2 -m 0 -n 16 -s 65536 -l 100   # Write
bash run.sh -r ON -d 2 -m 1 -n 16 -s 65536 -l 100   # WriteWithNotify
```

### 指定通信的卡

通过 `--dev_ids` (`-i`) 指定参与通信的物理设备ID列表：

```bash
# 卡0和卡3通信
bash run.sh -r ON -i 0,3 -m 3 -o 0 -s 65536 -l 16
```

### 批量跑用例

通过 `batch_run.sh` 批量跑 `cases/input_cases.csv` 里的用例，结果输出到 `cases/output_results.csv`:

```bash
# 批跑所有用例
bash batch_run.sh

# 只跑 case_id=3
bash batch_run.sh -caseid 3

# 跑多个 case
bash batch_run.sh -caseid 3,5,7

# 指定 dev_ids
bash batch_run.sh -caseid 3 -- 0,1
```

输入 CSV 格式 (含表头):
```
case_id,sqe_mode,bench_mode,sqe_bytes,loop_count
1,mc2_native,batch_write,32,1
2,mc2_native,batch_write,65535,100
3,hcomm,write_drain_loop,32,500
```

输出 CSV 格式 (含表头 + 字段说明):
```
case_id,sqe_mode,bench_mode,sqe_bytes,loop_count,assemble(us),commit/issue(us),drain(us),total(us),bw(GB/s),verify
(sqe_mode: mc2_native/hcomm, bench_mode: batch_write/write_drain_loop)
1,mc2_native,batch_write,32,1,0.34,0.00,0.10,0.44,72.73,PASS
```

### 命令行参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--dev_ids` | `-i` | `0,1` | 参与通信的物理设备ID列表 (逗号分隔) |
| `--dev_count` | `-d` | 2 | 设备数 (dev_ids 优先) |
| `--peer` | `-p` | 自动 | 目标 rank (rank号, 不是设备号) |
| `--sqe_mode` | `-m` | 1 | 0=Write, 1=WriteWithNotify, 2=hcomm, 3=mc2_native |
| `--bench_mode` | `-o` | 0 | 0=batch_write, 1=write_drain_loop (-m 2, 3) |
| `--num_sqe` | `-n` | 16 | 数据 SQE 数量 (仅 -m 0/1) |
| `--sqe_bytes` | `-s` | 65536 | 每个 SQE 传输的数据量 (字节) |
| `--buf_bytes` | `-b` | 104857600 (100MB) | 注册 workspace 大小 |
| `--db_interval` | `-k` | 1 | 每 K 个 SQE 敲一次 DoorBell (仅 -m 0/1) |
| `--loop_count` | `-l` | 100 | kernel 内循环次数 |
| `--verify` | `-g` | OFF | ON=开启正确性验证 (仅 -m 2, 3) |
| `--soc-version` | `-v` | ascend950pr_9599 | 芯片版本 |
| `--cmake-rebuild` | `-c` | OFF | ON=重新编译 |
| `--run-test` | `-r` | OFF | ON=运行测试 |

### 输出格式

```
[rank 0 (dev 0)] avg  assemble=0.27us commit=0.05us drain=10.05us total=10.37us bw=6.31GB/s (over 100 loops)
[rank 0 (dev 0)] last assemble=0.23us commit=0.05us drain=10.02us total=10.30us bw=6.36GB/s (last loop)
[rank 1 (dev 1)] VERIFY: PASS (6553600 bytes, 100 chunks)

========== Case Summary ==========
  mode:       mc2_native
  bench:      batch_write
  sqe_bytes:  65536
  loop:       100
  --- avg (over 100 loops) ---
  assemble: 0.27 us  (total (N writes))
  commit:   0.05 us  (1 call)
  drain:    10.05 us  (1 call)
  total:    10.37 us
  bw:       6.31 GB/s
  --- last (total, only 1 batch) ---
  ...
  verify:   PASS
==================================
```

benchMode=0 时显示 `assemble/commit/drain`，benchMode=1 时显示 `assemble/issue/drain`（issue=assemble，因 WriteWithCommit 合并了 Write+Commit）。

| 字段 | 含义 |
|------|------|
| `assemble` | WQE 组装 + DataCopy 到 SQ ring 时间 (含 S_MTE3/MTE3_S 同步) |
| `commit` | Commit 时间 (敲一次 DoorBell, 仅 benchMode=0) |
| `issue` | = assemble (WriteWithCommit 合并, 仅 benchMode=1) |
| `drain` | Drain 时间 (poll CQ 等硬件完成 + 更新 PICI) |
| `total` | assemble + commit/issue + drain |
| `bw` | 带宽 (avg=sqeBytes/avgTotalNs) |
| `VERIFY` | 正确性验证结果 (仅 -g ON) |

- `avg`：单次平均时间 (timing 累计值 / loopCount)
- `last`：benchMode=0 为总量 (only 1 batch), benchMode=1 为最后一轮单次值
- 每项标注 `(total (N writes))` / `(1 call)` / `(per write)` 表示是循环总量还是单次

## 正确性验证 (`-g ON`)

开启后验证 URMA P2P 通信是否真正成功——sender 的数据是否被 receiver 正确接收。支持 `-m 2` (hcomm) 和 `-m 3` (mc2_native)。

### 原理

```
sender (rank 0):
  1. kernel launch 前向 workspace 数据区填充按位置编码的 pattern
     每个 uint32 = chunk_index * 0x10000 + word_offset_in_chunk
     (trailing bytes: chunk_index * 0x5A + byte_offset)
  2. barrier → kernel 通过 URMA Write 把 pattern 搬到对端 workspace
  3. aclrtSynchronizeStream 等 kernel 完成 → barrier

receiver (rank 1):
  1. barrier 等 sender 填好 pattern
  2. barrier 等 sender kernel 完成 (URMA 写入已落盘)
  3. D2H 拷贝 workspace 数据区 → 逐字节比对 pattern
```

### Mode 差异

| | mc2_native (-m 3) | hcomm (-m 2) |
|---|---|---|
| 数据区偏移 | workspace + 350KB (P2P_STATE_SIZE) | workspace + 0 (buffer 基址) |
| 验证 chunk 数 | `min(loopCount, numSlots)` | `min(loopCount, numSlots)` |
| 每轮 Write 数 | 1 | 1 (对齐 mc2_native) |

### 验证范围

`numSlots = bufBytes / sqeBytes`。chunk 数取写入数与 numSlots 的较小值（超过会 wrap 覆盖）。

### 输出

```
[rank 1 (dev 1)] VERIFY: PASS (6553600 bytes, 100 chunks)
```

或失败时定位到具体字节:

```
[rank 1 (dev 1)] VERIFY: FAIL at byte 65536 (chunk 1, offset 0): expected 0x01, got 0x00
```

### 注意

- 验证逻辑在 kernel 外执行，**不影响 timing 打点**
- barrier 为自实现线程同步 (atomic 计数器 + generation 计数器, 可复用)
- 仅支持 `-m 2` (hcomm) 和 `-m 3` (mc2_native)
- 验证数据量受 workspace 大小限制 (`-b`)

## 工作原理

### Host 侧流程 (单进程多线程)

```
main()
  ├─ aclInit()
  ├─ HcclGetRootInfo(rootInfo)              # 主线程生成 rootInfo (需先 aclrtSetDevice)
  ├─ std::thread per device:                # 每个设备一个线程
  │   ├─ aclrtSetDevice + aclrtCreateContext
  │   ├─ HcclCommInitRootInfoConfig         # 初始化通信域
  │   ├─ setup_ctx():
  │   │   ├─ [1/6] aclrtMalloc workspace    # 分配 URMA 目标缓冲区 (GM)
  │   │   ├─ [2/6] HcclCommMemReg           # 注册 workspace, 使对端可 URMA 写入
  │   │   ├─ [3/6] build_channel_descs      # 枚举 UBC_CTP/UBC_TP 链路, 构建 ChannelDesc
  │   │   ├─ [4/6] HcclChannelAcquire       # 跨 rank 握手, 获取 ChannelHandle
  │   │   ├─ [5/6] resolve channel ptrs     # handle → device VA (兼容新旧 ABI)
  │   │   └─ [6/6] alloc table+state GM     # channelPtrTable + state 拷到 device
  │   ├─ 如果 -m 3 (mc2_native):
  │   │   └─ build_hccl_context():
  │   │       ├─ 从 device 读 ChannelEntity (aclrtMemcpy D2H)
  │   │       ├─ 解码 SQ/CQ/buffer/token 字段
  │   │       ├─ 填入 HcclAiRMAWQ / HcclAiRMACQ 结构体
  │   │       ├─ 设置 windowsIn[peer] (对端 workspace 地址)
  │   │       ├─ 设置 windowsOut[myRank] (PICI 窗口 GM, 含 1MB 前缀)
  │   │       └─ 拷 HcclCombinOpParam 到 device
  │   ├─ 如果 -g ON (verify):
  │   │   └─ sender 填 pattern 到 workspace 数据区 (H2D)
  │   ├─ barrier (等所有 rank 就绪)
  │   ├─ rank 0: <<<>>> launch → synchronize → 读 timing → 打印
  │   │  rank 1: (等待)
  │   ├─ 如果 -g ON (verify):
  │   │   └─ barrier → receiver D2H 读 workspace → 逐字节比对 pattern
  │   └─ join all threads
```

### mc2_native Kernel 侧流程 (-m 3)

与 mc2 算子版 `P2pUrmaSqeComm` 逻辑一致，API 拆分为 Write/Commit/Drain 三段:

```
InitRankComm(peer):
  1. GetURMASqInfoTensor  — 从 HcclCombinOpParam.sqs[peer] 读 SQ 信息到 UB
  2. GetURMACqInfoTensor  — 从 HcclCombinOpParam.cqs[peer] 读 CQ 信息到 UB
  3. GetIsFirstInComm     — 读 PICI 窗口的 isFirstFlag
  4. GetPICI              — 读 windowsOut[myRank]+1MB+512B*peer 的 sqPi/sqCi/cqCi
  5. InvalidateCqeStatus  — 首次通信时 CQ ring 全填 0xff
  6. PollCommCQUpdateSQCI — 清理残余 CQE + 敲 JFC DoorBell 更新硬件 cqCi
  7. GenerateCommWriteWithNotifySQE — 生成 SQE 模板 (opcode=0x5, 96B)
  8. UpdateCommWriteWithNotifySQE   — 填 channel 元数据 (tp_id/eid/token)

Write(peer, localAddr, remoteAddr, length):
  → SetCommWriteWithNotifySQE (cqe=1: 硬件完成后写 CQE)
  → SyncFunc<S_MTE3> (确保 S 写完, MTE3 才读)
  → PutCommNotifySQE (DataCopy SQE 到 SQ ring, 不敲 DoorBell)
  → SyncFunc<MTE3_S> (确保 MTE3 读完, 下一轮 S 才覆盖)

Commit():
  → SyncFunc<MTE3_S> (确保 SQE 落到 HBM)
  → SendJFSDoorBell (敲 JFS DoorBell, 通知硬件取 WQE)

Drain(peer):
  → PollCommCQUpdateSQCI (轮询 CQ 等完成 + 敲 JFC DoorBell)
  → UpdatePICI (写 sqPi/sqCi/cqCi 回 windowsOut, 供下次 launch 复用)

benchMode=0 (batch_write):
  loopCount × Write → Commit → Drain

benchMode=1 (write_drain_loop):
  loopCount × { WriteWithCommit(=Write+SendJFSDoorBell) → Drain }
```

### 流水线同步

打点区间内所有异步操作 (DataCopy MTE3/MTE2) 均有对应 SyncFunc 同步，测到的是真实执行时间:

| 位置 | 同步 | 作用 |
|------|------|------|
| Write: SetCommWriteWithNotifySQE 后 | S_MTE3 | 确保 S 标量写完, MTE3 才读 sqeTensor_ |
| Write: PutCommNotifySQE 后 | MTE3_S | 确保 MTE3 读完, 下一轮 S 才覆盖 sqeTensor_ |
| Commit: SendJFSDoorBell 前 | MTE3_S | 确保 SQE 落到 HBM, 硬件才取 WQE |
| CheckCqeStatus: DataCopyPad 后 | MTE2_S | 确保 CQE 搬完才能读 status |

### 关键数据结构

**HcclCombinOpParam** (Host 侧 `build_hccl_context` 构造)：

```
struct HcclCombinOpParam {
    rankId, rankDim;                     // rank 信息
    windowsIn[MAX_RANK_NUM];             // 各 peer 的 workspace 地址 (URMA 写入目的)
    windowsOut[MAX_RANK_NUM];            // 本 rank 的 PICI 窗口地址 (跨 launch 持久化 SQ/CQ 索引)
    sqs[MAX_RANK_NUM]: HcclAiRMAWQ {     // 每个 peer 的 SQ 信息
        sqVA, dbAddr, wqeSize, sqDepth,  //   SQ ring 地址 + doorbell + 深度
        tp_id, rmtEid[16],               //   通信路由元数据
        rmtObjId, rmtTokenValue,         //   远端 token (填入 SQE)
        localTokenId                     //   本端 token (填入 SQE)
    };
    cqs[MAX_RANK_NUM]: HcclAiRMACQ {     // 每个 peer 的 CQ 信息
        cqVA, cqeSize, cqDepth, dbAddr   //   CQ ring 地址 + doorbell + 深度
    };
};
```

**PICI 窗口** (每个 peer 512B)：

```
[sqPi]       — SQ 生产者索引 (kernel 写到哪了)
[sqCi]       — SQ 消费者索引 (硬件处理到哪了)
[cqPi]       — CQ 生产者索引 (硬件写到哪了)
[cqCi]       — CQ 消费者索引 (kernel 读到哪了)
[sqPiLinear] — SQ 线性计数 (不取模)
[cqCiLinear] — CQ 线性计数
```

跨 kernel launch 持久化 SQ/CQ 读写位置，kernel 结束时 `UpdatePICI` 写回，下次开始时 `GetPICI` 读出。

### Timing Slots

```
benchMode=0:  [0]=assemble(总量) [1]=commit [2]=drain [3]=total
              [4-7]=同 [0-3] (只有一批)
benchMode=1:  [0]=assemble累计 [1]=issue(=assemble) [2]=drain累计 [3]=total累计
              [4]=最后一轮assemble [5]=最后一轮issue [6]=最后一轮drain [7]=最后一轮total
```

## 与 mc2 算子版的对比

| 维度 | mc2 算子版 | 本直调版 (-m 3) |
|------|-----------|-----------------|
| 调用方式 | GE 图框架调度 aclnn | `<<<>>>` 直调 |
| HCCL 初始化 | `MC2().HcclGroup()` 框架自动 | `HcclGetRootInfo` + `HcclCommInitRootInfoConfig` |
| 通信上下文 | `GetHcclContext` → `HcclCombinOpParam` (框架创建) | `build_hccl_context` Host 侧手动构造 |
| SQE 接口 | `moe_distribute_base.h` | `moe_distribute_base_native.h` (逐字复制) |
| 多轮循环 | Host 侧多次 launch | kernel 内部循环, 一次 launch |
| PICI | windowsOut (框架分配) | 单独分配 GM |
| 编译依赖 | mc2 全量代码 | 仅 CANN headers |
| rank table | 需要 | 不需要 |

## 限制

- 单核执行 (blockDim=1, aivId==0)
- 单向发送: rank 0 发, rank 1 等 (URMA Write 单向)
- `sqeBytes` 需 ≤ `bufBytes`，建议 32B 对齐
- WriteWithNotify 每个 SQE 占 2 个 WQEBB，`loopCount` 需满足 `loopCount * 2 < sqDepth` (通常 256，超过触发自动 flush)
- 需要物理 UBC_CTP 或 UBC_TP 链路 (同机 NPU 间)
- `benchMode` 仅 `-m 2`(hcomm) 和 `-m 3`(mc2_native) 支持
- `--verify` 仅 `-m 2`(hcomm) 和 `-m 3`(mc2_native) 支持, 验证 chunk 数受 `bufBytes / sqeBytes` 限制
- `-n` (num_sqe) 和 `-k` (db_interval) 仅 `-m 0/1` 使用
