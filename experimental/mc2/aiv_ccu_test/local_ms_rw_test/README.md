# Local MS Read/Write 测试

本地读写 MemorySlice (MS) 性能测试工程，通过 `<<<>>>` 直调 Kernel，测量 UB↔MS 数据搬运带宽，并校验数据正确性。

## 文件目录

```
├── CMakeLists.txt                    # 构建配置
├── kernel/
│   └── local_ms_rw_kernel.h          # Kernel 核心逻辑（读/写/混合/先写后读 + 交织/非交织 + 计时 + 校验）
├── local_ms_rw.h                     # <<<>>> 调用函数声明
├── local_ms_rw.cpp                   # Kernel 入口 + <<<>>> 封装
├── test_local_ms_rw.cpp              # Host 侧测试代码
├── run_local_ms_rw.sh                # 编译运行脚本
├── design.md                         # 原始设计文档
├── local_ms_rw_impl.md               # 实现方案
└── WriteThenRead方案.md              # WriteThenRead 模式方案
```

## 环境要求

- **芯片**：Ascend 950PR (DAV_3510, UB 248KB)
- **CANN**：需安装并 source 环境变量
- **hcomm**：链接 hcomm 库（获取 MS 地址）
- **RANK_TABLE_FILE**：环境变量，指向 rank table 文件

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json
```

## 编译运行

### 一键编译运行

```shell
# 编译 + 运行（先写后读，4KB，非交织，单卡）
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -m 3 -i 0 -s 4096 -n 1
```

### 仅编译

```shell
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c ON -r OFF
```

### 仅运行（复用编译产物）

```shell
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c OFF -r ON -m 3 -i 0 -s 4096
```

### 选项说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Release |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-m` | mode0：0=写, 1=读, 2=读写混合, 3=先写后读 | 0 |
| `-i` | mode1：0=非交织, 1=交织 | 1 |
| `-s` | 数据量（字节） | 4096 |
| `-n` | 卡数 | 1 |

### 常用组合

```shell
# 先写后读 4KB（验证链路）
bash run_local_ms_rw.sh -c ON -r ON -m 3 -i 0 -s 4096

# 先写后读 64KB
bash run_local_ms_rw.sh -c OFF -r ON -m 3 -i 0 -s 65536

# 先写后读 120KB（接近单次搬运上限）
bash run_local_ms_rw.sh -c OFF -r ON -m 3 -i 0 -s 122880

# 纯写 4KB 交织
bash run_local_ms_rw.sh -c OFF -r ON -m 0 -i 1 -s 4096
```

## 模式说明

### mode0（读写模式）

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | WRITE | 纯写：UB → MS，4KB 分块循环搬运 |
| 1 | READ | 纯读：MS → UB，4KB 分块循环搬运 |
| 2 | READ_WRITE | 读写混合：每块先读后写，4KB 分块 |
| 3 | WRITE_THEN_READ | 先写后读：一次写全部 → Barrier → 一次读全部，打点外校验 |

### mode1（交织模式）

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | SEQUENTIAL | 非交织：连续写满 CCUA0(1.5MB) 再写 CCUA1 |
| 1 | INTERLEAVED | 交织：每 8 个 4KB 块轮转 CCUA0→1→2→3 |

## WriteThenRead 模式详解 (mode0=3)

### 流程

```
打点外：FillData 填充递增数据 (dataLocal[i] = i & 0xFF)
打点内：startCycle → DataCopy 写 → Barrier → DataCopy 读 → Barrier → endCycle
打点外：重读逐元素校验，结果写入 timeGm[2]
```

### 数据校验

用递增模式填充（每个位置值不同），可检测：
- 零值（MS 未写入）
- 脏数据（MS 残留旧值）
- 地址偏移（数据写错位置）
- 部分写入（部分块缺失）

### 数据量上限

WriteThenRead 模式一次搬运全部数据，受 UB 容量约束：

| 约束 | 上限 |
|------|------|
| 两个 buffer (dataBuf + readBuf) | ~120KB |
| UB 总容量 (950) | 248KB |

超过 120KB 会报错退出。其他模式（0/1/2）无此限制，按 4KB 分块循环搬运。

### 输出示例

```
[INFO] ===== 本地读写 MS 测试 =====
[INFO] rank_id=0, ranksize=1, mode0=WRITE_THEN_READ, mode1=SEQUENTIAL, data_size=4096 bytes (4.00 KB)
[INFO] 通信域初始化成功。
[INFO] 获取 TLV Handle...
[INFO] 查询 MS 地址...
[INFO] MSAddr0 (CCUA0): va=0x0000123456789000, size=1572864 bytes
[INFO] MSAddr1 (CCUA1): va=0x0000123456abc000, size=1572864 bytes
[INFO] MSAddr2 (CCUA2): va=0x0000123456def000, size=1572864 bytes
[INFO] MSAddr3 (CCUA3): va=0x0000123457012000, size=1572864 bytes
[INFO] 启动 Kernel (blockDim=1)...
[INFO] Kernel 执行完成。

===== 性能结果 =====
  模式:       WRITE_THEN_READ
  交织:       SEQUENTIAL
  数据量:     4.00 KB (4096 bytes)
  startCycle: 1234567890
  endCycle:   1234567990
  耗时:       100.00 ns (0.10 us)
  带宽:       40.9600 GB/s
  数据校验:   通过
====================
```

## 工作原理

### 整体架构

```
Host 侧 (test_local_ms_rw.cpp)
  │
  ├── aclInit / aclrtSetDevice / aclrtCreateStream
  ├── HcclCommInitClusterInfoConfig    ← 初始化通信域
  ├── CcuGetTlvHandle                  ← 获取 TLV 句柄
  ├── CcuGetMemInfo                    ← 查询 4 个 CCUA 的 MS 地址
  ├── aclrtMalloc(timeGm, 24)          ← 分配计时结果 GM
  ├── local_ms_rw_demo<<<>>>           ← 启动 Kernel
  ├── aclrtSynchronizeStream
  └── 读取 timeGm，打印耗时/带宽/校验结果
  │
Kernel 侧 (local_ms_rw_kernel.h)
  │
  ├── Init: SetGlobalBuffer + InitBuffer + Get<T>
  ├── FillData: 递增模式填充 (打点外)
  ├── GetSystemCycle → startCycle
  ├── WriteThenReadMode:
  │     DataCopy(MS, dataLocal)        ← UB → MS
  │     PipeBarrier<PIPE_ALL>()
  │     DataCopy(readLocal, MS)        ← MS → UB
  │     PipeBarrier<PIPE_ALL>()
  ├── GetSystemCycle → endCycle
  ├── VerifyResult: 重读逐元素校验 (打点外)
  └── timeGm 写入 startCycle, endCycle, verifyResult
```

### 计时

- 接口：`GetSystemCycle()`
- 频率：1000MHz，1 cycle = 1 ns
- 打点范围：仅包含 DataCopy + Barrier，不含 FillData 和校验
