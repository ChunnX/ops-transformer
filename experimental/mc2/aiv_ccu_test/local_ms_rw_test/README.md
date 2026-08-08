# Local MS Read/Write 测试

本地读写 MemorySlice (MS) 性能测试工程，通过 `<<<>>>` 直调 Kernel，测量 UB↔MS 数据搬运带宽，并校验数据正确性。

## 文件目录

```
├── CMakeLists.txt                    # 构建配置
├── kernel/
│   └── local_ms_rw_kernel.h          # Kernel 核心逻辑（读/写/混合/先写后读 + 计时 + 校验）
├── local_ms_rw.h                     # <<<>>> 调用函数声明
├── local_ms_rw.cpp                   # Kernel 入口 + <<<>>> 封装
├── test_local_ms_rw.cpp              # Host 侧测试代码
├── run_local_ms_rw.sh                # 单次编译运行脚本
├── batch_run.sh                      # 批量跑 CSV 用例脚本
├── cases/
│   └── cases_example.csv             # 示例用例文件
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
# 编译 + 运行（先写后读，MS0，预热关闭，正式循环 100 次，4KB，单卡）
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -m 3 -w 0 -u 0 -l 100 -s 4096 -n 1
```

### 仅编译

```shell
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c ON -r OFF
```

### 仅运行（复用编译产物）

```shell
bash run_local_ms_rw.sh -v ascend950pr_9599 -b Release -c OFF -r ON -m 3 -w 1 -u 3 -l 100 -s 4096
```

### 选项说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Release |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-m` | mode0：0=写, 1=读, 2=读写混合, 3=先写后读 | 0 |
| `-w` | warmup：0=不预热, 1=正式打点前执行一次相同操作 | 0 |
| `-u` | ms_id：选择测试的 MS，范围 0..3 | 0 |
| `-l` | loop_count：正式打点内循环次数 | 100 |
| `-s` | 数据量（字节） | 4096 |
| `-n` | 卡数 | 1 |
| `-a` | rank ID | 0 |

### 常用组合

```shell
# 先写后读 4KB（验证链路）
bash run_local_ms_rw.sh -c ON -r ON -m 3 -w 1 -u 0 -l 100 -s 4096

# 测 MS3，先写后读 64KB，正式循环 1000 次
bash run_local_ms_rw.sh -c OFF -r ON -m 3 -w 1 -u 3 -l 1000 -s 65536

# 先写后读 120KB（接近单次搬运上限）
bash run_local_ms_rw.sh -c OFF -r ON -m 3 -w 0 -u 0 -l 100 -s 122880

# 纯写 4KB
bash run_local_ms_rw.sh -c OFF -r ON -m 0 -w 0 -u 0 -l 100 -s 4096
```

### 批量运行

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json

# 一条命令搞定：自动编译 + 批跑 CSV 用例 + 回写结果
bash batch_run.sh cases/cases_example.csv

# 指定输出文件名
bash batch_run.sh cases/cases_example.csv cases/my_result.csv

# 只跑 id=3 的用例
bash batch_run.sh cases/cases_example.csv -caseid 3

# 只跑 id=3，指定输出文件
bash batch_run.sh cases/cases_example.csv cases/my_result.csv -caseid 3
```

脚本会自动检测：如果二进制不存在或源码有更新，自动重新编译；否则跳过编译直接跑。

`batch_run.sh` 参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| input.csv | 输入用例 CSV 文件 | 必填 |
| output.csv | 输出结果 CSV 文件 | input 文件名加 `_result` |
| -caseid N | 只跑 id=N 的用例（匹配 CSV 中 id 列） | 不传=跑全部 |

## 批量运行说明

### 输入 CSV 格式

首行为表头（字段顺序固定），每行一条用例：

```csv
id,mode0,warmup,ms_id,loop_count,data_size,rank_id,ranksize
0,write_then_read,1,0,100,8192,0,1
4,write,1,0,100,32,0,1
12,read,0,0,1,8192,0,1
```

| 字段 | 含义 | 取值 |
|------|------|------|
| id | 用例 ID | 从 0 开始，用于 `-caseid` 指定单跑 |
| mode0 | 读写模式 | `write` / `read` / `read_write` / `write_then_read`（也支持数字 0/1/2/3） |
| warmup | 预热 | 0=不预热, 1=预热一次 |
| ms_id | MS 编号 | 0..3 |
| loop_count | 正式打点循环次数 | 正整数 |
| data_size | 数据量（字节） | 32B ~ 1.5MB（1572864），需 32B 对齐；WTR 模式上限 120KB |
| rank_id | rank ID | 默认 0 |
| ranksize | 卡数 | 默认 1 |

### 输出 CSV 格式

在输入列后追加结果列，自动回写：

```csv
id,mode0,warmup,ms_id,loop_count,data_size,rank_id,ranksize,warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,bandwidth_gbs,verify_result,status
3,0,1,0,1000,4096,0,1,1234,5678,6789,11.11,11.11,40.96,N/A,OK
```

| 追加字段 | 含义 |
|----------|------|
| warmup_cycles | 预热耗时（cycle） |
| start_cycle | 打点起始 cycle |
| end_cycle | 打点结束 cycle |
| avg_cycles | 平均每轮 cycle 数 |
| avg_ns | 平均每轮耗时（ns） |
| bandwidth_gbs | 带宽（GB/s） |
| verify_result | 校验结果（PASS/FAIL/N/A，仅 mode0=3 有） |
| status | 执行状态（OK/ERROR） |

## 模式说明

### mode0（读写模式）

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | WRITE | 纯写：UB → MS，4KB 分块循环搬运（支持 32B ~ 1.5MB） |
| 1 | READ | 纯读：MS → UB，4KB 分块循环搬运（支持 32B ~ 1.5MB） |
| 2 | READ_WRITE | 读写混合：每块先读后写，4KB 分块 |
| 3 | WRITE_THEN_READ | 先写后读：一次写全部 → Barrier → 一次读全部，打点外校验（上限 120KB） |

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
[INFO] rank_id=0, ranksize=1, mode0=WRITE_THEN_READ, warmup=10, ms_id=0, loop_count=100, data_size=8192 bytes (8.00 KB)
[INFO] 通信域初始化成功。
[INFO] 获取 CCU TLV Handle...
[INFO] ccuHanle=0x...
[INFO] MSAddr0 (CCUA0): va=0x0000123456789000
[INFO] MSAddr1 (CCUA1): va=0x0000123456abc000
[INFO] MSAddr2 (CCUA2): va=0x0000123456def000
[INFO] MSAddr3 (CCUA3): va=0x0000123457012000
[INFO] 启动 Kernel (blockDim=1)...
[INFO] Kernel 执行完成。

===== MS 测试结果 =====
  模式:         WRITE_THEN_READ
  warmup:       10
  MS ID:        0
  loop_count:   100
  数据量:       8.00 KB (8192 bytes)
  warmupCycle:  1234
  startCycle:   1234567890
  endCycle:     1234567990
  耗时:         100.00 ns (0.10 us)
  平均:         1.0000 ns/loop
  带宽:         81.9200 GB/s
  数据校验:     通过
=======================
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
