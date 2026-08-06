# Local XN/CKE Read/Write 测试 (v2)

基于新版 hcomm 的 `HcommCcuVariableAlloc` / `HcommCcuEventAlloc` / `HcommCcuVariableGetAddr` / `HcommCcuEventGetAddr` 四个接口获取 CCU 资源地址，用 `<<<>>>` 直调 AIV kernel 做读写验证。

与 v1 (`local_XnCke_rw_test`) 的区别：v1 使用旧接口 `HcommCcuResAddrGet` + `CcuGetTlvHandle`/`CcuPutTlvHandle`，本版改用 hcomm 原生的 Alloc/AddrGet 流程。

## 资源映射

| hcomm 接口 | 资源类型 | 硬件寄存器 | 位宽 | kernel 访问方式 |
|-----------|---------|-----------|------|-----------------|
| `HcommCcuVariableAlloc` + `HcommCcuVariableGetAddr` | Variable | XN | 64 bit (8B) | `WriteGmByPassDCache` / `ReadGmByPassDCache` |
| `HcommCcuEventAlloc` + `HcommCcuEventGetAddr` | Event | CKE | 16 bit (2B) | `WriteGmByPassDCache` / `ReadGmByPassDCache` |

> XN/CKE 均为标量寄存器，kernel 使用 `WriteGmByPassDCache`/`ReadGmByPassDCache` 绕过 DCache 直接读写，避免 Cache 一致性问题。

## API 调用流程

```
HcommCcuInsCreateDefault(nullptr, 0, &ccuInsHandle)
  │
  ├── HcommCcuVariableAlloc(ccuInsHandle, dieId, xnNum, &varHandle)
  │     └── HcommCcuVariableGetAddr(varHandle, xnIndex, &xnAddr)
  │
  ├── HcommCcuEventAlloc(ccuInsHandle, dieId, ckeNum, &eventHandle)
  │     └── HcommCcuEventGetAddr(eventHandle, ckeIndex, &ckeAddr)
  │
  ├── kernel launch (xnAddr, ckeAddr)
  │
  └── HcommCcuInsDestroy(ccuInsHandle)
```

> **注意**：通信域初始化（`HcclCommInitClusterInfoConfig`）时会创建 CCU 实例并占用 dieId=0 的资源。若自行调 `HcommCcuInsCreateDefault` 创建新实例，需使用 `dieId=1`（dieId=0 资源已被占用）。

## 文件目录

```text
├── CMakeLists.txt
├── kernel/
│   ├── local_xn_rw_kernel.h        # XN only kernel
│   ├── local_cke_rw_kernel.h       # CKE only kernel
│   └── local_xncke_rw_kernel.h     # XN + CKE kernel
├── local_xncke_rw.h
├── local_xncke_rw.cpp              # 三个 kernel 入口 + host 侧统一封装
├── test_local_xncke_rw.cpp         # host 侧测试程序
├── run_local_xncke_rw.sh           # 单次编译运行脚本
├── batch_run.sh                    # 批量跑 CSV 用例脚本
├── cases/
│   └── cases_example.csv           # 示例用例文件
└── README.md
```

## 编译运行

### 单次运行

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json

# 编译 + 测试 XN 和 CKE，write_then_read，不预热
bash run_local_xncke_rw.sh -c ON -r ON -m 2 -k 2 -w 0 -i 1 -l 100
```

### 批量运行

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json

# 一条命令搞定：自动编译 + 批跑 CSV 用例 + 回写结果
bash batch_run.sh cases/cases_example.csv

# 或指定输出文件名
bash batch_run.sh cases/cases_example.csv cases/my_result.csv
```

脚本会自动检测：如果二进制不存在或源码有更新，自动重新编译；否则跳过编译直接跑。

## 批量运行说明

### 输入 CSV 格式

首行为表头（字段顺序固定），每行一条用例：

```csv
mode,res_kind,die_id,warmup_count,loop_count,pattern,rank_id,ranksize
2,2,1,10,100,0x5A5A000000000001,0,1
0,0,1,10,1000,0x5A5A000000000001,0,1
```

| 字段 | 含义 | 取值 |
|------|------|------|
| mode | 读写模式 | `write` / `read` / `write_then_read`（或 `wtr`），也支持数字 0/1/2 |
| res_kind | 资源类型 | `xn` / `cke` / `both`，也支持数字 0/1/2 |
| die_id | IO Die ID | 0 或 1（单 rank 默认用 1） |
| warmup_count | 预热次数 | 0=不预热, N=预热 N 轮 |
| loop_count | 正式打点循环次数 | 正整数 |
| pattern | 写入值 | 支持 0x 前缀，XN 用 64bit，CKE 用低 16bit |
| rank_id | rank id | 默认 0 |
| ranksize | rank 数量 | 默认 1 |

### 输出 CSV 格式

在输入列后追加结果列，自动回写：

```csv
mode,res_kind,...,ranksize,warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,verify_result,xn_readback,cke_readback,status
2,2,...,1,1234,5678,6789,11.11,11.11,PASS,0x5a5a...,0x0001,OK
```

| 追加字段 | 含义 |
|----------|------|
| warmup_cycles | 预热耗时（cycle） |
| start_cycle | 打点起始 cycle |
| end_cycle | 打点结束 cycle |
| avg_cycles | 平均每轮 cycle 数 |
| avg_ns | 平均每轮耗时（ns） |
| verify_result | 数据校验结果（PASS/FAIL/N/A） |
| xn_readback | XN 读回值 |
| cke_readback | CKE 读回值 |
| status | 执行状态（OK/ERROR） |

## 单次运行参数

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Debug |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-m` | mode：0=写, 1=读, 2=先写后读 | 2 |
| `-k` | res_kind：0=XN, 1=CKE, 2=BOTH | 2 |
| `-w` | warmup 次数：0=不预热, N=预热 N 轮 | 0 |
| `-i` | die_id：IO Die ID (0 或 1) | 1 |
| `-x` | xn_num：申请 Variable(XN) 资源数量 | 1 |
| `-X` | xn_index：使用第几个 XN 资源 (从 0 开始) | 0 |
| `-e` | cke_num：申请 Event(CKE) 资源数量 | 1 |
| `-E` | cke_index：使用第几个 CKE 资源 (从 0 开始) | 0 |
| `-l` | 循环次数 | 100 |
| `-p` | 写入 pattern，支持 `0x` 前缀；XN 用完整 64bit，CKE 用低 16bit | 0x5A5A000000000001 |
| `-a` | rank id | 0 |
| `-n` | ranksize | 1 |

## kernel 访问方式

| mode | XN (64bit) | CKE (16bit) |
|------|-----------|-------------|
| WRITE (0) | `WriteGmByPassDCache` 写 8B | `WriteGmByPassDCache` 写 2B |
| READ (1) | `ReadGmByPassDCache` 读 8B | `ReadGmByPassDCache` 读 2B |
| WRITE_THEN_READ (2) | 写 8B → 读 8B | 写 2B → 读 2B |

> `WRITE_THEN_READ` 模式会校验读回值是否等于 pattern（XN 比对完整 64bit，CKE 比对低 16bit）。
>
> kernel 拆为三个独立入口：`Aivccu_COMM_XN`（仅 XN）、`Aivccu_COMM_CKE`（仅 CKE）、`Aivccu_COMM_BOTH`（两者一起），host 侧按 `res_kind` 选择调哪个。

## 返回结果

`timeGm` 写回 6 个 `uint64_t`：

| 下标 | 含义 |
|------|------|
| 0 | startCycle |
| 1 | endCycle |
| 2 | verifyResult，0 表示通过 |
| 3 | XN readback (64bit) |
| 4 | CKE readback (低 16bit 有效) |
| 5 | warmupCycles |

`warmup > 0` 时，kernel 会在正式 `startCycle` 之前，对所选资源执行 N 轮相同操作预热，并把预热总耗时记录到 `warmupCycles`。`warmup = 0` 时不执行预热，`warmupCycles` 为 0。
