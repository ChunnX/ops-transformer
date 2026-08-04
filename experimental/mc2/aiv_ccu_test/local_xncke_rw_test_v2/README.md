# Local XN/CKE Read/Write 测试 (v2)

基于新版 hcomm 的 `HcommCcuVariableAlloc` / `HcommCcuEventAlloc` / `HcommCcuVariableAddrGet` / `HcommCcuEventAddrGet` 四个接口获取 CCU 资源地址，用 `<<<>>>` 直调 AIV kernel 做读写验证。

与 v1 (`local_XnCke_rw_test`) 的区别：v1 使用旧接口 `HcommCcuResAddrGet` + `CcuGetTlvHandle`/`CcuPutTlvHandle`，本版改用 hcomm 原生的 Alloc/AddrGet 流程。

## 资源映射

| hcomm 接口 | 资源类型 | 硬件寄存器 | 位宽 | kernel 访问方式 |
|-----------|---------|-----------|------|-----------------|
| `HcommCcuVariableAlloc` + `HcommCcuVariableAddrGet` | Variable | XN | 64 bit (8B) | `DataCopy` 8B 读写 / `SetValue` |
| `HcommCcuEventAlloc` + `HcommCcuEventAddrGet` | Event | CKE | 16 bit (2B) | `SetValue` / `GetValue` (2B) |

> CKE 为 16 位寄存器，kernel 使用 `SetValue`/`GetValue` 做按元素读写，避免 `DataCopy` 的对齐约束问题。

## API 调用流程

```
HcommCcuInsCreateDefault(nullptr, 0, &ccuInsHandle)
  │
  ├── HcommCcuVariableAlloc(ccuInsHandle, dieId, xnNum, &varHandle)
  │     └── HcommCcuVariableAddrGet(varHandle, xnIndex, &xnAddr)
  │
  ├── HcommCcuEventAlloc(ccuInsHandle, dieId, ckeNum, &eventHandle)
  │     └── HcommCcuEventAddrGet(eventHandle, ckeIndex, &ckeAddr)
  │
  ├── kernel launch (xnAddr, ckeAddr)
  │
  └── HcommCcuInsDestroy(ccuInsHandle)
```

## 文件目录

```text
├── CMakeLists.txt
├── kernel/
│   └── local_xncke_rw_kernel.h
├── local_xncke_rw.h
├── local_xncke_rw.cpp
├── test_local_xncke_rw.cpp
├── run_local_xncke_rw.sh
└── README.md
```

## 编译运行

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json

# 编译 + 同时测试 XN 和 CKE，write_then_read，不预热
bash run_local_xncke_rw.sh -c ON -r ON -m 2 -k 2 -w 0 -x 1 -X 0 -e 1 -E 0 -l 100
```

## 参数

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Debug |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-m` | mode：0=写, 1=读, 2=先写后读, 3=SetValue写常数 | 2 |
| `-k` | res_kind：0=XN, 1=CKE, 2=BOTH | 2 |
| `-w` | warmup：0=不预热, 1=正式打点前执行一次同类操作 | 0 |
| `-i` | die_id：IO Die ID (0 或 1) | 0 |
| `-x` | xn_num：申请 Variable(XN) 资源数量 | 1 |
| `-X` | xn_index：使用第几个 XN 资源 (从 0 开始) | 0 |
| `-e` | cke_num：申请 Event(CKE) 资源数量 | 1 |
| `-E` | cke_index：使用第几个 CKE 资源 (从 0 开始) | 0 |
| `-l` | 循环次数 | 100 |
| `-p` | 写入 pattern，支持 `0x` 前缀；XN 用完整 64bit，CKE 用低 16bit | 0x5A5A000000000001 |
| `-a` | rank id | 0 |
| `-n` | ranksize | 1 |

## 常用组合

```shell
# DataCopy 先写后读，只测 XN，申请 1 个资源，用第 0 个
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 0 -w 0 -x 1 -X 0 -l 100

# DataCopy 先写后读，只测 XN，预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 0 -w 1 -x 1 -X 0 -l 100

# 先写后读，只测 CKE，不预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 1 -w 0 -e 1 -E 0 -l 100

# 纯写 XN，预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 0 -k 0 -w 1 -x 1 -X 0 -l 100

# 纯读 CKE，不预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 1 -k 1 -w 0 -e 1 -E 0 -l 100

# SetValue 直接给 XN 写常数，不走 DataCopy
bash run_local_xncke_rw.sh -c OFF -r ON -m 3 -k 0 -w 0 -x 1 -X 0 -p 0x12345678

# 申请多个资源，测试第 3 个 XN
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 0 -w 0 -x 8 -X 3 -l 100

# 指定 die_id=1
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 2 -w 0 -i 1 -x 1 -X 0 -e 1 -E 0 -l 100
```

## kernel 访问方式

| mode | XN (64bit) | CKE (16bit) |
|------|-----------|-------------|
| WRITE (0) | `DataCopy` 写 8B | `SetValue` 写 2B |
| READ (1) | `DataCopy` 读 8B | `GetValue` 读 2B |
| WRITE_THEN_READ (2) | `DataCopy` 写 8B → `DataCopy` 读 8B | `SetValue` 写 2B → `GetValue` 读 2B |
| SET_VALUE (3) | `SetValue` 写 8B | `SetValue` 写 2B |

> `WRITE_THEN_READ` 模式会校验读回值是否等于 pattern（XN 比对完整 64bit，CKE 比对低 16bit）。

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

`warmup=1` 时，kernel 会在正式 `startCycle` 之前，对所选资源执行一次相同类型的操作，并把这一次的耗时记录到 `warmupCycles`。`warmup=0` 时不执行预热，`warmupCycles` 为 0。
