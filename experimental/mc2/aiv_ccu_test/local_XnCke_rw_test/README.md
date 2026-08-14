# Local XN/CKE Read/Write 测试

本工程按 `local_ms_rw_test` 的测试逻辑，直接通过 hcomm 暴露的 `HcommCcuResAddrGet` 获取 CCU XN/CKE 资源 VA，然后用 `<<<>>>` 直调 AIV kernel 做 8B 读写验证。

主测试路径使用 `DataCopy` 做 8B 读写；额外提供一个 `SetValue` 模式，用来验证“直接给资源地址对应位置赋常数值”的访问方式。

## 文件目录

```text
├── CMakeLists.txt
├── kernel/
│   └── local_xncke_rw_kernel.h
├── local_xncke_rw.h
├── local_xncke_rw.cpp
├── test_local_xncke_rw.cpp
└── run_local_xncke_rw.sh
```

## 资源类型

当前测试按 runtime 的 CCU 资源类型取值传给 `HcommCcuResAddrGet`：

| 资源 | resType | resId |
|------|---------|-------|
| CKE | 3 | `--cke_id` |
| XN | 4 | `--xn_id` |

这里的 `resId` 是硬件资源 ID。这个测试是最小映射验证用例，默认从 `0` 开始测。

## 编译运行

```shell
source /usr/local/Ascend/cann/set_env.sh
export RANK_TABLE_FILE=/path/to/rank_table.json

# 编译 + 同时测试 XN 和 CKE，DataCopy write_then_read，不预热
bash run_local_xncke_rw.sh -c ON -r ON -m 2 -k 2 -w 0 -x 0 -e 0 -l 100
```

## 参数

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Debug |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-m` | mode：0=DataCopy写, 1=DataCopy读, 2=DataCopy先写后读, 3=SetValue写常数 | 2 |
| `-k` | res_kind：0=XN, 1=CKE, 2=BOTH | 2 |
| `-w` | warmup：0=不预热, 1=正式打点前执行一次同类操作 | 0 |
| `-x` | XN 资源 ID | 0 |
| `-e` | CKE 资源 ID | 0 |
| `-l` | 循环次数 | 100 |
| `-p` | 写入 pattern，支持 `0x` 前缀 | 0x5A5A000000000001 |
| `-a` | rank id | 0 |
| `-n` | ranksize | 1 |

## 常用组合

```shell
# DataCopy 先写后读，只测 XN0，不预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 0 -w 0 -x 0 -l 100

# DataCopy 先写后读，只测 XN0，预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 0 -w 1 -x 0 -l 100

# DataCopy 先写后读，只测 CKE0，不预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 2 -k 1 -w 0 -e 0 -l 100

# DataCopy 纯写 XN0，预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 0 -k 0 -w 1 -x 0 -l 100

# DataCopy 纯读 CKE0，不预热
bash run_local_xncke_rw.sh -c OFF -r ON -m 1 -k 1 -w 0 -e 0 -l 100

# SetValue 直接给 XN0 写常数，不走 DataCopy
bash run_local_xncke_rw.sh -c OFF -r ON -m 3 -k 0 -w 0 -x 0 -p 0x12345678
```

`SetValue` 模式只测直接写常数动作，不做读回，也不打印 readback。

## 返回结果

`timeGm` 写回 6 个 `uint64_t`：

| 下标 | 含义 |
|------|------|
| 0 | startCycle |
| 1 | endCycle |
| 2 | verifyResult，0 表示通过 |
| 3 | XN readback |
| 4 | CKE readback |
| 5 | warmupCycles |

`DataCopy WRITE_THEN_READ` 模式会校验读回值是否等于 `pattern`。写入用的 8B 数据会在预热和正式计时之前放到 UB，本模式的正式打点区间只包含对 XN/CKE 地址的 `DataCopy` 写、barrier、`DataCopy` 读、barrier。读回后的 `GetValue` 在 `endCycle` 之后执行，不计入正式耗时。

`warmup=1` 时，kernel 会在正式 `startCycle` 之前，对所选资源执行一次相同类型的操作，并把这一次的耗时记录到 `warmupCycles`。`warmup=0` 时不执行预热，`warmupCycles` 为 0。
