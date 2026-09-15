# quantMatmulReduceScatter ST 编译运行指南

## 环境信息

| 项目 | 值 |
|------|-----|
| CANN 版本 | 9.2.0 |
| CANN 路径 | /home/l00947526/Ascend806/cann-9.2.0 |
| NPU 架构 | Ascend 950PR (dav-3510) |
| 编译器 | bisheng (CANN 9.2.0) |

## 编译

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0

cd mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter
bash run.sh --cli 2048 3584 4096 2 512   # 自动编译并运行
```

也可手动编译：

```bash
cd mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . --target apace_qmm_rs_urma_st --parallel 4
```

> **注意**: bisheng 编译器会优先使用 `$ASCEND_TOOLKIT_HOME` 注入的路径（如 /usr/local/Ascend/cann-9.1.0），
> 需通过 `-iquote` 显式指定 ops-tensor tensor_api 的 include 路径（CMakeLists 已配置），否则会命中
> CANN 内置旧版 tensor_api 头文件导致 `asc::te` 编译错误。

## 精度验证

### run.sh 用例模式（cases.csv，97 用例）

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0

cd mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter
bash run.sh 1                                      # 运行第 1 个用例
bash run.sh --skip-build 1-5                       # 批量运行第 1~5 个
bash run.sh --skip-build all                       # 运行全部 97 个
bash run.sh --csv cases.csv --skip-build 10        # 指定 csv 文件
bash run.sh --cli 2048 3584 4096 2 512             # 命令行模式（绕过 csv）
```

用例来源：`aclnnMatmulReduceScatterV2.xlsx` level0/level0_4p 页签（world≤4，MXFP8 约束过滤后 97 用例）。

### qrun 提交（NPU 排队）

```bash
qrun -n y00845930-精度验证 -t 1500 "cd /home/y00845930/dev/ops-transformer/mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter && export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0 && bash run.sh --skip-build 10"
```

## 性能采集 (msprof)

```bash
cd mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter
bash run.sh --perf --skip-build 10                 # 单用例性能采集
bash run.sh --perf --skip-build 1-12               # 批量采集
python3 scripts/parse_prof.py --all                # 解析全部性能数据
```

性能数据归档到 `tests/st/matmul_reduce_scatter/prof/`。
对比分析报告见 `tests/st/matmul_reduce_scatter/profiling/`。

## 自适应 headMSize 策略

| rankNum | headMSize | 说明 |
|---------|-----------|------|
| R=2     | 512       | 大 tile, 少 SyncAll, 通信效率高 |
| R=4     | 128       | 小 tile, 多 commTurn, 流水掩盖效果好 |
| R=4, M=16384 | 512  | **约束**: chunkM/headM ≤ 16（headM=128 时 chunkM=4096 → 32 tiles 会挂死） |

> **MXFP8 约束**: `CeilDiv(K, 64)` 必须为偶数（scale 对齐），K=2728 等奇数值被拦截。

## 验证结果

- **精度**：**97/97 ALL PASS**（来源 V2 xlsx level0_4p 17 用例 + V1 xlsx level0/level0_4p 80 用例）
- **性能对比**：vs 小算子级联基线 93/97 更快（中位 **1.23x**）；vs MC2 融合算子 88/97 更快（中位 **1.14x**）
- **隔离测试**：12 代表用例通算融合加速比中位 **1.20x**，流水掩盖率中位 **0.70**
- **日期**：2026-09-10 ~ 2026-09-12
- **详细报告**: [profiling/README.md](../../../tests/st/matmul_reduce_scatter/profiling/README.md)
