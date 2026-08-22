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

cd mc2/common/op_kernel/apace/tests/st/quant_matmul_reduce_scatter
bash run.sh --cli 2048 3584 4096 2 512   # 自动编译并运行
```

也可手动编译：

```bash
cd mc2/common/op_kernel/apace/tests/st/quant_matmul_reduce_scatter
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . --target quant_matmul_reduce_scatter --parallel 4
```

## 精度验证

### run.sh 一键模式

```bash
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh

cd mc2/common/op_kernel/apace/tests/st/quant_matmul_reduce_scatter
bash run.sh --cli 2048 3584 4096 2 512          # 单个用例
bash run.sh --skip-build --csv cases_12prof.csv  # 12 case 批量
```

### qrun 提交（NPU 排队）

```bash
qrun -n y00845930-精度验证 -t 600 "cd /home/y00845930/dev/ops-transformer/mc2/common/op_kernel/apace/tests/st/quant_matmul_reduce_scatter && source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh 2>/dev/null && bash run.sh --cli 2048 3584 4096 2 512 2>&1 | grep -E 'ALL PASS|FAIL'"
```

## 性能采集 (msprof)

```bash
cd mc2/common/op_kernel/apace/tests/st/quant_matmul_reduce_scatter
bash run_12cases_msprof.sh
```

性能数据归档到 `kernel/quant_matmul_reduce_scatter/profiling/st_direct/`。

## 自适应 headMSize 策略

| rankNum | headMSize | commTurn | 说明 |
|---------|-----------|----------|------|
| R<=2    | 512       | 2~4      | 大 tile, 少 SyncAll, 通信效率高 |
| R>=4    | 128       | 4~8      | 小 tile, 多 commTurn, 流水掩盖效果好 |

## 验证结果

- **精度**：12/12 ALL PASS（含 N=12288 大 case、R=4 多 rank）
- **性能**：与 cann_skills 仓完全一致（差异 <0.2%）
- **日期**：2026-08-22

## 性能数据 (msprof cross_rank_max, us)

| Case | Task Duration |
|------|---------------|
| M=2048 K=3584 N=4096 R=2 | 272.7 us |
| M=2048 K=3584 N=4096 R=4 | 179.5 us |
| M=4096 K=3584 N=4096 R=2 | 471.2 us |
| M=4096 K=3584 N=4096 R=4 | 307.2 us |
| M=4032 K=8192 N=4096 R=2 | 575.3 us |
| M=4032 K=8192 N=4096 R=4 | 615.9 us |
| M=8192 K=3584 N=4096 R=2 | — |
| M=8192 K=3584 N=4096 R=4 | — |
| M=8192 K=1024 N=8192 R=2 | 1622.4 us |
| M=8192 K=1024 N=8192 R=4 | 918.5 us |
| M=8192 K=3696 N=8192 R=2 | 1772.6 us |
| M=8192 K=3696 N=8192 R=4 | 1319.5 us |
| M=8192 K=1536 N=12288 R=2 | 2433.5 us |
| M=8192 K=1536 N=12288 R=4 | 1339.8 us |
