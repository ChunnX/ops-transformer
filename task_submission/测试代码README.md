# KvCacheTurboQuant 自测用例与测试代码说明

本目录包含 kv_cache_turbo_quant 算子的自测用例与测试代码，验收人可按以下步骤复现全部精度与性能测试结果。

## 目录结构

```
ASCEND_DEVELOP/
├── tests/
│   ├── gen_test_data.py                 # 测试数据与 golden 生成脚本（golden.py 的 CPU 移植）
│   ├── run_tests.sh                     # 一键自测脚本（环境 + 数据 + 运行 + 日志）
│   └── test_aclnn_kv_cache_turbo_quant  # 编译好的 aclnn 自测二进制（源码见算子目录 examples/）
├── test_data/                           # 5 个任务 case 的输入/期望输出 .bin（由脚本生成）
└── task_submission/                     # 交付件（步骤说明/报告/日志）
```

## 复现步骤

```bash
# 1. 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/data/ma-user/work/ASCEND_DEVELOP/custom_opp/vendors/custom_transformer
export LD_LIBRARY_PATH=/data/ma-user/work/ASCEND_DEVELOP/upstream_ot/build:$LD_LIBRARY_PATH

# 2. 生成数据（已完成可跳过）
python3 tests/gen_test_data.py --out /data/ma-user/work/ASCEND_DEVELOP/test_data

# 3. 一键测试（默认 200 次迭代）
bash tests/run_tests.sh 500
```

## 用例清单

| case | 类型 | 参数 | 判定 |
| --- | --- | --- | --- |
| gqa_decode_b1 | 精度+性能 | 1, 8, 128, 3 | 精度: 失配率 idx≤1e-5/qjl≤1.1e-5/norm≤1e-3/gamma≤1e-3; 性能 ≥10x |
| gqa_decode_b64 | 精度+性能 | 64, 8, 128, 3 | 同上 |
| gqa_prefill_t2048 | 精度+性能 | 2048, 8, 128, 3 | 同上 |
| gqa_mse2_t2048 | 精度+性能 | 2048, 8, 128, 2 | 同上 |
| gqa_mse4_t2048 | 精度+性能 | 2048, 8, 128, 4 | 同上 |

## 测试代码说明

- `gen_test_data.py`：与任务随附 `golden.py:calc_expect_func` 完全一致的 CPU torch 移植（fp32 数学、Lloyd-Max 质心、`_pack_bits` 打包序、clamp_min(1e-30) 零向量语义），生成 5 个 case 的输入与期望输出。
- `examples/test_aclnn_kv_cache_turbo_quant.cpp`（算子目录内）：aclnn 两段式接口调用；逐 case 比对 4 个输出（失配计数与失配率）；预热后按迭代次数计时（500 次均值）并对照任务书 eager 基线输出加速比；`KVTQ_DUMP=1` 时落盘算子输出供离线比对。
- host 侧 UT：算子目录 `tests/ut/op_host/` 下 infershape 6 例 + tiling 5 例。
