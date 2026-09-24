# KvCacheTurboQuant 自测用例与复现步骤

本目录的 `test_aclnn_kv_cache_turbo_quant.cpp` 是 aclnn 单算子自测程序：一次运行即完成
**精度对拍（9 个 case，与任务方 `golden.py` 的同算法在 host 重算）** 与 **性能计时（官方 5 个 shape）**。

## 1. 用例清单

| case | T (num_tokens) | H (num_kv_heads) | mse_bits | 说明 |
|---|---|---|---|---|
| gqa_decode_b1 | 1 | 8 | 3 | 官方 shape，GQA decode |
| gqa_decode_b64 | 64 | 8 | 3 | 官方 shape，GQA decode batch |
| gqa_prefill_t2048 | 2048 | 8 | 3 | 官方 shape，主指标（prefill） |
| gqa_mse2_t2048 | 2048 | 8 | 2 | 官方 shape，2-bit 主编码 |
| gqa_mse4_t2048 | 2048 | 8 | 4 | 官方 shape，4-bit 主编码 |
| perf_v4_small | 128 | 8 | 3 | 过程用（每核 1 tile） |
| edge_zero_vec | 4 | 4 | 3 | 全零向量保护（1/8 行置零） |
| edge_heads32 | 8 | 32 | 3 | 多 head（约束上限） |
| edge_heads4 | 8 | 4 | 3 | 少 head |

输入按确定性随机生成（bf16 往返，`value_range=[-2,2]`，与官方用例同口径）；
`rotation_matrix` / `qjl_matrix` 优先加载任务方 fixtures（见第 2 节），缺失时退回自造正交/高斯矩阵。

## 2. 官方 fixtures（推荐，与验收同源）

把任务包的 `rotation_matrix.bin`（128×128 fp32）与 `qjl_matrix.bin` 放到任意目录，用
`KVTQ_MAT_DIR` 指过去即可；程序会打印 `[MAT] 使用任务方 fixtures`：

```bash
KVTQ_MAT_DIR=/path/to/matrices ./kvtq_test
```

不带该变量时使用程序内自造的 QR 正交矩阵 + `U(-4,4)` 投影矩阵（**注意**：此时 §3.2 内积指标
不具参考性——QJL 解码常数 `√(π/2)/d` 假设 `S` 的元素方差为 1）。

## 2.1 dtype 变体覆盖

任务书参数表中的矩阵/输出 dtype 组合由两个环境变量覆盖（默认 `mat=fp32, out=bf16`，跑全部 case）：

```bash
# rotation/qjl_matrix 以 bf16 上传（golden 同步取 bf16 舍入后的矩阵）
KVTQ_MAT_DTYPE=bf16 KVTQ_MAT_DIR=<fixtures 目录> ./kvtq_test
# quant_norm/quant_gamma 以 fp16 输出（接口属性 out_dtype=1）
KVTQ_OUT_DTYPE=fp16 KVTQ_MAT_DIR=<fixtures 目录> ./kvtq_test
```

运行头部会打印 `[DTYPE] mat=.. out=..` 以确认本次口径。

## 3. 编译与运行

前置：CANN 已安装并 `source set_env.sh`；本算子的自定义包已安装
（`--install-path` 下的 `vendors/custom_transformer` 提供 `libcust_opapi.so` 与 op_api 头）。

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
source <install-path>/vendors/custom_transformer/bin/set_env.bash      # 提供 LD_LIBRARY_PATH

INC1=$ASCEND_HOME_PATH/include
API_INC=<install-path>/vendors/custom_transformer/op_api/include
API_LIB=<install-path>/vendors/custom_transformer/op_api/lib

g++ -O2 -std=c++17 test_aclnn_kv_cache_turbo_quant.cpp -o kvtq_test \
    -I$INC1 -I$API_INC -L$API_LIB -L$ASCEND_HOME_PATH/lib64 \
    -lcust_opapi -lascendcl -lnnopbase -lpthread

KVTQ_MAT_DIR=<fixtures 目录> ./kvtq_test
```

算子包本身的构建（在 ops-transformer 仓库内）：

```bash
cd ops-transformer
source ~/Ascend/ascend-toolkit/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=kv_cache_turbo_quant
# 产物：build_out/cann-ops-transformer-custom_linux-aarch64.run
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=<install-path> --quiet
```

## 4. 输出解读

```
[PASS]/[FAIL] <case> T=.. H=.. bits=.. | idx_diff=.. qjl_diff=.. norm_bit_diff=.. gamma_bit_diff=..
              | norm_ulp(max)=<最大偏差 ULP>/<超 2ULP 行数> gamma_ulp(max)=..
  [PERF] <case> single avg <us> us | HOST gws <us> us exec <us> us   # 单次下发+同步（含 host 下发）
  [PERF] <case> BATCH avg <us> us over 50 iters                      # 50 次连续下发+末尾一次同步（与官方基线同口径）
  [ACC] relMSE=.. (任务书<0.05, Lloyd-Max下限=..) relMSE_qjl=.. | 内积(单位q): ip95raw=.. ipRmsRel=..
==== summary: N passed, M failed ====
```

- **性能以 BATCH avg 为准**（与任务书 §3.5 的 PyTorch eager 基线同口径）：10× 目标 = 基线/10。
- **精度判据**：
  1. `idx` 差异率 ≤ 2e-2、`qjl` ≤ 1e-2（远严于任务书 §3.2 的容差）；
  2. `norm/gamma` 按 **2 ULP**（bf16 或 fp16 输出对应各自 ULP 宽度）+ 1e-6 绝对下限判定
     （实测 bf16 口径 norm 逐位一致、gamma 仅个位数行差 1 ULP）；
  3. 任务书 §3.2：`MSE/||x||² < 0.05`。**bits=2 档该阈值数学上不可达**——Lloyd-Max 标量量化
     理论下限为 0.1175 > 0.05，故该档按"下限的 1.15 倍"判（即要求达到该编码形式的最优）；bits≥3 直接按 0.05 判。
  4. 任务书 §3.2：内积相对误差 95th percentile < 0.1（解码含 QJL 项，`q` 取单位向量）。
- `[YERR]/[YROW]/[SPLIT]/[NZ]/[HST]/[FIND]` 是失败时的诊断输出（y 误差量级、坏行清单、workspace 内容比对）；
  设 `KVTQ_DIAG=1` 可强制打印。

## 5. 可复现性与注意点

- 同一台机器多次运行性能有 ±1~3% 漂移，建议重复 2~3 次取中位；prefill/mse2 的 10× 余量约 4%/2%，
  漂移较大时以多轮中位值判定。
- 若 `LoadError: libcust_opapi.so` → 未 source `vendors/custom_transformer/bin/set_env.bash`。
- 若 `GetWorkspaceSize`/`sync ret=507015` → workspace 分配异常，检查是否装了旧算子包。
- `[MAT] 任务方 fixtures 缺失，用合成矩阵` → 未设置 `KVTQ_MAT_DIR`。
