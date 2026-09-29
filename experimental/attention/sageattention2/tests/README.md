# SageAttention2 测试目录说明

本目录包含两个可独立编译运行的 C++ 测试；Python API 验收套件位于
`../python/test_python_api.py`（覆盖任务书 TC-01 ~ TC-10 与 L-B 精度门限，
复现步骤见仓库根 `README.md` 的「测试说明」）。

## 1. numeric_error_test.cpp —— 内核数值规格一致性测试

无 AscendC 依赖的可移植 C++17 宿主机测试，将内核数值构件的浮点误差界固化为
13 大类 29 项断言：FastReciprocal（最差 binade 包络 2^-14）、per-token INT8
量化往返（≤ scale/2）、E4M3 饱和舍入（≤ 2^-4）、E8M0 分组 scale 覆盖性与紧致性、
DEQF16(2^-11)×FP16 平衡 scale 的 2048 抵消恒等式（≤ 2^-9）、FP16(-ln448) 平移
常量位级一致与 P 满量程/饱和裕量、LSE 契约（raw log2 内核值 + 开源平滑校正与
逐头 GQA 广播，双精度参考 ≤ 1e-5）、smooth-K 整行不变性（FP32 内核算术精度
≤ 1e-4）、Causal Mask NZ 拓扑剪枝 bit-exact 与 ≥95% 向量指令 bypass、
qk_quant_gran 分组共享 scale（BroadcastGroupMaxima 覆盖性/无饱和/往返界）、
pv_accum_dtype=fp32+fp16 的部分和 FP16 舍入往返与累积误差界、smooth_v 居中
补偿的 MXFP8 重建误差包络（居中不展宽）、片上 Streaming Q 量化与 Host 路径
bit-exact 等价。

```bash
g++ -O2 -std=c++17 tests/numeric_error_test.cpp -o build/numeric_error_test
./build/numeric_error_test
```

全部通过时输出 `ALL NUMERIC ERROR TESTS PASSED`，任一断言失败即非零退出。

## 2. benchmark_aclnn.cpp —— aclnn 双路径性能基准

对标 `aclnnFusedInferAttentionScoreV5` 与 `aclnnSageAttention2`，运行任务书
P-01 ~ P-07 基准矩阵（HND，FP16/BF16，固定 `per_thread` + `fp32+fp16`）。
计时口径与验收一致：每个实现先 50 次预热（含首编译），同步后以一对 Device
Event 包住连续 100 次两阶段调用（GetWorkspaceSize + 执行），除以 100 得均值；
先 FIA 后 Sage，用例间 500 ms 热冷却。加速比门限 1.6（`>=` 达标）。

依赖：CANN 9.2.0-beta.1 环境与已安装的本算子自定义 OPP 包
（提供 `aclnnop/aclnn_sage_attention2.h`）。Atlas 950PR 真机上运行：

```bash
g++ -O2 -std=c++17 \
    -I$ASCEND_HOME_PATH/aarch64-linux/include \
    -I$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include \
    tests/benchmark_aclnn.cpp -o build/benchmark_aclnn \
    -L$ASCEND_HOME_PATH/aarch64-linux/lib64 -lascendcl -lopapi
./build/benchmark_aclnn --warmup 50 --repeats 100 --output build/aclnn_bench.json
```

stdout 输出逐用例对比表（含 Gate >=1.6x 判定），`--output` 将逐用例
`fia_time_ms / sage_time_ms / speedup / gate_passed` 写入 JSON。
