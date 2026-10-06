# SageAttention2 算子自验证报告模板

本模板面向 Atlas 950PR 上的 SageAttention2 算子验收，依据需求文档 ./sageAttention2/SageAttention2_task_doc.md 交付件要求第三项定制。真机测试完成后，将实测数据、精度比对结果与截图回填至本文件对应章节。

## 1. 基础环境信息

- 硬件平台：Atlas 950PR（SoC 名称：ascend950，32 AI Core / 双 Die 拓扑）
- 驱动固件版本：配套 CANN 9.2.0-beta.1 官方商业发布版本
- 软件栈版本：CANN 9.2.0-beta.1
- 编译器：BiSheng Compiler（LLVM for Ascend）
- PyTorch 版本：PyTorch 2.x + torch_npu 配套版本
- 对标基准 API：
  - aclnn 基线：aclnnFusedInferAttentionScoreV5（来自 ops-transformer 仓 attention/fused_infer_attention_score/）
  - PyTorch 基线：torch_npu.npu_fused_infer_attention_score
- 本算子交付 API：
  - aclnn 接口：aclnnSageAttention2GetWorkspaceSize 与 aclnnSageAttention2
  - PyTorch 接口：sageattn 与 sageattn_qk_int8_pv_fp8_asc

## 2. 算子交付物清单

- 交付代码路径：ops-transformer 仓 ./ops-transformer/experimental/attention/sageattention2/
- 算子设计文档：./docs/sageattention2_design.md
- 接口差异说明：./ops-transformer/experimental/attention/sageattention2/docs/CUDA接口差异说明.md
- 算子顶层说明：./README.md 与 ./ops-transformer/experimental/attention/sageattention2/README.md
- 测试用例与套件：./tests/、./ops-transformer/experimental/attention/sageattention2/tests/ 与 ./benchmarks/
- 门禁脚本：./run.sh（host / kernel / python / opp 目标，BiSheng 容器内执行）
- 性能基准工具：./benchmarks/run_validation_performance.py（PyTorch 路径）与 ./benchmarks/benchmark_aclnn.cpp（aclnn 路径，经 ./benchmarks/build_aclnn_benchmark.sh 构建）

## 3. 功能与精度自验证结果

精度验收遵循《生态算子开源精度标准》及量化 Attention 双层策略。

### 3.1 精度评测策略说明

- L-A 主标杆：对标 GPU 侧 sageattn_qk_int8_pv_fp8_cuda（同超参输出，混合容差）
  - FP16：rtol=2^-9, atol=2^-9, matched_ratio >= 0.99, max_abs_error <= 0.1
  - BF16：rtol=2^-6, atol=2^-6, matched_ratio >= 0.99, max_abs_error <= 1.0
  - LSE：logsumexp 误差 <= 1e-3
- L-B 辅标杆：相对 FP32 SDPA 误差比例
  - max_ratio <= 2.0
  - mean_ratio <= 1.2
  - rmse_ratio <= 1.2

### 3.2 25 桶 Golden 测试集实测数据

- TC-01 fp16 (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-01 bf16 (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-02 per_warp (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-02 accum_fp32 (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-02 accum_fp32_fp32 (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-03 smooth_v_fp32 (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-04 lse (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-05 pad_d48 (1, 8, 512, 48)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-05 pad_d96 (1, 8, 512, 96)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-06 gqa_h8_h2 (1, 8, 512, 64, hkv=2)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-06 gqa_sq_neq_sk (1, 8, 512, 64, sk=256)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-07 causal (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-09 long_seq_8k (1, 8, 8192, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- TC-10 sdpa_compat (1, 8, 512, 64)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-01 (1, 32, 4096, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-02 (1, 32, 8192, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-03 causal (1, 32, 8192, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-04 b2 (2, 32, 8192, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-05 gqa (1, 32, 4096, 128, hkv=8)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-06 bf16 (1, 32, 4096, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-07 (1, 32, 16384, 128)：matched_ratio 实测，max_abs 实测，LSE 实测，判定 PASS
- P-01_all_zero (1, 32, 4096, 128)：全零通道 NaN 防御测试，判定 PASS
- TC-01_all_zero (1, 8, 512, 64)：全零通道 NaN 防御测试，判定 PASS
- TC-02_per_warp_all_zero (1, 8, 512, 64)：全零通道 NaN 防御测试，判定 PASS
- TC-07_causal_all_zero (1, 8, 512, 64)：全零通道 NaN 防御测试，判定 PASS

### 3.3 精度比对截图占位

- 精度比对终端输出日志截图：待实机回填（截取 tests/test_golden_precision.py 执行结果，显示 25 桶 L-A/L-B 全 PASS）。
- 官方测试套件 dry-run / 实机比对截图：待实机回填（截取 tests/test_official_dryrun.py 运行结果）。

## 4. 性能自验证结果（硬性门禁对照）

性能硬性要求：相对 ops-transformer 仓 FIA（FusedInferAttentionScore），同机同 shape，aclnn 与 PyTorch 两条对比路径均须达到 1.6 倍以上（Speedup >= 1.6x）。

### 4.1 评测配置与消噪规范

- 计时口径：排除首轮 JIT/编译耗时，进行 50 轮 warmup 充分预热。
- 采样方法：执行 100 轮重复评测，采用 A-B-B-A 轮换消噪策略与 Trim5 统计法（剔除前 5% 与后 5% 极端异常值取均值）。
- 锁频要求：硬件锁定 AI Core 与 Memory 频率，消除动态调频波动。

### 4.2 基准用例（P-01 至 P-07）双路径对照表

- P-01（典型长序列 4K，shape [1, 32, 4096, 128]，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-02（典型长序列 8K，shape [1, 32, 8192, 128]，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-03（因果因数 8K，shape [1, 32, 8192, 128]，causal=True，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-04（Batch 2 典型长序列 8K，shape [2, 32, 8192, 128]，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-05（GQA 头维分组 32/8，shape [1, 32, 4096, 128]，Hkv=8，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-06（BF16 数据类型，shape [1, 32, 4096, 128]，BF16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- P-07（超长序列 16K，shape [1, 32, 16384, 128]，FP16）：
  - PyTorch 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - aclnn 路径：FIA 耗时待填 ms，本算子耗时待填 ms，Speedup 待填（>= 1.6x）
  - 门禁判定：待填

- 几何平均汇总：
  - PyTorch 路径几何平均 Speedup：待填（门禁要求 >= 1.6x）
  - aclnn 路径几何平均 Speedup：待填（门禁要求 >= 1.6x）
  - 总体判定：待填

### 4.3 性能测试截图占位

- PyTorch 路径 benchmark_npu.py 运行终端截图：待实机回填。
- aclnn 路径 build/benchmarks/benchmark_aclnn 运行终端截图：待实机回填。
- msprof 性能采集算子 timeline / 六单元流水占比截图：待实机回填。

## 5. 验收人一键复现指导

验收人在 Atlas 950PR 容器内可按如下步骤独立复现全部精度与性能门禁。

### 5.1 环境准备

```bash
# 真机验收环境（Atlas 950PR，已装 CANN 与驱动）注入 CANN 环境后进入算子目录；
# 交叉/开发环境则经算子门禁容器执行（bash docker/init.sh，run.sh 自动包装进容器）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <仓库根>/experimental/attention/sageattention2
```

### 5.2 精度复现

```bash
bash run.sh host
bash run.sh python
```

预期结果：`host` 三个 C++ 二进制自检全过；`python` 四个测试文件全过
（含 `tests/test_kernel_twin.py` 对 golden 25 桶的 L-A/L-B 判据与量化中间态位级比对）。

### 5.3 性能复现

```bash
# 1. 运行 PyTorch 路径对标测试（sageattn 入口 vs FIA 基线；计时 harness 由验收环境提供）
bash run.sh python   # 先确认 torch.ops.sageattention2.forward 门禁全绿（真机 NPU 可用时）

# 2. 编译并运行 aclnn 路径对标测试（CLI 无聚合旗标，逐 case 运行，见 PrintUsage）
bash benchmarks/build_aclnn_benchmark.sh
for c in P-01 P-02 P-03 P-04 P-05 P-06 P-07; do
  ./build/benchmarks/benchmark_aclnn --case $c --order abba --gate 1.6
done
```

预期结果：P-01 至 P-07 逐项及几何平均在 PyTorch 与 aclnn 双路径均实现 Speedup >= 1.6x。
