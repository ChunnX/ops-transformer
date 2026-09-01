# SageAttention2 阶段性交付自测报告

## 结论

- CANN 9.2.0-beta.1 / Ascend950PR_9579 上，v53 全量 145 个 tiling key 构建、安装成功。
- 480 个功能与精度正向用例全部完成真实 NPU 调用，输出 shape、dtype、NaN/Inf、LSE 返回与 LSE NaN/Inf 检查均通过：480/480。
- 官方功能用例通过：380/380。
- 官方负向参数校验通过：6/6。
- 14 个性能大形状逐进程有限值检查通过：14/14。
- 对 v52 随机 AICore ECC 的 4 个历史触发形状进行定向压力复测：20/20 通过；未再出现 507015 或 L0A ECC。
- 附件精度测试当前结果为 40/100。该结果不能作为任务书规定的最终精度验收结论，因为当前实例没有 CUDA GPU / GPU SageAttention2 golden；详情见“精度验收缺口”。

## 本阶段交付范围

本阶段关闭源码实现、PyTorch/aclnn 接口、参数校验、全 tiling-key 构建、功能矩阵、边界、有限值/LSE
和运行稳定性问题。任务方已确认暂时豁免最终精度损失和性能门禁，两项均保留测试入口并安排到后续阶段。

本报告中的“通过”仅表示上述阶段范围通过，不表示已经满足最终社区任务的 GPU L-A/L-B 精度或相对 FIA
的 1.6 倍性能要求。

## 环境与产物

- 设备：Ascend950PR_9579，NPU 0
- CANN：`/home/developer/Ascend/cann-9.2.0-beta.1`
- v53 OPP：`/workspace/sage_cann92_v53_l0stagefix_opp`
- 归档安装包：`sageattention2_cann92_v53_full145_l0stagefix.run`
- 安装包 SHA-256：`bfb1de1b3ed78f9632dce1282ba2954f5cfb5b5fd6d9eb7bfb507183524e8e88`
- 分支：`codex/sageattention2-dn-score-v1`
- 提交：`3d22a4f71 fix(sageattention2): stabilize FP8 attention results`
- 本地日志：`remote_logs/full_correctness_v53_20260824/`

## 修复内容

### D64 FP8 E4M3FN NaN

D64 HND 标量 FP8 编码路径中，scale 的 FP32 舍入可能使归一化绝对值略大于 448，手工编码后得到 E4M3FN 的 `0x7f`（NaN）。在编码前增加 448 上限钳位。D128 使用原生饱和 cast，不受此问题影响。

### return_lse 输出归一化错误

旧实现把在线 softmax 的行和 `sum` 原地替换为 `log(sum)`，随后又用这个值做输出归一化。修复为保留 `sum`，把 `max + log(sum)` 缓存在 `lastMax` 中，仅用于 LSE 写回。

### QK/PV L0A/L0B 双缓冲竞态

QK 每个 stage 的物理 tile 为 16 KiB，PV 每个 stage 为 32 KiB；旧实现按各自 tile 大小紧凑排布，但共享同一组 stage event。PV stage 0 因而跨越 QK stage 0/1，却只受 stage 0 token 保护，下一轮 QK MTE1 可能覆盖仍被 PV Cube 读取的高半区，随机触发 `A multi-bit ECC error occurs when CUBE reads L0A`。

修复后 QK 的 L0A/L0B stage stride 固定为物理缓冲区的一半，与 PV 的 stage 区域和 event 一一对应。该改动不改变 tile 数学、workspace 或接口。

## 验证结果

| 项目 | 结果 | 说明 |
|---|---:|---|
| v53 全 key 构建/安装 | 通过 | 145 个唯一 tiling key |
| 全 key shape/dtype/finite/LSE | 480/480 | functional 380 + accuracy 100；覆盖全部 145 key |
| 官方 functional JSON | 380/380 | 附件原测试入口 |
| 官方 negative JSON | 6/6 | head_dim、causal、GQA、gran、layout、accum 非法参数 |
| 性能大形状 finite | 14/14 | 每个用例独立 Python 进程 |
| ECC 定向压力 | 20/20 | 4 个历史触发形状，各 5 次独立进程 |
| 附件 accuracy JSON | 40/100 | FP16 0/52，BF16 40/48；不等同于任务书 L-A/L-B 验收 |

## 精度验收缺口

任务书规定：

- L-A 主验收应将 NPU `sageattn_qk_int8_pv_fp8_asc` 与 GPU `sageattn_qk_int8_pv_fp8_cuda` 在同一 `pv_accum_dtype`、`qk_quant_gran` 下比较。
- L-B 辅验收应比较 NPU 与 GPU-Sage 相对 FP32 SDPA 的误差比例，而不是直接要求 FP8 输出逐元素贴近 FP32 SDPA。

当前云实例只有 NPU，没有 CUDA GPU。附件测试在 GPU golden 不可用时，将 L-A 用例回退为 NPU 对 FP32 SDPA 的直接混合容差比较；标记为 L-B 的用例也执行相同的直接比较。因此 40/100 只能记录为“附件 NPU-only 回退测试结果”，不能证明或否定任务书规定的 L-A/L-B 精度验收。

完成最终数值精度验收仍需要一台带 CUDA GPU 的环境生成同配置 GPU SageAttention2 golden，保存输出后再在当前 NPU 实例计算 L-A 与 L-B 指标。

## 性能验收延期项

当前版本已经验证 14 个性能大形状可执行且输出有限，但本阶段不以这些调用的耗时作为通过条件。后续性能
阶段仍需使用 CANN 9.2.0-beta.1，在同机、同 shape、同 dtype 和同 warmup/迭代条件下，分别完成：

- PyTorch `sageattn_qk_int8_pv_fp8_asc` 对 `torch_npu.npu_fused_infer_attention_score`；
- `aclnnSageAttention2` 对 `aclnnFusedInferAttentionScoreV5`；
- P-01～P-07 每条和两路径几何平均的 1.6 倍门禁。

## 关键日志

- `build_cann92_sage_v53_l0stagefix.log`
- `install_cann92_sage_v53_l0stagefix.log`
- `finite_functional_accuracy480_cann92_v53.log`
- `pytest_official_cann92_v53_functional380.log`
- `pytest_official_cann92_v53_negative6.log`
- `finite_perf14_isolated_cann92_v53.log`
- `finite_ecc_regression20_cann92_v53.log`
- `pytest_official_cann92_v53_accuracy100.log`
- v52 故障对照：`finite_perf_retry10_cann92_v52.log`、`finite_perf14_isolated_repeat2_cann92_v52.log`

## 复现命令

```bash
source /home/developer/Ascend/cann-9.2.0-beta.1/set_env.sh
source /workspace/sage_cann92_v53_l0stagefix_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_LAUNCH_BLOCKING=1
export SAGEATTN_NPU_MODULE=sageattention2
export SAGEATTN_DEVICE=npu:0

# 代码仓内只执行代表性 smoke
python3 -m pytest -q -rA \
  experimental/attention/sageattention2/tests/pytest/test_sage_attention2_smoke.py

# 以下入口位于仓外独立交付附件 sageAttentionTest，不提交到算子代码仓
cd "/path/to/sageAttentionTest"
python3 -m pytest -c pytest.ini -q -rA \
  tests/pytest/test.py::test_sageattn_functional_json
python3 -m pytest -c pytest.ini -q -rA \
  tests/pytest/test.py::test_sageattn_negative_json

# 480 个正向配置的 finite/LSE 检查脚本与附件、日志一同在仓外归档
SAGEATTN_OFFICIAL_ROOT="$PWD" SAGEATTN_SUITES=functional,accuracy \
python3 /path/to/run_full_correctness_finite.py

# NPU-only 精度回退测试（仓外诊断项，不是本阶段门禁）
python3 -m pytest -c pytest.ini -q -rA \
  tests/pytest/test.py::test_sageattn_accuracy_json
```
