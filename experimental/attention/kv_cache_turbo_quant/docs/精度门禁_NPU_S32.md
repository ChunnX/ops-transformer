# NPU 精度门禁（任务书 §3.2）

**日期**：2026-09-23  
**证据**：`tests/logs/npu_precision_matrix_s32.log`

## 口径

| 指标 | 门槛 | 角色 |
|------|------|------|
| IP95 | &lt; 0.1 | **硬门槛** |
| relMSE | &lt; 0.05 | 参考；mse2 可 WARN |

小 shape 另保留 `npu_precision_cases` **打包 exact** 作回归哨兵。  
T=2048 允许相对 C++ 孪生 1～数字节 ULP，**以本门禁为准**。

## 本机结果（Mul+ReduceSum Kernel）

| case | relMSE | IP95 | 门 |
|------|-------:|-----:|:--:|
| gqa_decode_b1 | 0.0360 PASS | 0.0385 PASS | PASS |
| gqa_decode_b64 | 0.0342 PASS | 0.0421 PASS | PASS |
| gqa_prefill_t2048 | 0.0340 PASS | 0.0411 PASS | PASS |
| gqa_mse2_t2048 | 0.1161 WARN | 0.0728 PASS | PASS |
| gqa_mse4_t2048 | 0.0093 PASS | 0.0217 PASS | PASS |

**ALL PASS npu_precision_matrix_s32**

## 复现

```bash
# 编 export（同 npu_precision_cases 链接）
g++ -O2 -std=c++17 tests/npu_export_for_metric.cpp ... -o /tmp/npu_export_for_metric
cd tests && python3 run_npu_precision_matrix.py
```
