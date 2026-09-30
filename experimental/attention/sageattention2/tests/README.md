# SageAttention2 测试套件与验收复现说明

本目录包含 SageAttention2 算子的全量测试资产，涵盖 GPU Golden 复现导出、TC-01 至 TC-10 功能矩阵验收、L-A / L-B 双层精度门禁、C++ 原生内核数值规格一致性单测以及 P-01 至 P-07 双路径性能基准测试。

---

## 1. 目录资产概览

| 测试脚本 / 源码 | 类型 | 适用环境 | 说明 |
|---|---|---|---|
| `test_functional_matrix.py` | Python / unittest | CPU / NPU | TC-01 ~ TC-10 全量功能验收矩阵（参数组合、Pad、GQA、Causal、边界拦截等） |
| `test_precision_dual_level.py` | Python / unittest | CPU / NPU | L-A（GPU 对标）与 L-B（FP32 SDPA 相对误差比）双层精度门禁 |
| `numeric_error_test.cpp` | C++17 可移植单测 | 任意宿主机 / 容器 | 18 大类 50 项内核数学近似、反量化抵消、换底与溢出安全包络断言（含 [18] 宽路径分母折叠回归） |
| `benchmark_pytorch_runner.py` | Python 自动化压测 | NPU (Atlas 950PR) | P-01 ~ P-07 PyTorch 路径对标 `torch_npu.npu_fused_infer_attention_score` |
| `benchmark_aclnn.cpp` | C++ 独立基准程序 | NPU (Atlas 950PR) | P-01 ~ P-07 aclnn 路径对标 `aclnnFusedInferAttentionScoreV5`，两段式 Event 计时 |

---

## 2. GPU Golden 复现与导出指南（Task 3.1 / L-A 精度基准）

依据任务书规范，L-A 级精度验收以 GPU 原生参考实现为标杆。必须保证**同路径、同 accum、同 gran**（严禁使用 FP16-PV 或不同 accum 配置作为 golden）。
> 现状口径：本目录现有 Python 门禁在宿主机上以 `python/sageattention2/reference.py`（CPU golden）为标杆运行，GPU golden 导出文件当前无套件内消费端——真机 L-A 对标（GPU golden vs NPU 内核）按 todo.md 阶段 4 在验收环境执行。

### 2.1 环境要求
- **硬件平台**：NVIDIA Ada Lovelace（如 RTX 4090）或 Hopper（如 H100 / H800），具备原生 FP8 (E4M3) Tensor Core 算力；
- **CUDA 版本**：CUDA 12.1 及以上；
- **PyTorch 版本**：PyTorch 2.3.0 及以上；
- **开源依赖库**：`sageattention==2.2.0`（通过 `pip install sageattention==2.2.0` 安装）。

### 2.2 Golden 数据生成与导出脚本

在具备上述 GPU 环境的机器上运行以下脚本，导出标准化测试张量：

```python
import os
import math
import torch
from sageattention import sageattn_qk_int8_pv_fp8_cuda

def export_gpu_golden(
    output_dir="./golden_data",
    b=1, hq=32, hk=32, sq=4096, sk=4096, d=128,
    dtype=torch.float16,
    is_causal=False,
    qk_quant_gran="per_thread",
    pv_accum_dtype="fp32+fp16",
    smooth_k=True,
    smooth_v=False,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0"
    torch.manual_seed(42)

    # 构造输入张量 (HND 布局: [B, H, S, D])
    q = torch.randn(b, hq, sq, d, dtype=dtype, device=device)
    k = torch.randn(b, hk, sk, d, dtype=dtype, device=device)
    v = torch.randn(b, hk, sk, d, dtype=dtype, device=device)
    sm_scale = 1.0 / math.sqrt(d)

    # 调用 GPU 原生算子执行前向计算
    out_cuda, lse_cuda = sageattn_qk_int8_pv_fp8_cuda(
        q, k, v,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=True,
    )

    file_name = f"golden_b{b}_h{hq}_s{sq}_d{d}_{qk_quant_gran}_{pv_accum_dtype}_{'causal' if is_causal else 'dense'}.pt"
    save_path = os.path.join(output_dir, file_name)

    torch.save({
        "config": {
            "b": b, "hq": hq, "hk": hk, "sq": sq, "sk": sk, "d": d,
            "dtype": str(dtype), "is_causal": is_causal,
            "qk_quant_gran": qk_quant_gran,
            "pv_accum_dtype": pv_accum_dtype,
            "smooth_k": smooth_k, "smooth_v": smooth_v,
            "sm_scale": sm_scale,
        },
        "inputs": {"q": q.cpu(), "k": k.cpu(), "v": v.cpu()},
        "golden": {"out": out_cuda.cpu(), "lse": lse_cuda.cpu()}
    }, save_path)
    print(f"[SUCCESS] Exported golden data to {save_path}")

if __name__ == "__main__":
    # 生成基准配置 Golden
    export_gpu_golden(sq=4096, sk=4096, pv_accum_dtype="fp32+fp16", qk_quant_gran="per_thread")
    export_gpu_golden(sq=4096, sk=4096, pv_accum_dtype="fp32+fp16", qk_quant_gran="per_warp")
    export_gpu_golden(sq=4096, sk=4096, pv_accum_dtype="fp32+fp32", qk_quant_gran="per_thread")
    export_gpu_golden(sq=4096, sk=4096, pv_accum_dtype="fp32", qk_quant_gran="per_thread")
```

---

## 3. 功能矩阵验收用例集（TC-01 至 TC-10）执行指南

功能测试集严格对齐任务书各项要求，支持在昇腾 NPU 环境及 CPU 宿主机环境下运行：

### 3.1 执行命令

```bash
# 方式 1：使用 pytest 运行（推荐）
python3 -m pytest -v experimental/attention/sageattention2/tests/test_functional_matrix.py

# 方式 2：使用 unittest 运行
python3 -m unittest experimental/attention/sageattention2/tests/test_functional_matrix.py -v
```

### 3.2 用例明细与场景覆盖

| 编号 | 测试用例方法名 | 场景覆盖与输入规格 | 判定准则 |
|---|---|---|---|
| **TC-01** | `test_tc01_basic_functionality_and_dtypes` | 自动路由至 FP8 PV，FP16/BF16 数据类型，HND/NHD 布局互换 | 输出 Shape/Dtype 完全对齐，无 NaN/Inf |
| **TC-02** | `test_tc02_quant_gran_and_accum_dtypes` | 量化粒度（`per_thread`/`per_warp`）$\times$ 累加格式（`fp32`/`fp32+fp32`/`fp32+fp16`）全矩阵 | 全组合稳定输出有限数值 |
| **TC-03** | `test_tc03_smooth_k_and_smooth_v_behavior` | `smooth_k` 开关对比；低精度累加下开启 `smooth_v` 拦截 Warning | 正确触发 Warning 并安全忽略 `smooth_v` |
| **TC-04** | `test_tc04_lse_export_and_shapes` | `return_lse=True` 输出维度 `[B, Hq, Sq]`，FP32 数据类型 | 满足 Raw $\to$ Natural 对数换算与 GQA 广播校正 |
| **TC-05** | `test_tc05_head_dim_auto_pad_and_crop` | $D=48 \to 64$ 与 $D=96 \to 128$ 自动 Pad，末维连续性裁剪 | 输出 Shape 精确裁剪恢复原始特征维度 |
| **TC-06** | `test_tc06_gqa_and_asymmetric_sequence_lengths` | GQA 模式（$H_Q \% H_{KV} == 0$），非因果 $S_Q \ne S_K$ | 输出张量正确，各头注意力正确收敛 |
| **TC-07** | `test_tc07_causal_attention` | 因果掩码下 $S_Q == S_K$ | 下三角掩码严格生效，上三角概率全零 |
| **TC-08** | `test_tc08_strict_exception_interception` | 非法输入防御：$D>128$、末维非连续、非法枚举字符串、非空 `attn_mask`、因果 $S_Q \ne S_K$ | 抛出符合预期的 `ValueError` 或 `TypeError` |
| **TC-09** | `test_tc09_long_sequence_handling` | 长序列超千维 / 万维 ($S=1024 \dots 16384$) | 显存无爆栈，数值稳定性无溢出 |
| **TC-10** | `test_tc10_sdpa_seamless_substitution` | `replace_sdpa()` 补丁无缝拦截 `torch.nn.functional.scaled_dot_product_attention` | 替换后功能正常，恢复后还原原生实现 |

### 3.3 预期输出

```text
test_tc01_basic_functionality_and_dtypes (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc02_quant_gran_and_accum_dtypes (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc03_smooth_k_and_smooth_v_behavior (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc04_lse_export_and_shapes (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc05_head_dim_auto_pad_and_crop (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc06_gqa_and_asymmetric_sequence_lengths (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc07_causal_attention (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc08_strict_exception_interception (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc09_long_sequence_handling (test_functional_matrix.TestFunctionalMatrix) ... ok
test_tc10_sdpa_seamless_substitution (test_functional_matrix.TestFunctionalMatrix) ... ok

----------------------------------------------------------------------
Ran 10 tests in ...s  # 示意格式，实际耗时以运行环境为准

OK
```

---

## 4. 数值规格与精度门禁测试

### 4.1 C++ 内核数值规格测试 (`numeric_error_test.cpp`)

本测试无任何 AscendC 硬件库依赖，可直接在宿主机独立编译执行，将内核数值变换公式与浮点误差界固化为 18 大类 50 项可执行断言：

```bash
# 编译并运行
g++ -O2 -std=c++17 experimental/attention/sageattention2/tests/numeric_error_test.cpp -o /tmp/numeric_error_test
/tmp/numeric_error_test
```

**断言覆盖清单**：
1. `FastReciprocal` 浮点倒数逼近误差界 $\le 2^{-14}$；
2. INT8 对称动态量化往返误差 $\le \frac{\Delta}{2}$；
3. E4M3 饱和截断误差界 $\le 2^{-4}$；
4. E8M0 分组 Scale 覆盖紧致性；
5. DEQF16 ($2^{-11}$) 与平衡因子乘积的 $2048$ 抵消恒等式；
6. `LN_448`（$+6.104793$）以 $-\mathrm{LN\_448}$ 下移跟踪最大值的符号约束与满量程饱和裕度；
7. LSE 换算契约（Raw $\log_2$ 转自然对数并补偿平滑项）；
8. Key 序列平滑前后 Softmax 严格整行不变性；
9. Causal Mask NZ 拓扑剪枝位级一致性；
10. `qk_quant_gran` 分组共享 Scale 广播覆盖性；
11. `pv_accum_dtype="fp32+fp16"` 部分和舍入界与 $2^{-8}$ 平移防溢出包络（$\max|V| \le 146.2$ 有界）；
12. `smooth_v` 居中补偿不展宽 MXFP8 误差界；
13. 片上流式 Q 量化与 Host 预量化位级等价；
14. 同源 Normalizer 消除概率质量损失一致性验证；
15. 长序列 Delayed FP32 Buffering（FP16 对折叠累加）误差包络与溢出安全；
16. Chunk-wise 局部平滑零屏障在线 Softmax 与全序参考等价；
17. Task 3.4 / P5.2 同源 Normalizer（E4M3 回读值参与分母累加）概率质量守恒；
18. Task P0.1 宽路径多块分母折叠与 LSE 契约（dp=128, sk≥4096）：配对流水折叠 vs 逐块基准（≤1e-6）、融合 LSE（≤1e-5）、归一化输出（≤1e-5）、同源概率质量守恒（≤1e-5）、无折叠错误分母可检出（放大比 ≥1000x）、未初始化 DEN_UB NaN canary 经缺失 FIRST 分派的可检出性、FIRST 分派折叠对 UB 垃圾免疫且与逐块基准一致（≤1e-6）。

**宽路径（Wide Streaming）触发条件与覆盖口径**：宽路径由 op_host 侧 `dp == 128 && d == dp && sq % 128 == 0 && sk % 512 == 0 && sk >= 4096` 门控（正是 P-01~P-07 全部基准形状），其分母折叠链位于 `op_kernel/decoupled_tri_queue.h`（配对退休 `UpdateWidePair` / 单步退休 `FinalizeWideFp32`）与 `op_kernel/online_reduction_softmax.h`（`WideFoldDenomSingle` / `WideFoldDenomPair`）。上述 [18] 用例以可移植 C++ 模拟该折叠数据流（首块 `denom := blockSum` 初始化 + 其后 `denom = denom*ratio + blockSum`）固化其数值语义；**宿主 pytest（TC-01~TC-10、L-A/L-B）走 `python/sageattention2/reference.py` CPU golden 且输入 sk≤1024，到不了宽路径设备端 kernel**——宽路径 kernel 的端到端覆盖须在真机执行（长序列 TC-09 形状与 L-B 精度门禁含 P0.1 修复复测），本目录宿主资产仅提供上述折叠链数值规格防线。

**预期输出**：
```text
[PASS] Category 1: FastReciprocal error bound < 2^-14   # 以下为示意格式
[PASS] Category 2: Symmetric INT8 quantization round-trip
...
============================================================
ALL NUMERIC ERROR TESTS PASSED  # 示意格式：实际输出无计数后缀
============================================================
```

### 4.2 L-A / L-B 双层精度门禁测试 (`test_precision_dual_level.py`)

```bash
python3 -m unittest experimental/attention/sageattention2/tests/test_precision_dual_level.py -v
```

- **L-A 标杆（主标杆）**：
  - FP16：$\text{rtol} = 2^{-9}$, $\text{atol} = 2^{-9}$, $\text{matched\_ratio} \ge 0.99$, $\text{max\_abs\_error} \le 0.1$；
  - BF16：$\text{rtol} = 2^{-6}$, $\text{atol} = 2^{-6}$, $\text{matched\_ratio} \ge 0.99$, $\text{max\_abs\_error} \le 1.0$；
- **L-B 标杆（辅标杆）**：相对 FP32 SDPA 误差比：
  - $\text{MaxErrorRatio} \le 2.0$；
  - $\text{MeanErrorRatio} \le 1.2$；
  - $\text{RMSEErrorRatio} \le 1.2$。

---

## 5. P-01 至 P-07 性能对比测试指南

在 Atlas 950PR（CANN 9.2.0-beta.1）真机上执行性能基准测试。硬性门禁要求：**aclnn 与 PyTorch 双路径加速比均须 $\ge 1.6\times$ 官方基线 FIA**。

### 5.1 aclnn 路径基准测试 (`benchmark_aclnn.cpp`)

本程序以 C++ 直接调用底层 ACLNN 接口，基线算子调用 `aclnnFusedInferAttentionScoreV5`，被测算子调用 `aclnnSageAttention2`。两阶段调用（`GetWorkspaceSize` + `Execute`）均包含在一对 Device Event 计时区间内：

```bash
# 编译可执行程序
g++ -O2 -std=c++17 \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include \
    experimental/attention/sageattention2/tests/benchmark_aclnn.cpp \
    -o /tmp/benchmark_aclnn \
    -L$ASCEND_HOME_PATH/lib64 -lascendcl -lopapi

# 运行全量 P-01 ~ P-07 性能压测
/tmp/benchmark_aclnn --warmup 50 --repeats 100 --output aclnn_perf_results.json
```

**控制台输出样式**：
```text
==========================================================================
   SageAttention2 C++ aclnn Benchmark Suite (P-01 ~ P-07)
   Warmup: 50, Repeats: 100
==========================================================================
Case    Shape(HND)      FIA-aclnn(ms) Ours-aclnn(ms) aclnn Speedup    Gate >=1.6x
--------------------------------------------------------------------------
P-01    [1,32,4096,128]  <真机实测后回填>  <同左>        <同左>          <同左>
P-02    [1,32,8192,128]  <同左>        <同左>          <同左>          <同左>
...     ...              ...           ...            ...             ...
==========================================================================
注意：上表为输出格式示意。截至本文件修订时 P-01~P-07 双路径尚未在真机采数达标
（已有真机锚点：P-01 本算子 0.580 ms vs FIA 0.519 ms，约 0.89x，未达 1.6x 门禁），
严禁在无实测数据时回填任何数值或 PASS 判定。
```

### 5.2 PyTorch 路径基准测试 (`benchmark_pytorch_runner.py`)

在 PyTorch 层调用 `sageattn_qk_int8_pv_fp8_asc` 与 `torch_npu.npu_fused_infer_attention_score` 进行同等计时比对：

```bash
python3 experimental/attention/sageattention2/tests/benchmark_pytorch_runner.py pytorch_perf_results.json
```

**控制台输出样式**：
```text
==========================================================================
   SageAttention2 PyTorch Benchmark Runner (P-01 ~ P-07)
   Warmup: 50, Repeats: 100
==========================================================================
Case    Shape(HND)        FIA-PT(ms)    Ours-PT(ms)    PT Speedup    Gate >=1.6x
--------------------------------------------------------------------------
P-01    [1,32,4096,128]   <真机实测后回填>  <同左>        <同左>          <同左>
P-02 ~ P-07             ...           ...            ...             ...
==========================================================================
[SUCCESS] Results written to pytorch_perf_results.json
注意：同上，输出格式示意，数值一律以真机实测为准；当前门禁未达标。
```
