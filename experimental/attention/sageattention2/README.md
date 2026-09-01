# sageattention2 — SageAttention2（INT8 QK + FP8 PV）NPU 算子

昇腾 NPU（Atlas 950PR，CANN 9.2.0-beta.1）上的 SageAttention2 量化注意力算子，仅实现 **INT8 QK^T + FP8 PV** 路径（不含 FP16 PV / varlen / triton）。Kernel 基于 **AscendC + CATLASS**；性能基线为同仓 FIA（`FusedInferAttentionScore`，950PR 用 `aclnnFusedInferAttentionScoreV5` / `torch_npu.npu_fused_infer_attention_score`），两条接口路径均须 Speedup ≥ 1.6×。

## 目录结构

```text
experimental/attention/sageattention2/
├── sageattn_fp8/                        # AscendC + CATLASS kernel（MM1 INT8 QK^T / MM2 FP8 PV / chain 串联）
│   ├── mm1_qk_int8.cpp                  # MM1：INT8 QK^T matmul（独立 kernel）
│   ├── mm2_pv_fp8.cpp                   # MM2：FP8 E4M3 PV matmul（独立 kernel）
│   ├── sageattn_chain.cpp               # host 侧 MM1→softmax→MM2 串联
│   ├── fai_pipeline.cpp / fai_pipeline_kernel.h   # FA pipeline（读 S→softmax→MM2→rescale）
│   ├── fai_kernel.h / fai_tiling.h ...  # FA 主 kernel / tiling（device-side 断链，见 docs）
│   └── aclnn_pv/                        # aclnn 统一 FP8 PV 入口（op_host + op_kernel）
├── python/sageattention2/               # PyTorch 适配层（必选）
│   ├── core.py                          # sageattn / sageattn_qk_int8_pv_fp8_asc（Host 侧逻辑）
│   ├── quant.py                         # INT8 QK + FP8 V 量化（对标开源语义）
│   ├── backend.py                       # NPU kernel 后端注册 + CPU fallback（仅调试）
│   ├── sageattn_fp8_backend.py          # NPU kernel 执行入口（INT8 MM1 + FP8 MM2）
│   └── __init__.py
├── docs/                                # 设计文档 / 自验证报告
└── README.md
```

## 接口

### `sageattn`

```python
sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs)
```

自动分发入口：**NPU 上固定分发到 FP8 PV 实现**（`sageattn_qk_int8_pv_fp8_asc`，默认 `pv_accum_dtype="fp32+fp16"`）。可用 kwargs 覆盖下沉路径参数：`qk_quant_gran`、`pv_accum_dtype`、`smooth_k`、`smooth_v`。**不支持** `attn_mask`（非 `None` 报错）。

### `sageattn_qk_int8_pv_fp8_asc`

```python
sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND", is_causal=False,
    qk_quant_gran="per_thread", sm_scale=None, pv_accum_dtype="fp32+fp16",
    smooth_k=True, smooth_v=False, return_lse=False, **kwargs)
```

参数/默认值/返回值语义对标开源 `sageattn_qk_int8_pv_fp8_cuda`（函数名 `*_cuda` → `*_asc`）。

| 参数 | 取值 | 说明 |
|------|------|------|
| `tensor_layout` | `"HND"` / `"NHD"` | HND: `[B,H,S,D]`；NHD: `[B,S,H,D]` |
| `qk_quant_gran` | `"per_thread"` / `"per_warp"` | Q/K INT8 量化粒度（算法语义，非 AscendC 内置） |
| `pv_accum_dtype` | `"fp32"` / `"fp32+fp32"` / `"fp32+fp16"` | PV 两级累加；默认 `fp32+fp16`（默认验收配置） |
| `smooth_k` | `True` / `False` | K 序列维减均值；GQA 时对 LSE 做 correction |
| `smooth_v` | `True` / `False` | V 序列维减均值；`fp32`/`fp32+fp32`/`fp32+fp16` 下 **warning 并忽略**（对齐开源） |
| `return_lse` | `True` / `False` | 返回 `(o, lse)`；`lse` 形状 `[B,Hq,Sq]`，含 `÷1.44269504` 换算与 smooth_k 校正 |

**约束**：输入 FP16/BF16 且三者一致；`Hq % Hk == 0`；`stride(-1)==1`；head_dim∈(0,128]（内部 pad 到 64/128，>128 报错，输出裁回原始维度）。

### 环境变量 / 覆盖方式

| 变量 | 含义 |
|------|------|
| `SAGEATTN_NPU_MODULE` | 指定提供 NPU API 的模块名（默认尝试 `sageattention` / `sageattention2` / `sageattn` / `experimental.attention.sageattention2`） |
| `SAGEATTN_ALLOW_CPU_STUB` | `1` 时无 NPU 后端回退 CPU 参考实现（**仅流程调试，不作精度/性能验收**） |

## 构建与测试

**PyTorch 适配层**（`python/sageattention2/`）：纯 Python，无独立编译（Python 3.9+，需 `torch` / `torch_npu`）。

**AscendC + CATLASS kernel**（`sageattn_fp8/`）：需 CANN + AscendC 工具链，编译方式见 `sageattn_fp8/CMakeLists.txt`（与 CATLASS 样例一致）：

```bash
# 在 catlass 仓内（将本目录接入 examples 后）：
cd /workspace/catlass
bash scripts/build.sh sageattn_fp8 -DCATLASS_ARCH=3510
# 或本目录内：
mkdir -p build && cd build && cmake .. -DCATLASS_ARCH=3510 && make -j32
```

**aclnn 壳**（`sageattn_fp8/aclnn_pv/`）：目标 `aclnn_sageattn_pv`（host 校验 + tiling 结构 + kernel 占位，真 kernel 接驳见 docs 待办）。

**以下测试命令均在测试工程目录 `sageAttentionTest/` 下执行**（该工程含 `requirements.txt` / `cases/` / `tests/pytest/`）：

```bash
# 1. 安装依赖
pip install -r requirements.txt   # pytest / torch>=2.1.0 / numpy

# 2. 生成 500 条用例（前置）
python cases/generate_cases.py

# 3. 功能 + 精度（缺 NPU 实现会 skip）
python -m pytest -rA -s tests/pytest/test.py -m "functional or accuracy"

# 4. 冒烟 / 性能门禁（perf 需 NPU + torch_npu + FIA）
python -m pytest -rA -s tests/pytest/test.py -m smoke
python -m pytest -rA -s tests/pytest/test_perf.py -m perf
```

对接测试工程：`sageAttentionTest`（`sage_backend.load_sage_npu` 按 `SAGEATTN_NPU_MODULE` 加载本包）。接入方式：

```bash
# 在 sageAttentionTest 目录下，把本包加入 Python 路径并指定模块名
PYTHONPATH=<本包 python/ 目录的绝对路径> SAGEATTN_NPU_MODULE=sageattention2 python -m pytest -rA -s tests/pytest/test.py -m smoke
```

精度口径 L-A（vs GPU `sageattn_qk_int8_pv_fp8_cuda` 同 `pv_accum_dtype`）+ L-B（vs FP32 SDPA）；性能口径见任务书 P-01~P-07（双路径 ≥1.6× FIA，需在自验证报告填两路径计时表）。

## 详细设计

见 `docs/sageattention2_design.md`（AscendC/CATLASS 模块边界、量化语义、tiling、head_dim pad、两级累加、LSE 换算）。任务书口径唯一来源：`SageAttention2_task_doc.md`，冲突时以任务书为准。
