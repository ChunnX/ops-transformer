# FlashAttention 精度测试框架

## 一、架构概览

```
test_flash_attn.py (入口)
    │
    ├── case_loader.load_case_modules()        ← test_cases/*.py 加载 + 笛卡尔积展开
    ├── case_loader.normalize_params()         ← 补全默认值、TND/PA/quant 特殊处理
    │
    ▼   逐 case 循环
    ├── orchestrator.run_case()
    │       ├── data.flash_attn_inputs.generate()   ← 声明式张量生成
    │       │
    │       ├── golden.compute()     ← CPU 参考 (tforward / dequant + tforward)
    │       ├── primary.compute()    ← NPU/GPU 后端
    │       │
    │       └── compare.check_result()              ← 精度对比
    │
    └── reporter → 终端表格 + CSV
```

## 二、目录结构

```
pytests/
├── core/
│   ├── backends/
│   │   ├── base.py              # Backend 抽象基类
│   │   ├── cpu.py               # CPU Backend — 调用 tforward + dequant 流水线
│   │   ├── cpu_golden.py        # tforward / tforward_tnd / _attend (CPU 参考实现)
│   │   ├── quant_utils.py       # FP4 E2M1 / FP8 E8M0 解码与反量化 (Golden)
│   │   ├── gpu.py               # GPU Backend — flash_attn 库封装
│   │   ├── gpu_impl.py          # GPU 底层实现
│   │   └── npu.py               # NPU Backend — npu_flash_attn + print_metadata
│   ├── data.py                  # InputSpec 声明式张量生成 + build_flash_attn_params
│   ├── case_loader.py           # 用例加载、参数规范化、ID 解析
│   ├── orchestrator.py          # 编排器 — 数据生成→后端计算→精度对比→持久化
│   └── reporter.py              # Reporter — 终端表格 + CSV
├── test_cases/
│   ├── functional_stc.py        # 功能 STC 用例
│   ├── functional_redline_infer.py / functional_redline_train.py
│   ├── functional_redline_train_tnd.py
│   ├── performance_redline_infer.py / performance_redline_train.py
│   └── ...
├── utils/
│   ├── compare.py               # 精度对比 + 失败分布分析
│   ├── data.py                  # mask 生成、page_attn_for_bnsd、layout 转换
│   ├── perf_parser.py / perf_runner.py
│   └── precision_visual.py
├── tools/
│   └── xlsx_to_testcase.py     # Excel → test_cases 转换
├── test_flash_attn.py          # 主入口
└── readme.md
```

## 三、环境依赖

| 模式 | 依赖 |
|------|------|
| CPU-only | Python 3.8+, PyTorch 2.2+, einops, numpy |
| NPU | + torch_npu, cann_ops_transformer |
| GPU | + flash-attn, CUDA 12.0+ |

```bash
python -c "import torch_npu, cann_ops_transformer; print('NPU OK')"
python -c "import torch, flash_attn, einops; print('GPU OK')"
```

## 四、运行方式

```bash
# 全量精度测试
python test_flash_attn.py

# 指定 case
python test_flash_attn.py --case_id FP4_BNBD_01,TND_05

# GPU 模式
python test_flash_attn.py --case_id BASE_01 --use_gpu

# 只跑 metadata（打印分核）
python test_flash_attn.py --case_id TND_05 --meta_only

# 性能采集
python test_flash_attn.py --case_files performance_redline_train --perf_mode

# 三方对比
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu

# Golden 持久化
python test_flash_attn.py --save_golden ./golden_dump
python test_flash_attn.py --load_golden ./golden_dump --case_id BASE_01
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--case_files` | 用例文件，逗号分隔 |
| `--case_id` | 用例名，逗号分隔，支持前缀匹配 |
| `--device_id` / `--gpu_device` | 设备 ID |
| `--meta_only` | 只调 metadata，不跑 kernel |
| `--verbose_diff` | 输出全部超阈值元素 |
| `--visualize` / `--viz_dir` | 精度热力图 |
| `--fail_analysis` | 多维度失败分布分析 |
| `--save_golden` / `--load_golden` | golden 落盘 / 加载 |
| `--perf_mode` / `--perf_output` | 性能测试 |
| `--compare_mode` | 多后端对比 |
| `--graph_mode` | torchair 图模式 |

## 五、Layout 支持矩阵

| Layout | Shape | CPU | NPU | GPU | 说明 |
|--------|-------|-----|-----|-----|------|
| BNSD | (B, N, S, D) | ✓ | ✓ | ✓ | 默认 |
| BSND | (B, S, N, D) | ✓ | ✓ | ✓ | |
| TND | (total, N, D) | ✓ | ✓ | ✓ | 变长序列紧凑存储 |
| PA_BBND | (num_blocks, block_size, N2, D) | ✓* | ✓ | ✓ | Paged KV，D 不减半 |
| PA_BNBD | (num_blocks, N2, block_size, D) | ✓* | ✓ | ✓ | Paged KV，D 不减半 |

*CPU 对 PA 的支持通过 `_pa_to_bnsd` 重建 BNSD 实现，包括量化与非量化场景。

## 六、FP4 伪量化 Golden

### 6.1 设计规格

| 项 | 值 |
|---|---|
| PA 场景 | 默认开启 |
| 量化模式 | per-token-group，group_size = 32 |
| 量化对象 | 仅 K, V（Q 不量化） |
| D 对齐 | 32 对齐 |
| KV 精度 | FP4 E2M1（uint8 伪装，unpacked，D 不变） |
| Scale 精度 | FP8 E8M0（uint8 伪装，D/32） |
| Offset | 不支持 |

### 6.2 反量化公式

```python
KV_deq[..., d] = FP4_TABLE[KV_nibble[..., d]] * 2^(scale_byte[..., d//32] - 127)
#                  查表解码                     per-token-group 缩放（每组 32 个 D）
```

### 6.3 FP4 E2M1 值表

| nibble | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|--------|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|-----|
| float | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 | -0.0 | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |

### 6.4 执行流程

```
KV uint8 (PA layout)       Scale uint8 (D/32)
      ↓ _pa_to_bnsd              ↓ _pa_to_bnsd(PA_BBND)
KV uint8 (BNSD)            Scale uint8 (BNSD)
      ↓ fp4_to_float             ↓ e8m0_to_float
KV float32 (...,D)         Scale float32 (...,D/32)
                                 ↓ repeat_interleave(32, -1)
                           Scale broadcast (...,D)
      ↓ × 逐元素相乘 ──────────────────────────┘
KV_raw float32 (...,D)
      ↓ clamp + to(target_dtype)
KV_deq fp16/bf16 → .float() → tforward → attention 输出
```

### 6.5 相关模块

| 文件 | 职责 |
|------|------|
| `core/backends/quant_utils.py` | FP4/E8M0 值表、解码函数、`dequant_kv()` 反量化封装 |
| `core/backends/cpu.py` `_dequant_kv_to_tforward()` | 内联反量化流水线（PA 重建→解码→broadcast→clamp） |
| `core/case_loader.py` `quant_mode` 分支 | `quant_mode=1` 时推导 `D_group`/`DV_group`/`group_size`，校验 PA 强制 |
| `core/data.py` 量化 recipe | `gen_unpacked_fp4` / `gen_unpacked_e8m0` 生成 + `scale_k`/`scale_v` shape 定义 |

### 6.6 用例示例

```python
# 最小 fp4 用例
"FP4_BNBD_01": {**BASE_FP4, "Dtype": ["fp16"], "layout_q": ["BNSD"]}

# bf16 + BNBD
"FP4_BNBD_02": {**BASE_FP4, "Dtype": ["bf16"]}

# BBND（与 BNBD shape 不同，scale 相同）
"FP4_BBND_01": {**BASE_FP4, "layout_kv": ["PA_BBND"]}

# TND（变长 Q + PA KV）
"FP4_TND_01": {**BASE_FP4,
    "layout_q": ["TND"],
    "cu_seqlens_q":  [[0, 64, 128]],
    "cu_seqlens_kv": [[0, 1024, 2048]]}
```

`normalize_params` 中 `quant_mode >= 1` 时，block_table 不传则从 `seqused_kv` + `block_size` 自动生成。

## 七、执行流程

### 7.1 参数规范化

`normalize_params()` 关键逻辑：

| 条件 | 动作 |
|------|------|
| TND + 缺 seqused_q | 从 cu_seqlens_q 差分推导 |
| PA + 缺 block_table | 从 seqused_kv + block_size 自动生成 |
| quant_mode=1 | 推导 D_group=D//32, DV_group=DV//32, group_size=32 |
| quant_mode>=1 | 强制 layout_kv ∈ PA_BBND/PA_BNBD/PA_NZ |

### 7.2 张量生成

`InputSpec` 通过 shape token（`B`, `N1`, `S1`, `D`, `_V`, `total_s1`, `num_blocks`, `D_group` 等）声明式定义各 tensor 的维度，按 layout 分支生成。

`when` 条件互斥：`quant_mode==0` 生成 float KV，`quant_mode>=1` 生成 uint8 fp4 KV + uint8 e8m0 scale。

### 7.3 CPU Golden 路径

```
quant_mode == 0 → _to_tforward(K/V)         直接 layout 转换
quant_mode >= 1 → _dequant_kv_to_tforward   PA 重建 → 解码 → broadcast → clamp
两者统一 → tforward(q, k, v) → attention 输出
```

`_pa_to_bnsd()` 按 block_table 将物理 block 拷贝回逻辑序列。`d_override` 参数使同一函数可同时处理 KV（D=128）和 Scale（D/32=4）。

### 7.4 精度对比

`check_result()` 判定标准（fp16/bf16）：

```
单元素通过: |diff| ≤ max(|golden| × rtol, atol)
整体通过:   超阈值元素占比 ≤ red_range
```

## 八、mask_mode

| 值 | 名称 | 说明 |
|----|------|------|
| 0 | NO_MASK | 全 attention |
| 3 | CAUSAL | 右下对齐因果 mask |
| 4 | BAND_CAUSAL | + 滑动窗口，win_left/win_right 生效 |

## 九、已知缺口（未实现）

| # | 功能 | 位置 | 现状 | 阻塞原因 |
|---|------|------|------|----------|
| 1 | NPU 量化路径 | `core/backends/npu.py` | FP4 用例走 `cpu.py` dequant 能产出正确 golden，但 NPU 侧 `npu_flash_attn()` 接口不接收 `k_descale`/`v_descale` 参数，uint8 KV 直接传入会输出垃圾结果，**端到端 NPU vs CPU 对比无法进行** | 需 `npu_flash_attn` 接口扩展支持 scale 参数 |
| 2 | GPU 量化路径 | `core/backends/gpu.py` | GPU backend 零量化支持，无 dequant 流水线也无 scale 参数传递，FP4 用例走 GPU 路径无效 | 需 `flash-attn` 库支持量化 KV 参数或自行实现 GPU 反量化 |

> 修复这两项后，FP4 用例才能进入真正的端到端精度对比流程。
