# FlashAttention 精度测试框架使用手册

## 一、目录结构

```
pytests/
├── backends/                  # 后端实现
│   ├── cpu_impl.py            # CPU 参考实现（纯 PyTorch，golden reference）
│   ├── gpu_impl.py            # GPU 实现（基于 flash_attn 库）
│   └── npu_impl.py            # NPU 算子封装（npu_flash_attn + metadata）
├── test_cases/                # 测试用例定义
│   ├── base.py                    # 基础测试用例（BASE、BNSD、TND、PA 等）
│   ├── fia_stc.py                 # FIA STC 测试用例集（实际推理场景）
│   ├── functional_stc.py          # 单功能测试用例
│   ├── functional_redline_train.py      # flash_attn 训练红线用例
│   ├── functional_redline_train_tnd.py  # flash_attn varlen 训练红线TND用例
│   ├── functional_redline_infer.py      # FIA 推理红线用例
│   └── performance_redline_infer.py     # 性能红线推理用例
├── tools/                      # 开发工具
│   └── xlsx_to_testcase.py     # Excel → test case 转换器
├── test_flash_attn.py           # 主入口：命令行参数解析、流程编排、三方精度对比
├── utils/                      # 工具函数包
│   ├── data.py                  # 测试数据生成、mask 生成、layout 转换
│   ├── compare.py               # 精度对比、结果检查、失败分布分析
│   ├── io.py                    # Tensor 文件读写
│   └── precision_visual.py     # 精度可视化工具（热力图 PNG）
├── test_npu_metadata.py        # NPU metadata 算子独立测试
└── readme.md                   # 本文档
```

---

## 二、环境依赖

### 通用依赖

| 依赖 | 说明 |
|---|---|
| Python 3.8+ | — |
| PyTorch 2.2+ | CPU/GPU 计算基础 |
| `einops` | Layout 转换工具，`pip install einops` |
| `numpy` | 数值计算，`pip install numpy` |

### GPU 模式依赖（可选）

| 依赖 | 说明 |
|---|---|
| `flash-attn` | FlashAttention 库，`pip install flash-attn --no-build-isolation` |
| CUDA 12.0+ | GPU 环境 |

### NPU 模式依赖（可选）

| 依赖 | 说明 |
|---|---|
| `torch_npu` | PyTorch NPU 扩展 |
| `npu_ops_transformer` | 提供 `npu_flash_attn` 和 `npu_flash_attn_metadata` |

### 可视化依赖（可选）

| 依赖 | 说明 |
|---|---|
| `matplotlib` | 精度热力图，`pip install matplotlib` |

### 环境检查

```bash
# CPU-only 模式
python -c "import torch, einops, numpy; print('CPU OK')"

# GPU-only 模式
python -c "import torch, flash_attn, einops; print('GPU OK')"

# NPU-only 模式
python -c "import torch_npu, npu_ops_transformer; print('NPU OK')"

# 可视化功能
python -c "import matplotlib; print('Visual OK')"
```

---

## 三、运行模式

### 3.1 CPU Golden（默认）

所有模式下 CPU 参考实现作为 golden reference：

```bash
python test_flash_attn.py --case_id BASE_01
```

### 3.2 GPU-only 模式

```bash
python test_flash_attn.py --case_id BASE_01 --use_gpu
```

- 使用 `flash_attn` 库进行 GPU 计算
- 支持更多 layout（TND、PA_BBND、PA_BNBD）
- 不依赖 NPU 环境

### 3.3 NPU-only 模式

```bash
python test_flash_attn.py --case_id BASE_01 --device_id 0
```

- 使用 `npu_flash_attn` 算子
- 需要 `torch_npu` 和 `npu_ops_transformer`

### 3.4 三方对比模式（CPU vs GPU vs NPU）

```bash
# 实时三方对比（需要 GPU + NPU 环境）
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu

# GPU-only 三方对比（自动加载 NPU dump）
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu \
    --load_npu_dump ./dump_output/BASE_01/npu_out.txt

# 离线三方对比（从 dump 文件）
python test_flash_attn.py --case_id BASE_01 --compare_mode \
    --load_gpu_dump ./dump_output/BASE_01/gpu_out.txt \
    --load_npu_dump ./dump_output/BASE_01/npu_out.txt
```

---

## 四、命令行参数完整说明

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--case` | str | `all` | 选择测试集：`all`/`base`/`fia`/`TestCases`/`TestCasesFIA` |
| `--case_id` | str | `all` | 具体 case 名，多个用逗号分隔 |
| `--device_id` | int | `0` | NPU 设备 ID |
| `--gpu-device` | int | `0` | GPU 设备 ID |

### 计算模式参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--use_gpu` | flag | 关闭 | 使用 GPU 计算（不执行 NPU）|
| `--compare_mode` | flag | 关闭 | 启用三方详细对比（CPU vs GPU vs NPU）|

### 输出保存参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dump_tensors` | flag | 关闭 | 保存 Q/K/V 及输出为 txt 文件 |
| `--dump_dir` | str | `./dump_output` | Dump 文件保存根目录 |

### 离线对比参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--load_gpu_dump` | str | None | GPU dump 文件路径 |
| `--load_npu_dump` | str | None | NPU dump 文件路径 |

### 调试参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--verbose_diff` | flag | 关闭 | 输出全部超阈值元素对比表 |
| `--visualize` | flag | 关闭 | 生成精度热力图 PNG |
| `--viz_dir` | str | `./viz_output` | 热力图保存目录 |
| `--meta_only` | flag | 关闭 | 只调用 metadata 算子 |
| `--fail_analysis` | flag | 关闭 | 精度失败时分析各 batch/head/seq 的误差分布 |

### 失败分布分析 (`--fail_analysis`)

当精度对比 FAIL 时，自动输出失败元素在各维度的分布统计，帮助定位精度问题集中在哪个 batch 或 head。

**使用方式：**

```bash
python test_flash_attn.py --case_id case69 --fail_analysis --dump_tensors
```

**输出内容：**

1. **按 Batch 统计** — 每个 batch 的失败元素数、失败率、最大/平均绝对误差
2. **按 Head 统计** — Top-10 失败率最高的 head
3. **按 Seq 位置统计** — Top-10 失败率最高的 query 位置
4. **Batch vs seqused_kv 关联** — 将每个 batch 的失败率与其实际 KV 序列长度对照，标注超 3% 阈值的 batch
5. **错误位置连续性分析** — 选取失败率最高的 3 个 batch，展开 D 维度检测错误是随机分散还是连续集中

**报告文件：** 自动保存到 `<dump_dir>/<case_name>/fail_analysis.txt`

**示例输出：**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   失败元素分布分析: fia_stc/case69
  Shape: (16, 180, 253, 128)  FailElems: 6358919/93265920 (6.8181%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [按 Batch 维度统计] (共 16 个 batch)
   Batch     FailCnt       Total   FailRatio    MaxAbsErr   MeanAbsErr
       0      543210     5806080     9.3561%    0.03125000  0.00156250
       2        1024     5806080     0.0176%    0.00390625  0.00097656
  ...

  [Batch vs seqused_kv 关联分析]
   Batch   seqused_kv   FailRatio                  说明
       0         1357     9.3561%            ⚠ 超3%阈值
       2           35     0.0176%

  [错误位置连续性分析] (选取失败率最高的 3 个 batch，分析 D 维度连段)

  ── Batch 0 (FailRatio=9.3561%) ──
    最密集位置: head=23, seq=112, 失败D元素=47/128
    连续段数: 38  最长段: 3  平均段长: 1.2
    模式判定: 随机分散（典型精度累积误差）
    前8段: [2:3](2), [7:7](1), [15:15](1), [21:22](2), ...
    seq分段失败率(每15位置): [8.2%, 7.9%, 9.1%, ...]
```

---

## 五、测试用例配置

### 5.1 测试集划分

| 测试集 | 文件 | 说明 |
|---|---|---|---|
| `TestCases` | `base.py` | 基础功能测试（BNSD/BSND/TND/PA）|
| `TestCasesFIA` | `fia_stc.py` | FIA STC 场景（实际推理 case）|

### 5.2 用例字段说明

```python
TestCases = {
    "MY_CASE": {
        # ---- 必填 ----
        "B":        2,          # batch size（TND layout 下可省略）
        "N1":       8,          # query head 数
        "D":        128,        # head dim
        "layout_q": "BNSD",     # 输入 layout：BNSD / BSND / TND / PA_BBND / PA_BNBD
        "Dtype":    "fp16",     # 数据类型：fp16 / bf16

        # ---- 选填（有合理默认值）----
        "N2":        4,         # KV head 数，默认 = N1（MHA/GQA/MQA）
        "S1":        512,       # query 序列长度
        "S2":        512,       # KV 序列长度，默认 = S1
        "DV":        128,       # value head dim，默认 = D
        "layout_kv": "BNSD",    # KV layout，默认 = layout_q
        "layout_out":"BNSD",    # 输出 layout，默认 = layout_q
        "mask_mode": 3,         # mask 模式，见第六节
        "win_left":  2147483647,
        "win_right": 2147483647,

        # ---- Q/K/V 值域 ----
        "q_range": (-1.0, 1.0), # Q 均匀随机值域，省略则固定值 10
        "k_range": (-1.0, 1.0),
        "v_range": (0.0,  1.0),
    },
}
```

### 5.3 TND Layout 特殊配置

```python
"TND_CASE": {
    "N1":            8,
    "N2":            4,
    "D":             128,
    "layout_q":      "TND",
    "layout_kv":     "TND",
    "layout_out":    "TND",
    "Dtype":         "bf16",
    "mask_mode":     3,
    # 累积序列长度列表（B+1 个元素）
    "cu_seqlens_q":  [0, 128, 256, 512],   # 3 个请求，seqlen 分别为 128/128/256
    "cu_seqlens_kv": [0, 128, 256, 512],
}
```

### 5.4 Paged Attention 配置（GPU-only）

```python
"PA_CASE": {
    "B": 4, "N1": 8, "N2": 4, "S1": 256, "S2": 1024, "D": 128,
    "layout_q": "BNSD",
    "layout_kv": "PA_BBND",     # 或 PA_BNBD
    "layout_out": "BNSD",
    "Dtype": "bf16",
    "mask_mode": 3,
    "seqused_kv": [256, 256, 512, 512],
    "block_size": 256,          # flash_attn 要求：必须是 256 的倍数
    "block_table": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
}
```

---

## 六、mask_mode 取值说明

| 值 | 名称 | 说明 |
|---|---|---|
| `0` | NO_MASK | 不使能 mask（全 attention） |
| `3` | RIGHT_DOWN_CAUSAL | 右下对齐因果 mask |
| `4` | BAND_CAUSAL | BAND + CAUSAL 混合 |

---

## 七、精度判定标准

### 标准对比模式

单元素判定：
```
diff ≤ max(|cpu| × 0.5%,  0.000025)
```

整体判定：超阈值元素占比 ≤ 0.5%

### 三方对比模式（compare_mode）

详细统计指标：

| 指标 | 说明 |
|---|---|
| 大值统计 | `|value| ≥ small_value` 的元素统计 |
| 相对误差 | `diff / |golden|` |
| 分段误差 | 万分之一/千分之一/千分之五/百分之一 误差统计 |
| 小值统计 | `|value| < small_value` 的元素统计 |
| NaN/INF | NaN 和 ±INF 的数量及错误统计 |
| RMSE | 绝对误差均方根 |
| 均衡性偏差 | `|sum(positive) - sum(negative)|` |

---

## 八、使用示例

### 8.1 快速测试

```bash
# 运行所有 case（NPU 模式）
python test_flash_attn.py

# 运行指定 case
python test_flash_attn.py --case_id BASE_01,BNSD_01

# GPU-only 模式
python test_flash_attn.py --case_id BASE_01 --use_gpu

# 指定测试集
python test_flash_attn.py --case fia --case_id aclnnFusedInferAttentionScoreV5_FIA_51_160_20_59_256_case56
```

### 8.2 三方对比

```bash
# 实时三方对比（GPU + NPU）
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu

# GPU-only 三方对比（从 dump 加载 NPU）
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu \
    --dump_tensors --load_npu_dump ./dump_output/BASE_01/npu_out.txt

# 离线三方对比（无 GPU/NPU）
python test_flash_attn.py --case_id BASE_01 --compare_mode \
    --load_gpu_dump ./dump_output/BASE_01/gpu_out.txt \
    --load_npu_dump ./dump_output/BASE_01/npu_out.txt
```

### 8.3 调试功能

```bash
# 保存 tensor 到 txt
python test_flash_attn.py --case_id BASE_01 --dump_tensors --dump_dir ./debug

# 可视化精度热力图
python test_flash_attn.py --case_id BASE_01 --visualize --viz_dir ./viz

# 输出全部超阈值元素
python test_flash_attn.py --case_id BASE_01 --verbose_diff

# 组合调试
python test_flash_attn.py --case_id BASE_01 \
    --dump_tensors --visualize --verbose_diff
```

### 8.4 PA 用例测试（GPU-only）

```bash
python test_flash_attn.py --case_id PA_BBND_05,PA_BNBD_06 --use_gpu
```

---

## 九、精度可视化

### 9.1 实时可视化

```bash
python test_flash_attn.py --case_id BASE_01 --visualize
```

生成两类 PNG 文件：

| 文件 | 内容 |
|---|---|
| `{case}_heatmap_p{N}.png` | 逐 panel relErr 热力图（绿色=pass，红色=fail）|
| `{case}_passrate.png` | 各 panel 精度通过率棒状图 |

### 9.2 独立可视化（从 dump 文件）

```bash
# 先 dump 输出
python test_flash_attn.py --case_id BASE_01 --dump_tensors

# 再可视化
python precision_visual.py \
    --dump_dir ./dump_output \
    --case_id BASE_01 \
    --out_dir ./viz_output
```

`precision_visual.py` 参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dump_dir` | str | — | dump 根目录（与 `--case_id` 配合）|
| `--case_id` | str | — | case 名，逗号分隔 |
| `--cpu_txt` | str | — | 直接指定 cpu_out.txt |
| `--npu_txt` | str | — | 直接指定 npu_out.txt |
| `--out_dir` | str | `./viz_output` | 图片保存目录 |
| `--threshold` | float | `0.005` | 相对误差阈值 |
| `--max_panels` | int | `16` | 最大展示 panel 数 |

---

## 十、Layout 支持矩阵

| Layout | CPU | GPU | NPU | 说明 |
|---|---|---|---|---|
| **BNSD** | ✓ | ✓ | ✓ | (B, N, S, D) - 默认格式 |
| **BSND** | ✓ | ✓ | ✓ | (B, S, N, D) |
| **TND** | ✓ | ✓ | ✓ | (total_tokens, N, D) - 变长序列 |
| **PA_BBND** | ✗ | ✓ | ✗ | (num_blocks, block_size, N, D) - Paged KV |
| **PA_BNBD** | ✗ | ✓ | ✗ | (num_blocks, N, block_size, D) - Paged KV |
| **BSH** | ✓ | ✗ | ✓ | (B, S, H) |
| **SBH** | ✓ | ✗ | ✓ | (S, B, H) |

**说明**：
- PA (Paged Attention) 仅在 GPU 端通过 `flash_attn_with_kvcache` 实现
- NPU PA 支持待后续实现（算子层面已支持，需完善参数传递）
- CPU 未实现 PA golden reference

---

## 十一、常见问题

### Q1：运行时报 `ModuleNotFoundError: No module named 'flash_attn'`

**A**：GPU 模式需要安装 flash_attn：
```bash
pip install flash-attn --no-build-isolation
```

### Q2：运行时报 `ModuleNotFoundError: No module named 'torch_npu'`

**A**：NPU 模式需要 torch_npu 环境，或改用 GPU 模式：
```bash
python test_flash_attn.py --case_id BASE_01 --use_gpu
```

### Q3：PA case 运行报错 `block_size must be multiple of 256`

**A**：flash_attn_with_kvcache 要求 `block_size` 必须是 256 的倍数。调整用例：
```python
"block_size": 256,  # 不是 64
"S2": 1024,         # block数 × block_size
```

### Q4：三方对比时缺少 GPU/NPU 结果

**A**：使用 `--load_gpu_dump` 或 `--load_npu_dump` 加载预先 dump 的结果：
```bash
python test_flash_attn.py --case_id BASE_01 --compare_mode \
    --load_gpu_dump ./dump/BASE_01/gpu_out.txt \
    --load_npu_dump ./dump/BASE_01/npu_out.txt
```

### Q5：TND case 精度异常

**A**：检查 `cu_seqlens_q` 格式（必须包含前导 0）：
```python
"cu_seqlens_q": [0, 128, 256, 512],  # 长度 = batch_size + 1
```

### Q6：如何查看 metadata 信息

**A**：使用 `--meta_only` 模式：
```bash
python test_flash_attn.py --case_id BASE_01 --meta_only
```

---

## 十二、文件职责速查

| 文件 | 你需要改它吗？ | 典型改动 |
|---|---|---|---|
| `test_cases/base.py` | **经常** | 新增/修改测试 case |
| `test_cases/fia_stc.py` | **经常** | 添加 FIA STC 用例 |
| `test_flash_attn.py` | 偶尔 | 改参数、新增功能 |
| `utils/compare.py` | 很少 | 精度对比 / 报告 / 失败分析 |
| `utils/data.py` | 很少 | 测试数据生成 / mask / layout |
| `backends/cpu_impl.py` | 一般不改 | CPU golden 实现 |
| `backends/gpu_impl.py` | 很少 | GPU API 封装、PA 支持 |
| `backends/npu_impl.py` | 很少 | NPU 算子调用、metadata 解析 |
| `utils/precision_visual.py` | 很少 | 改热力图样式 |
