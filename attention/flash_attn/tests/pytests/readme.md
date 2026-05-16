# FlashAttention 精度测试框架使用手册

## 一、目录结构

```
pytests/
├── test_flash_attn.py    # 主入口：命令行参数解析、流程编排、精度报告
├── test_case.py          # 测试用例定义（在此添加/修改 case）
├── precision_visual.py   # 精度可视化工具（独立可运行，生成热力图 PNG）
├── test_utils.py         # 工具函数：生成 Q/K/V、mask、layout 转换等
├── npu_impl.py           # NPU 算子调用封装（npu_flash_attn + npu_flash_attn_metadata）
├── cpu_impl.py           # CPU 参考实现（纯 PyTorch 浮点计算，作为 golden）
└── README.md             # 本文档
```

---

## 二、环境依赖

| 依赖 | 说明 |
|---|---|
| Python 3.8+ | — |
| PyTorch | 需带 `torch_npu` 扩展 |
| `npu_ops_transformer` | 提供 `npu_flash_attn` 和 `npu_flash_attn_metadata` |
| `einops` | layout 转换工具，`pip install einops` |
| `matplotlib` | 精度可视化（可选），`pip install matplotlib` |

确认安装：
```bash
python -c "import torch_npu, npu_ops_transformer, einops; print('OK')"
python -c "import matplotlib; print('matplotlib OK')"  # 可视化功能可选
```

---

## 三、快速上手

### 3.1 运行所有 case

```bash
cd pytests
python test_flash_attn.py
```

### 3.2 运行指定 case（单个或多个，逗号分隔）

```bash
python test_flash_attn.py --case_id BASE_01
python test_flash_attn.py --case_id BASE_01,BNSD_01,TND_01
```

### 3.3 指定 NPU 设备

```bash
python test_flash_attn.py --case_id BASE_01 --device_id 1
```

---

## 四、命令行参数完整说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--case_id` | str | `all` | 要运行的 case 名，多个用逗号分隔，`all` 表示全部 |
| `--device_id` | int | `0` | NPU 卡号 |
| `--dump_tensors` | flag | 关闭 | 将 q/k/v 及 cpu/npu 输出保存为 txt 文件 |
| `--dump_dir` | str | `./dump_output` | `--dump_tensors` 的保存根目录 |
| `--verbose_diff` | flag | 关闭 | 精度失败时输出**全部**超阈值元素的逐元素对比表 |
| `--visualize` | flag | 关闭 | 生成 CPU vs NPU 精度热力图 PNG（需 matplotlib） |
| `--viz_dir` | str | `./viz_output` | `--visualize` 的图片保存目录 |
| `--meta-only` | flag | 关闭 | 只允许metadata算子 |

---

## 五、如何添加新的测试 Case

所有 case 在 `test_case.py` 的 `TestCases` 字典中定义，格式如下：

```python
TestCases = {
    "MY_CASE": {
        # ---- 必填 ----
        "B":        2,          # batch size（TND layout 下可省略）
        "N1":       8,          # query head 数
        "D":        128,        # head dim
        "layout_q": "BNSD",     # 输入 layout：BNSD / BSND / TND
        "Dtype":    "fp16",     # 数据类型：fp16 / bf16

        # ---- 选填（有合理默认值）----
        "N2":        4,         # KV head 数，默认 = N1（MHA/GQA/MQA）
        "S1":        512,       # query 序列长度，默认 = S2
        "S2":        512,       # KV 序列长度
        "DV":        128,       # value head dim，默认 = D
        "layout_kv": "BNSD",    # KV layout，默认 = layout_q
        "layout_out":"BNSD",    # 输出 layout，默认 = layout_q
        "mask_mode": 1,         # sparse 模式，见第六节
        "pre_tokens":  2147483647,
        "next_tokens": 2147483647,

        # ---- Q/K/V 值域（新功能）----
        "q_range": (-1.0, 1.0), # Q 均匀随机值域，省略则全部为固定值10
        "k_range": (-1.0, 1.0),
        "v_range": (0.0,  1.0),
    },
}
```

### TND layout 特殊字段

TND layout 不使用 B/S1/S2，改用 **累积序列长度列表**（cumulative seqlen）：

```python
"TND_CASE": {
    "N1":            8,
    "N2":            4,           # 可省略，默认 = N1
    "D":             128,
    "layout_q":      "TND",
    "layout_kv":     "TND",
    "layout_out":    "TND",
    "Dtype":         "bf16",
    "mask_mode":     1,
    # cu_seqused_q：累积 Q 序列长度列表。
    #   - 首元素为 0（可省略，框架自动补齐）
    #   - 列表长度 = batch_size + 1（有前导0时）
    #   - 示例 [0, 128, 256, 512] 表示 3 个请求，seqlen 分别为 128/128/256
    "cu_seqused_q":  [0, 128, 256, 512],
    "cu_seqused_kv": [0, 128, 256, 512],   # 省略则与 cu_seqused_q 相同（自注意力）
}
```

**TND Decode 场景**（Q=1/请求，KV 长度各异）：

```python
"TND_DECODE": {
    "N1": 32, "N2": 8, "D": 128,
    "layout_q": "TND", "layout_kv": "TND", "layout_out": "TND",
    "Dtype": "bf16", "mask_mode": 1,
    # 8 个请求，每请求 Q = 1 token（逐 token 解码）
    "cu_seqused_q":  [0, 1, 2, 3, 4, 5, 6, 7, 8],
    # 各请求的 KV cache 长度（累积值）
    "cu_seqused_kv": [0, 64, 128, 256, 384, 640, 896, 1408, 2432],
    "q_range": (-1.0, 1.0), "k_range": (-1.0, 1.0), "v_range": (-1.0, 1.0),
}
```

**TND Prefill + 因果掩码场景**（变长 batch，自注意力）：

```python
"TND_PREFILL": {
    "N1": 16, "N2": 4, "D": 128,
    "layout_q": "TND", "layout_kv": "TND", "layout_out": "TND",
    "Dtype": "fp16",
    "mask_mode": 2,   # CAUSAL 上三角因果 mask
    # 4 个请求，seqlen 分别为 128/256/384/512（Q=KV，自注意力）
    "cu_seqused_q":  [0, 128, 384, 768, 1280],
    "cu_seqused_kv": [0, 128, 384, 768, 1280],
    "q_range": (-0.5, 0.5), "k_range": (-0.5, 0.5), "v_range": (-0.5, 0.5),
}
```

> **关键参数说明**：`cu_seqused_q` 中的首元素 0 可以省略；框架会自动检测并补齐。
> per-batch seqlen 由相邻元素之差得到：`seqlen[i] = cu[i+1] - cu[i]`。

### Paged Attention 特殊字段

```python
"PA_CASE": {
    ...
    "layout_kv":  "PA_BBND",      # 或 PA_BNBD
    "seqused_kv": [256, 512],
    "block_size": 64,
    "block_table": [[0,1,2,3], [4,5,6,7]],
}
```

---

## 六、mask_mode（sparse_mode）取值说明

| 值 | 名称 | 说明 |
|---|---|---|
| `0` | BAND | 带状 mask，配合 `pre_tokens`/`next_tokens` |
| `1` | NO_MASK | 无 mask（全 attention） |
| `2` | CAUSAL | 上三角因果 mask |
| `3` | RIGHT_DOWN_CAUSAL | 右下对齐因果 mask |
| `4` | BAND_CAUSAL | BAND + CAUSAL 混合 |
| `5` | PREFIX | 系统前缀 attention，配合 `prefix` 字段 |
| `6` | DILATED | 膨胀 attention |
| `7` / `8` | BAND_KV_SPLIT | 带状 + KV 分段 |

---

## 七、精度判定标准

每个元素通过条件：
```
diff(cpu, npu) ≤ max(|cpu| × 0.5%,  0.000025)
```

整体通过条件：超阈值元素占比 ≤ 0.5%

精度报告示例（PASS）：
```
┌────────────────────────────────────────────────────────────────┐
│  精度报告: BASE_01
├────────────────────────────────────────────────────────────────┤
│  Shape       : (1, 1, 64, 128)
│  MaxAbsErr   : 0.00012207
│  MeanAbsErr  : 0.00003819
│  MaxRelErr   : 0.00098801
│  MeanRelErr  : 0.00031042
│  FailElems   : 0 / 8192  (0.0000%)
│  Threshold   : elemRelErr≤0.50%  failRatio≤0.50%
│  结论        : ✓ PASS
└────────────────────────────────────────────────────────────────┘
```

---

## 八、调试功能

### 8.1 保存 Q/K/V 和输出到文件（`--dump_tensors`）

```bash
python test_flash_attn.py --case_id BASE_01 --dump_tensors --dump_dir ./debug
```

会在 `./debug/BASE_01/` 下生成：

```
q.txt        ← Q tensor（展平为逐行 float32）
k.txt        ← K tensor
v.txt        ← V tensor
cpu_out.txt  ← CPU golden 输出
npu_out.txt  ← NPU 算子输出
```

每个文件首行为注释，记录 shape：
```
# shape: 1x1x64x128  total: 8192
0.12345678
-0.98765432
...
```

### 8.2 逐元素精度对比表（`--verbose_diff`）

```bash
python test_flash_attn.py --case_id BASE_01 --verbose_diff
```

精度失败时，从默认的"前10个"改为列出**全部**超阈值元素：

```
│  全部 37 个超阈值元素 (relErr > 0.50%):
│         idx             CPU             NPU       absErr       relErr
│           0  +0.12345678  +0.12500000  0.00154322    0.012500
│           5  -0.98760000  -0.99100000  0.00340000    0.003443
│         ...
```

### 8.3 精度可视化（`--visualize`）

生成 CPU vs NPU 逐元素相对误差热力图，直观定位 fail 区域。

```bash
# 运行测试 + 同步生成热力图
python test_flash_attn.py --case_id BASE_01 --visualize

# 自定义图片保存目录
python test_flash_attn.py --case_id BASE_01 --visualize --viz_dir ./my_viz

# TND case 可视化
python test_flash_attn.py --case_id TND_DECODE_05 --visualize --viz_dir ./viz
```

每个 case 生成两类 PNG 文件：

| 文件 | 内容 |
|---|---|
| `{case}_heatmap_p{N}.png` | 逐 panel 的 relErr 热力图，**绿色=pass，红色遮罩=fail** |
| `{case}_passrate.png` | 各 panel 精度通过率棒状图（红柱=有 fail，绿柱=全 pass）|

**热力图配色规则**：
- 颜色深度表示 relErr 大小（浅绿 → 深绿 → 黄 → 红）
- colorbar 上的黄色虚线 = 0.5% 阈值分界线
- fail 元素（relErr > 0.5%）额外叠加半透明红色遮罩，一目了然

**Panel 划分规则**（基于输出 shape）：

| 输出 shape | Panel 划分 | 每 panel 内容 |
|---|---|---|
| 4D `(B,N,S,D)` BNSD | B×N 个 panel | (S×D) 热力图 |
| 3D `(T,N,D)` TND | T 个 panel | (N×D) 热力图 |
| 3D `(B,S,H)` BSH | B 个 panel | (S×H) 热力图 |

### 8.4 独立运行可视化工具（`precision_visual.py`）

可在不重新跑算子的情况下，对已 dump 的 txt 文件进行二次可视化：

```bash
# 先 dump 输出
python test_flash_attn.py --case_id BASE_01,TND_01 --dump_tensors

# 再可视化（逗号分隔多个 case）
python precision_visual.py \
    --dump_dir ./dump_output \
    --case_id  BASE_01,TND_01 \
    --out_dir  ./viz_output

# 直接指定 txt 文件（单 case）
python precision_visual.py \
    --cpu_txt ./dump_output/BASE_01/cpu_out.txt \
    --npu_txt ./dump_output/BASE_01/npu_out.txt

# 调整阈值和最大 panel 数
python precision_visual.py \
    --dump_dir ./dump_output --case_id BASE_01 \
    --threshold 0.01 \
    --max_panels 32
```

`precision_visual.py` 完整参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dump_dir` | str | — | dump 根目录，与 `--case_id` 配合（与 `--cpu_txt` 二选一）|
| `--cpu_txt` | str | — | 直接指定 cpu_out.txt（与 `--npu_txt` 配对）|
| `--npu_txt` | str | — | 直接指定 npu_out.txt（`--cpu_txt` 模式下必填）|
| `--case_id` | str | — | case 名，逗号分隔（`--dump_dir` 模式下必填）|
| `--case_name` | str | 自动推断 | 图片标题名（`--cpu_txt` 模式下可选）|
| `--out_dir` | str | `./viz_output` | 图片保存目录 |
| `--threshold` | float | `0.005` | 相对误差阈值（默认 0.5%）|
| `--max_panels` | int | `16` | 热力图最多展示的 panel 数量 |

### 8.5 全功能组合示例

```bash
# 一次完成：测试 + 保存 txt + 生成热力图 + 逐元素对比表
python test_flash_attn.py \
    --case_id TND_DECODE_05,TND_PREFILL_06 \
    --dump_tensors --dump_dir ./debug \
    --visualize    --viz_dir  ./viz \
    --verbose_diff \
    --device_id 1
```

---

## 九、Q/K/V 值域配置

在 `test_case.py` 中为 case 添加 `q_range`/`k_range`/`v_range` 字段，控制随机值的分布范围（均匀分布）：

```python
"MY_CASE": {
    ...
    "q_range": (-1.0, 1.0),   # Q 值从 Uniform(-1, 1) 采样
    "k_range": (-0.5, 0.5),   # K 值从 Uniform(-0.5, 0.5) 采样
    "v_range": (0.0,  2.0),   # V 值从 Uniform(0, 2) 采样
}
```

**不设置**时，Q/K/V 全部为固定值 10（便于底层调试，消除随机性影响）。

> **注意**：fp16 有效范围约 ±65504，请勿设置超出此范围的值域，否则会产生 inf/nan。

---

## 十、常见问题

**Q：运行时报 `ModuleNotFoundError: No module named 'npu_ops_transformer'`**

A：需要先安装或编译 `npu_ops_transformer` 包并加入 `PYTHONPATH`。

**Q：精度全部为 0（NPU 输出全零）**

A：通常是 tiling 或 metadata 问题，开启内核调试打印：
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
python test_flash_attn.py --case_id BASE_01
```
观察 `[META raw]`、`[INIT AIC0]`、`[GetTaskMode]` 等打印。

**Q：TND case 运行报 shape 错误**

A：检查 `cu_seqused_q`，允许有或没有前导 `0`，框架会自动处理。`cu_seqused_q` 首元素为 0 时长度为 batch+1，不含时长度为 batch。

**Q：TND case 精度与预期不符（CPU/NPU 结果对不上）**

A：这通常是 TND tensor 形状错误导致的（total_tokens 与 max_seqlen 混用）。框架已修复此问题：
- CPU golden 使用 `total_tokens` 作为序列维度
- NPU 通过 `S1/S2`（max_seqlen）传递元数据

A：用 `--dump_tensors` 保存后检查 `q.txt` 首行 shape，确认 `S` 维度等于所有 batch 的 seqlen 总和。

**Q：`--visualize` 报 `No module named 'matplotlib'`**

A：安装 matplotlib：
```bash
pip install matplotlib
```
如果无法安装，改用 `--dump_tensors` 保存 txt 后，在有 matplotlib 的环境中单独运行 `precision_visual.py`。

**Q：热力图全绿但有少量红色遮罩，pass 率 99.x%**

A：这是正常的 fp16/bf16 精度波动。红色遮罩的位置帮助定位哪些 (seq, dim) 坐标容易出误差；可结合 `--verbose_diff` 查看具体数值。

**Q：如何只看汇总结果而屏蔽详细日志**

A：重定向标准错误（NPU driver 日志）：
```bash
python test_flash_attn.py 2>/dev/null
```

---

## 十一、文件职责速查

| 文件 | 你需要改它吗？ | 典型改动 |
|---|---|---|
| `test_case.py` | **经常** | 新增/修改测试 case |
| `test_flash_attn.py` | 偶尔 | 改精度阈值、新增 flag |
| `precision_visual.py` | 很少 | 改热力图样式、panel 划分规则 |
| `test_utils.py` | 很少 | 修改 Q/K/V 生成方式、mask 逻辑 |
| `npu_impl.py` | 很少 | 修改算子调用参数 |
| `cpu_impl.py` | 一般不改 | CPU golden 实现 |