# QLI v2 MX 调试 Dump 使用指南

本文面向负责 QLI v2 MXFP8/MXFP4 测试与问题定位的测试端大模型。当前插桩是临时调试代码，用于区分以下两类问题：

1. NPU 计算出的完整 score 已经错误。
2. 完整 score 正确，但 TopK 选择、value/index 配对或输出后处理错误。

## 1. 使用原则

- 插桩不限制固定的 batch、Q/K 长度、布局或 sparse_count。
- 每次只选择一行 `(b_idx, n2_idx, s1_idx)` 采集，避免 DumpTensor 数据量失控。
- Python 侧目标由环境变量控制；NPU kernel 侧目标由编译期宏控制。两者必须完全一致。
- 修改 NPU 目标行或 QK S2 窗口后，必须重新编译并安装算子。
- 不要为了让 Case 通过而修改比较阈值、过滤失败元素或放宽 TopK 规则。
- 每个 Case 使用独立输出目录，避免前一个 Case 的文件被覆盖。

## 2. 调试目标配置

### 2.1 Python/pytest 侧

运行前设置：

```bash
export QLIV2_DEBUG_DUMP=1
export QLIV2_DEBUG_B_IDX=0
export QLIV2_DEBUG_N2_IDX=0
export QLIV2_DEBUG_S1_IDX=4
export QLIV2_DEBUG_WATCH_INDICES=2492,2499
export QLIV2_DEBUG_DUMP_DIR="$PWD/qliv2_debug_seed3_b0_n20_s14"
```

`QLIV2_DEBUG_WATCH_INDICES` 只控制终端额外打印哪些 TopK index，不限制实际保存的数据。

### 2.2 NPU kernel 侧

默认值位于：

```text
attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_common.h
```

相关宏：

```cpp
QLIV2_DEBUG_TARGET_B_IDX
QLIV2_DEBUG_TARGET_N2_IDX
QLIV2_DEBUG_TARGET_S1_IDX
QLIV2_DEBUG_QK_S2_BEGIN
QLIV2_DEBUG_QK_S2_END
```

默认目标为 `b=0, n2=0, s1=4`，MXFP4 QK 的默认采集窗口为 `S2=[0, 4096)`。可修改默认值，或通过编译参数传入同名宏。

注意：

- 只修改 Python 环境变量不会改变 NPU 采集位置。
- 如果关注的 K index 不在 QK 窗口内，应调整 `QLIV2_DEBUG_QK_S2_BEGIN/END` 后重新编译。
- 完整 score 不受 QK 窗口限制；窗口只用于控制额外的 MXFP4 QK 中间结果数据量。

## 3. 运行测试

先确保服务器同步到包含调试插桩的提交，并重新编译、安装算子。然后只启用待定位 Case：

```bash
timeout 600 bash test_run.sh single 2>&1 | tee qliv2_dump.log
echo "exit_code=$?"
```

如果普通运行日志中没有 `DumpTensor:`，需要使用服务器 CANN 环境支持的 NPU 调试工具开启 DumpTensor 采集。源码已经打开 `ASCENDC_DUMP`，但日志采集方式仍取决于服务器环境。

## 4. 将 NPU DumpTensor 转成 NumPy

```bash
python attention/quant_lightning_indexer_v2/tests/pytest/parse_qliv2_debug_dump.py \
  qliv2_dump.log \
  -o qliv2_npu_dump
```

解析器会：

- 保存每条 descriptor 对应的原始 `.npy`。
- 为每份QK输出`[QK meta]`行，明确标注`blockId/aiCoreIdx/curS2Idx/localRow/gIdx`。
- 将 NPU BF16 sortable key 还原为可直接比较的 FP32 数值。
- 按 S2 起点拼接分段的完整 NPU score。
- 提取 kernel 内部 TopK index/value、scale、seqused 和实际 S1/S2 长度。
- 当 DumpTensor 被截断时输出 warning。

已有单独的 sortable-key 解码脚本：

```bash
python attention/quant_lightning_indexer_v2/tests/pytest/decode_qliv2_debug_dump.py \
  score_keys.npy -o score.npy
```

## 5. 主要输出文件

### 5.1 pytest 直接生成

| 文件 | 含义 |
| --- | --- |
| `cpu_score.npy` | CPU 目标行完整 score，包含有效 K 范围 |
| `cpu_score_effective.npy` | sparse_mode=3 下该行真正进入 NPU TopK 的有效前缀 |
| `cpu_topk_index.npy` | CPU TopK index |
| `cpu_topk_value.npy` | CPU TopK value，仅 return_value=1 |
| `cpu_topk_pairs.npy` | CPU 的完整 `(index, value)` 对 |
| `npu_topk_index_raw.npy` | 算子原始返回、pytest 排序前的 NPU index |
| `npu_topk_value_raw.npy` | 算子原始返回、pytest 排序前的 NPU value |
| `npu_topk_pairs_raw.npy` | pytest 排序前的完整 NPU `(index, value)` 对 |
| `npu_topk_index_sorted.npy` | pytest 按 value 降序重排后的 index |
| `npu_topk_value_sorted.npy` | pytest 重排后的 value |
| `query_raw.npy` / `key_raw.npy` | Q/K 的 packed 原始字节 |
| `query_decoded.npy` / `key_decoded.npy` | CPU 解码后的 Q/K |
| `q_scale_e8m0_raw.npy` / `k_scale_e8m0_raw.npy` | E8M0 原始编码 |
| `q_scale_decoded.npy` / `k_scale_decoded.npy` | CPU 解析后的 scale 数值 |
| `weight.npy` | 目标 query 行、目标 N2 对应的 G 组 weight |
| `debug_metadata.npz` | shape、布局、有效长度、G、量化模式等元数据 |

`return_value=0` 时没有 TopK value 文件是正常现象。

### 5.2 内核 DumpTensor 解析后生成

| 文件 | 含义 |
| --- | --- |
| `npu_score_keys.npy` | TopK 前 NPU score 的 uint16 sortable key |
| `npu_score.npy` | 解码后的完整 NPU score |
| `npu_topk_index_kernel.npy` | kernel TopK 或 LD 合并后的 index |
| `npu_topk_score_kernel.npy` | kernel TopK 或 LD 合并后的 value |
| `npu_query_scale_raw.npy` | NPU GM 中目标 Q 行的 scale 原始字节 |
| `npu_key_scale_raw.npy` | 非 PA 场景 K scale 原始字节 |
| `npu_seqused_q.npy` / `npu_seqused_k.npy` | NPU 实际读取的 seqused |
| `npu_act_s1_size.npy` / `npu_act_s2_size.npy` | NPU 运行时采用的有效 S1/S2 |
| `npu_block_table.npy` | PA 场景目标 batch 的 block table |
| `npu_qk_block*_aic*_s2_*_localrow_*_g_*.npy` | 按核、S2分片、AIV局部行和G组命名的QK数据 |
| `npu_qk_block*_aic*_s2_*_localrow_*_g_*_meta.npy` | 与QK数据一一对应的16字段寻址元数据 |
| `desc_*.npy` | 其他原始记录，包括分 G 的 MXFP4 QK 和 PA scale block |

## 6. 推荐定位顺序

### 阶段一：先检查边界

比较：

- `debug_metadata.npz` 中的 `q_valid_len/k_valid_len/effective_score_len`
- `npu_seqused_q.npy`、`npu_seqused_k.npy`
- `npu_act_s1_size.npy`、`npu_act_s2_size.npy`

若这里不一致，优先定位 seqused、cu_seqlens、cmp_residual_k、cmp_ratio 或 sparse mask 的长度计算，不要继续归因于精度。

### 阶段二：检查 scale 与 QK

先逐字节比较 Q/K scale 的 E8M0 编码，再比较解码值。确认：

- Q scale 对应目标 `N2 * G ... (N2 + 1) * G` 的 Q heads。
- 每 32 个 D 元素对应一个 E8M0 scale。
- ? PA ? NPU K scale dump ???? batch ??? N2?? N2 ?? 1 ?????
  `[K, N2, D/32]` reshape?????? `n2_idx`?
- PA K scale 按 block table 找到的物理 block 顺序正确。
- Q/K 的 scale 分组顺序与 CPU 一致。

MXFP4还可查看`npu_qk_block*_aic*_s2_*_localrow_*_g_*.npy`中按核和G保存的QK中间值。
QK dump只覆盖配置的S2窗口，其descriptor编码为：

```text
data_desc = 10000000 + blockId * 100000 + qkDumpBegin + gIdx
meta_desc = 100000000 + blockId * 100000 + qkDumpBegin + gIdx
```

16个uint32元数据字段依次为：

```text
version, blockId, aiCoreIdx, bIdx, n2Idx, curS2Idx,
curAivS1Idx, localRow, gIdx, qkDumpBegin, qkDumpLen,
qkBlockOffset, pingpong, qkGroupStride, qkRowStride, targetS1Idx
```

### 阶段三：逐元素比较完整 score

```python
import numpy as np

cpu = np.load("qliv2_debug_seed3_b0_n20_s14/cpu_score_effective.npy").astype(np.float32)
npu = np.load("qliv2_npu_dump/npu_score.npy").astype(np.float32)

length = min(cpu.size, npu.size)
diff_idx = np.flatnonzero(cpu[:length] != npu[:length])
print("shape:", cpu.shape, npu.shape)
print("mismatch_count:", diff_idx.size)
print("first_mismatch:", diff_idx[:32])
```

重点打印 watch index：

```python
for idx in (2492, 2499):
    print(idx, cpu[idx], npu[idx], npu[idx] - cpu[idx])
```

判断：

- 如果目标 index 的 NPU score 已经与 CPU 不同，问题位于 scale/QK/weighted-reduce/score 搬运链路。
- 如果完整 score 一致，但 NPU TopK 集合错误，问题位于 TopK、LD 合并或 index offset。

### 阶段四：区分 kernel TopK 与输出后处理

依次比较：

1. `npu_topk_index_kernel.npy` 与从 `npu_score.npy` 重算出的 TopK。
2. `npu_topk_index_kernel.npy` 与 `npu_topk_index_raw.npy`。
3. `npu_topk_pairs_raw.npy` 与 `npu_topk_pairs_sorted.npy`。

结论：

- kernel TopK 已错：检查 TopK 输入长度、sortable key、分段 TopK 和 LD 合并。
- kernel TopK 正确但 raw 输出错：检查 GM 搬出、output offset 和输出布局。
- raw 输出正确但 sorted 输出错：检查 pytest 的 value 排序和 gather 维度。
- index 正确但旁边 value 不属于该 index：检查 value/index 是否在同一次排序或搬运中保持配对。

## 7. PA_BBND 特别检查

PA 场景至少核对：

1. CPU 的 `pa_physical_block_ids.npy`。
2. NPU 的 `npu_block_table.npy`。
3. descriptor `520000 + logical_block` 对应的 K scale 物理块。
4. `keyDequantScaleStride0` 是否按 E8M0 元素寻址且满足打包对齐。

不要直接按逻辑 K token 将 PA scale 当作连续 GM 数据。

## 8. 常见误区

- Python target 已修改，但 NPU 宏仍是默认值。
- 修改宏后没有重新编译算子。
- 使用同一个 dump 目录连续运行多个 Case，文件被覆盖。
- 把 `npu_topk_value_sorted.npy` 当作算子原始输出。
- 直接把 uint16 sortable key 当成 BF16 bit pattern。
- sparse_mode=3 时用完整 K 长度比较 NPU 的有效 score 前缀。
- QK 关注 index 超出 `QLIV2_DEBUG_QK_S2_BEGIN/END`。
- DumpTensor 出现截断 warning 后仍把缺失数据当作真实结果。
- 只看最终 fulfill_percent，不先比较同一 index 的完整 score。

## 9. 提交给分析模型的信息

至少提供：

```text
case 名称：
随机 seed：
quant_mode：
layout_query/layout_key：
b/n2/s1：
q_valid_len/k_valid_len/effective_score_len：
sparse_count/sparse_mode/cmp_ratio/return_value：
CPU/NPU score shape：
score mismatch_count：
首批 mismatch index：
watch index 的 CPU/NPU score：
kernel TopK 是否包含 watch index：
raw TopK 是否包含 watch index：
scale raw 是否逐字节一致：
DumpTensor 是否出现截断 warning：
```

推荐同时提供：

- `debug_metadata.npz`
- `cpu_score_effective.npy`
- `npu_score.npy`
- CPU/NPU TopK pairs
- Q/K scale raw 与 decoded 文件
- `qliv2_dump.log`

拿到这些数据后，应先给出“边界、score 计算、TopK、输出后处理”四个阶段各自是否通过，再提出修改方案。
