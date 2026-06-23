# SMLA Stride Debug

本文档用于说明如何使用`smla_stride_debug.py`确认`SparseFlashMla`是否命中`GetOptionalInputStride`获取KV stride0的路径。

## 使用场景

脚本默认优先验证TND + SCFA场景，构造0轴非连续的`ori_kv`和`cmp_kv`视图调用`SparseFlashMla`，再使用对应的contiguous拷贝调用一次，最后比较两次输出。

TND格式下，KV输入为`[T, N2, D]`。脚本会在N轴扩展底层storage，再切回原始`N2`，使KV的T轴stride0大于contiguous布局下的stride0，从而验证算子是否真正读取并使用输入stride0。

如果算子正确使用输入stride0，非连续KV视图和contiguous拷贝的输出应一致；如果未命中stride0路径，非连续KV视图可能按连续内存偏移读取KV，输出会和contiguous拷贝不一致。

## 一键执行

在已安装包含`SparseFlashMla`、`SparseFlashMlaMetadata`以及PyTorch API的CANN包后，进入仓库根目录执行：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py --device-id 0 --launch-blocking
```

默认等价于：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
  --layout tnd \
  --launch-blocking
```

如需指定设备卡号，修改`--device-id`：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py --device-id 1 --launch-blocking
```

如果需要在源码仓内临时注册`torch_extension`，可显式传入`--torch-extension-path`：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
  --layout tnd \
  --torch-extension-path torch_extension \
  --launch-blocking
```

## 关键输出

命中新stride路径时，重点观察以下关键字：

```text
layout=TND
contiguous=False
expected ori_kv stride0=
expected cmp_kv stride0=
[compare] scfa: finite=True
[result] PASS: non-contiguous KV view outputs match contiguous KV outputs.
```

其中`expected ori_kv stride0`和`expected cmp_kv stride0`中，非连续视图的stride0应大于对应contiguous stride0，例如：

```text
[stride-check] expected ori_kv stride0=2048, contiguous stride0=512
[stride-check] expected cmp_kv stride0=2048, contiguous stride0=512
```

未命中stride路径或结果异常时，常见关键字包括：

```text
differs from contiguous output
max_abs=
Segmentation fault
RuntimeError
```

## 常用参数

TND优先验证命令：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
  --layout tnd \
  --batch 2 \
  --q-len 38 \
  --ori-len 166 \
  --cmp-len 41 \
  --topk 512 \
  --cmp-ratio 4 \
  --num-heads-q 64 \
  --num-heads-kv 1 \
  --head-dim 512 \
  --kv-pad 3 \
  --launch-blocking
```

如需回归BSND场景，显式指定`--layout bsnd`：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
  --layout bsnd \
  --batch 2 \
  --launch-blocking
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device-id` | `0` | NPU设备号。 |
| `--layout` | `tnd` | 调测布局。默认优先验证TND；可选`tnd`或`bsnd`。 |
| `--batch` | `2` | batch维度。TND下用于生成多段cu_seqlens；BSND下必须大于等于2才能验证0轴stride。 |
| `--q-len` | `38` | 每个batch的Q序列长度。 |
| `--ori-len` | `166` | 每个batch的ori_kv序列长度。 |
| `--cmp-len` | `41` | 每个batch的cmp_kv序列长度。 |
| `--topk` | `512` | cmp_sparse_indices的topk。 |
| `--cmp-ratio` | `4` | SCFA压缩比例。 |
| `--kv-pad` | `3` | TND下在KV的N轴增加padding；BSND下在KV的S轴增加padding，用于构造0轴非连续视图。 |
| `--torch-extension-path` | 不传 | 指定源码仓内`torch_extension`路径；不传时使用已安装的`cann_ops_transformer`包。 |
| `--launch-blocking` | 不启用 | 启用同步执行，便于定位报错位置。 |

## 注意事项

- 默认`--layout tnd`，优先验证TND格式。
- TND下脚本会传入`cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`，并同时打印这些输入。
- TND下即使`batch=1`也可以通过T轴stride0验证非连续KV；BSND下默认`--batch=2`用于验证0轴stride。
- 默认不传`--torch-extension-path`，适用于已安装内部CANN包的场景。
- 若提示找不到`cann_ops_transformer`，说明当前Python环境未安装PyTorch API包，需先切换到包含该包的环境，或使用`--torch-extension-path`临时注册源码接口。
