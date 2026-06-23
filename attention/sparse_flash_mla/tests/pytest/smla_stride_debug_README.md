# SMLA Stride Debug

本文档用于说明如何使用`smla_stride_debug.py`确认`SparseFlashMla`是否命中`GetOptionalInputStride`获取KV stride0的路径。

## 使用场景

脚本构造BSND + SCFA场景，使用0轴带padding的`ori_kv`和`cmp_kv`非连续视图调用`SparseFlashMla`，再使用对应的contiguous拷贝调用一次，最后比较两次输出。

如果算子正确使用输入stride0，非连续KV视图和contiguous拷贝的输出应一致；如果未命中stride0路径，非连续KV视图可能按连续内存偏移读取KV，输出会和contiguous拷贝不一致。

## 一键执行

在已安装包含`SparseFlashMla`、`SparseFlashMlaMetadata`以及PyTorch API的CANN包后，进入仓库根目录执行：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py --device-id 0 --launch-blocking
```

如需指定设备卡号，修改`--device-id`：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py --device-id 1 --launch-blocking
```

如果需要在源码仓内临时注册`torch_extension`，可显式传入`--torch-extension-path`：

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
  --torch-extension-path torch_extension \
  --launch-blocking
```

## 关键输出

命中新stride路径时，重点观察以下关键字：

```text
contiguous=False
expected ori_kv stride0=
expected cmp_kv stride0=
[compare] scfa: finite=True
[result] PASS: non-contiguous KV view outputs match contiguous KV outputs.
```

其中`expected ori_kv stride0`和`expected cmp_kv stride0`中，非连续视图的stride0应大于对应contiguous stride0，例如：

```text
[stride-check] expected ori_kv stride0=86528, contiguous stride0=84992
[stride-check] expected cmp_kv stride0=22528, contiguous stride0=20992
```

未命中stride路径或结果异常时，常见关键字包括：

```text
differs from contiguous output
max_abs=
Segmentation fault
RuntimeError
```

## 常用参数

```bash
python -u attention/sparse_flash_mla/tests/pytest/smla_stride_debug.py \
  --device-id 0 \
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

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device-id` | `0` | NPU设备号。 |
| `--batch` | `2` | batch维度，必须大于等于2才能验证0轴stride。 |
| `--q-len` | `38` | Q序列长度。 |
| `--ori-len` | `166` | ori_kv序列长度。 |
| `--cmp-len` | `41` | cmp_kv序列长度。 |
| `--topk` | `512` | cmp_sparse_indices的topk。 |
| `--cmp-ratio` | `4` | SCFA压缩比例。 |
| `--kv-pad` | `3` | 在KV的S轴增加padding，用于构造0轴非连续视图。 |
| `--torch-extension-path` | 不传 | 指定源码仓内`torch_extension`路径；不传时使用已安装的`cann_ops_transformer`包。 |
| `--launch-blocking` | 不启用 | 启用同步执行，便于定位报错位置。 |

## 注意事项

- 默认`--batch=2`用于验证0轴stride；`--batch=1`时即使底层storage带padding，PyTorch也可能认为view是contiguous，不能用于确认0轴stride路径。
- 默认不传`--torch-extension-path`，适用于已安装内部CANN包的场景。
- 若提示找不到`cann_ops_transformer`，说明当前Python环境未安装PyTorch API包，需先切换到包含该包的环境，或使用`--torch-extension-path`临时注册源码接口。
