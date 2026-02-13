# QuantizeAddLayerNorm

## 功能描述
计算 (x1 + x2 + bias)，再做 LayerNorm，最后量化输出为 int8；可选输出 x = round(x1 + x2 + bias) 的浮点结果。

## 输入
- **x1**, **x2**: 输入张量，支持 float16 / bfloat16 / float32
- **gamma**, **beta**, **bias**: LayerNorm 的 gamma、beta 与加性 bias，与 x1/x2 最后一维同形状
- **scales**: 量化 scale，与 normalized_shape 一致
- **zero_points** (可选): 量化零点

## 输出
- **y**: int8 量化结果，与 x1 同 shape
- **x**: 浮点结果 round(x1 + x2 + bias)，与 x1 同 shape（始终输出）

## 属性
- **dtype**: 量化输出数据类型，固定 DT_INT8
- **axis**: 归一化轴，默认 -1；-65535 表示 mul_scale 模式 (y = x_norm * scale + zp)，否则为 div 模式 (y = x_norm / scale + zp)
- **epsilon**: LayerNorm 的 epsilon，默认 1e-5
- **additional_output**: 是否输出浮点 x，默认 false（kernel 始终写 x）

## 参考
迁移自 CANN ops-nn/norm/quantize_add_layer_norm。
