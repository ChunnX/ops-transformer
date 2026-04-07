# TransQuantParamV2

## 功能

将 FLOAT32 的 `scale`（及可选的 `offset`）打包为 UINT64 量化参数（与 AscendC 实现中 `round_mode==0` 分支一致）。

## 输入输出

| 名称 | 说明 |
|------|------|
| `scale` | FLOAT32，1D；或符合 infershape 约束的 `(1,n)` 等形状 |
| `offset` | 可选，FLOAT32，与 `scale` 可广播/对齐的形状 |
| `y` | UINT64，与 infershape 推断的输出形状一致 |

## 属性

- `round_mode`：图模式下可选；本仓库 **aclnnTransQuantParamV2** 封装内部将 **roundMode 固定为 0**，validation 仅覆盖该路径。

## Python 示例（编译并安装自定义 OPP 后）

```python
import torch
import kernel_gen_ops

out = kernel_gen_ops.trans_quant_param_v2(scale, offset)  # offset 可为 None
```
