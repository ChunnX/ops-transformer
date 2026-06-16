# mamba2_rmsnormgated算子说明

### 功能和实现说明

基于状态空间模型（SSM）的因果卷积，实现MambaV2 Prefill阶段的因果卷积计算。计算流程包含kernel_size=4的depthwise conv1d和SiLU激活。本算子采用纯Vector实现conv1d，并融合bias和SiLU运算以提升性能。

**计算流**  

<img src="https://raw.gitcode.com/user-images/assets/7673863/f90c8afa-f740-40c3-a2d4-f470a301b60a/image.png" height="300">

### 自定义Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| x   | BSD   | FP32   |
| w   | D   | FP32   |
| z   | BSD    | FP32   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| out   | BSD   | FP32   |

**参数说明：**  
B: batch size  
S: sequence len   
D: dimension
额外需要参数
G: ngroups
E: eps

**调用方式**

```
import npu_ops_transformer_ext

out = torch.ops.npu_ops_transformer_ext.mamba2_rmsnormgated(x, z, w, G, E)
```

**测试方法**

见当前目录tests/

```
python test_rmsnormgated.py
```
