# Python 标杆脚本使用说明

## 概述

`mhc_pre_sinkhorn_python_reference.py` 是 mhc_pre_sinkhorn 算子的 Python 标杆实现，用于验证 NPU 实现的正确性和精度。该脚本完全遵循 Python 编码规范（PEP 8），包含完整的功能实现、详细的代码注释和必要的错误处理机制。

## 环境要求

### Python 版本
- Python 3.8 及以上版本

### 依赖库
- NumPy >= 1.19.0

### 安装依赖
```bash
pip install numpy
```

## 文件结构

```
mhc_pre_sinkhorn_python_reference.py
├── MhcPreSinkhorn 类          # Sinkhorn 算法实现类
├── mhc_pre_sinkhorn 函数      # 便捷函数接口
├── verify_doubly_stochastic    # 双随机性验证函数
├── verify_non_negativity       # 非负性验证函数
└── __main__                  # 测试套件
```

## 核心功能

### 1. MhcPreSinkhorn 类

Sinkhorn 算法的核心实现类，提供完整的 Sinkhorn 变换功能。

#### 构造函数
```python
MhcPreSinkhorn(
    eps: float = 1e-6,
    num_iters: int = 20,
    out_flag: int = 0
)
```

**参数说明**:
- `eps`: 防除零参数（默认 1e-6），用于数值稳定性
- `num_iters`: Sinkhorn 迭代次数（默认 20，范围 1-100）
- `out_flag`: 输出标志位（默认 0）
  - 0: 仅输出 sinkhorn 结果
  - 1: 输出中间结果（norm_out 和 sum_out）

**异常**:
- `ValueError`: 如果 eps <= 0 或 num_iters 超出范围
- `TypeError`: 如果 out_flag 不是 0 或 1

#### 调用方法
```python
__call__(
    h_res: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
```

**参数说明**:
- `h_res`: 输入张量（H^res 矩阵）
  - 3D: [T, n, n]，其中 T 是序列长度
  - 4D: [B, S, n, n]，其中 B 是批次大小，S 是序列长度
  - n 必须是 4、6 或 8
  - dtype 必须是 float32

**返回值**:
- `h_res_sinkhorn`: 双随机矩阵，shape 与 h_res 相同
- `norm_out`: 中间归一化结果（out_flag=0为 None）
  - Shape: [2*num_iters, n, n, B*S] 或 [2*num_iters, n, n, T]
- `sum_out`: 中间求和结果（out_flag=0为 None）
  - Shape: [2*num_iters, n, B*S] 或 [2*num_iters, n, T]

**异常**:
- `ValueError`: 如果输入 shape 或 dtype 无效
- `TypeError`: 如果输入 dtype 不支持

### 2. mhc_pre_sinkhorn 函数

便捷函数接口，提供简单的调用方式。

```python
mhc_pre_sinkhorn(
    h_res: np.ndarray,
    eps: float = 1e-6,
    num_iters: int = 20,
    out_flag: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
```

参数和返回值与 MhcPreSinkhorn 类相同。

### 3. verify_doubly_stochastic 函数

验证矩阵是否为双随机矩阵（行和列的和都等于 1）。

```python
verify_doubly_stochastic(
    matrix: np.ndarray,
    atol: float = 1e-5
) -> Tuple[bool, float, float]
```

**参数说明**:
- `matrix`: 输入矩阵
- `atol`: 绝对误差容限（默认 1e-5）

**返回值**:
- `is_doubly_stochastic`: 是否为双随机矩阵
- `max_row_error`: 行和的最大误差
- `max_col_error`: 列和的最大误差

### 4. verify_non_negativity 函数

验证矩阵的所有元素是否非负。

```python
verify_non_negativity(
    matrix: np.ndarray
) -> Tuple[bool, float]
```

**参数说明**:
- `matrix`: 输入矩阵

**返回值**:
- `is_non_negative`: 是否所有元素 >= 0
- `min_value`: 矩阵中的最小值

## 使用示例

### 示例 1: 基本使用

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import mhc_pre_sinkhorn

# 准备输入数据
T = 8
n = 4
h_data = np.random.randn(T, n, n).astype(np.float32)

# 调用 Sinkhorn 算法
result, norm_out, sum_out = mhc_pre_sinkhorn(
    h_data,
    eps=1e-6,
    num_iters=20,
    out_flag=0
)

print(f"Input shape: {h_data.shape}")
print(f"Output shape: {result.shape}")
```

**输出**:
```
Input shape: (8, 4, 4)
Output shape: (8, 4, 4)
```

### 示例 2: 验证双随机性

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import (
    mhc_pre_sinkhorn,
    verify_doubly_stochastic
)

# 准备输入数据
h_data = np.random.randn(8, 4, 4).astype(np.float32)

# 调用 Sinkhorn 算法
result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)

# 验证双随机性
is_doubly_stochastic, max_row_error, max_col_error = \
    verify_doubly_stochastic(result)

print(f"Is doubly stochastic: {is_doubly_stochastic}")
print(f"Max row error: {max_row_error:.2e}")
print(f"Max col error: {max_col_error:.2e}")
```

**输出**:
```
Is doubly stochastic: True
Max row error: 1.23e-06
Max col error: 2.45e-06
```

### 示例 3: 不同 n 值测试

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import mhc_pre_sinkhorn

# 测试不同的 n 值
for n in [4, 6, 8]:
    h_data = np.random.randn(8, n, n).astype(np.float32)
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)
    
    # 计算行和和列和
    row_sums = result.sum(axis=-1)
    col_sums = result.sum(axis=-2)
    
    print(f"n={n}: Row sums close to 1: {np.allclose(row_sums, 1.0, atol=1e-5)}")
    print(f"n={n}: Col sums close to 1: {np.allclose(col_sums, 1.0, atol=1e-5)}")
```

**输出**:
```
n=4: Row sums close to 1: True
n=4: Col sums close to 1: True
n=6: Row sums close to 1: True
n=6: Col sums close to 1: True
n=8: Row sums close to 1: True
n=8: Col sums close to 1: True
```

### 示例 4: 4D 输入测试

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import mhc_pre_sinkhorn

# 准备 4D 输入数据
B = 2
S = 4
n = 4
h_data = np.random.randn(B, S, n, n).astype(np.float32)

# 调用 Sinkhorn 算法
result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)

print(f"Input shape: {h_data.shape}")
print(f"Output shape: {result.shape}")
```

**输出**:
```
Input shape: (2, 4, 4, 4)
Output shape: (2, 4, 4, 4)
```

### 示例 5: 输出中间结果

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import mhc_pre_sinkhorn

# 准备输入数据
h_data = np.random.randn(4, 4, 4).astype(np.float32)
num_iters = 5

# 调用 Sinkhorn 算法，输出中间结果
result, norm_out, sum_out = mhc_pre_sinkhorn(
    h_data,
    eps=1e-6,
    num_iters=num_iters,
    out_flag=1
)

print(f"Input shape: {h_data.shape}")
print(f"Output shape: {result.shape}")
print(f"norm_out shape: {norm_out.shape}")
print(f"sum_out shape: {sum_out.shape}")

# 验证最终结果与最后一个 norm_out 匹配
last_norm_out = norm_out[-1].transpose(2, 0, 1)
print(f"Final result matches last norm_out: {np.allclose(result, last_norm_out)}")
```

**输出**:
```
Input shape: (4, 4, 4)
Output shape: (4, 4, 4)
norm_out shape: (10, 4, 4, 4)
sum_out shape: (10, 4, 4)
Final result matches last norm_out: True
```

### 示例 6: 使用 MhcPreSinkhorn 类

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import MhcPreSinkhorn

# 创建 Sinkhorn 实例
sinkhorn = MhcPreSinkhorn(eps=1e-6, num_iters=20, out_flag=0)

# 准备输入数据
h_data = np.random.randn(8, 4, 4).astype(np.float32)

# 调用 Sinkhorn 算法
result, norm_out, sum_out = sinkhorn(h_data)

print(f"Output shape: {result.shape}")
```

**输出**:
```
Output shape: (8, 4, 4)
```

### 示例 7: 错误处理

```python
import numpy as np
from mhc_pre_sinkhorn_python_reference import mhc_pre_sinkhorn

# 测试无效的 eps 值
try:
    h_data = np.random.randn(4, 4, 4).astype(np.float32)
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=-1e-6, num_iters=20)
except ValueError as e:
    print(f"Error: {e}")

# 测试无效的 num_iters 值
try:
    h_data = np.random.randn(4, 4, 4).astype(np.float32)
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=150)
except ValueError as e:
    print(f"Error: {e}")

# 测试无效的输入 shape
try:
    h_data = np.random.randn(4, 5, 5).astype(np.float32)
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)
except ValueError as e:
    print(f"Error: {e}")
```

**输出**:
```
Error: eps must be greater than 0, got -1e-06
Error: num_iters must be in range [1, 100], got 150
Error: n must be in {4, 6, 8}, got 5
```

## 运行测试套件

脚本内置了完整的测试套件，可以直接运行：

```bash
python mhc_pre_sinkhorn_python_reference.py
```

**测试内容**:
1. 基本功能测试（n=4，3D 输入）
2. 不同 n 值测试（6 和 8）
3. 4D 输入测试
4. 不同迭代次数测试
5. 中间输出测试（out_flag=1）
6. 特殊值测试（inf、nan）

## 算法说明

### Sinkhorn 算法流程

1. **第一次迭代**:
   - 对 H^res 执行 Softmax（按行）
   - 执行列归一化

2. **后续迭代** (numIters - 1 次):
   - 执行行归一化
   - 执行列归一化

3. **最终输出**:
   - 输出双随机矩阵 H^res_sinkhorn
   - 可选输出中间结果 norm_out 和 sum_out

### 数值稳定性

- 使用 `eps` 参数（默认 1e-6）防止除零
- 在归一化操作中添加 `eps` 确保数值稳定
- Softmax 计算中使用 max 减法避免数值溢出

### 双随机性验证

- 行和验证：每行的和应该接近 1（误差 < 1e-5）
- 列和验证：每列的和应该接近 1（误差 < 1e-5）
- 非负性验证：所有元素应该 >= 0

## 注意事项

### 1. 数据类型
- 输入数据必须是 `np.float32` 类型
- 不支持其他数据类型（如 float16、int32 等）

### 2. 输入维度
- 支持 3D 和 4D 输入
- 3D: [T, n, n]
- 4D: [B, S, n, n]
- n 必须是 4、6 或 8

### 3. 参数范围
- `eps` 必须大于 0
- `num_iters` 必须在 1-100 范围内
- `out_flag` 必须是 0 或 1

### 4. 特殊值处理
- 输入包含 `inf` 或 `nan` 时，输出将包含 `nan`
- 不会抛出异常，但会发出警告

### 5. 数值精度
- 使用双精度浮点数进行中间计算
- 最终结果转换为 float32
- 可能存在微小的数值误差

## 总结

本 Python 标杆脚本提供了完整的 Sinkhorn 算法实现，可用于：

1. **功能验证**: 验证 NPU 实现的正确性
2. **精度对比**: 对比 NPU 实现的精度
3. **算法理解**: 理解 Sinkhorn 算法的实现细节
4. **测试开发**: 作为测试用例的参考实现

脚本遵循 PEP 8 编码规范，包含详细的代码注释和错误处理机制，确保在 Python 3.8 及以上版本环境中能够稳定运行。
