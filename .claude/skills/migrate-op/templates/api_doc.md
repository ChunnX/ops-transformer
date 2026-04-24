# API 文档模板

api_desc.md 使用中文，严格遵循以下结构：

```markdown
# aclnn<OpName>

## 功能描述

### 算子功能
（一句话描述算子功能）

### 计算公式
$$公式$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.<opname>()` 函数形式提供：

​```python
def <opname>(<params>):
    """..."""
​```

## 使用案例
​```python
import torch
import kernel_gen_ops
# 完整可运行的调用示例
​```

## 约束与限制
- 支持的数据类型
- 支持的数据格式
- 其他算子特有约束
```
