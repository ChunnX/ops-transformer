# API 文档模板

`question/api_desc.md` 使用中文：

````markdown
# aclnn<OpName>

## 功能描述

说明算子的用户可见功能。

## 接口定义

```python
kernel_gen_ops.<kernel_name>(...)
```

## 参数说明

列出输入张量、属性参数和返回值。

## 使用示例

```python
import torch
import kernel_gen_ops

# 构造输入并调用 kernel_gen_ops.<kernel_name>(...)
```

## 约束与限制

- 支持 dtype
- 支持 format/layout
- shape、index、attr 的合法范围
````
