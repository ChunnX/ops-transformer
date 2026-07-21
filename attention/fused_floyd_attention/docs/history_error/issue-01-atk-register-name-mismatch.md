# ATK Executor 注册名不匹配

## 现象

```
KeyError: 'ascend_function_ffa'
```

运行 ATK 测试时，两个用例全部失败，报找不到 `ascend_function_ffa`。

## 根因

yaml/json 中 `api_type: ascend_function_ffa`，但 `run.sh` 指定的
`executor_aclnnFusedFloydAttention_new_golden.py:47` 注册的是：

```python
@register("execute_aclnnFusedFloydAttention")   # ❌ 名字对不上
```

正确注册名（`ascend_function_ffa`）其实在另一个未被引用的文件
`executor_aclnnFusedFloydAttention.py:50` 里。

## 修复

**文件**：`executor_aclnnFusedFloydAttention_new_golden.py:47`

```python
# 改前
@register("execute_aclnnFusedFloydAttention")
# 改后
@register("ascend_function_ffa")
```

## 次要问题（修完上面会触发）

1. **`import struct` 缺失**（new_golden.py:31 用了 `struct` 但未 import）→ 补 `import struct`。
2. **基类警告**：new_golden 继承 `BaseApi`，ATK 期望 `AclnnBaseApi`；当前 cpu 标杆模式下非阻塞，后续对齐。
