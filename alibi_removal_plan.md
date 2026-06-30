# Alibi 特性下线及函数删除修改计划

## 1. 背景说明
当前策略要求下线 Alibi 特性，并删除相关函数 `GetActualSeqLengthsQData` 和 `GetActualSeqLengthsKvData`。

## 2. 代码依赖分析
在 `actual_seq_len_checker.cpp` 文件中：
- `CheckFeatureAlibi` 函数调用了 `GetActualSeqLengthsQData` 和 `GetActualSeqLengthsKvData`。
- **`CheckFeatureFP8GQAFullquant` 函数也调用了这两个函数**（用于校验 FP8 GQA 全量化场景下的序列长度范围）。

因此，直接删除这两个 `Get` 函数会导致 `CheckFeatureFP8GQAFullquant` 编译失败。

## 3. 修改方案

### 步骤 1: 删除 `CheckFeatureAlibi` 函数
完全删除 `CheckFeatureAlibi` 函数的定义（约在文件第 311-339 行）。

### 步骤 2: 移除 `CheckCrossFeature` 中的调用
在 `CheckCrossFeature` 函数中，移除对 `CheckFeatureAlibi` 的调用逻辑：
```cpp
// 修改前
ge::graphStatus ActualSeqLenChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAlibi(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (enableFullQuant_) {
        // ...
    }
    return ge::GRAPH_SUCCESS;
}

// 修改后
ge::graphStatus ActualSeqLenChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        // ...
    }
    return ge::GRAPH_SUCCESS;
}
```

### 步骤 3: 处理 `CheckFeatureFP8GQAFullquant` 的依赖 (关键)
由于 `CheckFeatureFP8GQAFullquant` 仍在使用这两个函数，我们有两个选择：

- **方案 A (推荐)**: **保留** `GetActualSeqLengthsQData` 和 `GetActualSeqLengthsKvData` 函数，仅删除 `CheckFeatureAlibi`。这样改动最小且安全。
- **方案 B**: 如果强制要求删除这两个 `Get` 函数，则需要将 `CheckFeatureFP8GQAFullquant` 中获取序列长度的逻辑**内联**，重新实现这两个 `Get` 函数的内部逻辑。

## 4. 待确认事项
请确认是否接受**方案 A**（仅删除 Alibi 相关调用，保留 Get 函数供 FP8 使用）？还是必须采用**方案 B**彻底删除 Get 函数？