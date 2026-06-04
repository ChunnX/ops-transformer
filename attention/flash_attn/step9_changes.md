# Step 9: 热路径对象构造优化

## 优化目标
消除 ProcessVec2DSplit 和 ProcessVec2_C2GmUpdateUb 热路径循环内每轮迭代构造 DataCopyParams/DataCopyPadParams 对象的开销。

## 问题分析

### 原始代码问题
在两个热路径函数的 for 循环内部，每轮迭代都构造 DataCopyParams 和 DataCopyPadParams 对象：

```cpp
for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    // ... 其他代码 ...
    
    DataCopyParams dataCopyParams;      // 每轮构造
    DataCopyPadParams dataCopyPadParams; // 每轮构造
    dataCopyParams.blockCount = subMRealsize;
    dataCopyParams.dstStride = ...;     // 常量
    dataCopyParams.srcStride = ...;     // 常量
    dataCopyParams.blockLen = ...;      // 常量
    
    DataCopyPad(..., dataCopyParams, dataCopyPadParams);
}
```

**性能问题：**
- 对象构造展开后是大量初始化 Store 指令
- dstStride、srcStride、blockLen 在循环中是常量，但每轮都重新赋值
- 每轮迭代产生 ~10+ 个不必要的 Store 指令

## 优化方案

### 核心思路
将对象构造和常量字段初始化移到循环外，循环内仅修改变化字段（blockCount）。

### 修改内容

#### 1. ProcessVec2DSplit 函数优化

**优化前：**
```cpp
for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    // ... 计算 subMRealsize ...
    
    if (constInfo.dSizeV == dTemplateAlign64) {
        DataCopy(...);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = subMRealsize;
        dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
        dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        DataCopyPad(..., dataCopyParams, dataCopyPadParams);
    }
}
```

**优化后：**
```cpp
DataCopyParams dataCopyParams;
DataCopyPadParams dataCopyPadParams;
if (constInfo.dSizeV != dTemplateAlign64) {
    dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
    dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
}

for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    // ... 计算 subMRealsize ...
    
    if (constInfo.dSizeV == dTemplateAlign64) {
        DataCopy(...);
    } else {
        dataCopyParams.blockCount = subMRealsize;  // 仅修改变化字段
        DataCopyPad(..., dataCopyParams, dataCopyPadParams);
    }
}
```

#### 2. ProcessVec2_C2GmUpdateUb 函数优化

**优化前：**
```cpp
for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    // ... 计算 subMRealsize ...
    
    if (constInfo.dSizeV == dTemplateAlign64) {
        DataCopy(...);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = subMRealsize;
        dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        DataCopyPad(..., dataCopyParams, dataCopyPadParams);
    }
}
```

**优化后：**
```cpp
DataCopyParams dataCopyParams;
DataCopyPadParams dataCopyPadParams;
if (constInfo.dSizeV != dTemplateAlign64) {
    dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
    dataCopyParams.srcStride = 0;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
}

for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    // ... 计算 subMRealsize ...
    
    if (constInfo.dSizeV == dTemplateAlign64) {
        DataCopy(...);
    } else {
        dataCopyParams.blockCount = subMRealsize;  // 仅修改变化字段
        DataCopyPad(..., dataCopyParams, dataCopyPadParams);
    }
}
```

## 优化效果

### 性能提升
- **消除每轮迭代的对象构造开销**：减少 ~10+ 个 Store 指令/轮
- **减少冗余赋值**：常量字段只初始化一次
- **提升循环体执行效率**：循环内代码更精简

### 代码变更统计
- 1 file changed, 16 insertions(+), 10 deletions(-)

## 技术细节

### 为什么有效
1. **对象构造外提**：避免每轮迭代的构造函数调用和初始化 Store
2. **常量字段预初始化**：dstStride、srcStride、blockLen 在循环中不变，只需设置一次
3. **变化字段延迟赋值**：blockCount 依赖 subMRealsize，必须在循环内更新

### 条件判断优化
使用 `if (constInfo.dSizeV != dTemplateAlign64)` 条件判断，只在需要时才初始化参数对象，避免不必要的开销。

## 验证
- Commit: 925449b97
- Branch: flash_attn
- 已推送到远程仓库

## 注意事项
1. 此优化遵循 P9 原则：Hot Loop 内不构造对象
2. 与 Step 3（Buffer 索引优化）和 Step 8（Event ID 拆标量）共同减少热路径开销
3. 优化后代码更易维护，逻辑更清晰
