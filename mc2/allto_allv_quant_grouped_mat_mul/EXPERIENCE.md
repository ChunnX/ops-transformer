# MC2算子开发经验总结

## 问题解决记录

### 1. 栈变量失效问题

**问题描述**：
`GET_TILING_DATA(tilingData, tilingGM)` 创建栈变量，`scheduler.Init()` 返回后栈内存失效，所有指向栈变量的指针变成悬垂指针。

**影响场景**：
- 非MX场景：`LaunchCommBeforeCompute` 中使用 `sendCnt` 栈变量指针，导致通信参数错误
- MX场景：`LaunchScaleBeforeCompute` 中使用 `sendCnt` 栈变量指针
- PermuteScale中使用 `taskTilingInfo_->ubSize`，导致数据拷贝长度错误

**解决方案**：
在 `Init` 方法中拷贝需要跨函数使用的数据到成员变量数组：

```cpp
// hccl_a2av_op.h Init方法中
for (int i = 0; i < e_ * rankDim_; i++) {
    recvCounts_[i] = taskTilingInfo_->recvCnt[i];
    sendCounts_[i] = taskTilingInfo_->sendCnt[i];
}
ubSize_ = taskTilingInfo_->ubSize;

// 成员变量定义
uint64_t recvCounts_[MAX_EXPERT_SIZE] = {0UL};
uint64_t sendCounts_[MAX_EXPERT_SIZE] = {0UL};
uint64_t ubSize_ = 0UL;
```

**经验教训**：
- 凡是需要在 `Process()` 中使用的数据，必须在 `Init` 中拷贝到成员变量
- 不要保存指向栈变量的指针，要拷贝数据本身

### 2. workspace重复偏移问题

**问题描述**：
scheduler 已传入偏移后的 workspace (`computeWorkspaceGm_ = workspaceGM + workspaceOffset`)，compute类又加偏移：
```cpp
groupListGm_ = isA2avGmmFlag ? workspaceGM_ + CalcGroupListOffset(...) : workspaceGM_;
```

**影响场景**：
共享专家计算时 workspace 越界崩溃

**解决方案**：
compute类直接使用传入的 workspace，不加额外偏移：
```cpp
groupListGm_ = workspaceGM_;  // scheduler已传入偏移后的地址
```

删除无用的 `CalcGroupListOffset` 方法。

**经验教训**：
- scheduler 采用 workspaceOffset 方式管理布局后，compute类不应该再加偏移
- 子模块拿到的 workspace 已经是偏移后的，直接使用即可

### 3. DataCopyPad API用法

**问题描述**：
PermuteScale 中 DataCopyPad 参数不正确导致编译失败。

**正确用法**（参考项目其他算子）：

| 操作 | 参数数量 | API签名 |
|------|---------|---------|
| GM→UB | 4参数 | `DataCopyPad(LocalTensor, GlobalTensor, DataCopyParams, DataCopyPadParams)` |
| UB→GM | 3参数 | `DataCopyPad(GlobalTensor, LocalTensor, DataCopyParams)` |

**示例代码**：
```cpp
DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLength), 0U, 0U};
DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};

// GM->UB
DataCopyPad(scaleUbTensor, globalBuffer_[offset], dataCopyParams, dataCopyPadParams);

// UB->GM
DataCopyPad(globalBuffer_[offset], scaleUbTensor, dataCopyParams);
```

### 4. PermuteScale偏移计算

**问题描述**：
srcOffset 和 dstOffset 没有乘 axis，导致偏移是 token 数而不是字节偏移。

**解决方案**：
```cpp
uint64_t srcOffset = (posBefore == 0U) ? 0UL : recvSumCntBefore[posBefore - 1U] * axis;
uint64_t dstOffset = (posAfter == 0U) ? 0UL : recvSumCntAfter[posAfter - 1U] * axis;
```

**经验教训**：
- 偏移计算要明确单位（token数 vs 字节偏移）
- GlobalTensor下标是元素单位，需要根据数据类型确认是否需要转换

### 5. InitBuffer位置

**问题描述**：
`pipe->InitBuffer(scaleTBuf_, ubSize)` 在 PermuteScale 每次调用时执行，不符合最佳实践。

**解决方案**：
将 InitBuffer 移到 InitScaleBuffer 方法中：
```cpp
__aicore__ inline void InitScaleBuffer(...) {
    // ...
    TPipe *pipe = GetTPipePtr();
    pipe->InitBuffer(scaleTBuf_, ubSize_);  // 只在初始化时调用一次
}
```

**经验教训**：
- InitBuffer 应该在初始化阶段完成，不要在每次处理时重复调用

### 6. TBuf位置参数

**正确写法**：
```cpp
TBuf<QuePosition::VECIN> scaleTBuf_;  // 使用VECIN，不是VE
```

### 7. 变量命名规范

**问题**：
`rIndex`、`e` 等缩写不清晰，容易和成员变量 `e_` 混淆。

**规范**：
```cpp
// 不要使用
for (uint32_t rIndex = 0U; rIndex < rankDim_; rIndex++) {
    for (uint32_t e = 0U; e < e_; e++) {  // e和e_容易混淆
        ...
    }
}

// 正确使用
for (uint32_t rankIndex = 0U; rankIndex < rankDim_; rankIndex++) {
    for (uint32_t expertIdx = 0U; expertIdx < e_; expertIdx++) {
        ...
    }
}
```

### 8. 类成员声明顺序

**规范**：G.FMT.08-CPP 遵循传统顺序：public → protected → private

```cpp
class MyClass {
public:
    // 公共方法
    
protected:
    // 保护方法
    
private:
    // 私有方法
    // 私有成员变量
};
```

### 9. 注释格式规范

**规范**：G.CMT.01-CPP 注释符与注释内容间留有1个空格

```cpp
/* *
 * Copyright...
 * See LICENSE...
 *  */  // 结尾有空格
```

## 通用开发建议

1. **栈变量生命周期**：任何在Init中获取的栈变量指针，Process中都不能直接使用，必须拷贝数据
2. **workspace管理**：采用workspaceOffset统一管理，子模块不再额外计算偏移
3. **API参数检查**：参考项目中其他算子的写法，确认参数数量和类型
4. **单位一致性**：偏移、长度计算要明确单位（字节、元素个数），保持一致
5. **命名清晰**：避免缩写，使用完整词汇如 rankIndex、expertIdx

## 相关文件

```
mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/
├── scheduler/a2av_gmm_scheduler.h    # workspaceOffset管理
├── communication/hccl_a2av_op.h      # recvCounts_/sendCounts_/ubSize_
├── compute/quant_grouped_matmul.h    # 直接使用传入的workspace
└── common/a2av_common_tiling.h       # MAX_EXPERT_SIZE常量
```