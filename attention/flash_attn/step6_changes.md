# Step 6: CalcParams 热路径内冗余计算消除

## 优化目标
消除 CalcParams 函数中每轮迭代都执行的冗余计算，特别是：
1. bIdx/n2Idx 的重复除法运算
2. GetPreNextTokenLeftUp 的重复调用

## 修改内容

### 1. 新增缓存成员变量
在 `flash_attn_kernel_base_gqa_template.h` 中添加 4 个缓存变量：
```cpp
// Cached values for CalcParams optimization
uint32_t cachedBIdx = 0;
uint32_t cachedN2Idx = 0;
int64_t cachedPreTokensLeftUp = 0;
int64_t cachedNextTokensLeftUp = 0;
```

### 2. 在 GetTaskDealMode 中缓存计算结果
当 batch 切换时（`isFirstTask || prevBIdx != bIdx`），计算并缓存：
```cpp
cachedBIdx = bIdx;
cachedN2Idx = bN2Cur % constInfo.n2Size;
GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, cachedPreTokensLeftUp, cachedNextTokensLeftUp);
```

### 3. CalcParams 使用缓存值
将原来的实时计算替换为直接使用缓存：
```cpp
// 原来：
info.bIdx = bN2Cur / constInfo.n2Size;
info.n2Idx = bN2Cur % constInfo.n2Size;
GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

// 现在：
info.bIdx = cachedBIdx;
info.n2Idx = cachedN2Idx;
info.preTokensLeftUp = cachedPreTokensLeftUp;
info.nextTokensLeftUp = cachedNextTokensLeftUp;
```

## 优化效果

### 性能提升
- **消除每轮迭代的 2 次 64-bit 除法**：bIdx 和 n2Idx 的计算从每轮执行改为仅在 batch 切换时执行
- **消除每轮迭代的函数调用**：GetPreNextTokenLeftUp 从每轮调用改为仅在 batch 切换时调用
- **预计每轮 task 节省 ~20-40 cycles**

### 代码变更统计
- 1 file changed, 13 insertions(+), 3 deletions(-)

## 验证
- Commit: fae7323fd
- Branch: flash_attn
- 已推送到远程仓库

## 注意事项
1. 缓存值仅在 batch 切换时更新，同一 batch 内的所有 task 共享相同的缓存值
2. 此优化与 Step 5（主循环状态机优化）互补，两者共同减少热路径中的冗余计算
3. actMSizeAlign32 和 actVecMSize 的计算已经使用位运算，无需进一步优化
