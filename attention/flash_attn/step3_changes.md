# Step 3: softmax 状态 Buffer 数组索引优化 [P1]

## 严重度
🟠 高优先级

## 涉及文件
- `flash_attn_block_vec_noquant_gqa.h`

## 现状问题
```cpp
TBuf<> softmaxMaxBuf[PRELOAD_N + 1];  // 深度=3
TBuf<> softmaxSumBuf[PRELOAD_N + 1];
TBuf<> softmaxExpBuf[PRELOAD_N + 1];
```
访问方式 `% (PRELOAD_N + 1)` 即 `% 3`，3 不是 2 的幂，编译器无法优化为位运算，每次取模产生 Scalar 除法开销。

## 修改方案：方案 A — 深度从 3 抬到 4，用位掩码索引

### 1. Buffer 声明深度改为 4
```cpp
TBuf<> softmaxMaxBuf[4];
TBuf<> softmaxSumBuf[4];
TBuf<> softmaxExpBuf[4];
```

### 2. 所有 16 处取模替换为位掩码
`% (PRELOAD_N + 1)` → `& 3`

涉及行：176-178, 351-354, 512-514, 641, 649, 767, 775, 1064, 1072, 1083

### 3. SoftmaxInitBuffer() 补充 index [3] 初始化
```cpp
tPipe->InitBuffer(softmaxSumBuf[3], 256);
tPipe->InitBuffer(softmaxMaxBuf[3], 256);
tPipe->InitBuffer(softmaxExpBuf[3], 256);
```

## 修改统计
- 1 file changed, 22 insertions(+), 22 deletions(-)

## 额外 UB 开销
3 个 TBuf（每个 256B）= 768B，换取消模运算的 Scalar 除法开销

## 实施状态
✅ 已完成（commit 924354d61）
