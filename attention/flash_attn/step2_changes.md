# Step 2: constInfo 成员变量 → 局部变量 [P4]

## 严重度
🔴 最高优先级

## 涉及文件
- `flash_attn_kernel_base_gqa_template.h`
- `flash_attn_block_vec_noquant_gqa.h`

## 现状问题
热路径函数中大量通过 `constInfo.xxx` / `this->constInfo.xxx` 访问成员变量。每次访问生成 `this+offset` Load，跨函数调用后必须重新 Load（别名分析保守）。

## 修改内容

### 1. FlashAttention() 入口处拷贝热路径字段到局部变量
```cpp
const uint32_t n2Size = constInfo.n2Size;
const uint32_t gSize = constInfo.gSize;
const float scaleValue = constInfo.scaleValue;
const uint8_t subBlockIdx = constInfo.subBlockIdx;
const bool isKvContinuous = constInfo.isKvContinuous;
```

### 2. CreateTask() 签名增加参数传递
```cpp
CreateTask(..., uint32_t n2Size, uint32_t gSize, float scaleValue, uint8_t subBlockIdx, bool isKvContinuous)
```

### 3. CalcParams() 签名增加参数，内部使用参数替代 constInfo
- `constInfo.n2Size` → `n2Size`
- `constInfo.gSize` → `gSize`
- `constInfo.isKvContinuous` → `isKvContinuous`
- `constInfo.subBlockIdx` → `subBlockIdx`

### 4. ProcessVec1Nd() 入口提升局部变量
```cpp
const float scaleValue = constInfo.scaleValue;
const uint8_t subBlockIdx = constInfo.subBlockIdx;
```
- 8处 `static_cast<T>(constInfo.scaleValue)` → `static_cast<T>(scaleValue)`
- 1处 `constInfo.subBlockIdx` → `subBlockIdx`

### 5. ProcessVec1Dn() 入口提升局部变量
```cpp
const float scaleValue = constInfo.scaleValue;
const uint8_t subBlockIdx = constInfo.subBlockIdx;
```
- 2处 `static_cast<T>(constInfo.scaleValue)` → `static_cast<T>(scaleValue)`
- 3处 `constInfo.subBlockIdx` → `subBlockIdx`

## 修改统计
- 2 files changed, 38 insertions(+), 24 deletions(-)

## 实施状态
✅ 已完成（commit d3727b9f1）
