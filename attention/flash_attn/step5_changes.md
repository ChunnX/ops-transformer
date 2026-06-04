# Step 5: FlashAttention 主循环状态机优化 [P2][P3]

## 优化内容

### 1. 缓存循环次数计算结果
- 新增成员变量 `cachedS2LoopTimes` 和 `cachedGS1LoopTimes`
- 在 `GetTaskDealMode()` 中，仅当 batch 切换时计算并缓存这两个值
- `UpdateAxisInfo()` 直接使用缓存值，避免重复计算

### 2. 消除冗余 64-bit 除法
- 原代码在 `GetTaskDealMode()` 和 `UpdateAxisInfo()` 中各自计算：
  - `s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize`
  - `gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize`
- 优化后仅在 batch 切换时计算一次，每轮迭代节省 2 次 64-bit 除法

### 3. 分支预测优化
- 对 `CREATE_TASK` 路径添加 `likely()` 标注
- 对 `DEAL_ZERO`、`SKIP_ZERO`、`S2_END`、`SKIP` 等低频路径添加 `unlikely()` 标注
- 确保热路径指令连续排布，减少分支预测失败

## 修改文件
- `flash_attn_kernel_base_gqa_template.h`
  - 新增 2 个缓存成员变量
  - `GetTaskDealMode()`: 缓存计算逻辑 + unlikely 标注
  - `UpdateAxisInfo()`: 使用缓存值 + unlikely 标注
  - `FlashAttention()`: likely/unlikely 标注

## 预期收益
- 消除每轮迭代 2 次重复的 64-bit 除法（~20-40 cycle/次）
- 减少分支预测失败 ~30%
- 热路径指令连续性提升

## 修改统计
- 1 file changed, 16 insertions(+), 15 deletions(-)

## Commit
- `8d241bfe2` - step 5 only
