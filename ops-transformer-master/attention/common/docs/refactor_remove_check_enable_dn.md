# 重构记录：移除 CheckEnableDN 及 dnFlag_ 死代码

## 1. 修改背景
在 `fia_tiling_nonquant_gqa.cpp` 文件中，`CheckEnableDN()` 函数在入口处直接 `return false;`，导致后续的条件判断逻辑成为死代码。同时，依赖该函数返回值的成员变量 `dnFlag_` 实际上恒为 `false`。

为了保持代码整洁，消除冗余逻辑，计划将 `CheckEnableDN()` 函数及其相关的 `dnFlag_` 变量和所有依赖该变量的死代码分支彻底移除。

## 2. 涉及文件
- `op_host/arch35/fia_tiling_nonquant_gqa.cpp`
- `op_host/arch35/fia_tiling_nonquant_gqa.h` (需同步修改声明和成员变量)

## 3. 具体修改点

### 3.1 `fia_tiling_nonquant_gqa.cpp`

#### A. 删除 `CheckEnableDN()` 函数定义
- **位置**: `FiaTilingNonQuantArch35::CheckEnableDN()`
- **操作**: 直接删除整个函数体。

#### B. 简化 `SplitPolicy()` 函数
- **位置**: `FiaTilingNonQuantArch35::SplitPolicy()`
- **操作**: 
  1. 删除 `dnFlag_ = CheckEnableDN();`。
  2. 删除依赖 `dnFlag_` 的死代码分支：
     ```cpp
     // 删除以下代码
     if (dnFlag_ && CheckQKVActualSeqLengthsRight() && fiaInfo_->qkHeadDim == fiaInfo_->vHeadDim &&
         fiaInfo_->qkHeadDim == DSIZE_64) {
         sInnerFactor_ = SINNER_256;
     }
     ```

#### C. 简化 `CheckS1OutSplit()` 函数
- **位置**: `FiaTilingNonQuantArch35::CheckS1OutSplit()`
- **操作**: 从条件判断中移除 `|| dnFlag_`。
  ```cpp
  // 修改前:
  if (fiaInfo_->sysPrefixFlag || fiaInfo_->kvPaddingSizeFlag || fiaInfo_->qPaddingSizeFlag || dnFlag_ ||
      fiaInfo_->learnableSinkFlag || fiaInfo_->enableAlibiPse) {
      return false;
  }
  
  // 修改后:
  if (fiaInfo_->sysPrefixFlag || fiaInfo_->kvPaddingSizeFlag || fiaInfo_->qPaddingSizeFlag ||
      fiaInfo_->learnableSinkFlag || fiaInfo_->enableAlibiPse) {
      return false;
  }
  ```

#### D. 简化 `CalcWorkspaceSize()` 函数
- **位置**: `FiaTilingNonQuantArch35::CalcWorkspaceSize()`
- **操作**: 简化条件表达式，因为 `!dnFlag_` 恒为 true，`dnFlag_` 恒为 false。
  ```cpp
  // 修改前:
  if ((!dnFlag_ && dSize > DSIZE_128) || (dnFlag_ && dSize > DSIZE_192)) {
  
  // 修改后:
  if (dSize > DSIZE_128) {
  ```

### 3.2 `fia_tiling_nonquant_gqa.h`

#### A. 删除函数声明
- **操作**: 删除 `bool CheckEnableDN();` 声明。

#### B. 删除成员变量
- **操作**: 删除 `bool dnFlag_{false};` (或类似形式的成员变量定义)。

## 4. 预期效果
- 代码量减少，逻辑更清晰。
- 消除编译器可能产生的死代码警告。
- 行为与修改前完全一致（因为原本就是恒定返回 `false`）。