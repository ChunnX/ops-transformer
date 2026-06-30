# 重构记录：移除未调用的 `CheckQKVActualSeqLengthsRight`

## 1. 修改背景
在 `fia_tiling_nonquant_gqa.cpp` 文件中，`CheckQKVActualSeqLengthsRight()` 函数仅在 `SplitPolicy()` 函数中被调用，且其调用受 `dnFlag_` 变量保护（`if (dnFlag_ && CheckQKVActualSeqLengthsRight() && ...)`）。

由于 `CheckEnableDN()` 恒返回 `false` 导致 `dnFlag_` 恒为 `false`，基于 C++ 短路求值机制，`CheckQKVActualSeqLengthsRight()` 永远不会被执行。如果在执行《移除 CheckEnableDN 及 dnFlag_ 死代码》的重构计划后，该调用所在的 `if` 分支被删除，那么 `CheckQKVActualSeqLengthsRight()` 将成为完全未被调用的死代码，应当一并删除以保持代码整洁。

## 2. 涉及文件
- `op_host/arch35/fia_tiling_nonquant_gqa.cpp`
- `op_host/arch35/fia_tiling_nonquant_gqa.h`

## 3. 具体修改点

### 3.1 `fia_tiling_nonquant_gqa.cpp`
- **删除 `CheckQKVActualSeqLengthsRight` 函数定义**：
  直接删除 `bool FiaTilingNonQuantArch35::CheckQKVActualSeqLengthsRight()` 的整个函数体。

### 3.2 `fia_tiling_nonquant_gqa.h`
- **删除函数声明**：
  在类的 `private` 区域中，删除以下声明：
  ```cpp
  bool CheckQKVActualSeqLengthsRight();
  ```

## 4. 前置依赖
本修改依赖于 `docs/refactor_remove_check_enable_dn.md` 的执行。只有当 `dnFlag_` 相关的 `if` 死代码分支被删除后，此函数才彻底变为无调用点的死代码。

## 5. 预期效果
- 进一步消除 `FiaTilingNonQuantArch35` 类中的冗余死代码。
- 不影响其他算子模板（如 FullQuant、MLA 等），因为它们各自实现了自己的同名函数。