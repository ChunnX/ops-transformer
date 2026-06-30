# 重构记录：移除未调用的 `GetPreNextTokensLeftUp` 与 `FixParamWithRowInvalid`

## 1. 修改背景
在排查 `fia_tiling_nonquant_gqa.cpp` 代码时发现，`GetPreNextTokensLeftUp` 和 `FixParamWithRowInvalid` 这两个函数虽然在当前类 `FiaTilingNonQuantArch35` 中进行了声明和定义，但**在该文件内部并未被任何地方调用**。

经过全局代码搜索确认，这两个函数的同名/同功能实现仅在 `fia_tiling_fullquant_gqa.cpp`、`fia_tiling_fullquant_mx.cpp` 以及 `fia_tiling_nonquant_mla.cpp` 等其他派生类中被调用。因此，在 `FiaTilingNonQuantArch35` 中的这两个函数属于冗余的死代码，应当予以删除以保持代码整洁。

## 2. 涉及文件
- `op_host/arch35/fia_tiling_nonquant_gqa.cpp`
- `op_host/arch35/fia_tiling_nonquant_gqa.h`

## 3. 具体修改点

### 3.1 `fia_tiling_nonquant_gqa.cpp`
- **删除 `GetPreNextTokensLeftUp` 函数定义**：
  直接删除 `void FiaTilingNonQuantArch35::GetPreNextTokensLeftUp(...)` 的整个函数体。
- **删除 `FixParamWithRowInvalid` 函数定义**：
  直接删除 `void FiaTilingNonQuantArch35::FixParamWithRowInvalid(...)` 的整个函数体。

### 3.2 `fia_tiling_nonquant_gqa.h`
- **删除函数声明**：
  在类的 `public` 或 `protected` 区域中，删除以下两行声明：
  ```cpp
  void GetPreNextTokensLeftUp(int64_t actualSeqLengthQ, int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                              int64_t &nextTokensLeftUp);
  void FixParamWithRowInvalid(int64_t &actualSeqLengthQ, int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                              int64_t &nextTokensLeftUp);
  ```

## 4. 预期效果
- 消除 `FiaTilingNonQuantArch35` 类中未被引用的冗余函数，减少维护成本。
- 避免对代码阅读者产生误导（以为该逻辑在当前类中生效）。
- 不影响其他算子模板（FullQuant、MLA 等）的正常逻辑，因为它们各自实现了自己的同名函数。