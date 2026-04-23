## Goal

将两个低比特算子（quant_all_reduce 和 quant_reduce_scatter）从GET模式改为PUT模式，且**不使用AIV分组**（使用原始分核逻辑）。原始GET版本代码可回退到 bd4c56e9 查看。

## Instructions

- 目标3：修改为PUT模式，先不使用AIV分组
- 用户确认GET版本精度保证正确，不需要验证
- 直接修改当前版本，保留PUT模式和目标1/2的改进（DataCopyPad、动态块大小）
- 持续调试直到精度正确

## Discoveries

1. **scale偏移计算错误**：原来固定为 `SCALE_BLCOK_BYTES/sizeof(ScalesType)`（32B），但应该跟随x块大小和量化比例：
   - TG格式（float scale）：128个x对应1个scale，比例=128
   - MX格式（fp8_e8m0 scale）：64个x对应2个scale（MX_SCALES_LAST_DIM=2），比例=32

2. **Mc2CcTilingConfig类型错误**：不能使用未定义的`OP_TYPE_PUT`，应使用已定义的`OP_TYPE_ALL_TO_ALL = 8`

3. **ReduceScatter PUT写入位置逻辑**：经过多次迭代调试，最终正确逻辑：
   - 本rank的第srcRankId分区数据 → 写入rank srcRankId Win Area的**第rankId分区位置**
   - 这样rank j Win Area的第k分区位置存储rank k的第j分区数据（rank k对rank j分区的贡献）
   - **关键修正**：`winXOffset = hcclContext_->rankId * xSliceSizeNums + curXOffset`（而非`srcRankId * xSliceSizeNums`）

4. **ExecuteReduceScatter读取位置**：从本卡Win Area第srcRankId分区读取数据做ReduceSum（读取位置正确）

## Accomplished

**已完成**：
- quant_all_reduce：PUT模式+原始分核，精度全部PASS (100%) ✓
- quant_reduce_scatter：PUT模式+原始分核，精度全部PASS (100%) ✓
  - 修复写入位置：rankId分区而非srcRankId分区
  - 修复MX格式scale比例：64个x对应2个scale（比例=32而非64）

**测试结果**：
- level0所有用例均PASS（TG格式和MX格式）

## Relevant files / directories

- `/home/m00947052/ops-transformer/mc2/quant_reduce_scatter/op_host/op_tiling/quant_reduce_scatter_tiling.cpp` - Mc2CcTilingConfig使用OP_TYPE_ALL_TO_ALL
- `/home/m00947052/ops-transformer/mc2/quant_reduce_scatter/op_kernel/mte_comm.h`：
  - `ComputeBlockParams()`：scale偏移计算改为动态量化比例（TG=128，MX=32）
  - `PutDataToAllWin<isReduceScatter>()`：写入逻辑 `winXOffset = hcclContext_->rankId * xSliceSizeNums + curXOffset`
- `/home/m00947052/ops-transformer/mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h` - PUT模式注释更新

## 下一步

- 提交代码到git
- 根据用户要求决定是否继续目标4（AIV分组）