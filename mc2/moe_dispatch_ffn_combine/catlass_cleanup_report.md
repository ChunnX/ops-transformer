# Catlass 库清理报告

## 概述
本次清理针对 `/home/mace/workspace/ops-transformer/mc2/moe_dispatch_ffn_combine/catlass` 目录，删除了未被项目引用的文件，保留了所有被使用的依赖文件。

## 统计信息

| 指标 | 数值 |
|------|------|
| 删除前文件总数 | 226 个 |
| 删除后文件总数 | 99 个 |
| 已删除文件数 | 127 个 |
| 压缩比 | 43.8% |

## 入口文件

项目中直接引用 catlass 的文件：
1. `op_kernel/tile_copy.hpp` - 自定义的 tile copy 扩展实现
2. `op_kernel/moe_dispatch_ffn_combine_catlass_impl.h` - MOE Dispatch FFN Combine 的 catlass 实现

## 删除的目录

| 目录 | 文件数 | 说明 |
|------|--------|------|
| `catlass/conv/` | 14 个 | 卷积操作模块，未被使用 |
| `catlass/epilogue/block/` | 27 个 | 尾操作块实现，未被使用 |
| `catlass/epilogue/fusion/` | 12 个 | 操作融合框架，未被使用 |
| `catlass/gemm/kernel/` | 35 个 | GEMM 内核实现，未被使用 |
| `catlass/gemv/` | 11 个 | 矩阵向量乘法模块，未被使用 |
| `catlass/epilogue/tile/` | 15 个 | 保留 2 个文件，删除 15 个 |

## 保留的模块

### 1. 核心模块 (99 个文件)
- **arch/** (4 个文件) - 架构相关抽象
- **detail/** (6 个文件) - 实现细节和工具宏
- **epilogue/tile/** (2 个文件) - 保留 copy_gm_to_ub.hpp 和 copy_ub_to_gm.hpp
- **gemm/block/** (40 个文件) - 块级矩阵乘法实现
- **gemm/device/** (1 个文件) - 设备级接口
- **gemm/tile/** (19 个文件) - 瓦片级操作
- **gemm/** (4 个文件) - GEMM 核心头文件
- **layout/** (4 个文件) - 内存布局定义
- **tla/** (8 个文件) - 模板布局抽象层
- **其他** (11 个文件) - 核心头文件

## 备份位置

原始 catlass 库已备份至：
```
/home/mace/workspace/ops-transformer/mc2/moe_dispatch_ffn_combine/bakcatlass
```

## 验证方法

可以通过编译项目来验证删除是否正确：
```bash
cd /home/mace/workspace/ops-transformer/mc2/moe_dispatch_ffn_combine
# 如果编译通过，则说明依赖关系完整
```

## 注意事项

1. **条件编译**：部分文件使用 `CATLASS_ARCH` 宏进行条件编译，当前保留了 A2 (2201) 和 A5 (3510) 两个架构的文件
2. **未来扩展**：如果需要使用 conv、gemv 或其他 epilogue 功能，可以从备份目录恢复
3. **依赖验证**：建议在删除备份前进行完整编译测试

## 文件清单

### 保留的完整文件列表 (99 个)

#### catlass/arch/ (4 个)
```
catlass/arch/arch.hpp
catlass/arch/cross_core_sync.hpp
catlass/arch/local_tensor_buffer.hpp
catlass/arch/resource.hpp
```

#### catlass/detail/ (6 个)
```
catlass/detail/alignment.hpp
catlass/detail/callback.hpp
catlass/detail/dependent_false.hpp
catlass/detail/kernel_adapter.hpp
catlass/detail/macros.hpp
catlass/detail/tag_to_layout.hpp
```

#### catlass/epilogue/tile/ (2 个)
```
catlass/epilogue/tile/copy_gm_to_ub.hpp
catlass/epilogue/tile/copy_ub_to_gm.hpp
```

#### catlass/gemm/block/ (40 个)
```
catlass/gemm/block/block_dequant.hpp
catlass/gemm/block/block_mmad.hpp
catlass/gemm/block/block_mmad_dynamic_common.hpp
catlass/gemm/block/block_mmad_fa_pv.hpp
catlass/gemm/block/block_mmad_fa_qk.hpp
catlass/gemm/block/block_mmad_fag_ab.hpp
catlass/gemm/block/block_mmad_fag_abt.hpp
catlass/gemm/block/block_mmad_fag_atb.hpp
catlass/gemm/block/block_mmad_fai_pv_head_tail.hpp
catlass/gemm/block/block_mmad_fai_pv_normal.hpp
catlass/gemm/block/block_mmad_fai_pv_split_row_tla.hpp
catlass/gemm/block/block_mmad_fai_pv_tla.hpp
catlass/gemm/block/block_mmad_fai_qk_head_tail.hpp
catlass/gemm/block/block_mmad_fai_qk_normal.hpp
catlass/gemm/block/block_mmad_fai_qk_split_row_tla.hpp
catlass/gemm/block/block_mmad_fai_qk_tla.hpp
catlass/gemm/block/block_mmad_gemm.hpp
catlass/gemm/block/block_mmad_hstu_pv_split_row_tla.hpp
catlass/gemm/block/block_mmad_hstu_qk_split_row_tla.hpp
catlass/gemm/block/block_mmad_mla_pv.hpp
catlass/gemm/block/block_mmad_mla_pv_tp1_spec.hpp
catlass/gemm/block/block_mmad_mla_qk.hpp
catlass/gemm/block/block_mmad_mla_qk_tp1_spec.hpp
catlass/gemm/block/block_mmad_multi_batch_tla.hpp
catlass/gemm/block/block_mmad_mx_tla.hpp
catlass/gemm/block/block_mmad_pingpong.hpp
catlass/gemm/block/block_mmad_pingpong_bias.hpp
catlass/gemm/block/block_mmad_pingpong_dequant_tla.hpp
catlass/gemm/block/block_mmad_pingpong_full_loadA.hpp
catlass/gemm/block/block_mmad_pingpong_full_loadA_tla.hpp
catlass/gemm/block/block_mmad_pingpong_full_loadB_tla.hpp
catlass/gemm/block/block_mmad_pingpong_preloadk_tla.hpp
catlass/gemm/block/block_mmad_pingpong_slice_k.hpp
catlass/gemm/block/block_mmad_pingpong_tla.hpp
catlass/gemm/block/block_mmad_pingpong_with_prologue.hpp
catlass/gemm/block/block_mmad_preload.hpp
catlass/gemm/block/block_mmad_preload_async.hpp
catlass/gemm/block/block_mmad_preload_async_with_callback.hpp
catlass/gemm/block/block_mmad_preload_async_with_callback_tla.hpp
catlass/gemm/block/block_mmad_preload_tla.hpp
catlass/gemm/block/block_mmad_small.hpp
catlass/gemm/block/block_swizzle.hpp
```

#### catlass/gemm/device/ (1 个)
```
catlass/gemm/device/device_gemm.hpp
```

#### catlass/gemm/tile/ (19 个)
```
catlass/gemm/tile/cast_int4_to_int8.hpp
catlass/gemm/tile/cast_int8_to_fp16.hpp
catlass/gemm/tile/copy_gm_to_l1_a2.hpp
catlass/gemm/tile/copy_gm_to_l1_a5.hpp
catlass/gemm/tile/copy_gm_to_ub_a2.hpp
catlass/gemm/tile/copy_l0c_to_dst.hpp
catlass/gemm/tile/copy_l0c_to_gm_a2.hpp
catlass/gemm/tile/copy_l0c_to_gm_a5.hpp
catlass/gemm/tile/copy_l0c_to_ub_a5.hpp
catlass/gemm/tile/copy_l1_to_bt_a2.hpp
catlass/gemm/tile/copy_l1_to_bt_a5.hpp
catlass/gemm/tile/copy_l1_to_fb_a2.hpp
catlass/gemm/tile/copy_l1_to_l0a_a2.hpp
catlass/gemm/tile/copy_l1_to_l0a_a5.hpp
catlass/gemm/tile/copy_l1_to_l0b_a2.hpp
catlass/gemm/tile/copy_l1_to_l0b_a5.hpp
catlass/gemm/tile/copy_ub_to_gm_a2.hpp
catlass/gemm/tile/tile_copy.hpp
catlass/gemm/tile/tile_copy_tla.hpp
catlass/gemm/tile/tile_mmad.hpp
catlass/gemm/tile/tile_traits.hpp
```

#### catlass/gemm/ 其他 (4 个)
```
catlass/gemm/dispatch_policy.hpp
catlass/gemm/gemm_type.hpp
catlass/gemm/helper.hpp
catlass/gemm_coord.hpp
```

#### catlass/layout/ (4 个)
```
catlass/layout/layout.hpp
catlass/layout/matrix.hpp
catlass/layout/tensor.hpp
catlass/layout/vector.hpp
```

#### catlass/ 其他 (7 个)
```
catlass/catlass.hpp
catlass/conv_coord.hpp
catlass/coord.hpp
catlass/debug.hpp
catlass/matrix_coord.hpp
catlass/numeric_size.hpp
catlass/status.hpp
```

#### tla/ (8 个)
```
tla/int_tuple.hpp
tla/layout.hpp
tla/numeric/integer_sequence.hpp
tla/numeric/integral_constant.hpp
tla/numeric/math.hpp
tla/tensor.hpp
tla/tuple.hpp
tla/type_traits.hpp
```

---
*报告生成时间: 2026-04-11*
*备份位置: /home/mace/workspace/ops-transformer/mc2/moe_dispatch_ffn_combine/bakcatlass*
