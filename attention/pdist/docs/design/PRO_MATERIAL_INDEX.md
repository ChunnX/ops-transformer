# PyPTO-Pro 全部资料索引

> **自动生成时间**: 2026-09-01
> **说明**: 本索引 §A/§C 由扫描命令动态生成，§B 为官方指定算子固定清单。§C 扫描 `tutorials/`。每次执行须重新扫描 §A/§C，§B 以官方最新指定清单为准。
> **缓存根**: `$PYPTO_DEVKIT_DIR` = `.devkit/`（本仓库 orchestrator 目录下；注意 `pro_ops/` 与 `docs/` 为指向 `.devkit/.repo-pypto/` 的 symlink，用 `find` 枚举须加 `-L`）

---

## §A API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）

> 搜索范围：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 递归获取所有 `.md` 文档，本次扫描共 **314** 个。

### A.1 API 总索引

| 文档 | 路径 |
|------|------|
| PyPTO-Pro API 总索引 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/index.md` |

### A.2 API 根目录（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/index.md` |

### A.3 SIMD-API（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/index.md` |

### A.4 SIMD-API/basic_data_structures（12 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | AccToVecMode | `docs/pypto_pro/api/SIMD-API/basic_data_structures/AccToVecMode.md` |
| 2 | DataType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/DataType.md` |
| 3 | MemorySpace | `docs/pypto_pro/api/SIMD-API/basic_data_structures/MemorySpace.md` |
| 4 | PipeType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/PipeType.md` |
| 5 | Ptr | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Ptr.md` |
| 6 | ReluPreMode | `docs/pypto_pro/api/SIMD-API/basic_data_structures/ReluPreMode.md` |
| 7 | Scalar | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Scalar.md` |
| 8 | Tensor | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Tensor.md` |
| 9 | TensorLayout | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TensorLayout.md` |
| 10 | TilePad | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TilePad.md` |
| 11 | TileType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md` |
| 12 | index | `docs/pypto_pro/api/SIMD-API/basic_data_structures/index.md` |

### A.5 SIMD-API/operation（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/index.md` |

### A.6 SIMD-API/operation/atomic_operations（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/atomic_operations/index.md` |
| 2 | store_atomic | `docs/pypto_pro/api/SIMD-API/operation/atomic_operations/store_atomic.md` |

### A.7 SIMD-API/operation/cache_control（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | dcci | `docs/pypto_pro/api/SIMD-API/operation/cache_control/dcci.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/cache_control/index.md` |

### A.8 SIMD-API/operation/controlflow（3 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/controlflow/index.md` |
| 2 | range | `docs/pypto_pro/api/SIMD-API/operation/controlflow/range.md` |
| 3 | section_vector_section_cube | `docs/pypto_pro/api/SIMD-API/operation/controlflow/section_vector_section_cube.md` |

### A.9 SIMD-API/operation/matrix_computation（7 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/index.md` |
| 2 | matmul | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul.md` |
| 3 | matmul_acc | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_acc.md` |
| 4 | matmul_mx | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_mx.md` |
| 5 | matmul_mx_acc | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_mx_acc.md` |
| 6 | phase | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/phase.md` |
| 7 | set_mm_layout_transform | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/set_mm_layout_transform.md` |

### A.10 SIMD-API/operation/memory_data_movement（10 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/index.md` |
| 2 | insert | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/insert.md` |
| 3 | load | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/load.md` |
| 4 | load_tile | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/load_tile.md` |
| 5 | move | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/move.md` |
| 6 | ssbuf_load | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/ssbuf_load.md` |
| 7 | ssbuf_store | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/ssbuf_store.md` |
| 8 | store | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/store.md` |
| 9 | store_tile | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/store_tile.md` |
| 10 | subview | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/subview.md` |

### A.11 SIMD-API/operation/memory_vector_computation（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | fillpad | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fillpad.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/index.md` |

### A.12 SIMD-API/operation/memory_vector_computation/comparison（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | comparison | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/comparison/comparison.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/comparison/index.md` |

### A.13 SIMD-API/operation/memory_vector_computation/composite_computation（3 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | axpy | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/axpy.md` |
| 2 | fused_mul_add | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/fused_mul_add.md` |
| 3 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/index.md` |

### A.14 SIMD-API/operation/memory_vector_computation/elementwise（16 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | abs | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/abs.md` |
| 2 | add | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/add.md` |
| 3 | and_ | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/and_.md` |
| 4 | div | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/div.md` |
| 5 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/index.md` |
| 6 | log | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/log.md` |
| 7 | maximum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/maximum.md` |
| 8 | minimum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/minimum.md` |
| 9 | mul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/mul.md` |
| 10 | neg | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/neg.md` |
| 11 | relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/relu.md` |
| 12 | shl | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shl.md` |
| 13 | shr | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shr.md` |
| 14 | sqrt | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/sqrt.md` |
| 15 | sub | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/sub.md` |
| 16 | xor | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/xor.md` |

### A.15 SIMD-API/operation/memory_vector_computation/fused_vector_computation（13 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | add_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/add_relu.md` |
| 2 | add_relu_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/add_relu_cast.md` |
| 3 | addc | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/addc.md` |
| 4 | fused_mul_add_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/fused_mul_add_relu.md` |
| 5 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/index.md` |
| 6 | mul_add_dst | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/mul_add_dst.md` |
| 7 | mul_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/mul_cast.md` |
| 8 | partadd | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partadd.md` |
| 9 | partmax | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmax.md` |
| 10 | partmin | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmin.md` |
| 11 | partmul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmul.md` |
| 12 | sub_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/sub_relu.md` |
| 13 | sub_relu_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/sub_relu_cast.md` |

### A.16 SIMD-API/operation/memory_vector_computation/math_functions（15 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | argmax | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmax.md` |
| 2 | argmin | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmin.md` |
| 3 | exp | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/exp.md` |
| 4 | expand_div | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_div.md` |
| 5 | expand_max | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_max.md` |
| 6 | expand_min | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_min.md` |
| 7 | expand_mul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_mul.md` |
| 8 | expand_sub | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_sub.md` |
| 9 | expands | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expands.md` |
| 10 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/index.md` |
| 11 | max | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/max.md` |
| 12 | min | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/min.md` |
| 13 | recip | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/recip.md` |
| 14 | rsqrt | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/rsqrt.md` |
| 15 | sum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/sum.md` |

### A.17 SIMD-API/operation/memory_vector_computation/scatter_gather（6 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | gather | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gather.md` |
| 2 | gatherb | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gatherb.md` |
| 3 | gathermask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gathermask.md` |
| 4 | histogram | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/histogram.md` |
| 5 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/index.md` |
| 6 | scatter | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/scatter.md` |

### A.18 SIMD-API/operation/memory_vector_computation/selection（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/selection/index.md` |
| 2 | select | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/selection/select.md` |

### A.19 SIMD-API/operation/memory_vector_computation/sorting（4 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/index.md` |
| 2 | mrgsort | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort.md` |
| 3 | mrgsort2 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort2.md` |
| 4 | sort32 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/sort32.md` |

### A.20 SIMD-API/operation/memory_vector_computation/transpose_and_element_access（6 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | getval | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/getval.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/index.md` |
| 3 | set_stride | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_stride.md` |
| 4 | set_validshape | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_validshape.md` |
| 5 | setval | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/setval.md` |
| 6 | transpose | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/transpose.md` |

### A.21 SIMD-API/operation/memory_vector_computation/type_conversion（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/type_conversion/cast.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/type_conversion/index.md` |

### A.22 SIMD-API/operation/memory_vector_computation/vector_mask（5 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/index.md` |
| 2 | reset_mask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/reset_mask.md` |
| 3 | set_mask_count | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_mask_count.md` |
| 4 | set_mask_norm | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_mask_norm.md` |
| 5 | set_vec_mask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_vec_mask.md` |

### A.23 SIMD-API/operation/quantization（3 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | dequant | `docs/pypto_pro/api/SIMD-API/operation/quantization/dequant.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/quantization/index.md` |
| 3 | quant | `docs/pypto_pro/api/SIMD-API/operation/quantization/quant.md` |

### A.24 SIMD-API/operation/resource_management（6 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | addptr | `docs/pypto_pro/api/SIMD-API/operation/resource_management/addptr.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/resource_management/index.md` |
| 3 | make_ptr | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_ptr.md` |
| 4 | make_tensor | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tensor.md` |
| 5 | make_tile | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tile.md` |
| 6 | make_tile_group | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tile_group.md` |

### A.25 SIMD-API/operation/synchronization（6 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | barrier | `docs/pypto_pro/api/SIMD-API/operation/synchronization/barrier.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/synchronization/index.md` |
| 3 | mutex_lock_mutex_unlock | `docs/pypto_pro/api/SIMD-API/operation/synchronization/mutex_lock_mutex_unlock.md` |
| 4 | set_cross_core_wait_cross_core | `docs/pypto_pro/api/SIMD-API/operation/synchronization/set_cross_core_wait_cross_core.md` |
| 5 | sync_all | `docs/pypto_pro/api/SIMD-API/operation/synchronization/sync_all.md` |
| 6 | sync_src_sync_dst | `docs/pypto_pro/api/SIMD-API/operation/synchronization/sync_src_sync_dst.md` |

### A.26 SIMD-API/operation/system_variables（5 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | get_block_idx | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_block_idx.md` |
| 2 | get_block_num | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_block_num.md` |
| 3 | get_subblock_idx | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_subblock_idx.md` |
| 4 | get_subblock_num | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_subblock_num.md` |
| 5 | index | `docs/pypto_pro/api/SIMD-API/operation/system_variables/index.md` |

### A.27 SIMD-API/operation/transpose_and_element_access（2 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | fill_index | `docs/pypto_pro/api/SIMD-API/operation/transpose_and_element_access/fill_index.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/transpose_and_element_access/index.md` |

### A.28 SIMD-API/operation/utilities（5 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | Array | `docs/pypto_pro/api/SIMD-API/operation/utilities/Array.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/utilities/index.md` |
| 3 | make_tuple | `docs/pypto_pro/api/SIMD-API/operation/utilities/make_tuple.md` |
| 4 | struct | `docs/pypto_pro/api/SIMD-API/operation/utilities/struct.md` |
| 5 | struct_array | `docs/pypto_pro/api/SIMD-API/operation/utilities/struct_array.md` |

### A.29 SIMD-API/operation/vf_computation（3 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/index.md` |
| 2 | mask_reg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_reg.md` |
| 3 | reg_tensor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reg_tensor.md` |

### A.30 SIMD-API/operation/vf_computation/advanced_computation（6 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | de_interleave | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/de_interleave.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/index.md` |
| 3 | interleave | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/interleave.md` |
| 4 | pack | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/pack.md` |
| 5 | unpack | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/unpack.md` |
| 6 | unsqueeze | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/unsqueeze.md` |

### A.31 SIMD-API/operation/vf_computation/basic_arithmetic（25 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | abs | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/abs.md` |
| 2 | add | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/add.md` |
| 3 | addc | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/addc.md` |
| 4 | adds | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/adds.md` |
| 5 | div | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/div.md` |
| 6 | exp | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/exp.md` |
| 7 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/index.md` |
| 8 | leaky_relu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/leaky_relu.md` |
| 9 | ln | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/ln.md` |
| 10 | log | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log.md` |
| 11 | log10 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log10.md` |
| 12 | log2 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log2.md` |
| 13 | max | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/max.md` |
| 14 | maxs | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/maxs.md` |
| 15 | min | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/min.md` |
| 16 | mins | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mins.md` |
| 17 | mul | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mul.md` |
| 18 | mull | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mull.md` |
| 19 | muls | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/muls.md` |
| 20 | neg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/neg.md` |
| 21 | prelu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/prelu.md` |
| 22 | relu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/relu.md` |
| 23 | sqrt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/sqrt.md` |
| 24 | sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/sub.md` |
| 25 | subc | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/subc.md` |

### A.32 SIMD-API/operation/vf_computation/comparison_and_selection（9 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | eq | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/eq.md` |
| 2 | ge | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/ge.md` |
| 3 | gt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/gt.md` |
| 4 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/index.md` |
| 5 | le | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/le.md` |
| 6 | lt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/lt.md` |
| 7 | ne | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/ne.md` |
| 8 | select | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/select.md` |
| 9 | squeeze | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/squeeze.md` |

### A.33 SIMD-API/operation/vf_computation/composite_computation（7 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | abs_sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/abs_sub.md` |
| 2 | axpy | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/axpy.md` |
| 3 | exp_sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/exp_sub.md` |
| 4 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/index.md` |
| 5 | mul_add_dst | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_add_dst.md` |
| 6 | mul_dst_add | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_dst_add.md` |
| 7 | muls_cast | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/muls_cast.md` |

### A.34 SIMD-API/operation/vf_computation/data_movement（21 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | clear_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/clear_spr.md` |
| 2 | create_addr_reg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/create_addr_reg.md` |
| 3 | full | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/full.md` |
| 4 | gather | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/gather.md` |
| 5 | get_mask_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/get_mask_spr.md` |
| 6 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/index.md` |
| 7 | load | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load.md` |
| 8 | load_align | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_align.md` |
| 9 | load_unalign | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign.md` |
| 10 | load_unalign_init | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign_init.md` |
| 11 | load_unalign_pre | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign_pre.md` |
| 12 | mem_bar | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/mem_bar.md` |
| 13 | move | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/move.md` |
| 14 | scatter | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/scatter.md` |
| 15 | squeeze_store_unalign | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/squeeze_store_unalign.md` |
| 16 | squeeze_store_unalign_post | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/squeeze_store_unalign_post.md` |
| 17 | store | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store.md` |
| 18 | store_align | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_align.md` |
| 19 | store_unalign | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_unalign.md` |
| 20 | store_unalign_post | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_unalign_post.md` |
| 21 | unalign_reg_for_store | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/unalign_reg_for_store.md` |

### A.35 SIMD-API/operation/vf_computation/logical_computation（7 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | and_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/and_.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/index.md` |
| 3 | not_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/not_.md` |
| 4 | or_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/or_.md` |
| 5 | shift_left | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/shift_left.md` |
| 6 | shift_right | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/shift_right.md` |
| 7 | xor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/xor.md` |

### A.36 SIMD-API/operation/vf_computation/mask_operations（4 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | create_mask | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/create_mask.md` |
| 2 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/index.md` |
| 3 | mask_gen_with_reg_tensor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/mask_gen_with_reg_tensor.md` |
| 4 | update_mask | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/update_mask.md` |

### A.37 SIMD-API/operation/vf_computation/reduction（5 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/index.md` |
| 2 | pair_reduce_sum | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/pair_reduce_sum.md` |
| 3 | reduce_max | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_max.md` |
| 4 | reduce_min | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_min.md` |
| 5 | reduce_sum | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_sum.md` |

### A.38 SIMD-API/operation/vf_computation/sorting_and_indexing（3 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | arange | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/arange.md` |
| 2 | histograms | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/histograms.md` |
| 3 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/index.md` |

### A.39 SIMD-API/operation/vf_computation/special_reg_access（7 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | get_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_ctrl_spr.md` |
| 2 | get_saturation_flag | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_saturation_flag.md` |
| 3 | get_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_spr.md` |
| 4 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/index.md` |
| 5 | reset_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/reset_ctrl_spr.md` |
| 6 | set_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/set_ctrl_spr.md` |
| 7 | set_saturation_flag | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/set_saturation_flag.md` |

### A.40 SIMD-API/operation/vf_computation/type_conversion（4 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | astype | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/astype.md` |
| 2 | bit_cast | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/bit_cast.md` |
| 3 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/index.md` |
| 4 | truncate | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/truncate.md` |

### A.41 SIMD-API/operation/vf_computation/types（18 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | BinType | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/BinType.md` |
| 2 | CastLayout | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/CastLayout.md` |
| 3 | DataCopyMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/DataCopyMode.md` |
| 4 | DuplicatePos | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/DuplicatePos.md` |
| 5 | HistType | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/HistType.md` |
| 6 | IndexOrder | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/IndexOrder.md` |
| 7 | LoadDist | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/LoadDist.md` |
| 8 | MaskPattern | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MaskPattern.md` |
| 9 | MaskWidth | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MaskWidth.md` |
| 10 | MemBarMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MemBarMode.md` |
| 11 | MergeMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MergeMode.md` |
| 12 | PackPart | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/PackPart.md` |
| 13 | SaturateMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SaturateMode.md` |
| 14 | SaturationFlagMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SaturationFlagMode.md` |
| 15 | SqueezeMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SqueezeMode.md` |
| 16 | StoreDist | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/StoreDist.md` |
| 17 | VFRoundMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/VFRoundMode.md` |
| 18 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/index.md` |

### A.42 SIMT-API（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/SIMT-API/index.md` |

### A.43 SIMT-API/atomic（12 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | atomic_add | `docs/pypto_pro/api/SIMT-API/atomic/atomic_add.md` |
| 2 | atomic_and | `docs/pypto_pro/api/SIMT-API/atomic/atomic_and.md` |
| 3 | atomic_cas | `docs/pypto_pro/api/SIMT-API/atomic/atomic_cas.md` |
| 4 | atomic_dec | `docs/pypto_pro/api/SIMT-API/atomic/atomic_dec.md` |
| 5 | atomic_exch | `docs/pypto_pro/api/SIMT-API/atomic/atomic_exch.md` |
| 6 | atomic_inc | `docs/pypto_pro/api/SIMT-API/atomic/atomic_inc.md` |
| 7 | atomic_max | `docs/pypto_pro/api/SIMT-API/atomic/atomic_max.md` |
| 8 | atomic_min | `docs/pypto_pro/api/SIMT-API/atomic/atomic_min.md` |
| 9 | atomic_or | `docs/pypto_pro/api/SIMT-API/atomic/atomic_or.md` |
| 10 | atomic_sub | `docs/pypto_pro/api/SIMT-API/atomic/atomic_sub.md` |
| 11 | atomic_xor | `docs/pypto_pro/api/SIMT-API/atomic/atomic_xor.md` |
| 12 | index | `docs/pypto_pro/api/SIMT-API/atomic/index.md` |

### A.44 SIMT-API/execution（7 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | block_dim | `docs/pypto_pro/api/SIMT-API/execution/block_dim.md` |
| 2 | block_idx | `docs/pypto_pro/api/SIMT-API/execution/block_idx.md` |
| 3 | grid_dim | `docs/pypto_pro/api/SIMT-API/execution/grid_dim.md` |
| 4 | index | `docs/pypto_pro/api/SIMT-API/execution/index.md` |
| 5 | launch | `docs/pypto_pro/api/SIMT-API/execution/launch.md` |
| 6 | linear_thread_idx | `docs/pypto_pro/api/SIMT-API/execution/linear_thread_idx.md` |
| 7 | thread_idx | `docs/pypto_pro/api/SIMT-API/execution/thread_idx.md` |

### A.45 SIMT-API/scalar_compute（23 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | abs | `docs/pypto_pro/api/SIMT-API/scalar_compute/abs.md` |
| 2 | cast | `docs/pypto_pro/api/SIMT-API/scalar_compute/cast.md` |
| 3 | ceil | `docs/pypto_pro/api/SIMT-API/scalar_compute/ceil.md` |
| 4 | cos | `docs/pypto_pro/api/SIMT-API/scalar_compute/cos.md` |
| 5 | exp | `docs/pypto_pro/api/SIMT-API/scalar_compute/exp.md` |
| 6 | exp2 | `docs/pypto_pro/api/SIMT-API/scalar_compute/exp2.md` |
| 7 | floor | `docs/pypto_pro/api/SIMT-API/scalar_compute/floor.md` |
| 8 | fma | `docs/pypto_pro/api/SIMT-API/scalar_compute/fma.md` |
| 9 | index | `docs/pypto_pro/api/SIMT-API/scalar_compute/index.md` |
| 10 | isinf | `docs/pypto_pro/api/SIMT-API/scalar_compute/isinf.md` |
| 11 | isnan | `docs/pypto_pro/api/SIMT-API/scalar_compute/isnan.md` |
| 12 | log | `docs/pypto_pro/api/SIMT-API/scalar_compute/log.md` |
| 13 | log1p | `docs/pypto_pro/api/SIMT-API/scalar_compute/log1p.md` |
| 14 | log2 | `docs/pypto_pro/api/SIMT-API/scalar_compute/log2.md` |
| 15 | max | `docs/pypto_pro/api/SIMT-API/scalar_compute/max.md` |
| 16 | min | `docs/pypto_pro/api/SIMT-API/scalar_compute/min.md` |
| 17 | rint | `docs/pypto_pro/api/SIMT-API/scalar_compute/rint.md` |
| 18 | round | `docs/pypto_pro/api/SIMT-API/scalar_compute/round.md` |
| 19 | rsqrt | `docs/pypto_pro/api/SIMT-API/scalar_compute/rsqrt.md` |
| 20 | sin | `docs/pypto_pro/api/SIMT-API/scalar_compute/sin.md` |
| 21 | sqrt | `docs/pypto_pro/api/SIMT-API/scalar_compute/sqrt.md` |
| 22 | tanh | `docs/pypto_pro/api/SIMT-API/scalar_compute/tanh.md` |
| 23 | trunc | `docs/pypto_pro/api/SIMT-API/scalar_compute/trunc.md` |

### A.46 Utils-API（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/Utils-API/index.md` |

### A.47 Utils-API/debugging（5 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | dump_data | `docs/pypto_pro/api/Utils-API/debugging/dump_data.md` |
| 2 | index | `docs/pypto_pro/api/Utils-API/debugging/index.md` |
| 3 | printf | `docs/pypto_pro/api/Utils-API/debugging/printf.md` |
| 4 | pto_assert | `docs/pypto_pro/api/Utils-API/debugging/pto_assert.md` |
| 5 | trap | `docs/pypto_pro/api/Utils-API/debugging/trap.md` |

### A.48 Utils-API/python_syntax_sugar（4 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | const | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/const.md` |
| 2 | index | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/index.md` |
| 3 | max | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/max.md` |
| 4 | min | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/min.md` |

---

## §B 官方指定算子样例

> **重要**：以下为官方明确允许 agent 开发算子时参考的算子代码，是**唯一的算子写法参考来源**。`$PYPTO_DEVKIT_DIR/pro_ops/` 下其余文件**不得**作为样例参考或索引对象（orchestrator 资源缓存准备时已按清单清理，仅保留清单内文件）。

> **本次核对结果（2026-09-01）**：清单共 13 项，缓存实际存在 **12 项**；1 项缺失——`pro_ops/matmul/test_matmul_8K_example.py`（清单 #2，"matmul 入门"）。经与缓存内完整上游快照 `.devkit/.repo-pypto/python/tests/st/pypto_pro/frontend/matmul/` 比对，**上游仓库（gitcode.com/cann/pypto @ d6e9b83）本身已无此文件**（该目录仅含 4 个 perf 样例，无大小写变体）——属清单与上游版本漂移，非装配失败；重新装配无法恢复。该缺失样例为 matmul 入门教程性质，与本算子（纯 Vector 规约，无 Cube 计算）无参考关系；其余 12 个样例完整在场。已按模板规则记入 EXPLORE_REPORT §8 风险评估并上报 orchestrator 裁决（更新清单或维持现状）。

| # | 算子名称 | 缓存相对路径 | 类型 | 状态 |
|---|---------|-------------|------|------|
| 1 | add | `pro_ops/element_wise/test_add.py` | elementwise | 在缓存 |
| 2 | matmul_8k_example | `pro_ops/matmul/test_matmul_8K_example.py` | matmul 入门 | **缺失（上游已无此文件）** |
| 3 | matmul_perf_asw_4k | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset.py` | matmul 性能 | 在缓存 |
| 4 | matmul_perf_asw_8k_k128 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset.py` | matmul 性能 | 在缓存 |
| 5 | matmul_perf_asw_4k_dynamic | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | 在缓存 |
| 6 | matmul_perf_asw_8k_k128_dynamic | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | 在缓存 |
| 7 | fa_perf_tkv_preload | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | FlashAttention 生产级 | 在缓存 |
| 8 | fa_tilingkey_attn_mask | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | FlashAttention 教学 | 在缓存 |
| 9 | fa_with_mask | `pro_ops/fa/test_fa_with_mask.py` | FlashAttention 性能（mask + NBuffer + auto_mutex） | 在缓存 |
| 10 | flex_attention | `pro_ops/fa/test_flex_attention.py` | FlexAttention 性能 | 在缓存 |
| 11 | quant_lightning_indexer_vf | `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` | VF TopK | 在缓存 |
| 12 | layernorm_tile_group_vf | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | VF LayerNorm | 在缓存 |
| 13 | softmax_tile_group_vf | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | VF Softmax | 在缓存 |

缓存与清单无多余文件（非清单 .py 已在装配后清理；其余子目录为空目录，不含任何 .py）。

---

## §C 教程与设计指南（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials`）

> 扫描命令：`find -L $PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials -name "*.md" | sort`，本次共 **40** 个。

| # | 来源 | 文档 | 路径 |
|---|------|------|------|
| 1 | tutorials | aclnn_operator_project_development | `docs/pypto_pro/tutorials/advanced_programming/aclnn_operator_project_development.md` |
| 2 | tutorials | advanced_ai_core_programming_model | `docs/pypto_pro/tutorials/advanced_programming/advanced_ai_core_programming_model.md` |
| 3 | tutorials | ai_framework_operator_adaptation | `docs/pypto_pro/tutorials/advanced_programming/ai_framework_operator_adaptation.md` |
| 4 | tutorials | aot_compilation_optimization | `docs/pypto_pro/tutorials/advanced_programming/aot_compilation_optimization.md` |
| 5 | tutorials | auto_parallel_pipeline | `docs/pypto_pro/tutorials/advanced_programming/auto_parallel_pipeline.md` |
| 6 | tutorials | hardware_implementation | `docs/pypto_pro/tutorials/advanced_programming/hardware_implementation.md` |
| 7 | tutorials | index | `docs/pypto_pro/tutorials/advanced_programming/index.md` |
| 8 | tutorials | operator_graph_integration_development | `docs/pypto_pro/tutorials/advanced_programming/operator_graph_integration_development.md` |
| 9 | tutorials | superkernel | `docs/pypto_pro/tutorials/advanced_programming/superkernel.md` |
| 10 | tutorials | functional_debugging | `docs/pypto_pro/tutorials/debugging_and_optimization/functional_debugging.md` |
| 11 | tutorials | index | `docs/pypto_pro/tutorials/debugging_and_optimization/index.md` |
| 12 | tutorials | performance_optimization | `docs/pypto_pro/tutorials/debugging_and_optimization/performance_optimization.md` |
| 13 | tutorials | index | `docs/pypto_pro/tutorials/index.md` |
| 14 | tutorials | introduction | `docs/pypto_pro/tutorials/introduction.md` |
| 15 | tutorials | JIT_compilation | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/JIT_compilation.md` |
| 16 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/index.md` |
| 17 | tutorials | offline_binary_compilation | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/offline_binary_compilation.md` |
| 18 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/index.md` |
| 19 | tutorials | kernel_function | `docs/pypto_pro/tutorials/operator_development/kernel_function.md` |
| 20 | tutorials | Cube_matrix_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Cube_matrix_computation.md` |
| 21 | tutorials | Python_programming_overview | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Python_programming_overview.md` |
| 22 | tutorials | Reg_vector_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` |
| 23 | tutorials | Tile_vector_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Tile_vector_computation.md` |
| 24 | tutorials | TilingData | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/TilingData.md` |
| 25 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/index.md` |
| 26 | tutorials | multi_core_partitioning_and_Tiling | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` |
| 27 | tutorials | tail_block_handling | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` |
| 28 | tutorials | tiling_key | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tiling_key.md` |
| 29 | tutorials | abstract_hardware_architecture | `docs/pypto_pro/tutorials/programming_paradigm/abstract_hardware_architecture.md` |
| 30 | tutorials | index | `docs/pypto_pro/tutorials/programming_paradigm/index.md` |
| 31 | tutorials | programming_paradigm_overview | `docs/pypto_pro/tutorials/programming_paradigm/programming_paradigm_overview.md` |
| 32 | tutorials | simt_programming | `docs/pypto_pro/tutorials/programming_paradigm/simt_programming.md` |
| 33 | tutorials | Add_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/Add_operator.md` |
| 34 | tutorials | CV_fused_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/CV_fused_operator.md` |
| 35 | tutorials | HelloWorld | `docs/pypto_pro/tutorials/quick_start/SIMD/HelloWorld.md` |
| 36 | tutorials | Matmul_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/Matmul_operator.md` |
| 37 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/SIMD/index.md` |
| 38 | tutorials | Add_operator | `docs/pypto_pro/tutorials/quick_start/SIMT/Add_operator.md` |
| 39 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/SIMT/index.md` |
| 40 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/index.md` |
