# haojie0618 PR5419/PR6858 error 日志扫描总结

## 范围

- 当前分支：`haojie0618`
- PR refs：`upstream/pr5419` (`d42020149cc12129c1a56dee419251d1630ae468`), `upstream/pr6858` (`d9dbb03a5c45175ffc5afa10b907533cebd1c0a2`)
- 扫描方式：分别以 `upstream/master` 与 PR head 的 merge-base 为基线，扫描 touched C/C++ 源文件中新增或修改的 error 级别日志宏。
- 输出目录：`zsl_huawei_workspace/整改表格/haojie0618_pr5419_pr6858/`

## 扫描文件

PR5419 touched 源文件：

- `gmm/common/cgmct/block/block_mmad_mx.h`
- `gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h`
- `gmm/common/cgmct/utils/coord_utils.h`
- `gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp`
- `gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h`
- `gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h`
- `gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp`
- `gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_infershape.cpp`
- `gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp`
- `gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h`

PR6858 touched 源文件：

- `gmm/common/cgmct/block/block_mmad_mx.h`
- `gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp`
- `gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_mxfp4_weight_nz.h`

## 结果

- 整改日志数量：8
- 排除/未改候选数量：8
- 修改源码文件：
  - `gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp`
  - `gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp`

## 说明

- PR5419 的用户可修正参数类日志已改为 `EZ0009`、`EZ0012`、`EZ0013`、`EZ0020`、`EZ0027` 模板。
- PR5419 中 `CheckXXX failed`、`CalL1Tiling failed`、`CalcLeftL1Size failed`、`L1 space overflow` 按 skill 归为内部汇总/资源类日志，进入排除表。
- PR6858 diff 中有 3 处新增 OP_LOGE 候选，但当前 `haojie0618` 分支没有这些新增检查代码；为遵守“不改变代码逻辑”，本次未把 PR6858 的检查逻辑新增到源码，只在排除表记录为当前分支不适用。

## 验证

- `git diff --check -- gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp`：通过。
- 全文件 reason 句式扫描存在既有历史命中；对本次 diff 新增行扫描禁用模式：无命中。
