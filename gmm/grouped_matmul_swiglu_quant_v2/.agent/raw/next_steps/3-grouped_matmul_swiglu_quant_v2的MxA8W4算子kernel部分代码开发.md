# grouped_matmul_swiglu_quant_v2 MxA8W4算子kernel部分代码开发

该任务要求你完成grouped_matmul_swiglu_quant_v2算子伪量化MxA8W4数据流的kernel部分代码开发，当前的代码中kernel已经准备好了该数据流的入口函数，入口代码文件为gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp，其中注释MxA8W4场景的部分即为需要补充的kernel实现，你需要将kernel部分的代码实现在gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/weight_quant_basic_block目录下。下面该任务的详细说明：

- 读取agent工作流元文档理解项目情况。
- 文档[srs_gmmsq.md](./raw/development/srs_gmmsq.md)和[srs_coding_design.md](./raw/development/srs_coding_design.md)是该算子实现的设计文档和详细编码文档，请参考该文档进行编码。
- 目录[design](./raw/design/)中包含了算子原型的定义和规格约束，以及初版的kernel实现说明，总体来说grouped_matmul_swiglu_quant_v2算子MxA8W4数据流实现与grouped_matmul_finalize_routing算子的MxA8W4数据流实现高度类似，你可以参考代码gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block进行实现，或者基于已总结的[代码改写文档](./raw/docs/)进行编写，代码改写文档的优先级更高，如果需要再查看原始代码。
- 总体kernel代码实现比较复杂，整体架构分为controller、basic_block、cube/vector compute等几个部分，因此请逐步实现并检查代码，确保符合设计文档。
- 由于kernel部分代码无法直接运行进行验证，你只需要调用新实现的kernel类入口，并编译成功该任务就算成功。

> 注意：为了完成这个复杂的任务，请进行编排规划，通过subagent进行具体任务的执行。

下面是编译kernel二进制的方法：

``` shell
bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul_swiglu_quant_v2 -j12
```

## 任务完成状态

### Kernel部分 - 🟡 进行中
- [x] 创建gmmsq_weight_quant_cube_compute_tools.h（从GMMFR复制）
- [x] 创建gmmsq_mx_a8w4_vf.h（反量化+SwiGLU+Quant VF函数）
- [x] 创建gmmsq_weight_quant_vec_compute.h（VecCompute层）
- [x] 创建gmmsq_weight_quant_cube_compute.h（CubeCompute层）
- [x] 创建gmmsq_weight_quant_vcv_basic_block.h（BasicBlock层）
- [x] 创建gmmsq_weight_quant_resplit_controller.h（Controller层）
- [x] 更新apt.cpp kernel入口，连接完整Controller调用
- [x] 复制common头文件（basic_block_config.h, basic_block_vf_mx.h, tool.h）
- [x] 修复namespace解析问题（所有kernel文件的using声明）
- [x] 修复模板参数问题（hasBias参数添加）
- [ ] 修复MicroAPI调用问题（VF函数中的ShiftRight、Cast等API签名不匹配AscendC 9.1.0）
- [ ] 编译验证opkernel二进制

### 待解决问题
1. **VF函数MicroAPI错误**: `gmmsq_mx_a8w4_vf.h`中的`ComputeMaxExpVf`、`ComputeScaleVf`、`ComputeDataForQuantTargetFp8Vf`函数使用了不匹配的MicroAPI调用（ShiftRight with uint32_t, Cast with FP8类型转换）。需要参考AscendC 9.1.0文档修正API签名。
2. **Build状态**: ophost编译✅通过，opkernel编译❌失败（VF函数MicroAPI错误）
