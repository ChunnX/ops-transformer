# 环境信息

**算子**: quantMatmulReduceScatter   **状态**: ✅ 通过   **时间**: 2026-08-22

## 硬件

| 项 | 值 |
|----|----|
| 芯片型号 | Ascend 950PR |
| SocVersion | ASCEND950 |
| NPU Arch | dav-3510 |
| 设备数 | 4 |

## CANN

- ASCEND_HOME_PATH: `/home/l00947526/Ascend806/cann-9.2.0`
- 版本: `9.2.0`
- CPU 架构目录: `x86_64-linux`

## 编译器与库

- ✅ bisheng: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/ccec_compiler/bin/bisheng`
- ✅ kernel_operator.h: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/asc/include/kernel_operator.h`
- ✅ libascendcl.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libascendcl.so`
- ✅ libplatform.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libplatform.so`
- ✅ libhccl.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libhccl.so`

## MC2 框架 (CANN built-in)

框架路径: `/home/l00947526/Ascend806/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common`

- ✅ Apace 通信框架: `apace/block/aiv_comm/` (CollectiveComm, AllToAll PUT, TeamBarrier)
- ✅ Blaze MatMul 框架: `blaze/gemm/` (BlockMmad, BlockScheduler, DispatchPolicy)
- ✅ Tensor API: `tensor_api/` (FragmentTensor, Te::MakeCopy)
- ✅ AllToAll 通信实现: `apace/block/aiv_comm/all_to_all/`
- ✅ QuantMatmulMxKernel: `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`
- ✅ 参考算子: all_to_all_quant_matmul, all_gather_quant_matmul
- ✅ `--npu-arch=dav-3510`（Ascend 950PR）
- ✅ `ASC_DEVKIT_MAJOR=9` 条件编译可用

## 检查汇总

- 错误: 0
- 警告: 0
