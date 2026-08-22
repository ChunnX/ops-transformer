# 环境检查报告

**算子**: quantMatmulReduceScatter   **状态**: ✅ 通过   **时间**: 2026-08-15T00:00:00+08:00

## 硬件

| 项 | 值 |
|----|----|
| 芯片型号 | Ascend 950PR |
| SocVersion | ASCEND950 |
| 设备数 | 4 |

> NPU Arch / `--npu-arch` 编译参数由 Architect 在 Step 2 通过 `/npu-arch` skill 查得后写入 DESIGN.md，本章节不填。

## CANN

- ASCEND_HOME_PATH: `/home/l00947526/Ascend806/cann-9.2.0`
- 版本: `9.2.0`
- CPU 架构目录: `x86_64-linux`

## 编译器与库

> 标 ✅ 的前提是 `test -x <path>` 通过（对可执行文件）或 `test -f <path>` 通过（对头文件/.so）；存在但不可执行 → ❌。

- ✅ bisheng: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/bin/bisheng`
- ✅ kernel_operator.h: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/asc/include/kernel_operator.h`
- ✅ libregister.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libregister.so`
- ✅ libascendcl.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libascendcl.so`

## asc-devkit

- ✅ 路径: `./asc-devkit`
- API 文档: 2828 个
- 示例: 325 个
- CMake 配置: ✅

## 检查汇总

- 错误: 0
- 警告: 0

### MC2 框架可用性确认

- ✅ Apace 框架: `/home/l00947526/Ascend806/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`
- ✅ Blaze 框架: `ops_nn/ascendc/common/blaze/` + `ops-tensor/include/blaze/`
- ✅ AllToAll 通信实现: `apace/block/aiv_comm/all_to_all/`
- ✅ QuantMatmulMxKernel: `apace/kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`
- ✅ 参考算子: all_to_all_quant_matmul, all_gather_quant_matmul
- ✅ `--npu-arch=dav-3510`（Ascend 950PR）
- ✅ `ASC_DEVKIT_MAJOR=9` 条件编译可用
