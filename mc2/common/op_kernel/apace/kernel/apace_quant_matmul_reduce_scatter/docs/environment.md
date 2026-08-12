# 环境检查报告

**算子**: apace_quant_matmul_reduce_scatter_v2   **状态**: ✅ 通过   **时间**: 2026-08-11T10:00:00+08:00

## 硬件

| 项 | 值 |
|----|----|
| 芯片型号 | Ascend950PR |
| SocVersion | ASCEND950 |
| 设备数 | 4 |

> NPU Arch / `--npu-arch` 编译参数由 Architect 在 Step 2 通过 `/npu-arch` skill 查得后写入 DESIGN.md，本章节不填。

## CANN

- ASCEND_HOME_PATH: `/home/l00947526/Ascend806/cann-9.2.0`
- 版本: `9.2.0`
- CPU 架构目录: `x86_64-linux`

## 编译器与库

- ✅ bisheng: `/home/l00947526/Ascend806/cann-9.2.0/tools/bisheng_compiler/bin/bisheng`
- ✅ kernel_operator.h: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/asc/include/kernel_operator.h`
- ✅ libregister.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/devlib/linux/x86_64/libregister.so`
- ✅ libascendcl.so: `/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/lib64/libascendcl.so`

## asc-devkit

- ✅ 路径: `/home/y00845930/cann_skills/plugins-official/ops-direct-invoke/asc-devkit`
- API 文档: 2803 个
- 示例: 324 个
- CMake 配置: ✅

## 检查汇总

- 错误: 0
- 警告: 0
