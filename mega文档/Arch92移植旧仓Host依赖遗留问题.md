# Arch92 移植旧仓 Host 依赖遗留问题

## 背景

目标仓 `ops-transformer_api` 是较老但支持 Arch92 的代码仓。当前移植方式只复制 `mc2/mega_moe` 算子目录，不同步较新仓的公共模块和 opbase。仅复制算子目录不能保证所有外部依赖都存在。

## 当前临时处理

- [aclnn_mega_moe.cpp](../mc2/mega_moe/op_api/aclnn_mega_moe.cpp) 暂时屏蔽 Arch22 专用的 `ConvertTensorListToInt4`、架构判断及 `int32` 权重转 `int4` 逻辑，移除不再使用的 `op_const_def.h` 引用。缺省权重 scale 的占位类型固定为 `ACL_FLOAT8_E8M0`，四个 `CreateEmptyTensor` 调用继续保留。这与原代码在 Arch92 上的执行分支一致。
- 同一文件移除未使用的 `mc2_3rd_matmul_util.h` 引用及相应的 `using namespace Ops::Transformer`。这个头文件实际位于源仓 `mc2/3rd/common/op_host/op_api/`，只复制算子目录不会带上它。
- `mega_moe_infershape.cpp` 不使用 `op_const_def.h`，此前已移除该文件中的无用引用。

## 遗留问题

1. 当前 Host 预处理不再支持 Arch22。恢复 Arch22 时，需要解除上述代码屏蔽，恢复 `op_const_def.h` 并提供匹配版本的 opbase；也可以按目标仓现有的 `NpuArch::DAV_2201` 接口适配。必须恢复 Arch22 的 `ACL_UINT64` 缺省 scale 类型和 `int32` 到 `int4` 权重转换，不能仅恢复架构判断。
2. [MegaMoe CMake](../mc2/mega_moe/op_host/CMakeLists.txt) 仍声明 `mc2/common`、`mc2/3rd` 依赖；Kernel 和 tiling 也引用算子目录外的公共文件。目标仓需要逐项核对这些依赖。清理两个未使用的 Host include 不代表整个算子已能只靠 `mc2/mega_moe` 独立编译。
3. 本次只做源码依赖清理，尚未在目标旧仓执行 CANN 编译、模拟器或设备回归。后续编译错误应对照目标仓实际公共接口定位，避免直接删除仍在使用的头文件或逻辑。
