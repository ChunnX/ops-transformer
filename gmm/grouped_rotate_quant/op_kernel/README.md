# GroupedRotateQuant Ascend950 外层 kernel

当前设备入口已接入 ops-tensor 的 `Blaze::Attention::Kernel::GroupedRotateQuantKernel`。分组元数据解析和调度为新增实现，矩阵计算、Fixpipe 和 MX 后处理复用 RotateQuant 组件。**源码接线已完成，真实 CANN 设备编译、出包及 Ascend950 功能/精度/性能验证尚未完成。**

仅面向 Ascend950 / DAV_3510，支持范围沿用 ACLNN 直调和 Torch 单算子调用；不增加图模式能力。

## 文件与调用关系

| 文件 | 职责 |
| --- | --- |
| `grouped_rotate_quant_apt.cpp` | 与 OpDef 一致的设备入口；单 key=0；MIX AIC:AIV=1:2；建立 TPipe |
| `arch35/grouped_rotate_quant_kernel_arch35.h` | 读取 TilingData，检查版本，跳过全空输入，包装参数并调用一次后端 |
| `arch35/grouped_rotate_quant_kernel_args.h` | 算子自身的外层/内层参数契约，不冒充已有 Blaze Params |
| `arch35/grouped_rotate_quant_tensor_api.h` | 组装共享 MMAD/Epilogue 和分组 Kernel，逐字段转换库侧 Params |
| `arch35/grouped_rotate_quant_tiling_data.h` | Host/Device 共用的 128 字节、8 字节对齐协议 |

```text
grouped_rotate_quant(x, rotation, alpha, groupList, y, scale, workspace, tiling)
  -> GroupedRotateQuantKernel<DTYPE_X, DTYPE_Y>
    -> RunGroupedRotateQuant<TensorApiBackend<...>>
      -> Backend::Kernel::operator()(const KernelArguments &)
        -> Blaze::Attention::Kernel::GroupedRotateQuantKernel
          -> GroupResolver + BlockSchedulerGroupedRotateQuant
          -> BlockMmad<BlockRotateQuant<>> + BlockEpilogueRotateQuant
```

输出类型由 OpDef 的六种 dtype 组合选择，无额外模板 key；x/rotation 类型一致，scale 为 E8M0。入口不借用其它算子的转置、量化输入或 split-K 分支。

## TilingData 评估与约定

- 保留 v1 字段、顺序和 128 字节长度；增加显式布局与偏移断言。没有增加每组数组，组信息仍来自 TensorList 描述符和设备 groupList。
- L1 预算修正为 `4*mTile*R + 4*R*R`，覆盖 A 和 rotation 双缓冲。L0A/L0B/L0C 分别为 `4*mTile*R`、`4*R*R`、`8*mTile*R`。共享 rotation 也保守预留两份。
- `runtimeM=1` 时 `tileCount=0` 表示任务数待设备计算，不能当成空输入。只有 `totalElements=0` 才走外层全空快捷路径。
- `splitItem` 是归一化后的列表模式。单 group 时它可能为 0，而 `runtimeM` 仍为 1；后端必须以 `runtimeM` 判断组长来源。
- `totalM/totalElements/tileCount` 使用 64 位；每个实际 R 选择自己的 rotation profile，不能假定所有组共用一个 R、N 或 M。
- UB 预算仍以现有 RotateQuant epilogue 的固定布局为前提。实际分组后端若改变 buffer 或 workspace，必须同步更新 planner，不能把当前预算当成已通过 NPU 验证。

## 后端接入契约

`TensorApiBackend<XType,YType>` 已提供 `AVAILABLE=true` 及实际组件绑定。库侧参数与算子 tiling ABI 分离，禁止 reinterpret_cast 成单组 RotateQuant tiling。TilingData 的版本、长度、字段和资源预算保持不变。

1. `xList/rotationList/yList/scaleList` **始终是四个独立 TensorList 描述符地址**，即使列表长度为 1。只有 `alpha/groupList` 是普通数据地址；未提供时外层统一传 nullptr。
2. 一次后端调用覆盖本核全部 group；AIC 和 AIV 均进入后端。解析 groupList、检查组长/容量、跨组调度、同步和 rotation 缓存管理属于后端职责。禁止在此入口循环启动单组 RotateQuant。
3. `KernelArguments::tiling` 引用当前调用中的只读数据；后端必须在返回前完成消费，不得保留引用供后续调用使用。
4. 保持 FP32 累加、BF16 中间值、clamp 和 MX 量化的 RotateQuant 语义；FP4 打包、E8M0 padding、异常值和舍入不能借用不同量化输入算子的默认行为。需要的浮点控制状态及恢复由真实后端统一管理。
5. 当前无用户 workspace，入口不会向后端传递系统 workspace 基址；不分配全量中间结果。新增用户 scratch 时必须同时修改 Host workspace 规划和参数契约。
6. 后端模板覆盖 FP16/BF16 × FP4_E2M1/FP8_E4M3FN/FP8_E5M2 六种组合。所有 AIC/AIV 在计算及发布同步事件前执行相同的描述符/组表检查；设备断言不等同于 GetWorkspaceSize 阶段的同步参数错误码。

## 分组执行与依赖

- 每组任务拼成全局序列，以 `globalTile % usedCoreNum` 分配，空组和本核没有任务的组仍推进组遍历。
- 两个 AIV 交替处理完整 tile；每个 tile 都回复 ACK，包括本组最后一个。AIC 在组末消费 ACK 并排空本核流水后，才允许改变 MMAD 的 L1 布局和 rotation 缓存。
- `BlockMmad::Init(params, firstAiv)` 继承本核跨组 AIV 奇偶位置；原单组入口 `Init(params)` 的行为不变。
- 不新增全量中间结果或用户 workspace。首版选择明确的组边界等待，性能需要设备实测。

本地开发依赖同级 `ops-tensor_gmm_rotate_quant`，其中包含 PR 465 及分组扩展。也可以给 CMake 显式传入 `-DOPS_TENSOR_SOURCE_DIR=/path/to/ops-tensor_gmm_rotate_quant`，覆盖默认依赖解析。显式目录和开发同级目录不会被构建脚本 checkout。正式发布前需将分组库改动集成到可用的 ops-tensor 提交，再更新默认依赖版本；原 `deea03a7` 不包含此后端。

## 验证

`../tests/ut/op_kernel/run_outer_contract_tests.py` 使用普通 C++17 编译器验证实际外层及参数适配器：六种 dtype 组合、四种 split 模式、两种 groupList 编码、runtimeM 的零任务数哨兵、独立输出描述符、可选输入、全空输入及 ABI 布局。底层组件使用记录替身。

ops-tensor 的 `tests/ut/host/grouped_rotate_quant` 另行运行真实分组解析器/调度器，以及使用记录型计算组件和三个 Host 线程模拟 AIC/AIV 事件的真实 Kernel 编排。两类测试都不执行 MMAD/量化设备指令，不替代 CANN 编译或 NPU 数值和性能验证。

```text
python tests/ut/op_kernel/run_outer_contract_tests.py --compiler g++ --ops-tensor /path/to/ops-tensor_gmm_rotate_quant
python tests/ut/op_kernel/run_outer_contract_tests.py --compiler <zig.exe> --compiler-arg c++
```
