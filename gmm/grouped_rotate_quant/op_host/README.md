# GroupedRotateQuant Host 与 tiling 设计

目标平台为 Ascend 950 / DAV_3510。本目录实现算子定义、shape/dtype 推导、元数据校验及 Tensor API 路径的 tiling；Tensor API kernel 尚待实现和联调。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `grouped_rotate_quant_def.cpp` | 注册动态 x/rotation、可选 alpha/group_list、动态 y/scale；注册十个属性及 ascend950 配置 |
| `grouped_rotate_quant_context.h` | GE context 适配、IR 实例索引、属性读取和 dtype/format 校验 |
| `grouped_rotate_quant_model.h` | 输入输出 shape 契约、符号推导、整数溢出检查；与 GE context 解耦 |
| `grouped_rotate_quant_infershape.cpp` | 注册 InferShape / InferDataType，分别处理 y 与 scale 两个动态列表 |
| `op_tiling/arch35/grouped_rotate_quant_tiling.h/.cpp` | CompileInfo、TilingBaseClass 实现、平台读取、workspace、tilingData 保存及注册入口 |
| `op_tiling/arch35/grouped_rotate_quant_planner.h` | 资源约束、三种 R 的分块配置、静态任务数及启动核数计算 |
| `../op_kernel/arch35/grouped_rotate_quant_tiling_data.h` | Host/device 共用的固定 128 字节协议；当前仅有协议头文件 |

根 CMake 通过 `op_host/CMakeLists.txt` 的 `add_modules_sources(OP_API_INDEPENDENT ON ...)` 接入 API、推导及 tiling，替代 API 阶段的临时源文件注册。当前只有一个 Ascend950 策略，直接调用 `GroupedRotateQuantTiling950::DoTiling()`，无需 GMM 的多模板回退或量化输入 scale tiling。

## 接口与推导

输入顺序为 `x, rotation, alpha, group_list`；输出为 `y, scale`。属性顺序严格对应 L0：

`split_item, group_type, group_list_type, axis, round_mode, scale_alg, dst_type_max, trans, dst_type, tuning_config`。

支持独立 Tensor（group_type=-1）与 M 分组（group_type=0），覆盖 split_item=0/1/2/3，列表最多 1024 组。独立模式允许各组 N、R、rank 不同；M 分组要求二维且 N 相同。旋转矩阵支持组间共享或逐组提供，每个为 `[R,R]` 或 `[N/R,R,R]`，R 为 32/64/128。其余 dtype、round、scale_alg、alpha、axis、trans 约束与现有 op_api 一致。

动态输出的扁平索引必须通过 `GetIrOutputInstanceInfo(Y/SCALE)->GetInstanceStart()` 获取。不能把 scale 误当作扁平索引 1：例如三个 y 后面才是第一个 scale。

| 输入/输出表示 | y 的 shape 推导 |
| --- | --- |
| 独立 Tensor | 逐项保留 x 的 shape |
| 多 x、多 y | 逐项保留 `[M_g,N]` |
| 多 x、单 y | `[sum(M_g),N]` |
| 单 x、单 y | 保留 `[M_total,N]` |
| 单 x、多 y，E>1 | 每项 `[-1,N]`，Host 不读取设备 group_list |

每个 scale 的 shape 为对应 y 的 `[...,ceil(N/64),2]`。已知 rank 下的未知维度 -1 在推导阶段保留；未知 rank 不在本版支持范围。tiling 需要具体输入和输出描述符，且 ND 的 origin/storage shape 一致。

ACLNN 路径已由调用方提供具体输出 shape，L0 按这些 shape 分配 Tensor，并不依赖上述符号输出分配。**单 x、多 y 的图模式还需要图框架提供可分配的具体输出 shape；本 Host 实现不把设备 group_list 声明为 Host 值依赖，也不完成数据依赖的图模式输出分配。**

## 分块与资源约束

组内保持 RotateQuant 的旋转语义。记旋转矩阵边长为 R，列方向块数为 B=N/R：

- 二维 rotation 或 B=1：共用一个旋转矩阵，将组内输入视作 `[M_g*B,R]`，任务数为 `ceil(M_g*B/mTile)`。
- 三维 rotation 且 B>1：各列块使用不同旋转矩阵，任务数为 `ceil(M_g/mTile)*B`。
- 不合并不同专家的计算或量化范围；组间任务组合成一个全局任务序列。

K/N 方向基础块均为 R，不进行 K 累加分块；M 方向块长 16 对齐。对每个被使用的 R，资源上限同时满足：

```text
L0A: 4*mTile*R <= L0A_size         # 两份 FP16/BF16 A
L0B: 4*R*R <= L0B_size            # 两份 FP16/BF16 rotation
L0C: 8*mTile*R <= L0C_size        # 保守沿用半 L0C 限制
L1:  4*mTile*R + 2*R*R <= L1_size # A 双缓冲 + rotation 单缓冲
UB:  mTile*R <= 16384             # 每个 AIV 处理完整 tile
UB:  52736 + 32*mTile <= UB_size  # 固定区域 + scale 行缓冲
```

UB 模型对应当前 tensor 仓 `block_epilogue_rotate_quant.h` 的固定布局：BF16 输入 32768 B、输出 16384 B、三个 1024 B 辅助区、512 B scale 区，以及最后的 scale 排布区域。后续若修改 epilogue 内存布局，必须同时更新这里的资源模型。

对相同 R 的各组累计工作量 `M_g*(N_g/R)`，再除以 AIC 核数，16 对齐并截到资源上限，得到该 R 的 mTile。此策略使多个小专家共同参与核负载估计。单 x 的 M 分组使用 tuning_config[0]（非零时）或 `ceil(M_total/E)` 估计工作量；该估计只影响 mTile。

多 x 的组长来自描述符，Host 可以计算准确任务总数，启动 `min(tileCount,AIC_num)` 个核。单 x 的 M 分组启动全部 AIC，由 device group_list 决定各组任务。全空输入的 Host tiling 使用一个核并标记 totalElements=0；ACLNN 的全空快捷路径仍不提交计算。

workspace 预留 16 MiB 系统空间。计划中的计算沿用 L0C→UB→GM 路径，不申请原 RotateQuant Host 中的全量 M*N FP32 临时区；后续 kernel 若新增 GM 临时数据，必须同步扩展 workspace 规划。

## Host/device 协议

tilingData 仅保存列表长度、分组/量化属性、输入总容量、runtimeM 标记、启动核数，以及 R=32/64/128 三个配置。未使用配置和保留字段清零。dtype 由编译描述符选择，tilingKey=0。每组 shape 已存在于 GM TensorList 描述符，避免重复序列化大数组。

后续 kernel 必须完成以下工作：

1. 校验协议版本，按动态 TensorList 描述符解析 x/rotation/y/scale 地址及逻辑 shape；独立模式支持 rank 1..7 的 M 展平。
2. group_list_type=0 时对累计端点作差，=1 时读取组长并累计。单 x 的每组 M 始终来自设备组表；多 x 可以没有组表，有组表时仍需检查值与描述符一致。
3. 在读写之前验证组长非负、累计值未溢出、总 M 完整覆盖，以及各独立输出的 M/容量与组表一致。所有参与同步的 AIC/AIV 必须采用一致的有效性判断与退出路径。
4. 从实际 R 选择配置，按上述两种组内公式计算真实任务数、尾块和地址偏移；tuning_config 估计值不能参与实际 shape/循环边界计算。地址与总量运算使用 64 位；复用旧 block/epilogue 时需审查内部窄整数是否覆盖本协议的范围。
5. 跳过空组、维持跨组任务分配；scheduler 对外提供块 shape/coord。组间切换 R、shape 或 rotation 地址时重新初始化相关 block/epilogue 状态，并正确排空同步与缓存，不能沿用上一专家的旋转矩阵缓存。
6. 量化 scale 的分组、奇数 scale padding、FP4 打包、alpha/round/scale_alg 行为需与 RotateQuant 对齐。

因此本协议可以用于后续复用 RotateQuant 的 block MMAD 和 epilogue，但不能直接传给现有单组 `KernelRotateQuant` 或其 scheduler。

## 验证

`tests/ut/op_host` 包含元数据/分块模型测试，以及通过仓库 context faker 执行的注册入口测试，覆盖两种旋转布局、四种 split 模式、动态 M、空组、三种 R、异构 shape、128 字节协议、输出容量、dtype、平台及缓冲不足。

本地 Windows 验证使用 g++ C++17 与最小 GTest/GE 接口替身执行模型及 Host 入口检查；替身用于检查本地 C++ 语法与控制流程，不能代替真实 CANN SDK 编译、仓库 UT 和 NPU 精度/性能测试。接入 Tensor API kernel 后仍需完成这些验证。
