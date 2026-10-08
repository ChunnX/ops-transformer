# GroupedRotateQuant op_api 契约

本目录实现公开两段式接口、参数校验、输入连续化、L0 launcher 提交及双输出回写。
目标为 Ascend 950（DAV_3510）的 MXFP4/MXFP8 路径。
已生成 `GroupedRotateQuant` 的 op_host 定义、推导与 Ascend950 tiling；尚需实现 Tensor API kernel，才能执行非空计算。Host/device 协议见 [Host 与 tiling 设计](../op_host/README.md)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `aclnn_grouped_rotate_quant.h/.cpp` | 公开 ABI、executor 生命周期、Contiguous、双输出 ViewCopy |
| `grouped_rotate_quant_checker.h/.cpp` | 仅校验 Host 可见元数据，归一化属性 |
| `grouped_rotate_quant.h/.cpp` | 内部参数结构、输出分配及 L0 注册/提交 |

## 输入与分组

- `x`、`rotation`、`yOut`、`scaleOut` 均使用 TensorList；最多 1024 组，列表元素不可为空指针。
- `alpha` 沿用 RotateQuant 的可选 BF16 `[1]` Tensor，所有组共享。
- `groupType=-1`：每个 x 独立执行 RotateQuant，支持 rank 1..7；输出与 x 逐项对应。
- `groupType=0`：沿 M 轴分组，x/y 必须为二维，各组 N 相同；输入/输出可分别采用分组列表或拼接 Tensor。
- 每组 rotation 是 `[R,R]` 或 `[N/R,R,R]`，`R=32/64/128`。
  rank-3 的首维保持原 RotateQuant 的列方向旋转块语义。
  多个专家使用多个列表元素表示；列表仅一个元素时，该旋转 Tensor 在组间共享。
- `groupType=1/2`（N/K 分组）及 `groupListType=2`（稀疏组表）明确返回参数错误。

`splitItem` 与 GMM V5 一样，公开入口的 0/1 表示分组输出、2/3 表示拼接输出。
传给 L0 前根据输入/输出列表长度归一化：

| 内部 splitItem | x | yOut / scaleOut |
| --- | --- | --- |
| 0 | E 个 Tensor | E 个 Tensor |
| 1 | 1 个拼接 Tensor | E 个 Tensor |
| 2 | E 个 Tensor | 1 个拼接 Tensor |
| 3 | 1 个拼接 Tensor | 1 个拼接 Tensor |

单组列表存在等价表示，归一化方式与 GMM V5 相同。`groupType=-1` 只允许分组输出。
M 分组且 x 为单 Tensor 时必须提供 `groupListOptional`；多 x 时可从描述符得到组长，组表可省略。

## groupList 的运行时契约

`groupListOptional` 是设备 INT64 ND `[E]` Tensor。`groupListType=0` 为各组累计结束位置，
`groupListType=1` 为各组行数。组长须非负，允许空组，累计终点/组长之和须等于总 M。
若 x 或 y 使用独立 Tensor，组长还须与对应描述符的 M 一致。

op_api 校验 rank、dtype、E、声明的存储范围、列表长度及输入输出总 M；
不解引用组表数据，不进行 D2H 拷贝，也不使用 tuningConfig 推测实际组长。
**设备组值与各输出容量的一致性必须由后续 kernel 处理；op_api 返回成功并不证明设备组值有效。**
`tuningConfigOptional[0]` 仅为 tiling 提供预期 M，校验范围与 GMM V5 一致。

## 量化和输出

- 输入 x/rotation 为同一 FLOAT16 或 BFLOAT16 类型；所有输出 y 使用同一 MX 类型。
- MXFP4：FLOAT4_E2M1，scaleAlg 为 0/2，roundMode 为 rint/round/floor。
- MXFP8：FLOAT8_E4M3FN 或 FLOAT8_E5M2，scaleAlg 为 0/1，roundMode 为 rint。
- scaleAlg=2 时 dstTypeMax 属于 `[6,12]`；其他情况必须为 0。alpha 的设备值契约与 RotateQuant 相同。
- axis 指向每个 x 的末维，内部归一化为 -1；trans 仅支持 false。
- y 的逻辑末维仍是 N，FP4 不使用 INT32 打包接口；scale 为 E8M0，shape 为 `[...,ceil(N/64),2]`。
- L0 根据已校验的输出描述符分配连续 Tensor，aclnn 层分别对 y/scale 的每个非空元素做 ViewCopy。
  对于单 x、多 y，Host 无法从设备组表推导各 `M_i`，因此这里不调用依赖组值的 INFER_SHAPE。
- 所有参数校验完成后，只有全部输出为空才返回零 workspace；混合空组交给 kernel 跳过。

## op_host 与 kernel 采用的接口顺序

注册类型名：`GroupedRotateQuant`。

| 类别 | 顺序 |
| --- | --- |
| 输入 | x（动态 TensorList）、rotation（动态 TensorList）、alpha（可选 Tensor）、groupListOptional（可选 Tensor） |
| 输出 | y（动态 TensorList）、scale（动态 TensorList） |
| 属性 | splitItem、groupType、groupListType、axis、roundMode、scaleAlg、dstTypeMax、trans、dstType、tuningConfig |

整数属性使用 int64；`dstTypeMax` 提交为 float；`trans` 为 bool；`roundMode` 为字符串；
`dstType` 是输出 y 的 DataType 枚举值；`tuningConfig` 是可选整数列表。
L0 始终提交一个融合算子任务，不在 Host 循环调度 E 次 RotateQuant。

根目录 CMake 已通过 op_host 中的 `add_modules_sources(OP_API_INDEPENDENT ON ...)` 统一接入 API、推导与 tiling，并删除临时 API 源文件注册。
`tests/ut/op_api` 为元数据校验测试，可在具备 CANN/opbase 的仓库 UT 环境运行。
