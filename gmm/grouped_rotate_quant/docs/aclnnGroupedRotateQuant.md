# aclnnGroupedRotateQuant

[接口声明](../op_api/aclnn_grouped_rotate_quant.h) · [参数校验](../op_api/grouped_rotate_quant_checker.cpp) · [op_api 实现说明](../op_api/README.md)

> 开发状态：op_api、op_host、Ascend950 tiling 及分组 Tensor API kernel 已完成源码接线。Host/device 约定见 [Host 与 tiling 设计](../op_host/README.md)，本地 ops-tensor 依赖与验证范围见 [kernel 说明](../op_kernel/README.md)。真实 CANN 设备编译、出包、功能精度、性能及确定性仍待验证。

## 产品支持情况

| 产品 | 当前状态 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | 目标平台；op_api 接受 DAV_3510，完整算子待适配验证 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 不支持 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 不支持 |
| Atlas 200I/500 A2 推理产品 | 不支持 |
| Atlas 推理系列产品 | 不支持 |
| Atlas 训练系列产品 | 不支持 |

## 功能说明

GroupedRotateQuant 对多个分组分别执行分块旋转、可选的 clamp 和 MX 量化，输出量化数据及对应的块缩放因子。适用于多个专家或多个独立张量的旋转量化场景。

首版提供两种分组方式：

- `groupType=-1`：`x` 中每个 Tensor 独立执行 RotateQuant，输出逐项对应，允许不同的 shape。
- `groupType=0`：二维输入沿 M 维分组，各组具有相同的列数 N，支持输入和输出分别采用多个 Tensor 或单个拼接 Tensor。

分组接口参考 GMM V5 的 TensorList、`splitItem` 和 `groupList` 约定；组内计算沿用 RotateQuant 的 MX 路径。当前不支持 N/K 分组、稀疏分组、INT8/INT32 量化输出及输出转置。

### 符号与三种分组

| 符号 | 含义 |
| --- | --- |
| E | 专家/独立计算组数，取值范围为 [1, 1024] |
| M_g | 第 g 组的行数；独立高维输入按末维以外的维度展平得到行数 |
| N_g | 第 g 组的末维长度；M 分组时所有 N_g 相同，记为 N |
| R_g | 第 g 组的旋转矩阵边长，取值为 32、64 或 128 |
| B_g | 第 g 组沿列方向的旋转块数，等于 N_g / R_g |
| X_g、Z_g、V_g、P_g | 第 g 组的输入、旋转结果、clamp 后结果、量化结果 |

专家分组决定各组使用哪些输入行和旋转 Tensor；旋转分块把每行切成长度为 R_g 的块；MX 量化进一步以每行连续 32 个元素为一组共享 scale。这三者具有不同的含义。

### 1. 分块旋转

第 g 组内，对每一行、每一个旋转块分别进行矩阵乘法：

$$
Z_g[m,bR_g+j]=\sum_{k=0}^{R_g-1}X_g[m,bR_g+k]\,A_g^{(b)}[k,j]
$$

其中，`0 <= m < M_g`，`0 <= b < B_g`，`0 <= j < R_g`。旋转结果保持该组输入的逻辑 shape。

- 当 `rotation[g]` 为 `[R_g, R_g]` 时，同一组所有列块共享该矩阵。
- 当 `rotation[g]` 为 `[B_g, R_g, R_g]` 时，第 b 个列块使用第 b 个矩阵。
- 当 `rotation` 列表长度为 1 时，所有组共享该 Tensor；长度为 E 时，每组使用对应元素。

三维旋转 Tensor 的首维 B_g 表示**列方向旋转块数**。逐专家旋转矩阵由 TensorList 的 E 个元素表示。

### 2. 可选 clamp

沿每行的末维，以连续 32 个元素为一个量化块。对第 q 块，定义：

$$
a_{g,m,q}=\max_{0\le t<32}|Z_g[m,32q+t]|
$$

当提供 `alpha` 且 `0 < alpha < 1` 时：

$$
V_g[m,32q+t]=\operatorname{clamp}\bigl(Z_g[m,32q+t],-\alpha a_{g,m,q},\alpha a_{g,m,q}\bigr)
$$

`alpha` 为所有组共享的一个 BFLOAT16 标量 Tensor。仅 `0 < alpha < 1` 时执行 clamp，其余值及空指针均跳过，沿用 RotateQuant 的设备判定。op_api 仅检查其描述符，不读取数值。

### 3. MX 量化

每 32 个 clamp 后的元素共同计算一个缩放因子，量化结果写回对应位置：

$$
A_{g,m,q}=\max_{0\le t<32}|V_g[m,32q+t]|
$$

$$
S_{g,m,q}=\operatorname{MXScale}(A_{g,m,q},\mathrm{dstType},\mathrm{scaleAlg},\mathrm{dstTypeMax})
$$

$$
P_g[m,32q+t]=\operatorname{Cast}_{\mathrm{dstType},\mathrm{roundMode}}\left(V_g[m,32q+t]/S_{g,m,q}\right)
$$

`dstType` 由 `yOut` 的数据类型确定；scale 使用 FLOAT8_E8M0 编码。各算法的目标语义如下：

| scaleAlg | yOut 类型 | scale 计算方式 |
| --- | --- | --- |
| 0 | FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2 | 共享指数算法；对普通有限非零最大值，`shared_exp = floor(log2(A)) - emax`，`S = 2^shared_exp` |
| 1 | FLOAT8_E4M3FN、FLOAT8_E5M2 | 由块最大绝对值与目的类型最大值之比得到浮点 scale，再按 RotateQuant 的 E8M0 指数取整规则转换 |
| 2 | FLOAT4_E2M1 | 使用 `dstTypeMax` 选择 scale；6.0/7.0 采用 RotateQuant 对应的尾数阈值分支，其余有效值采用浮点 scale 到 E8M0 的转换分支 |

`scaleAlg=0` 的 emax 对 FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2 分别为 2、8、15。`scaleAlg=1/2` 的详细数值规则参考 [RotateQuant 的 MX 量化说明](../../../../../rotate_quant_blaze/ops-nn_rotate_quant_blaze/matmul/rotate_quant/docs/aclnnRotateQuant.md)。实现采用 FP32 累加，再经 Fixpipe 转为 BF16 后执行 clamp 和 MX 量化；全零块、NaN/Inf、指数边界及 scale padding 复用 RotateQuant 实现，仍需设备精度验证。

### 4. 输出布局

对任意一个输出 Tensor，若 `yOut[i].shape = [..., N_i]`，则：

```text
scaleOut[i].shape = [..., ceil(N_i / 64), 2]
```

有效量化块 q 的 scale 对应 `scaleOut[i][..., q / 2, q % 2]`，其中 `/` 为整数除法。一个末维长度为 2 的 scale 对应连续 64 个数据中的两个 32 元素块。

例如，`yOut[i]` 为 `[5, 128]` 时，`scaleOut[i]` 为 `[5, 2, 2]`；`yOut[i]` 为 `[5, 96]` 时，`scaleOut[i]` 为 `[5, 2, 2]`，每行仅有 3 个有效 scale。后一种情况的末尾多余 scale 槽位不对应有效输入块，调用方不应将其作为有效量化参数。

## 函数原型

接口采用[两段式调用](../../../docs/zh/context/two_phase_api.md)。首先调用 `aclnnGroupedRotateQuantGetWorkspaceSize` 获取 workspace 大小及执行器，然后调用 `aclnnGroupedRotateQuant` 提交计算。

```cpp
aclnnStatus aclnnGroupedRotateQuantGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *rotation,
    const aclTensor *alpha,
    const aclTensor *groupListOptional,
    int64_t splitItem,
    int64_t groupType,
    int64_t groupListType,
    int64_t axis,
    const char *roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    bool trans,
    const aclIntArray *tuningConfigOptional,
    aclTensorList *yOut,
    aclTensorList *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnGroupedRotateQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnGroupedRotateQuantGetWorkspaceSize

### 参数说明

表中 shape 和非连续支持均针对 TensorList 的各个元素；所有 Tensor 使用 ND 存储格式。非空指针的 Tensor 可以是空 Tensor，前提是满足下述 shape 约束；空 Tensor 与空 TensorList、空指针具有不同含义。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x（const aclTensorList*） | 输入 | 各组输入数据 X | 必选；列表及元素指针非空，长度 1～1024；允许符合约束的空 Tensor；列表内 dtype 一致 | FLOAT16、BFLOAT16 | ND | 独立模式：1～7 维；M 分组：二维 `[M_i,N]` 或单个 `[M,N]` | √ |
| rotation（const aclTensorList*） | 输入 | 旋转矩阵 A | 必选；长度为 1 或 E；dtype 与 x 相同；R 为 32/64/128 且整除对应 N；二维 Tensor 非空，三维仅在 N=0 时可为 `[0,R,R]` | 同 x | ND | 二维 `[R,R]` 或三维 `[N/R,R,R]` | √ |
| alpha（const aclTensor*） | 输入 | 所有组共享的 clamp 系数 | 可为空指针；非空时必须为 `[1]`，不支持空 Tensor；仅设备值满足 `0 < alpha < 1` 时 clamp，其余值及空指针跳过 | BFLOAT16 | ND | 一维 `[1]` | √ |
| groupListOptional（const aclTensor*） | 输入 | 各组 M 的设备分组信息 | 独立模式必须为空指针；M 分组单 x 时必选、多 x 时可省略；提供时不支持空 Tensor | INT64 | ND | 一维 `[E]` | √ |
| splitItem（int64_t） | 输入 | 输出分组方式 | 0/1 表示 E 个输出 Tensor，2/3 表示一个拼接输出；输入布局由 x 的列表长度确定；独立模式仅支持 0/1 | - | - | - | - |
| groupType（int64_t） | 输入 | 分组维度/模式 | -1：独立 Tensor；0：M 分组；其余值不支持 | - | - | - | - |
| groupListType（int64_t） | 输入 | groupList 的编码方式 | 0：累计结束位置；1：各组行数；groupList 为空时仍须为 0/1 | - | - | - | - |
| axis（int64_t） | 输入 | 量化轴 | -1，或每个 x 的最后一维索引；所有输入必须同时满足，内部归一化为 -1 | - | - | - | - |
| roundMode（const char*） | 输入 | 量化转换舍入模式 | 空指针默认使用 `"rint"`；FP4 支持 `"rint"`、`"round"`、`"floor"`，FP8 仅支持 `"rint"` | - | - | - | - |
| scaleAlg（int64_t） | 输入 | MX scale 算法 | FP4 支持 0/2，FP8 支持 0/1 | - | - | - | - |
| dstTypeMax（double） | 输入 | 算法 2 的目的范围参数 | scaleAlg=2 时取有限值 [6,12]，否则必须为 0；内部以 float 属性传递 | - | - | - | - |
| trans（bool） | 输入 | 是否转置输出 | 当前仅支持 false | - | - | - | - |
| tuningConfigOptional（const aclIntArray*） | 输入 | Host 侧调优提示 | 可为空指针或空数组；非空数组的第 0 项为预期 M，范围见约束说明；其余项预留 | INT64 | - | 一维整数数组 | - |
| yOut（aclTensorList*） | 输出 | 量化结果 P | 必选；列表及元素指针非空；列表内 dtype 一致；支持符合约束的空 Tensor；调用前提供正确的 shape 和输出存储 | FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND | 与输入相应，M 维按 splitItem 拆分或拼接；1～7 维 | √ |
| scaleOut（aclTensorList*） | 输出 | 各 32 元素块的缩放因子 S | 必选；列表长度与 yOut 相同且逐项对应；支持符合约束的空 Tensor；调用前提供正确的 shape 和输出存储 | FLOAT8_E8M0 | ND | `[...,ceil(N_i/64),2]`；2～8 维，M 分组为三维 | √ |
| workspaceSize（uint64_t*） | 输出 | 所需设备 workspace 字节数 | 必选，指针非空；成功返回后读取 | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 包含执行流程的执行器 | 必选，指针非空；成功返回后用于第二段接口 | - | - | - | - |

### 返回值

返回 `aclnnStatus`，常见返回码如下；公共框架返回码参见 [aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

| 返回码 | 数值 | 描述 |
| --- | --- | --- |
| ACLNN_SUCCESS | 0 | 本阶段成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选指针或 TensorList 元素为空；M 分组单 x 未提供 groupList |
| ACLNN_ERR_PARAM_INVALID | 161002 | 平台、dtype、shape、格式、存储描述、列表长度或属性组合不满足约束 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | 执行器创建失败 |
| ACLNN_ERR_INNER_NULLPTR | 561103 | 连续化、内部 Tensor 分配、算子提交或输出回写准备失败 |

op_api 不读取设备 `groupListOptional`、`alpha` 的数值，因此返回成功不表示这两者的设备值已通过检查。

## aclnnGroupedRotateQuant

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace（void*） | 输入 | 设备 workspace 地址；workspaceSize 为 0 时可为空指针 |
| workspaceSize（uint64_t） | 输入 | 第一段接口返回的 workspace 字节数 |
| executor（aclOpExecutor*） | 输入 | 第一段接口返回的执行器，必须非空；同一执行器仅调用一次第二段接口 |
| stream（aclrtStream） | 输入 | 执行任务的 Stream |

### 返回值

返回 `aclnnStatus`。`executor` 为空时返回 `ACLNN_ERR_PARAM_NULLPTR`，其余执行错误由公共执行框架返回，详见 [aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

第二段接口提交异步任务。读取输出或释放计算依赖的设备内存前，需要确保对应 Stream 上的计算已完成。

## 约束说明

### 1. 独立 Tensor 模式

当 `groupType=-1` 时：

- E 等于 x 的列表长度；`groupListOptional` 必须为空指针，`splitItem` 仅可取 0/1。
- `yOut`、`scaleOut` 均包含 E 个 Tensor；`yOut[g].shape` 必须等于 `x[g].shape`。
- 各组允许不同的 rank、行数、列数和旋转块大小；所有 x/rotation 的 dtype 相同，所有 yOut 的 dtype 相同。
- rank 为 1 时视为单行参与计算，输出保留一维 shape；不同 rank 的输入应使用 `axis=-1`。
- 共享 rotation 必须同时满足每个 x 的旋转 shape 约束。例如，不同 N 的组可以共享兼容的二维矩阵；共享三维 rotation 时，首维必须同时等于各组的 N/R。

### 2. M 分组及 splitItem

当 `groupType=0` 时，x 和 yOut 的元素均为二维 Tensor，所有组的 N 一致。总行数 M 为输入行数之和，也必须等于输出行数之和。

若提供 groupList，E 由其 shape `[E]` 确定；否则 E 由 x 的列表长度确定，且此时 x 必须包含多于一个 Tensor。单 x 即使只有一组，也必须提供 groupList。

下表列出 E>1 时的四种布局。公开接口中的 0/1 等价地选择分组输出，2/3 等价地选择拼接输出；op_api 再根据 x 的列表长度归一化内部属性。

| 输入布局 | 输出布局 | 公开 splitItem | 内部归一化值 | groupListOptional |
| --- | --- | --- | --- | --- |
| E 个 `[M_g,N]` | E 个 `[M_g,N]` | 0 或 1 | 0 | 可省略 |
| 1 个 `[M,N]` | E 个 `[M_g,N]` | 0 或 1 | 1 | 必须提供 |
| E 个 `[M_g,N]` | 1 个 `[M,N]` | 2 或 3 | 2 | 可省略 |
| 1 个 `[M,N]` | 1 个 `[M,N]` | 2 或 3 | 3 | 必须提供 |

`scaleOut` 的列表长度、行拆分方式与 yOut 一致，元素 shape 根据对应 yOut 推导。E=1 时，拆分与拼接在列表长度上等价；内部仍按实际列表长度归一化。

### 3. groupList 的值域与运行时约束

`groupListOptional` 是设备 Tensor。两种编码都按列表顺序排列各组：

- `groupListType=0`：元素为累计结束位置。设 `s_0=0`，则第 g 组起点为前一组结束位置，行数为相邻结束位置之差。结束位置须非负且单调不减，最后一个值必须等于总 M。
- `groupListType=1`：第 g 个元素直接等于 M_g；每项须非负，所有元素之和必须等于总 M，起点由前缀和得到。

例如，三组行数为 `[2,0,3]`、总 M=5 时，累计编码为 `[2,2,5]`，行数编码为 `[2,0,3]`。允许零行组；组表必须完整覆盖全部输入行。

当输入或输出以多个 Tensor 表示时，组表中的每个 M_g 还必须与对应 Tensor 的行数一致。op_api 检查描述符、列表长度及总行数等 Host 可见信息，不做设备组表的 D2H 拷贝，也无法在此阶段证明组值与分组输出容量逐项一致。非空计算由分组 kernel 在数据读写前检查运行时组边界；全空快捷路径不读取设备组表。

选择分组输出时，调用方需事先提供各个 `[M_g,N]` 输出 Tensor。若各 M_g 仅在设备侧已知，而 Host 只知道总 M，可选择拼接输出（splitItem=2/3），分配 `[M,N]` 的 yOut 和 `[M,ceil(N/64),2]` 的 scaleOut。

### 4. 数据类型和量化属性组合

| x / rotation | yOut | scaleOut | roundMode | scaleAlg | dstTypeMax |
| --- | --- | --- | --- | --- | --- |
| FLOAT16 或 BFLOAT16 | FLOAT4_E2M1 | FLOAT8_E8M0 | rint、round、floor | 0 | 0 |
| FLOAT16 或 BFLOAT16 | FLOAT4_E2M1 | FLOAT8_E8M0 | rint、round、floor | 2 | [6,12] 中的有限值 |
| FLOAT16 或 BFLOAT16 | FLOAT8_E4M3FN、FLOAT8_E5M2 | FLOAT8_E8M0 | rint | 0 或 1 | 0 |

不同组不能分别选择不同的输出 dtype、scaleAlg、roundMode 或 alpha。FP4 输出的逻辑末维仍为 N，接口使用 FLOAT4_E2M1 Tensor，不使用 INT32 打包输出形式。

`trans` 仅支持 false；输入 Tensor 的非连续视图支持不代表提供了额外的逻辑转置属性。

### 5. 空 Tensor、非连续 Tensor 和调优提示

- 所有 TensorList 必须包含 1～1024 个有效 Tensor 指针；零元素 TensorList 和 shape 为 `[0]` 的 groupList 不支持。
- 允许 M_g=0 或符合旋转约束的 N=0。即使全部输出为空，也会先完成参数及平台检查，再返回 workspaceSize=0 和有效执行器。
- 混合空组与非空组会按常规路径提交计算，kernel 跳过空组并继续调度其余组。
- 非连续 Tensor 通过输入连续化、输出 ViewCopy 适配；Tensor 的 shape、stride、viewOffset 及声明的存储范围须有效。此类适配可能增加 workspace 和数据搬运开销。
- `tuningConfigOptional[0]` 为预期每组行数，合法范围为 `[0,max(x[i].M)]`。独立高维 x 的 M 是末维以外各维的乘积，一维 x 的 M 为 1；单个 M 分组 x 的该上界是总 M。
- tuningConfig 是调优提示，不能替代实际 groupList；第 1 项及之后的项当前仅预留，未定义 GMM V5 中其他量化/打包开关的含义。

### 6. 确定性

内外层源码已接线，但尚未完成设备确定性验证，也未承诺通过确定性开关选择实现。

## 调用示例

### 示例参数

采用两组 M 分组、共享旋转矩阵、单输入/单输出，参数如下：

| 参数 | 示例值 |
| --- | --- |
| x | 1 个 FLOAT16 ND Tensor，shape 为 `[5,128]` |
| rotation | 1 个 FLOAT16 ND Tensor，shape 为 `[64,64]`；两组及每行的两个旋转块共享 |
| alpha | nullptr，跳过 clamp |
| groupListOptional | INT64 ND 设备 Tensor，shape 为 `[2]`，设备值为 `[2,5]` |
| splitItem / groupType / groupListType | 3 / 0 / 0；两组行数为 2 和 3 |
| axis / roundMode / scaleAlg / dstTypeMax / trans | -1 / `"rint"` / 0 / 0.0 / false |
| tuningConfigOptional | nullptr |
| yOut | 1 个 FLOAT8_E4M3FN ND Tensor，shape 为 `[5,128]` |
| scaleOut | 1 个 FLOAT8_E8M0 ND Tensor，shape 为 `[5,2,2]` |

下例展示已有设备 Tensor 时的两段式调用。调用方需先初始化 ACL、设置 Ascend 950 设备、创建 Stream，并按上表创建 Tensor/TensorList、分配设备存储及准备输入数据。头文件来自本算子的 op_api 目录。**非空执行需要包含分组后端的 ops-tensor 版本及真实 CANN 编译产物；本示例尚未完成设备验证。**

```cpp
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_grouped_rotate_quant.h"

// Tensor/TensorList 和 Stream 均由调用方持有，并满足上表约束。
int RunGroupedRotateQuantExample(
    const aclTensorList *x,
    const aclTensorList *rotation,
    const aclTensor *groupList,
    aclTensorList *y,
    aclTensorList *scale,
    aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnGroupedRotateQuantGetWorkspaceSize(
        x, rotation, nullptr, groupList,
        3, 0, 0, -1, "rint", 0, 0.0, false,
        nullptr, y, scale, &workspaceSize, &executor);
    if (status != ACLNN_SUCCESS) {
        return static_cast<int>(status);
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        const auto allocStatus = aclrtMalloc(
            &workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (allocStatus != ACL_SUCCESS) {
            // 尚未提交第二段接口时，释放第一段创建的执行器。
            aclDestroyAclOpExecutor(executor);
            return static_cast<int>(allocStatus);
        }
    }

    status = aclnnGroupedRotateQuant(workspace, workspaceSize, executor, stream);
    // 即使提交返回错误，也先等待已提交到该 Stream 的任务结束。
    const auto syncStatus = aclrtSynchronizeStream(stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (status != ACLNN_SUCCESS) {
        return static_cast<int>(status);
    }
    return static_cast<int>(syncStatus);
}
```

示例函数同步后，调用方可读取 y/scale，再按资源所有权规则销毁 TensorList、Tensor 和设备存储，最后销毁 Stream。第二段调用后的同一 executor 不应重复用于提交计算。

若改成分组输出，可保持 x、rotation 和 groupList 不变，将 splitItem 改为 0 或 1，并将 yOut 改为两个 shape 分别为 `[2,128]`、`[3,128]` 的 Tensor，scaleOut 对应改为 `[2,2,2]`、`[3,2,2]`。

## 参考资料

- [RotateQuant 接口与 MX 计算说明](../../../../../rotate_quant_blaze/ops-nn_rotate_quant_blaze/matmul/rotate_quant/docs/aclnnRotateQuant.md)：组内旋转、clamp、量化的参考；该链接依赖当前两个代码项目的相对目录布局。
- [GroupedMatmul V5 接口说明](../../grouped_matmul/docs/aclnnGroupedMatmulV5.md)：TensorList、splitItem、groupList 的接口参考；本算子的支持范围以本文和当前接口代码为准。
- [GroupedRotateQuant op_api 契约](../op_api/README.md)：参数校验、L0 提交及后续 op_host/kernel 接入约定。

## 修订记录

| 日期 | 版本 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| 2026-10-08 | 0.1 | Codex | 首版接口文档；描述独立 Tensor/M 分组、旋转与 MX 量化语义、两段式接口、约束及调用示例；注明当前仅完成 op_api |

| 2026-10-08 | 0.2 | Codex | 补齐 Host/tiling 代码及协议说明；更新开发状态，kernel 尚待实现和联调 |
