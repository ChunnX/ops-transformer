# MegaMoe arch35 Tiling 重整报告

## 1. 背景与范围

本次重整服务于 MegaMoe arch35 MTE 路径的 Shared/MoE 独立量化能力。原有 tiling 逻辑以单一
`groupedMatmulMode` 为中心，同时包含输入判空、TensorList 解析、跨输入校验、属性校验、平台资源读取、
buffer 计算、tiling key 和 workspace 设置。随着 MoE 与 shared 需要分别推导 GMM 模式，继续在原有流程上增加
分支会造成校验边界不清、相同元数据重复读取和主流程持续膨胀。

本次工作的范围是：

1. 建立 MoE/shared 两侧独立的 host tiling 参数模型；
2. 重整输入、属性和 shape/dtype/format 校验；
3. 为两侧量化模式生成 tiling key 所需信息；
4. 将 tiling 主入口整理为有明确依赖顺序的阶段编排；
5. 同步 tiling data、workspace 条件和 arch35 host UT 中受字段及 key 变化影响的内容。

本次没有引入 trace、动态 tileN、FullLoad 或其他性能实验，也没有在 arch22 中使用新增的 shared 量化属性。
Layered/URMA 只做公共字段改名和既有语义适配，不在本次实现两侧不同量化。

## 2. 重整原则

### 2.1 校验和取值只做一次

能够在校验时确定的结果直接写入中间参数或 `MegaMoeTilingData`，后续阶段消费已校验结果，不再次从
`TilingContext` 推导同一值。例如：

- TensorList 组织方式和专家数在 weight 参数校验时确定；
- MoE/shared GMM mode 在各自量化参数校验时确定；
- `bs`、`h`、`topK` 和 `hiddenDim` 在 shape 校验时写入 tiling data；
- activation 属性使用同一次读取结果完成参数校验和 tiling data 设置；
- 通信域、专家数和容量属性在对应校验函数内写入 tiling data。

### 2.2 先校验，后写入依赖结果

公共指针和属性指针在入口统一判空；动态 TensorList 中的单个 tensor 仍在遍历时校验。平台核数、拓扑类型、
通信域和容量等值均先通过合法性检查，再写入 tiling data 或用于后续计算。

### 2.3 按业务阶段组织，而不是按新增需求打补丁

MoE/shared 共用同一套校验流程，以 `ExpertParams` 描述某一侧的输入索引、专家数、activation dtype 和 GMM mode。
需要区分两侧时传入对应的 `ExpertParams`，不为每个新组合复制一套校验函数。

### 2.4 保持主流程可读

`MegaMoeTilingFuncImplPublic` 只表达阶段顺序。输入细节、平台信息和最终结果落盘分别由有完整语义的函数负责，
避免主入口同时承担大量局部计算。

## 3. 数据模型

### 3.1 `ExpertWeightTensorGroupInputs`

描述一侧专家的四组动态输入：

- `weight1`；
- `weight2`；
- `weight_scales1`；
- `weight_scales2`；
- 用于错误日志的专家侧名称。

它只保存输入定位信息，不保存重复的 input name/index 映射结构。

### 3.2 `ExpertParams`

在输入索引之外，保存该侧校验后得到的：

- 专家数；
- GMM1 activation dtype；
- `MegaMoeGmmMode` 协议值。

### 3.3 `MegaMoeExpertParams`

同时持有 `moe`、`shared` 两份 `ExpertParams`，以及双方共同使用的 TensorList 组织方式
`isPerExpertWeightTensor`。TensorList 组织方式仍保持 MoE/shared 一致，但两侧 dtype、format、专家数和 GMM mode
分别解析、分别校验。

## 4. 当前 tiling 主流程

```text
MegaMoeTilingFuncImplPublic
  |
  +-- 获取并检查 MegaMoeTilingData
  |
  +-- CheckAndSetEntryParams
  |     +-- 固定输入/输出指针判空
  |     +-- 属性指针判空
  |     `-- 校验并写入 topoType
  |
  +-- CheckAndSetInput
  |     +-- 解析 weight TensorList 和 shared 是否存在
  |     +-- 分别推导 MoE/shared activation dtype 与 GMM mode
  |     +-- 校验 shape、dtype、format、scale
  |     +-- 校验并写入通信域和专家数
  |     `-- 校验并写入 bs、h、topK、hiddenDim
  |
  +-- CheckAndSetPlatformParams
  |     +-- 校验 AIC/AIV 数量
  |     +-- 读取 UB 容量
  |     `-- 设置 blockDim、scheduleMode 和核心数
  |
  +-- CheckAttrAndSetTilingData
  |     +-- activation 参数
  |     +-- combine 量化属性
  |     +-- capacity 属性
  |     `-- 其余派生字段
  |
  +-- SetAdaptiveBufferConfigs
  |
  `-- FinalizeTiling
        +-- 生成并设置 tiling key
        +-- 计算并设置 workspace
        `-- 输出 tiling/workspace/peermem 调试信息
```

重整后 `MegaMoeTilingFuncImplPublic` 仅约 20 行有效代码；本轮新增或调整的阶段函数均不超过 50 行。

## 5. Weight 与 TensorList 校验

### 5.1 两种 TensorList 组织方式

当前继续支持两种既有组织：

1. 一个动态输入 tensor 对应一个专家。此时单个 weight 为二维，TensorList 长度就是专家数；
2. 一个 tensor 内连续存放多个专家。此时 weight 为三维，tensor 的专家维给出专家数。

`isPerExpertWeightTensor` 从 MoE `weight1` 的维数解析，shared 使用相同组织方式。两侧分别计算专家数，不能用
MoE 专家数代替 shared 专家数。

### 5.2 shared 是否启用

shared 的 `weight1`、`weight2`、`weight_scales1`、`weight_scales2` 必须同时为空或同时有效。四组输入全部有效时，
由 `shared_weight1` 计算 shared 专家数；全部为空时将 shared 专家数设置为 0。后续 shared 校验统一以该结果为入口，
不再在各函数内重复判断 shared 是否存在。

### 5.3 每侧内部一致性

MoE 和 shared 分别校验：

- W1/W2 dtype 相同；
- W1/W2 与该侧 activation dtype 能组成受支持的 GMM mode；
- W1/W2 format 满足对应模式约束；
- W1/W2/scale TensorList 的 tensor 数量与该侧专家数一致；
- 同一 TensorList 内各 tensor 的 shape、dtype、format 一致；
- scale dtype 为 E8M0，且 scale shape 与对应 weight shape 满足既有 MXFP 布局关系。

两侧之间不再校验 dtype 或 format 相等。双方只保留数学接口必须一致的维度关系：输入 H、GMM1 输出
`hiddenDim`、GMM2 输出 H。W1/W2 自身的矩阵乘维度校验已经保证 H，因此跨组 shape 校验只比较独立量化不会改变的
FFN expansion 维度，避免重复比较同一维度。

## 6. GMM mode 推导

`MegaMoeGmmMode` 统一表示 host、tiling data 和 kernel 使用的专家 GMM 模式：

| 模式 | 协议值 | GMM1 activation | weight | weight format |
| --- | ---: | --- | --- | --- |
| `A8W8_ND` | 0 | E5M2/E4M3FN | 同 activation dtype | ND |
| `A8W4_NZ` | 1 | E4M3FN | E2M1 | NZ_C0_32 |
| `A8W8_NZ` | 2 | E4M3FN | E4M3FN | NZ |
| `A4W4_NZ` | 4 | E2M1 | E2M1 | W1 NZ、W2 NZ_C0_32 |

枚举沿用已有协议值，但按语义分组排列。MoE 和 shared 分别执行相同的推导逻辑，并写入 `moeGmmMode` 与
`sharedGmmMode`。无 shared 专家时，shared 量化 dtype 继承 MoE dtype，仅用于形成稳定的模板选择信息。

## 7. 属性和容量校验

属性校验按真实依赖顺序执行：

1. 通信域属性给出 `epWorldSize`；
2. 专家属性使用 `epWorldSize` 校验总专家数并计算 `moeExpertPerRank`；
3. combine 属性确定通信量化模式；
4. capacity 属性使用 `bs`、`topK`、专家数、通信域和输出 dtype 大小计算最小 buffer 与最大输出规模。

`numMaxTokensPerRank`、`topkWeightsType`、`maxRecvTokenNum` 通过检查后直接写入最终字段。activation 参数同样使用
一次读取完成合法性检查和 `actMode`、`actSubMode`、clamp、alpha、beta 设置。

## 8. Tiling key 与 workspace

tiling key 增加以下独立选择信息：

- MoE activation dtype；
- shared activation dtype；
- MoE weight1 dtype/format；
- shared weight1 dtype/format；
- 原有 dispatch mode、combine mode、通信模式和 topK weight prefetch。

模板选择列表只枚举支持的 MTE 双侧组合；URMA 只保留两侧相同的既有组合。tiling key 数值发生变化，因此 arch35
host UT 的期望 key 同步更新，并增加 `MoE A4W4 + shared A8W8` 的 mixed 配置用例。

workspace 中原 `groupedMatmulMode` 判断改为 `moeGmmMode`。这是因为当前 workspace 中的 dispatch、MoE GMM
中间结果和同步区均属于 routed MoE 路径。shared 独立量化所需的本地量化 buffer 和 shared GMM 配置仍需在后续
kernel 双配置接通阶段按实际数据流补齐，不能仅凭 host mode 字段推断为已经完成。

## 9. 兼容性

- 相同量化配置仍由 MoE/shared 两份 mode 得到相同模板语义；
- `isPerExpertWeightTensor` 的既有两种组织保持不变；
- 最终输出仍为 BF16，combine 输出规则未改变；
- arch22 不读取或使用新增 shared 量化属性；
- URMA 不接受两侧不同的量化配置；
- tiling key 数值不保持旧值，相关 UT 必须随 key 定义更新。

## 10. 当前实现边界与后续工作

本轮完成的是独立量化所需的 host tiling 模型、校验、tiling data、key 选择和相关公共字段重整。当前 kernel 入口已
具备两侧 dtype/format 模板参数，但 MTE kernel 内部仍主要使用 MoE 的 `DTYPE_WEIGHT1`、`DispatchQuantOutType` 和
`moeGmmMode`。后续必须继续完成：

1. kernel 入口将 shared 模板类型真正传入 shared GMM 实例；
2. MoE/shared 分别构造 `GmmExecutionConfig`；
3. shared activation dtype 与 MoE 不同时，从 BF16 `x` 独立生成 shared 输入及 scale；
4. shared GMM1、activation、GMM2 使用 shared traits，并保持 A4W4 的 GMM2 A8W4 派生语义；
5. 按真实 shared 数据流复核 workspace、地址偏移、同步和空专家路径；
6. 补齐完整组合、空 shared、两种 TensorList 组织、越界与死锁测试。

## 11. 检查状态

本轮已进行：

- `git diff --check`；
- arch22 文件未修改检查；
- 已删除名称和旧字段引用检查；
- 主入口及本轮新增/调整函数长度检查；
- tiling 阶段顺序和已校验数据依赖的静态复核。

按开发环境约定，本地未执行完整编译、host UT 或板上测试；这些结果需要在编译机和设备环境补充。
