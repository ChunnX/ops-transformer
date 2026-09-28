# GridSample（aclnnGridSampler2D / aclnnGridSampler3D）测试需求分析 Spec

| 算子名称 | aclnnGridSampler2D / aclnnGridSampler3D |
|---|---|
| 接口类型 | aclnn |
| 目标产品 | Ascend 950PR/Ascend 950DT |
| 生成日期 | 2026-09-22 |
| 源文档 | docs/aclnnGridSampler2D.md、docs/aclnnGridSampler3D.md（算子仓 ops-cv image/grid_sample，补充：README.md） |

> **接口类型判定记录**：config.json 不存在（interface_type 视为 null）、golden 文件不存在 → 目录扫描：`inputs/grid_sample/docs/aclnnGridSampler2D.md` 与 `docs/aclnnGridSampler3D.md` 均存在 → **接口类型 = aclnn**。算子为同一 GridSample 算子的 2D/3D 两个 aclnn 接口，本 spec 同时覆盖。
>
> **目标产品支持确认**：两份接口文档"产品支持情况"章节均标注 `Ascend 950PR/Ascend 950DT：支持`，且无 950 专属限制段落（910/A2/A3/310b/310p 的专属限制不适用于目标产品）。

---

## 1. 验收标准

### 1.1 精度验收

- **precision_standard**：新精度2.1
- **来源**：文档未显式定义，采用默认值（两份 aclnn 接口文档均无"精度要求/精度标准/atol/rtol"章节）

> 说明：文档"约束说明"章节存在精度风险提示，作为用例设计注意项而非验收阈值：
> - 2D：当 grid 输入值 × 图片（长或宽）> 24 位二进制数（16777216）时，采样点可能存在误差，精度可能产生偏差（来源：aclnnGridSampler2D.md"约束说明"）。
> - 2D：grid 含大量超出 [-1, 1] 范围数据时，zeros/border 填充策略下计算结果值大量重复（来源：aclnnGridSampler2D.md"约束说明"）。

### 1.2 性能验收

- **网络规格**：平均 1.2x，单 case 底线 0.8x
- **功能 case**：单 case 底线 0.5x
- **来源**：文档未显式定义，采用默认值（两份 aclnn 接口文档均无"性能要求/性能基准/性能验收"章节）

---

## 2. 用例场景

> 依据函数原型（两段式接口第一段）判定参数类别：`input`/`grid`（`const aclTensor*`）→ Tensor；`interpolationMode`/`paddingMode`（`int64_t` 基本类型）→ 属性 Attribute；`alignCorners`（`bool` 基本类型）→ 属性 Attribute；`out`（`aclTensor*`）→ 输出 Tensor。`workspaceSize`/`executor` 为两段式接口框架机制参数，不作为测试因子。两接口原型结构一致，仅维度/dtype/format/枚举值域存在 2D/3D 差异，下文分接口列出。

### 2.1 L0 测试场景

#### 2.1.1 离散因子值覆盖

**aclnnGridSampler2D**

- **input**（输入 Tensor）：
  - dimensions：覆盖 [4]（固定，来源：参数表"维度(shape)"列 "4"；shape 为 (N, C, H_in, W_in)）
  - dtype：覆盖 [float16, float32, float64, bfloat16]（来源：参数表"数据类型"列 "FLOAT32、FLOAT16、DOUBLE、BFLOAT16"）
  - format：覆盖 [ND]（来源：参数表"数据格式"列 "ND"）
- **grid**（输入 Tensor）：
  - dimensions：覆盖 [4]（固定，来源：参数表"维度(shape)"列 "4"；shape 为 (N, H_out, W_out, 2)，最后一维固定为 2）
  - dtype：由 input 推导（来源：参数表"使用说明"列 "数据类型与入参 input 的数据类型一致"）
  - format：覆盖 [ND]（来源：参数表"数据格式"列 "ND"）
- **interpolationMode**（输入属性 Attribute，原型 `int64_t`）：
  - dtype：[int64] 固定
  - value：覆盖 [0, 1, 2]（0：bilinear、1：nearest、2：bicubic；来源：参数表"使用说明"列）
- **paddingMode**（输入属性 Attribute，原型 `int64_t`）：
  - dtype：[int64] 固定
  - value：覆盖 [0, 1, 2]（0：zeros、1：border、2：reflection；来源：参数表"使用说明"列）
- **alignCorners**（输入属性 Attribute，原型 `bool`）：
  - dtype：[bool] 固定
  - value：覆盖 [true, false]（来源：参数表"使用说明"列）
- **out**（输出 Tensor）：
  - dimensions：[4]，由 input/grid 推导（N、C 取自 input，H_out、W_out 取自 grid；来源：参数表"使用说明"列）
  - dtype：由 input 推导（来源：参数表"使用说明"列 "数据类型与 input 的数据类型一致"）
  - format：覆盖 [ND]（来源：参数表"数据格式"列 "ND"）

**aclnnGridSampler3D**

- **input**（输入 Tensor）：
  - dimensions：覆盖 [5]（固定，来源：参数表"维度(shape)"列 "5"；shape 为 (N, C, D_in, H_in, W_in)）
  - dtype：覆盖 [float16, float32, float64, bfloat16]（来源：参数表"数据类型"列 "FLOAT16、FLOAT32、DOUBLE、BFLOAT16"）
  - format：覆盖 [NCDHW, NDHWC, ND]（来源：参数表"数据格式"列 "NCDHW、NDHWC、ND"；ND 按 NCDHW 处理，NDHWC 仅当"数据类型为 FLOAT32 && interpolationMode 为 0 && C 轴为 4 的倍数"时支持）
- **grid**（输入 Tensor）：
  - dimensions：覆盖 [5]（固定，来源：参数表"维度(shape)"列 "5"；shape 为 (N, D_out, H_out, W_out, 3)，最后一维固定为 3）
  - dtype：由 input 推导（来源：参数表"使用说明"列 "数据类型与入参 input 的数据类型一致"）
  - format：覆盖 [ND]（来源：参数表"数据格式"列 "ND"）
- **interpolationMode**（输入属性 Attribute，原型 `int64_t`）：
  - dtype：[int64] 固定
  - value：覆盖 [0, 1]（仅 0：bilinear、1：nearest，**3D 不支持 bicubic**；来源：参数表"使用说明"列）
- **paddingMode**（输入属性 Attribute，原型 `int64_t`）：
  - dtype：[int64] 固定
  - value：覆盖 [0, 1, 2]（0：zeros、1：border、2：reflection；来源：参数表"使用说明"列）
- **alignCorners**（输入属性 Attribute，原型 `bool`）：
  - dtype：[bool] 固定
  - value：覆盖 [true, false]（来源：参数表"使用说明"列）
- **out**（输出 Tensor）：
  - dimensions：[5]，由 input/grid 推导（N、C 轴取自 input，D、H、W 轴取自 grid；来源：参数表"使用说明"列）
  - dtype：由 input 推导（来源：参数表"使用说明"列 "数据类型与 input 保持一致"）
  - format：由 input 推导（来源：参数表"使用说明"列 "数据格式和数据类型与 input 保持一致"）

#### 2.1.2 InfNan 覆盖

- 浮点 dtype：覆盖 inf/-inf/nan 三值，适用 dtype = input/grid 的 [float16, float32, float64, bfloat16]（2D/3D 相同；来源：参数表"数据类型"列，两份文档均未声明不支持 inf/nan）

#### 2.1.3 存在性覆盖

- 不适用：aclnn 函数原型中 input、grid、interpolationMode、paddingMode、alignCorners、out 全部为必传参数（来源：函数原型，无默认值/可选标记；README"参数说明"表中属性的"可选"指 proto 层默认值，aclnn 原型层面全必选）

#### 2.1.4 维度边界覆盖

- min 维度 = max 维度 = 4（2D 接口，input/grid 固定 4 维；来源：参数表"维度(shape)"列 "4"）；min 维度 = max 维度 = 5（3D 接口，input/grid 固定 5 维；来源：参数表"维度(shape)"列 "5"）
- 固定维度算子，无维度数变化边界；维度内边界（shape 各轴取值边界，如 N/C 为 0 的空 Tensor、H/W 最小值 1）由下游 st-design 在 shape 因子中实现（参数表"使用说明"列：2D input/grid/out 支持空 Tensor，但 2D input 最后两维维度值不能为 0；3D input/grid/out 支持空 Tensor）

#### 2.1.5 格式交叉覆盖

- aclnnGridSampler2D：不适用（input/grid/out 数据格式仅 ND 单一格式，来源：参数表"数据格式"列）
- aclnnGridSampler3D：input 为多格式参数，覆盖 [NCDHW, NDHWC, ND] 每种至少一次（来源：参数表"数据格式"列 "NCDHW、NDHWC、ND"）；其中 NDHWC 受条件约束"FLOAT32 && interpolationMode=0 && C 轴为 4 的倍数"（来源：参数表"使用说明"列）；grid 仅 ND、out 格式由 input 推导，不独立交叉

### 2.2 L1 测试场景

#### 2.2.1 Pairwise 组合

- **aclnnGridSampler2D** 参与组合的离散因子：input.dtype × interpolationMode.value × paddingMode.value × alignCorners.value
  - 不独立参与：grid.dtype（约束与 input 一致）、out.dtype/out.dimensions（由 input/grid 推导）、input.format/grid.format/out.format（单值 ND）、input.dimensions/grid.dimensions（固定 4）
  - 组合约束（来源：aclnnGridSampler2D.md 错误码表 ACLNN_ERR_PARAM_INVALID）：interpolationMode=2（bicubic）时 input/grid/out 数据类型须为 FLOAT32 或 FLOAT16，[bicubic × float64]、[bicubic × bfloat16] 为非法组合（下游按约束排除或作异常用例）
- **aclnnGridSampler3D** 参与组合的离散因子：input.dtype × input.format × interpolationMode.value × paddingMode.value × alignCorners.value
  - 不独立参与：grid.dtype（约束与 input 一致）、out.dtype/out.format/out.dimensions（由 input/grid 推导）、grid.format（单值 ND）、input.dimensions/grid.dimensions（固定 5）
  - 组合约束（来源：参数表"使用说明"列）：[NDHWC × 非 FLOAT32]、[NDHWC × interpolationMode≠0]、[NDHWC × C 非 4 的倍数] 不支持

#### 2.2.2 数据范围覆盖（范围描述级）

- 浮点 dtype（input/grid，2D/3D 相同）：覆盖极值 / 零 / ±1 / 最小正数 / inf / nan / ±0（文档未声明排除 inf/nan）
- 属性参数 value 无连续值域（interpolationMode/paddingMode/alignCorners 均为离散枚举，已在 L0 覆盖）
- 文档语义边界（来源：两份文档"功能说明/约束说明/使用说明"列原文）：
  - grid 归一化坐标基准区间 [-1, 1]（(-1,1) 左上角、(1,-1) 右下角），覆盖区间内常规值、边界值 ±1 及超范围越界值（越界坐标按 paddingMode 语义处理，是 paddingMode 三种模式的功能触发点）
  - 2D：input 的 H×W < INT32 最大值；input 最后两维维度值不能为 0
  - 3D：input 的 D×H×W < INT32 最大值
  - 2D：grid 输入值 × 图片（长或宽）> 16777216（24 位二进制数）时采样点存在误差（精度风险提示，数据范围设计时规避或单独验证）

#### 2.2.3 随机填充

- 覆盖目标不足时随机 shape 补充至 ≥500 条（shape 随机泛化须满足：2D input (N,C,H_in,W_in) 最后两维 ≥1、grid (N,H_out,W_out,2) 末维=2、N 一致；3D input (N,C,D_in,H_in,W_in)、grid 末维=3、N 一致；来源：参数表"使用说明"列 shape 约束）

---

## 补充说明（供下游 st-design 参考，非 spec 强制输出）

- **非连续 Tensor**：2D/3D 接口的 input、grid、out 参数表"非连续Tensor"列均为 √，支持非连续输入（来源：参数表"非连续Tensor"列）。
- **空 Tensor**：2D/3D 接口的 input、grid、out 均声明"支持空Tensor"（来源：参数表"使用说明"列）；2D 另有"input 的 shape 最后两维的维度值不能为 0"限制。
- **确定性计算**：aclnnGridSampler2D/aclnnGridSampler3D 默认确定性实现（来源：两份文档"约束说明"章节），无确定性/非确定性 tiling 分支区分需求。
- **文档内部差异 [待确认]**：aclnnGridSampler2D 错误码表声明"interpolationMode 为 bicubic 时 dtype 不是 FLOAT32/FLOAT16 报错"，而 A2/A3 产品段落声明 BFLOAT16 亦支持 bicubic；两者存在文档内部不一致。目标产品 950PR/950DT 无专属说明，建议以第一段接口实际校验行为为准，bicubic × bfloat16 组合在 950 上需实测确认。
