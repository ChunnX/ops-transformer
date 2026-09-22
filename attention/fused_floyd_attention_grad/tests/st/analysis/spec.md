# aclnnFusedFloydAttentionGrad 测试需求分析 Spec

| 算子名称 | aclnnFusedFloydAttentionGrad |
|---|---|
| 接口类型 | aclnn |
| 目标产品 | Ascend 950PR/Ascend 950DT（用户指定 A5；接口文档"产品支持情况"当前标注 950PR/950DT 不支持，本次需求为新增支持 950PR/950DT 的验证，见 §2.3 补充说明） |
| 生成日期 | 2026-09-21 |
| 源文档 | docs/aclnnFusedFloydAttentionGrad.md（接口文档）；README.md（参数/约束佐证）；tests/assets/spec.py + impl/（TTK TestSpec 精度标准）；op_host/fused_floyd_attention_grad_tiling_common.cpp（约束代码佐证） |

---

## 1. 验收标准

### 1.1 精度验收

- **precision_standard**：stat_rel_err（TTK 标准引用标识；bfloat16/float16/float32 三种 dtype 均适用）
- **来源**：接口文档 docs/aclnnFusedFloydAttentionGrad.md 未显式定义精度验收章节；采用算子仓 TTK TestSpec 显式定义——`tests/assets/spec.py` tolerance：`{"bfloat16": {"standard": "stat_rel_err"}, "float16": {"standard": "stat_rel_err"}, "float32": {"standard": "stat_rel_err"}}`；执行细则见 `tests/assets/impl/compare.py` 自定义三方对比（对标 ATK cv_fused_double_benchmark）：NPU 输出 vs fp64 高精度 golden vs 原 dtype 标杆，判定 4 项 ratio——max_re_ratio≤10、avg_re_ratio≤2、rmse_ratio≤2、number_count_ratio≤2，全部满足才 Pass

### 1.2 性能验收

- **网络规格**：平均 1.2x，单 case 底线 0.8x
- **功能 case**：单 case 底线 0.5x
- **来源**：接口文档无"性能要求/性能基准/性能验收"章节，采用默认值；另文档"约束说明"含性能语义边界：当 M\*D≥65536 或 K\*D≥65536 时出现明显性能下降（建议小算子拼接替换），性能用例需覆盖该边界两侧

---

## 2. 用例场景

### 2.1 L0 测试场景

**离散因子值覆盖**（shape 各轴为独立离散因子，全部输入 Tensor 固定 5 维 ND）

- query（输入 Tensor，原型 `const aclTensor*`）：
  - dimensions：[B,H,N,M,D]，各轴离散取值——B∈{1, 2, 2048}、H∈{1, 2, 256}、N∈{16, 128, 1024, 1048576}、M∈{128, 256, 1024, 1048576}、D∈{32, 64, 128}（全枚举）（来源：参数表"维度(shape)"列 "[B,H,N,M,D]" + 约束说明 "B：1~2K；H：1~256；N：16~1M且N%16==0；M：128~1M且M%128==0；K：128~1M且K%128==0；D：32/64/128"；max 边界受内存约束由 st-design 调整）
  - dtype：覆盖 [float16, bfloat16]（来源：参数表"数据类型"列 "FLOAT16、BFLOAT16"）
  - format：[ND] 固定（来源：参数表"数据格式"列 "ND"）
- key1（输入 Tensor）：dimensions：[B,H,N,K,D]，B/N/D 轴由 query 约束一致（约束说明"query与key1的第0/2/4轴需相同"），K∈{128, 256, 1024, 1048576}；dtype：[float16, bfloat16]（来源："数据类型"列 + "数据类型与query/value1/key2/value2的数据类型一致"）；format：[ND] 固定
- value1（输入 Tensor）：dimensions：由 key1 推导（约束说明"key1与value1 shape需相同"）；dtype：[float16, bfloat16]（同源约束）；format：[ND] 固定
- key2（输入 Tensor）：dimensions：[B,H,K,M,D]，B/H 轴由 query 约束一致，K/M 为独立因子（取值同上）；dtype：[float16, bfloat16]；format：[ND] 固定
- value2（输入 Tensor）：dimensions：由 key2 推导（约束说明"key2与value2 shape需相同"）；dtype：[float16, bfloat16]；format：[ND] 固定
- dy（输入 Tensor）：dimensions：由 query 推导（约束说明"query与dy/attentionIn shape需相同"）；dtype：[float16, bfloat16]；format：[ND] 固定
- attenMaskOptional（输入 Tensor，可选）：dimensions：[B,1,N,1,K]，B/N/K 轴由 query/key1 推导，第 1/3 轴固定为 1（来源：参数表"维度(shape)"列 "[B,1,N,1,K]"）；dtype：覆盖 [bool, uint8]（来源：参数表"数据类型"列 "BOOL、UINT8"）；format：[ND] 固定；value 语义：1=该位不参与计算、0=该位参与计算（来源：参数表"使用说明"列）
- softmaxMax（输入 Tensor）：dimensions：[B,H,N,M,8]，B/H/N/M 轴由 query 推导、尾轴固定 8（来源：参数表"维度(shape)"列 "[B,H,N,M,8]"）；dtype：[float32] 固定（来源：参数表"数据类型"列 "FLOAT"）；format：[ND] 固定
- softmaxSum（输入 Tensor）：dimensions：由 softmaxMax 推导（约束说明"softmaxMax与softmaxSum shape需相同"）；dtype：[float32] 固定；format：[ND] 固定
- attentionIn（输入 Tensor）：dimensions：由 query 推导（约束说明"query与dy/attentionIn shape需相同"）；dtype：[float16, bfloat16]（来源："数据类型和shape类型与query保持一致"）；format：[ND] 固定
- scaleValue（输入属性 Attribute，原型为基本类型 `double scaleValue`）：
  - dtype：[float64] 固定（来源：函数原型 `double scaleValue` + 参数表"数据类型"列 "DOUBLE"；固定单值，不作为离散覆盖维度参与 Pairwise）
  - value：无离散枚举（文档未定义取值范围列），典型取值 {1.0（默认值）、1/√D（调用示例 `1.0/sqrt(128)`）}（来源：README.md "默认值为1.0" + 接口文档调用示例），完整值域见 L1 数据范围
- dqOut（输出 Tensor）：dimensions/dtype：由 query 推导（shape 同 query；op_host infershape 确认输出 dtype=输入 dtype）；format：[ND] 固定
- dk1Out/dv1Out（输出 Tensor）：dimensions/dtype：由 key1/value1 推导；format：[ND] 固定
- dk2Out/dv2Out（输出 Tensor）：dimensions/dtype：由 key2/value2 推导；format：[ND] 固定

**InfNan 覆盖**
- 浮点 dtype：覆盖 inf/-inf/nan 的 dtype 列表 [float16, bfloat16]（query/key1/value1/key2/value2/dy/attentionIn）与 [float32]（softmaxMax/softmaxSum）（来源：参数表"数据类型"列；接口文档未声明"不支持 inf/nan"，故不排除）
- 注：softmaxMax/softmaxSum/attentionIn 为 forward 中间结果，TTK golden 的 customize_inputs（tests/assets/impl/inputs.py）会用 fp64 forward 结果 in-place 覆写这三个输入——直接注入的 InfNan 会被覆写，该场景需通过 query/key/value/scale 间接构造或调整 golden 适配（下游 st-design/执行时确认）

**存在性覆盖**
- 可选参数：attenMaskOptional，exist 取 [true, false]（来源：参数表/原型命名 "attenMaskOptional" + README "可选输入"；nullptr 表示无 mask）
- 其余 Tensor 参数全部必选（必选仅 [true]）；scaleValue 为属性参数（值传递，无指针存在性概念），其"默认 1.0"体现为 value 覆盖

**维度边界覆盖**
- min 维度：5 维（所有输入/输出 Tensor 固定 5 维 ND，无 0 维/标量形态）；max 维度：5 维——维度数无离散变化，边界体现为 shape 各轴取值边界：B=1（min）/2048（max）、H=1/256、N=16/1M、M=128/1M、K=128/1M、D∈{32,64,128} 全枚举；对齐粒度边界：N%16==0 最小 16、M%128==0 与 K%128==0 最小 128（来源：参数表"维度(shape)"列 5 维模板 + 约束说明；op_host tiling_common.cpp 佐证 "Headdim only supports 32/64/128"、"16-byte alignment"、"128-byte alignment"）

**格式交叉覆盖**
- 不适用：所有 Tensor 参数仅支持 ND 单一格式（来源：参数表"数据格式"列均为 "ND"，无 ≥2 种格式的参数）

### 2.2 L1 测试场景

**Pairwise 组合**
- 参与组合的离散因子：query.dtype [float16, bfloat16] × D [32, 64, 128] × B [1, 2, 2048] × H [1, 2, 256] × N [16, 128, 1024, 1048576] × M [128, 256, 1024, 1048576] × K [128, 256, 1024, 1048576] × attenMaskOptional.exist [true, false] × attenMaskOptional.dtype [bool, uint8] × scaleValue.value（离散典型值）
- 不独立参与组合（由其他参数推导或约束一致）：
  - key1/value1/key2/value2/dy/attentionIn 的 dtype（参数表"使用说明"列：五组 QKV 输入数据类型一致，由 query.dtype 推导）
  - value1.dimensions（=key1）、dy/attentionIn.dimensions（=query）、softmaxSum.dimensions（=softmaxMax）、key1 的 B/N/D 轴（=query 第 0/2/4 轴）、attenMaskOptional 的 B/N/K 轴（=query/key1）、softmaxMax/softmaxSum 的 B/H/N/M 轴（=query）
  - softmaxMax/softmaxSum.dtype（float32 固定单值）、全部 format（ND 单值）、scaleValue.dtype（float64 固定，属性参数）
  - 全部输出参数 dimensions/dtype（由对应输入推导）
- 注：B/H/N/M/K 为大空间因子，Pairwise 锚点组合由 st-design 实现；max 边界锚点受内存约束调整

**数据范围覆盖**（范围描述级，不列具体数值分段）
- 浮点 dtype（float16/bfloat16：query/key1/value1/key2/value2/dy/attentionIn）：覆盖极值 / 零 / ±1 / 最小正数 / inf / nan / ±0
- 浮点 dtype（float32：softmaxMax/softmaxSum）：覆盖极值 / 零 / ±1 / 最小正数 / inf / nan / ±0（注：golden customize_inputs 存在 forward 覆写行为，注入值会被覆写，需注意 golden 适配，见 §2.1 InfNan 注）
- 属性 scaleValue（float64）：覆盖零 / ±1 / 极值 / 最小正数 / ±0 + 文档语义边界（默认值 1.0、示例 1/√128）
- 整型/布尔 dtype（bool/uint8：attenMaskOptional）：覆盖 0 / 1（语义边界：1=该位不参与计算、0=该位参与计算）及 uint8 类型边界值（0/1/255）
- 文档语义边界（来源：接口文档"约束说明"）：B∈[1,2K]、H∈[1,256]、N∈[16,1M]且 N%16==0、M∈[128,1M]且 M%128==0、K∈[128,1M]且 K%128==0、D∈{32,64,128}；性能语义边界 M\*D≥65536 或 K\*D≥65536

**随机填充**
- 覆盖目标不足时随机 shape 补充至 ≥500 条：B/H/N/M/K 在文档约束范围内随机取值并保持对齐约束（N%16==0、M%128==0、K%128==0、D∈{32,64,128}）
- 随机数据支持正态分布输入（ATK 对齐场景：mean∈[-100,100]、std∈[1,25]，来源：tests/assets/impl/inputs.py ND 哨兵分布策略）

### 2.3 补充说明（算子特性，供下游 st-design/执行参考）

1. **产品支持差异**：接口文档"产品支持情况"当前标注 Ascend 950PR/950DT 不支持（支持 A3/A2 系列）；本 spec 按用户需求以 950PR/950DT（A5）为目标产品开展测试需求分析（新增支持验证场景），下游执行需在 950PR/950DT 环境部署
2. **确定性**：约束说明"该接口不支持确定性"——确定性比对场景不适用
3. **非连续**：输入非连续由 aclnn 适配层 AutoContiguous 支持（op_host/fused_floyd_attention_grad_def.cpp 各参数 `.AutoContiguous()`，参数表"非连续Tensor"列输入均标 √）；输出仅支持连续（约束说明"该接口不支持非连续输出"）；注意参数表输出行"非连续Tensor"列亦标 √，与约束说明存在文档矛盾 `[待确认]`，建议下游以非连续输出异常用例实测定界
4. **golden 适配**：TTK TestSpec 已就绪（tests/assets/spec.py，ACLNN 模式）；softmaxMax/softmaxSum/attentionIn 三个输入由 golden customize_inputs 以 fp64 forward 结果 in-place 覆写，测试数据构造需与之协同
5. **性能语义边界**：M\*D≥65536 或 K\*D≥65536 时明显性能下降（文档建议小算子拼接替换），性能 case 需覆盖边界两侧（如 D=128×M=512 与 D=128×M=256 对比）
