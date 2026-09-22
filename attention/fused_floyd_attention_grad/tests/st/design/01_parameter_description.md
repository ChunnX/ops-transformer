# aclnnFusedFloydAttentionGrad 参数说明

## 基本信息

- **算子名称**：aclnnFusedFloydAttentionGrad
- **目标产品系列**：Ascend 950PR/Ascend 950DT（用户指定 A5 新增支持需求；接口文档"产品支持情况"当前标注 950PR/950DT 不支持，本设计按新增支持验证场景开展）
- **功能说明**：训练场景下，计算 Floyd 注意力的反向输出。FloydAttn 相较于传统 FA，主要是计算 qk/pv 注意力时会额外将 seq 作为 batch 轴从而转换为 batchMatmul。正向公式：S=Mask(scale*(Q*K1^T + Q*K2^T), atten_mask)，P=Softmax(S)，Y=(P*V1+P*V2)；反向输出 dQ、dK1、dV1、dK2、dV2（dV1=P^T*dY，dV2=P^T*dY，dQ=((dS)*K1)/√d+((dS)*K2)/√d，dK1=((dS)^T*Q)/√d，dK2=((dS)^T*Q)/√d）。
- **接口特性**（来源：接口文档"约束说明"章节）：该接口不支持确定性；该接口不支持非连续输出；与 PyTorch 配合使用时需保证 CANN 相关包与 PyTorch 相关包版本匹配；当 M*D≥65536 或 K*D≥65536 时会出现明显性能下降（使用建议：此时建议使用小算子拼接替换实现，非硬性约束）。
- **两段式接口**：aclnnFusedFloydAttentionGradGetWorkspaceSize（入参校验 + workspace 计算）→ aclnnFusedFloydAttentionGrad（执行）。错误码：161001 ACLNN_ERR_PARAM_NULLPTR（必选参数空指针）、161002 ACLNN_ERR_PARAM_INVALID（dtype/格式不在支持范围）、561002 ACLNN_ERR_INNER_TILING_ERROR（tiling 约束违反，来源：op_host/fused_floyd_attention_grad_tiling_common.cpp CheckInputShapeValid 佐证）。

## 参数列表

| 参数名 | 类型 | 输入/输出 | 必选/可选 | 维度/长度 | 数据格式 | 数据类型 | 取值范围 | 空tensor | 非连续 | 补充说明 |
|--------|------|-----------|-----------|-----------|----------|----------|----------|----------|--------|----------|
| query | aclTensor | 输入 | 必选 | 5维 [B,H,N,M,D]，轴约束见 R1~R6 | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（各轴下界均为正：B≥1、H≥1、N≥16、M≥128、D∈{32,64,128}，shape 不含 0 轴） | 支持 | 数据类型与 key1/value1/key2/value2 一致（R15）；shape 约束见 R7/R10/R11 |
| key1 | aclTensor | 输入 | 必选 | 5维 [B,H,N,K,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（同上） | 支持 | 数据类型与 query/value1/key2/value2 一致（R15）；第 0/2/4 轴（B/N/D）与 query 相同（R7） |
| value1 | aclTensor | 输入 | 必选 | 5维 [B,H,N,K,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（同上） | 支持 | 数据类型与 query/key1/key2/value2 一致（R15）；shape 与 key1 相同（R8） |
| key2 | aclTensor | 输入 | 必选 | 5维 [B,H,K,M,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（同上） | 支持 | 数据类型与 query/key1/value1/value2 一致（R15） |
| value2 | aclTensor | 输入 | 必选 | 5维 [B,H,K,M,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（同上） | 支持 | 数据类型与 query/key1/value1/key2 一致（R15）；shape 与 key2 相同（R9） |
| dy | aclTensor | 输入 | 必选 | 5维 [B,H,N,M,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型 | 不支持（同上） | 支持 | shape 与 query 相同（R10）；数据类型与 query 一致（R17，推断自：dy 为正向输出 Y 的梯度，Y=(P*V1+P*V2) 与 attentionIn 同源且 attentionIn dtype 与 query 保持一致；文档未显式声明，[待确认]） |
| attenMaskOptional | aclTensor | 输入 | 可选（推断自：参数名含 Optional 后缀 + 错误码 161002 仅约束其存在时的 dtype/格式；README.md 标注"可选输入"） | 5维 [B,1,N,1,K]，第 1/3 轴固定为 1（R13） | ND | BOOL、UINT8 | 元素值域 {0,1}：取值为 1 代表该位不参与计算，为 0 代表该位参与计算（R20，来源：参数表"使用说明"列） | 不支持（轴下界均为正） | 支持 | exist=[true,false]，传入空指针表示无 mask |
| softmaxMax | aclTensor | 输入 | 必选 | 5维 [B,H,N,M,8]，尾轴固定 8（R14） | ND | FLOAT | 依赖具体数据类型（注意力正向计算的中间输出） | 不支持（轴下界均为正） | 支持 | shape 与 softmaxSum 相同（R12）；B/H/N/M 轴与 query 一致（R14） |
| softmaxSum | aclTensor | 输入 | 必选 | 5维 [B,H,N,M,8]（R14） | ND | FLOAT | 依赖具体数据类型（注意力正向计算的中间输出） | 不支持（轴下界均为正） | 支持 | shape 与 softmaxMax 相同（R12） |
| attentionIn | aclTensor | 输入 | 必选 | 5维 [B,H,N,M,D] | ND | FLOAT16、BFLOAT16 | 依赖具体数据类型（注意力正向计算的最终输出） | 不支持（轴下界均为正） | 支持 | shape 与 query 相同（R11）；数据类型和 shape 类型与 query 保持一致（R16，来源：参数表"使用说明"列） |
| scaleValue | double | 输入 | 必选（值传递属性，无指针存在性概念；README.md 标注"可选属性，默认值为 1.0"，默认值体现为 value 覆盖） | - | - | DOUBLE | 默认值 1.0；调用示例 1.0/sqrt(128)；无离散枚举，完整值域见 02_test_factors.yaml value_range_float64（覆盖 dtype 完整边界 + 默认值 1.0 + 典型 1/√D∈[0.0884,0.1768]） | - | - | 值传递，无功能分支（模式#9 快速跳过路径：scale 作为 Muls 的 scalar 值直接参与计算，不同取值不触发执行路径分支） |
| dqOut | aclTensor | 输出 | 必选 | 5维 [B,H,N,M,D] | ND | FLOAT16、BFLOAT16 | 不适用（输出参数） | - | 不支持（约束说明"该接口不支持非连续输出"；注意：参数表"非连续Tensor"列输出行标 √ 与约束说明矛盾，以约束说明为准，[待确认]） | shape 与 query 相同（R19）；数据类型与 query 一致（R18，来源：op_host/fused_floyd_attention_grad_infershape.cpp SetOutputDataType(0, GetInputDataType(0))） |
| dk1Out | aclTensor | 输出 | 必选 | 5维 [B,H,N,K,D] | ND | FLOAT16、BFLOAT16 | 不适用（输出参数） | - | 不支持（同上） | shape 与 key1 相同（R19）；数据类型与 query 一致（R18） |
| dv1Out | aclTensor | 输出 | 必选 | 5维 [B,H,N,K,D] | ND | FLOAT16、BFLOAT16 | 不适用（输出参数） | - | 不支持（同上） | shape 与 value1 相同（R19）；数据类型与 query 一致（R18） |
| dk2Out | aclTensor | 输出 | 必选 | 5维 [B,H,K,M,D] | ND | FLOAT16、BFLOAT16 | 不适用（输出参数） | - | 不支持（同上） | shape 与 key2 相同（R19）；数据类型与 query 一致（R18） |
| dv2Out | aclTensor | 输出 | 必选 | 5维 [B,H,K,M,D] | ND | FLOAT16、BFLOAT16 | 不适用（输出参数） | - | 不支持（同上） | shape 与 value2 相同（R19）；数据类型与 query 一致（R18） |

## 约束关系

> 中间符号定义（shape 轴语义变量，均需在 02_test_factors.yaml intermediate 区段定义）：
> B=batch（`_b`）、H=head_num（`_h`）、N=query 头数维（`_n`，对应 host 代码 s1 前的 n2/head 维）、M=query 序列维（`_m`，host 代码 s1）、K=key/value 序列维（`_k`，host 代码 s2/s3 共用约束）、D=head_dim（`_d`）。
> 说明：host 代码 tiling_common.cpp 中变量 (b, n, s1, s2, s3, d) 与文档 (B, H, N, M, K, D) 一一对应（softmaxMax 模板 [b,n,s1,s2,8]=[B,H,N,M,8]，attenMask 模板 [b,1,s1,1,s3]=[B,1,N,1,K]）。

### R1: B 轴取值范围约束
涉及参数: _b.value
条件因子: 无
来源: 接口文档"约束说明"章节 "B：取值范围为1~2K"
B ∈ [1, 2048]。

### R2: H 轴取值范围约束
涉及参数: _h.value
条件因子: 无
来源: 接口文档"约束说明"章节 "H：取值范围为1~256"
H ∈ [1, 256]。

### R3: N 轴取值范围与对齐约束
涉及参数: _n.value
条件因子: 无
来源: 接口文档"约束说明"章节 "N：取值范围为16~1M且N%16==0"；op_host tiling_common.cpp "The third dimension of the query only supports 16-byte alignment"
N ∈ [16, 1048576] 且 N % 16 == 0。

### R4: M 轴取值范围与对齐约束
涉及参数: _m.value
条件因子: 无
来源: 接口文档"约束说明"章节 "M：取值范围为128~1M且M%128==0"；op_host tiling_common.cpp "the fourth dimension only supports 128-byte alignment"
M ∈ [128, 1048576] 且 M % 128 == 0。

### R5: K 轴取值范围与对齐约束
涉及参数: _k.value
条件因子: 无
来源: 接口文档"约束说明"章节 "K：取值范围为128~1M且K%128==0"；op_host tiling_common.cpp "The fourth dimension of the key1 only supports 128-byte alignment"
K ∈ [128, 1048576] 且 K % 128 == 0。

### R6: D 轴枚举约束
涉及参数: _d.value
条件因子: 无
来源: 接口文档"约束说明"章节 "D：取值范围为32/64/128"、"D只支持32/64/128"；op_host tiling_common.cpp "Headdim only supports 32/64/128."
D ∈ {32, 64, 128}。

### R7: query 与 key1 的第 0/2/4 轴相同
涉及参数: query.shape → key1.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "query与key1的第0/2/4轴需相同"
query.shape[0]==key1.shape[0]（B）、query.shape[2]==key1.shape[2]（N）、query.shape[4]==key1.shape[4]（D）。

### R8: key1 与 value1 shape 相同
涉及参数: key1.shape → value1.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "key1与value1 shape需相同"；op_host tiling_common.cpp CheckSameShape(key1Shape, value1Shape)

### R9: key2 与 value2 shape 相同
涉及参数: key2.shape → value2.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "key2与value2 shape需相同"；op_host tiling_common.cpp CheckSameShape(key2Shape, value2Shape)

### R10: query 与 dy shape 相同
涉及参数: query.shape → dy.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "query与dy/attentionIn shape需相同"；op_host tiling_common.cpp CheckSameShape(queryShape, dyShape)

### R11: query 与 attentionIn shape 相同
涉及参数: query.shape → attentionIn.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "query与dy/attentionIn shape需相同"；op_host tiling_common.cpp CheckSameShape(queryShape, attentionInShape)

### R12: softmaxMax 与 softmaxSum shape 相同
涉及参数: softmaxMax.shape → softmaxSum.shape
条件因子: 无
来源: 接口文档"约束说明"章节 "softmaxMax与softmaxSum shape需相同"

### R13: attenMaskOptional shape 模板约束
涉及参数: attenMaskOptional.shape
条件因子: attenMaskOptional.exist
来源: 接口文档参数表"维度(shape)"列 "[B,1,N,1,K]"；op_host tiling_common.cpp CheckAttentionMaskShape(b, s1, s3)
存在时（exist=true）shape 必须为 [B,1,N,1,K]（B/N 轴与 query 一致，K 轴与 key1 一致，第 1/3 轴固定为 1）；exist=false 时不适用。

### R14: softmaxMax/softmaxSum shape 模板约束
涉及参数: query.shape → softmaxMax.shape, softmaxSum.shape
条件因子: 无
来源: 接口文档参数表"维度(shape)"列 "[B,H,N,M,8]" + "使用说明"列 "输出的shape类型为[B,H,N,M,8]"；op_host tiling_common.cpp CheckSoftmaxMaxAndSumShape(b, n, s1, s2, 8)
softmaxMax.shape = softmaxSum.shape = [B,H,N,M,8]（B/H/N/M 轴与 query 一致，尾轴固定 8）。

### R15: 五输入数据类型一致
涉及参数: query.dtype → key1.dtype, value1.dtype, key2.dtype, value2.dtype
条件因子: 无
来源: 接口文档参数表"使用说明"列 "数据类型与key1/value1/key2/value2的数据类型一致"（query/key1/value1/key2/value2 五行互指）
query、key1、value1、key2、value2 的数据类型必须完全相同（FLOAT16 或 BFLOAT16）。

### R16: attentionIn 数据类型与 query 一致
涉及参数: query.dtype → attentionIn.dtype
条件因子: 无
来源: 接口文档参数表"使用说明"列 "数据类型和shape类型与query保持一致"

### R17: dy 数据类型与 query 一致
涉及参数: query.dtype → dy.dtype
条件因子: 无
来源: 推断（文档未显式声明）：dy 为正向输出 Y=(P*V1+P*V2) 的梯度，Y 即 attentionIn（数据类型与 query 保持一致，R16），梯度数据类型与输出数据类型一致。[待确认]

### R18: 输出数据类型与 query 一致
涉及参数: query.dtype → dqOut.dtype, dk1Out.dtype, dv1Out.dtype, dk2Out.dtype, dv2Out.dtype
条件因子: 无
来源: op_host/fused_floyd_attention_grad_infershape.cpp（SetOutputDataType(0..4, GetInputDataType(0))，源码佐证）；接口文档参数表输出行"数据类型"列 FLOAT16、BFLOAT16 与输入一致

### R19: 输出 shape 与对应输入相同
涉及参数: query.shape → dqOut.shape; key1.shape → dk1Out.shape; value1.shape → dv1Out.shape; key2.shape → dk2Out.shape; value2.shape → dv2Out.shape
条件因子: 无
来源: 接口文档参数表"维度(shape)"列（dqOut=[B,H,N,M,D]=query；dk1Out/dv1Out=[B,H,N,K,D]=key1/value1；dk2Out/dv2Out=[B,H,K,M,D]=key2/value2）

### R20: attenMaskOptional 元素值域约束
涉及参数: attenMaskOptional.value_range_bool, attenMaskOptional.value_range_uint8
条件因子: 无
来源: 接口文档参数表"使用说明"列 "取值为1代表该位不参与计算，为0代表该位参与计算"
元素值域 {0,1}：0=该位参与计算，1=该位不参与计算（bool/uint8 两种 dtype 均为二值语义）。

## 约束提取反向校验表

| R{n} | 约束项 | 原文片段 | 符号化 | 转录结果 | 状态 |
|------|--------|---------|--------|---------|------|
| R1 | B 轴范围 | "B：取值范围为1~2K" | [1, 2048] | B ∈ [1, 2048] | EXACT_MATCH |
| R2 | H 轴范围 | "H：取值范围为1~256" | [1, 256] | H ∈ [1, 256] | EXACT_MATCH |
| R3 | N 轴范围与对齐 | "N：取值范围为16~1M且N%16==0" | [16, 1048576] 且 %16==0 | N ∈ [16, 1048576] 且 N%16==0 | EXACT_MATCH |
| R4 | M 轴范围与对齐 | "M：取值范围为128~1M且M%128==0" | [128, 1048576] 且 %128==0 | M ∈ [128, 1048576] 且 M%128==0 | EXACT_MATCH |
| R5 | K 轴范围与对齐 | "K：取值范围为128~1M且K%128==0" | [128, 1048576] 且 %128==0 | K ∈ [128, 1048576] 且 K%128==0 | EXACT_MATCH |
| R6 | D 轴枚举 | "D：取值范围为32/64/128" | D ∈ {32,64,128} | D ∈ {32, 64, 128} | EXACT_MATCH |
| R7 | query/key1 轴一致 | "query与key1的第0/2/4轴需相同" | 第0/2/4轴相等 | query.shape[0/2/4] == key1.shape[0/2/4] | EXACT_MATCH |
| R8 | key1/value1 shape | "key1与value1 shape需相同" | shape 相同 | key1.shape == value1.shape | EXACT_MATCH |
| R9 | key2/value2 shape | "key2与value2 shape需相同" | shape 相同 | key2.shape == value2.shape | EXACT_MATCH |
| R10 | query/dy shape | "query与dy/attentionIn shape需相同" | shape 相同 | query.shape == dy.shape | EXACT_MATCH |
| R11 | query/attentionIn shape | "query与dy/attentionIn shape需相同" | shape 相同 | query.shape == attentionIn.shape | EXACT_MATCH |
| R12 | softmaxMax/Sum shape | "softmaxMax与softmaxSum shape需相同" | shape 相同 | softmaxMax.shape == softmaxSum.shape | EXACT_MATCH |
| R13 | attenMask shape 模板 | "维度(shape)列 [B,1,N,1,K]" | shape=[B,1,N,1,K] | 存在时 shape=[B,1,N,1,K] | EXACT_MATCH |
| R14 | softmax shape 模板 | "输出的shape类型为[B,H,N,M,8]" | shape=[B,H,N,M,8] | [B,H,N,M,8] 尾轴固定 8 | EXACT_MATCH |
| R15 | 五输入 dtype 一致 | "数据类型与key1/value1/key2/value2的数据类型一致" | dtype 相同 | 五输入 dtype 完全相同 | EXACT_MATCH |
| R16 | attentionIn dtype | "数据类型和shape类型与query保持一致" | dtype == query | attentionIn.dtype == query.dtype | EXACT_MATCH |
| R17 | dy dtype | — | — | dy.dtype == query.dtype（推断自：dy 为 Y 的梯度，Y dtype=query） | INFERENCE |
| R18 | 输出 dtype | — | — | 输出 dtype == query.dtype（推断自：infershape.cpp SetOutputDataType(GetInputDataType(0))） | INFERENCE |
| R19 | 输出 shape | "维度(shape)列 dqOut=[B,H,N,M,D] 等" | shape=对应输入 | 各输出 shape == 对应输入 shape | EXACT_MATCH |
| R20 | mask 元素值域 | "取值为1代表该位不参与计算，为0代表该位参与计算" | 元素 ∈ {0,1} | 元素值域 {0,1}（0=参与，1=不参与） | EXACT_MATCH |
