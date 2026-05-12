# arch35校验逻辑对比分析报告

## 一、代码中发现但文档缺失的校验逻辑

### 1. queryRope和keyRope必须同时存在或不存在
**代码位置**: flash_attention_score_tiling_regbase.cpp:234-237  
**错误信息**: "query_rope and key_rope should be present or absent at the same time, check this."  
**文档缺失**: 文档未提及此约束。  
**建议补充**: 在约束说明中增加"queryRopeOptional和keyRopeOptional必须同时存在或同时为空指针"。

### 2. s1Size和s2Size不能超过INT32_MAX
**代码位置**: flash_attention_score_tiling_regbase.cpp:250-254  
**错误信息**: "s1Size[%ld] and s2Size[%ld] config error, both should not greater than max int value."  
**文档缺失**: 文档第574行提到T取值范围1~1M，但未明确限制s1Size和s2Size必须小于INT32_MAX。  
**建议补充**: 明确说明Sq和Skv不得超过INT32_MAX（约2.1亿）。

### 3. gSize不能为0
**代码位置**: flash_attention_score_tiling_regbase.cpp:255  
**错误信息**: "gSize is zero"  
**文档缺失**: 文档未说明gSize（group size）的约束。  
**建议补充**: 说明gSize = N1/N2必须大于0，且N1必须是N2的整数倍。

### 4. n2Size不能为0
**代码位置**: flash_attention_score_tiling_regbase.cpp:256  
**错误信息**: "n2Size is zero"  
**文档缺失**: 文档第576行提到N取值范围1~256，但未明确说明N2不能为0。  
**建议补充**: 明确说明N2（key/value的head数）必须大于0。

### 5. dSize必须大于0
**代码位置**: flash_attention_score_tiling_regbase.cpp:257-260  
**错误信息**: "query or key dSize is not support <= 0" 和 "value's dSize is not support <= 0"  
**文档缺失**: 文档第578行提到D取值范围1~768，但校验逻辑更严格。  
**建议补充**: 明确说明query/key的dSize必须大于0，value的dSize必须大于0且不超过query/key的dSize。

### 6. dSizeV不能超过dSize
**代码位置**: flash_attention_score_tiling_regbase.cpp:261-262  
**错误信息**: "value's dSize is larger than query's dSize"  
**文档缺失**: 文档第569行提到"kD >= vD"，但表述不够清晰。  
**建议补充**: 明确说明"value的dSize不能超过query/key的dSize"。

### 7. N1必须是N2的整数倍
**代码位置**: flash_attention_score_tiling_regbase.cpp:263-265  
**错误信息**: "n1Size [%ld] should be a multiple of n2Size [%ld]"  
**文档缺失**: 文档未明确说明此约束。  
**建议补充**: 明确说明N1必须是N2的整数倍，即支持GQA（Grouped Query Attention）时N1/N2必须为整数。

### 8. TND格式下actualSeqQLen必须等于actualSeqKvLen的batch数
**代码位置**: flash_attention_score_tiling_regbase.cpp:324-325  
**错误信息**: "VarLen scene, q is not equal kv."  
**文档缺失**: 文档未明确说明TND格式下actualSeqQLenOptional和actualSeqKvLenOptional的batch数必须相等。  
**建议补充**: 在TND相关约束中补充"actualSeqQLenOptional和actualSeqKvLenOptional的长度（batch数）必须相等"。

### 9. TND格式下T必须大于等于所有batch的sequence长度之和
**代码位置**: flash_attention_score_tiling_regbase.cpp:329-335  
**错误信息**: "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld)."  
**文档缺失**: 文档第574行只提到T取值范围1~1M，未说明T必须大于等于实际sequence长度之和。  
**建议补充**: 明确说明"TND格式下，query的T轴大小必须大于等于所有batch的Sq之和，key的T轴大小必须大于等于所有batch的Skv之和"。

### 10. TND格式下s1Size必须大于0
**代码位置**: flash_attention_score_tiling_regbase.cpp:360-361  
**错误信息**: "s1Size should be larger than 0."  
**文档缺失**: 文档未明确说明TND格式下至少有一个batch的sequence长度大于0。  
**建议补充**: 说明"TND格式下，至少有一个batch的Sq长度必须大于0"。

### 11. headNum必须与query的shape匹配
**代码位置**: 
- flash_attention_score_tiling_regbase.cpp:362-365 (TND格式)
- flash_attention_score_tiling_regbase.cpp:457-460 (BSND格式)
- flash_attention_score_tiling_regbase.cpp:477-480 (BNSD格式)

**错误信息**: "head_num is [%ld], but got query dim1 [%ld]" 或 "head_num is [%ld], but got query dim2 [%ld]"  
**文档缺失**: 文档第348行只说明headNum代表N轴长度，但未明确必须与query的shape一致。  
**建议补充**: 明确说明"headNum必须等于query的N轴大小"。

### 12. H不能为0
**代码位置**: flash_attention_score_tiling_regbase.cpp:428  
**错误信息**: "H is zero."  
**文档缺失**: 文档未明确说明H（Head-Size）不能为0。  
**建议补充**: 说明"H = N*D必须大于0"。

### 13. H1必须是N1的整数倍
**代码位置**: flash_attention_score_tiling_regbase.cpp:429-431  
**错误信息**: "h1 [%ld] should be a multiple of n1Size [%ld]."  
**文档缺失**: 文档未明确说明此约束。  
**建议补充**: 说明"在BSH/SBH格式下，H1必须是N1的整数倍，从而可以计算D = H1/N1"。

### 14. hRope必须是n1Size的整数倍
**代码位置**: flash_attention_score_tiling_regbase.cpp:432-434  
**错误信息**: "hRope [%ld] should be a multiple of n1Size [%ld]."  
**文档缺失**: 文档未提及queryRope的shape约束。  
**建议补充**: 说明"queryRopeOptional的H轴必须是N1的整数倍"。

### 15. PSE inner mode的shape约束
**代码位置**: flash_attention_score_tiling_regbase.cpp:626-628  
**错误信息**: "pse inner mode, unsupported pse shape"  
**文档缺失**: 文档第613行只提到"pseType为2或3的时候，数据类型需为FLOAT32, 对应shape支持范围是[B,N]或[N]"，但缺少维度校验说明。  
**建议补充**: 明确说明"pseType为2或3时，realShiftOptional的维度必须是1或2，shape为[N]或[B,N]"。

### 16. PSE batchsize必须为1或等于B
**代码位置**: flash_attention_score_tiling_regbase.cpp:581-583  
**错误信息**: "pse batchsize should be 1 or %ld, but got %ld"  
**文档缺失**: 文档第162行列出shape支持[B,N,Sq,Skv]，但未明确说明B轴可以是1或等于实际batchsize。  
**建议补充**: 明确说明"realShiftOptional的B轴可以是1（所有batch共享）或等于query的batchsize"。

### 17. attenMask压缩场景的shape限制
**代码位置**: flash_attention_score_tiling_regbase.cpp:696-709  
**错误信息**: 
- "In the attenmask compression, please set the atten_mask_shape to [2048,2048]."
- "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048]."

**文档缺失**: 文档未明确说明attenMask压缩场景下shape必须固定为[2048,2048]或[3072,2048]。  
**建议补充**: 详细说明各sparseMode下attenMask压缩的shape要求。

### 18. dropMask shape大小校验
**代码位置**: flash_attention_score_tiling_regbase.cpp:783-787  
**错误信息**: "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld"  
**文档缺失**: 文档第182行提到dropMask维度为"0、1"，但未说明具体shape大小要求。  
**建议补充**: 明确说明"dropMaskOptional的元素数量必须大于等于(B*N1*S1*S2)/8（按bit打包）"。

### 19. sink的dtype校验
**代码位置**: flash_attention_score_tiling_regbase.cpp:902-905  
**错误信息**: "invalid sink dtype[%s], only support float."  
**文档部分缺失**: 文档第250行说明sink数据类型支持FLOAT32，但缺少错误说明。  
**建议补充**: 在约束说明中强调"sinkOptional仅支持FLOAT32类型"。

### 20. sink的shape必须为[N1]
**代码位置**: flash_attention_score_tiling_regbase.cpp:914-916  
**错误信息**: "invalid sink shape [%s], sink only support [n,]."  
**文档缺失**: 文档第252行说明shape为[N1]，但缺少维度校验说明。  
**建议补充**: 明确说明"sinkOptional必须是1维Tensor，且元素数量必须等于N1"。

### 21. sink场景下其他tensor的dtype限制
**代码位置**: flash_attention_score_tiling_regbase.cpp:925-927  
**错误信息**: "invalid input dtype, other tensor`s dtype only support float16 and bf16."  
**文档缺失**: 文档未说明当sink存在时，query/key/value的dtype限制。  
**建议补充**: 说明"当sinkOptional存在时，query/key/value的数据类型仅支持FLOAT16和BFLOAT16"。

### 22. dScaleQ/K/V的shape和dtype校验
**代码位置**: flash_attention_score_tiling_regbase.cpp:801-871  
**错误信息**: 
- "invalid dScaleQ dtype[%s], only support float32."
- "invalid dScaleQ dimNum [%ld], only support 4 dims."
- "invalid dScaleQ dimNump[%ld][%ld][%ld][%ld], only support [B, N1, ceil(S1/128), 1]"

**文档缺失**: 文档第218-243行说明dScaleQ/K/V的shape，但缺少具体校验说明。  
**建议补充**: 详细说明量化参数的完整约束：
- dScaleQOptional: dtype必须是FLOAT32，shape必须是[B, N1, ceil(S1/128), 1]
- dScaleKOptional: dtype必须是FLOAT32，shape必须是[B, N2, ceil(S2/256), 1]
- dScaleVOptional: dtype必须是FLOAT32，shape必须是[B, N2, ceil(S2/512), 1]

### 23. band场景preTokens和nextTokens的交集校验
**代码位置**: flash_attention_score_tiling_regbase.cpp:1268-1271  
**错误信息**: "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block."  
**文档部分缺失**: 文档第598行提到"band场景，preTokens和nextTokens之间必须要有交集"，但缺少详细的校验逻辑说明。  
**建议补充**: 详细说明交集校验逻辑：
- 当preTokens < 0且nextTokens < 0时，无有效数据块，报错
- 其他情况的详细校验规则

### 24. kvStartIdx - qStartIdx的范围校验
**代码位置**: flash_attention_score_tiling_regbase.cpp:1713-1715  
**错误信息**: "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld."  
**文档缺失**: 文档未说明qStartIdx和kvStartIdx的差值范围限制。  
**建议补充**: 说明"kvStartIdx - qStartIdx必须在INT32_MIN到INT32_MAX范围内，以避免kernel侧类型转换截断"。

### 25. PSE alibi只支持causal sparse type
**代码位置**: 
- flash_attention_score_tiling_regbase.cpp:1729-1733
- flash_attention_score_tiling_varlen.cpp:158-161
- flash_attention_score_tiling_basic.cpp:134-135

**错误信息**: 
- "Pse alibi only support same S1 S2 when S1 lager than 1024"
- "Pse alibi only support causal sparse type."

**文档缺失**: 文档第610-614行提到alibi位置编码压缩，但未说明sparse type限制。  
**建议补充**: 明确说明"当s1Size > 1024且使用alibi压缩时，只支持causal类型的sparseMode（sparseMode=1或3）"。

### 26. VarLen场景不支持sparseMode 5（PREFIX）
**代码位置**: flash_attention_score_tiling_varlen.cpp:474-477  
**错误信息**: "Var len not support sparse mode %ld."  
**文档缺失**: 文档未说明TND（VarLen）格式下不支持sparseMode=5。  
**建议补充**: 说明"TND格式下不支持sparseMode=5（PREFIX模式），只支持sparseMode=6（PREFIX_COMPRESS）"。

### 27. Inner PSE不支持sparseMode 7（RIGHT_DOWN_CAUSAL_BAND）
**代码位置**: flash_attention_score_tiling_varlen.cpp:130-131  
**错误信息**: "INNER Pse does not support sparse mode 7."  
**文档缺失**: 文档未说明pseType=2或3时不支持RIGHT_DOWN_CAUSAL_BAND模式。  
**建议补充**: 说明"pseType为2或3时，不支持sparseMode=7（RIGHT_DOWN_CAUSAL_BAND）"。

### 28. Inner PSE sparseMode 8的特殊约束
**代码位置**: flash_attention_score_tiling_varlen.cpp:105-118  
**错误信息**: "Inner pse sparse mode 8 is only supported when actualSeqLenData[0] %ld + qStartIdx %ld - actualSeqLenKvData[0] %ld - kvStartIdx %ld == 0."  
**文档缺失**: 文档未说明pseType=2或3且sparseMode=8时的特殊约束。  
**建议补充**: 说明"pseType为2或3且sparseMode=8时，第一个batch的Sq + qStartIdx必须等于Skv + kvStartIdx，且所有batch的Sq必须等于Skv"。

### 29. VarLen场景下RIGHT_DOWN_CAUSAL的约束
**代码位置**: flash_attention_score_tiling_varlen.cpp:319-324  
**错误信息**: "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row."  
**文档缺失**: 文档未说明TND格式下sparseMode=2（RIGHT_DOWN_CAUSAL）时所有batch的Sq不能超过Skv。  
**建议补充**: 说明"TND格式下，sparseMode=2时，每个batch的Sq长度不能超过对应的Skv长度"。

### 30. VarLen场景下BAND模式的nextTokens校验
**代码位置**: flash_attention_score_tiling_varlen.cpp:257-261  
**错误信息**: "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row."  
**文档缺失**: 文档未说明TND格式下BAND模式的详细校验逻辑。  
**建议补充**: 说明"TND格式下，sparseMode=3时，每个batch的Sq - nextTokens不能超过对应的Skv"。

### 31. VarLen场景下RIGHT_DOWN_CAUSAL_BAND的约束
**代码位置**: flash_attention_score_tiling_varlen.cpp:276-296  
**错误信息**: 
- "RightDownCausal_Band mode: pre_tokens[%ld] is smaller than last valid s2[%ld] or next_tokens[%ld] is larger than 0"
- "Batch[%ld] s1[%ld] is larger than s2[%ld]."

**文档缺失**: 文档未说明sparseMode=7（RIGHT_DOWN_CAUSAL_BAND）的完整约束。  
**建议补充**: 详细说明该模式的约束：
- preTokens必须大于等于最后一个有效batch的Skv长度
- nextTokens必须小于等于0
- 所有batch的Sq不能超过Skv

### 32. VarLen场景下BAND_LEFT_UP_CAUSAL的约束
**代码位置**: flash_attention_score_tiling_varlen.cpp:339-349  
**错误信息**: 
- "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row."
- "Band_LeftUpCausal mode: pre_tokens[%ld] is smaller than first valid s2[%ld]."

**文档缺失**: 文档未说明sparseMode=8（BAND_LEFT_UP_CAUSAL）的完整约束。  
**建议补充**: 详细说明该模式的约束：
- 第一个有效batch的Sq - nextTokens不能超过对应的Skv
- preTokens必须大于等于第一个有效batch的Skv长度

### 33. prefixN的shape必须为[B]
**代码位置**: 
- flash_attention_score_tiling_regbase.cpp:1639-1650
- flash_attention_score_tiling_varlen.cpp:408-418

**错误信息**: "prefixN is invalid, it should be the same size as bSize[%ld], but it is %ld."  
**文档缺失**: 文档第262行说明prefix维度为"0、1"，但未明确说明必须与batchsize相等。  
**建议补充**: 说明"prefixOptional必须是1维Tensor，且长度必须等于batchsize"。

### 34. prefixN的取值范围校验（非TND场景）
**代码位置**: flash_attention_score_tiling_regbase.cpp:1660-1674  
**错误信息**: "batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]"  
**文档部分缺失**: 文档第599-600行说明了prefix的N值取值范围，但缺少每个batch的详细校验说明。  
**建议补充**: 明确说明"每个batch的prefixN值必须满足：
- 当Sq > Skv时：prefixN在[0, Skv]范围内
- 当Sq <= Skv时：prefixN在[Skv-Sq, Skv]范围内"

### 35. prefixN的取值范围校验（TND场景）
**代码位置**: flash_attention_score_tiling_varlen.cpp:428-446  
**错误信息**: "batch[%ld] prefixN=%ld is invalid, should be in range of [0, %ld]" 或 "batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]"  
**文档缺失**: 文档未说明TND格式下prefixN的每个batch独立校验。  
**建议补充**: 说明"TND格式下，每个batch的prefixN值根据该batch的Sq和Skv进行独立校验"。

### 36. HIFLOAT8的严格约束
**代码位置**: flash_attention_score_tiling_regbase.cpp:511-521  
**错误信息**: "HIFLOAT8 can only support layout:BSND without any optional inputs..."  
**文档缺失**: 文档未说明HIFLOAT8输入类型的严格限制。  
**建议补充**: 详细说明HIFLOAT8的完整约束：
- 只支持BSND layout
- 不能有任何可选输入（attenMask、pse、dropMask、rope）
- 必须满足特定shape组合

---

## 二、文档已有但代码可能缺失或不一致的校验逻辑

### 1. preTokens/nextTokens超过INT32范围的处理
**文档**: 未明确说明  
**代码**: flash_attention_score_tiling_regbase.cpp:141-158 会自动重置为INT32_MAX或INT32_MIN，仅warning而非error  
**建议**: 文档应说明"preTokens和nextTokens超过INT32范围时会自动调整，不会报错"

### 2. softmaxOutLayout的校验
**文档**: 第609行说明支持空字符串和"same_as_input"  
**代码**: 未找到明确的校验逻辑  
**建议**: 补充代码校验，确保softmaxOutLayout参数有效

### 3. seed和offset在非950PR/DT产品上的行为
**文档**: 第606-607行只说明950PR/DT的行为  
**代码**: 未找到其他产品的处理逻辑  
**建议**: 补充校验或在文档中明确说明其他产品不支持seed/offset参数

---

## 三、建议优先补充到文档的校验逻辑（按重要性排序）

### 高优先级（影响功能正确性）：
1. queryRope和keyRope必须同时存在或不存在
2. N1必须是N2的整数倍（GQA约束）
3. sink场景下query/key/value仅支持FP16/BF16
4. kvStartIdx - qStartIdx的范围限制
5. TND格式下不支持sparseMode=5
6. PSE alibi只支持causal sparse type
7. Inner PSE不支持RIGHT_DOWN_CAUSAL_BAND
8. VarLen场景下各sparseMode的详细约束
9. dScaleQ/K/V的完整shape和dtype约束
10. attenMask压缩场景的固定shape要求

### 中优先级（影响易用性）：
1. TND格式下T必须大于等于所有sequence长度之和
2. headNum必须与query shape匹配
3. prefixN的shape和取值范围详细校验
4. PSE inner mode的维度约束
5. dropMask的shape大小要求

### 低优先级（边界情况）：
1. HIFLOAT8的严格约束
2. 各参数不能为0的校验
3. band场景preTokens/nextTokens交集的详细说明