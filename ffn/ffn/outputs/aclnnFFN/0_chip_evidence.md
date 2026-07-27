# 芯片支持静态分析

## 产品型号映射配置

- 配置文件: C:\Users\c60107599\.cannbot\skills\aclnn-doc-gen\config\product_mapping.yml

## 映射命中摘要

| 原始标识 | 类型 | 映射产品型号 |
|---|---|---|
| `ascend910b` | AddConfig | Atlas A2 训练系列产品/Atlas A2 推理系列产品 |
| `ascend910_93` | AddConfig | Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| `ascend310p` | AddConfig | Atlas 推理系列产品 |
| `kirinx90` | AddConfig | 映射表中无对应条目 |
| `kirin9030` | AddConfig | 映射表中无对应条目 |
| `ASCEND310P` | SocVersion | Atlas 推理系列产品 |
| `__NPU_ARCH__==3003` | __NPU_ARCH__ | 映射表中无对应条目 |
| `__NPU_ARCH__==3113` | __NPU_ARCH__ | 映射表中无对应条目 |
| `__CCE_AICORE__==220` | __CCE_AICORE__ | 映射表中无对应条目 |
| `__CCE_AICORE__==200` | __CCE_AICORE__ | 映射表中无对应条目 |

## 规则1：AddConfig 注册声明

**ffn/op_host/ffn_def.cpp:163** — `ascend910b`

```cpp
161:             .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
162: 
163:         this->AICore().AddConfig("ascend910b", aicoreConfig);
164:         this->AICore().AddConfig("ascend910_93", aicoreConfig);
165:         aicoreConfig.Input("x")
166:             .ParamType(REQUIRED)
167:             .DataType({ge::DT_FLOAT16})
168:             .Format({ge::FORMAT_FRACTAL_NZ})
```

**ffn/op_host/ffn_def.cpp:164** — `ascend910_93`

```cpp
162: 
163:         this->AICore().AddConfig("ascend910b", aicoreConfig);
164:         this->AICore().AddConfig("ascend910_93", aicoreConfig);
165:         aicoreConfig.Input("x")
166:             .ParamType(REQUIRED)
167:             .DataType({ge::DT_FLOAT16})
168:             .Format({ge::FORMAT_FRACTAL_NZ})
169:             .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
```

**ffn/op_host/ffn_def.cpp:240** — `ascend310p`

```cpp
238:             .Format({ge::FORMAT_FRACTAL_NZ})
239:             .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
240:         this->AICore().AddConfig("ascend310p", aicoreConfig);
241: 
242:         OpAICoreConfig config_kirin = GetKirinCoreConfig();
243:         this->AICore().AddConfig("kirinx90", config_kirin);
244:         this->AICore().AddConfig("kirin9030", config_kirin);
245:     }
```

**ffn/op_host/ffn_def.cpp:243** — `kirinx90`

```cpp
241: 
242:         OpAICoreConfig config_kirin = GetKirinCoreConfig();
243:         this->AICore().AddConfig("kirinx90", config_kirin);
244:         this->AICore().AddConfig("kirin9030", config_kirin);
245:     }
246: 
247: private:
248:     OpAICoreConfig GetKirinCoreConfig() const
```

**ffn/op_host/ffn_def.cpp:244** — `kirin9030`

```cpp
242:         OpAICoreConfig config_kirin = GetKirinCoreConfig();
243:         this->AICore().AddConfig("kirinx90", config_kirin);
244:         this->AICore().AddConfig("kirin9030", config_kirin);
245:     }
246: 
247: private:
248:     OpAICoreConfig GetKirinCoreConfig() const
249:     {
```

## 规则2：运行时 NpuArch/SocVersion 分支

**ffn/op_host/ffn_tiling.cpp:591** — `ASCEND310P`

```cpp
589: {
590:     const uint32_t coreNum = compileInfoPtr->coreNum;
591:     is310P = compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P;
592: 
593:     OP_CHECK_IF(
594:         (coreNum == 0 || compileInfoPtr->ubSize == 0 || compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 ||
595:          compileInfoPtr->l0ASize == 0 || compileInfoPtr->l0BSize == 0),
596:         OPS_REPORT_VECTOR_INNER_ERR(
```

**ffn/op_host/op_api/aclnn_ffn.cpp:719** — `ASCEND310P`

```cpp
717: static aclnnStatus CheckFormat(const FFNParams &ffnParams)
718: {
719:     bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
720:     op::Format xFormat = ffnParams.x->GetStorageFormat();
721:     op::Format weight1Format = ffnParams.weight1->GetStorageFormat();
722:     op::Format weight2Format = ffnParams.weight2->GetStorageFormat();
723:     op::Format yFormat = ffnParams.y->GetStorageFormat();
724:     bool isXFormatValid = xFormat < Format::FORMAT_END &&
```

**ffn/op_host/op_api/aclnn_ffn.cpp:755** — `ASCEND310P`

```cpp
753:     CHECK_RET(CheckFmapWeightShape(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
754:     CHECK_RET(CheckBias(ffnParams), ACLNN_ERR_PARAM_INVALID);
755:     if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
756:         auto xDimNum = ffnParams.x->GetViewShape().GetDimNum();
757:         CHECK_COND(xDimNum == DIM_LIMIT_LOWER, ACLNN_ERR_PARAM_INVALID,
758:                    "FFN only supports x dim 2 on ASCEND310P, but x dim is %zu", xDimNum);
759:         CHECK_COND(ffnParams.innerPrecise == HIGH_PERFORMANCE, ACLNN_ERR_PARAM_INVALID,
760:                    "FFN is only support high performance now!");
```

**ffn/op_host/op_api/aclnn_ffn.cpp:787** — `ASCEND310P`

```cpp
785:         CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());
786: 
787:         bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
788:         // only in 310P, tensor should be transdata to NZ format
789:         if (!is310P) {
790:             return ACLNN_SUCCESS;
791:         }
792: 
```

**ffn/op_host/op_api/aclnn_ffn.cpp:986** — `ASCEND310P`

```cpp
984:                                      const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
985: {
986:     CHECK_COND(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P, ACLNN_ERR_PARAM_INVALID,
987:                "aclnnFFNGetWorkspaceSize and aclnnFFN are not supported on Ascend310P Soc. Please use "
988:                "aclnnFFNV2GetWorkspaceSize and aclnnFFNV2!");
989:     ActiveType activationType = GetActiveType(activation);
990:     CHECK_COND(activationType != ActiveType::INVALID_TYPE, ACLNN_ERR_PARAM_INVALID,
991:                "the activation type %s is not supported by ffn operator, please select right activation according to "
```

## 规则3：编译期架构宏分支

**ffn/op_kernel/ffn.cpp:17** — `__NPU_ARCH__==3003`

```cpp
15: 
16: #include "kernel_operator.h"
17: #if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
18: #include "ffn_high_performence.h"
19: #include "ffn_quant.h"
20: #include "ffn_antiquant.h"
21: #include "ffn_antiquant_msd.h"
22: #include "ffn_glu.cpp"
```

**ffn/op_kernel/ffn.cpp:17** — `__NPU_ARCH__==3113`

```cpp
15: 
16: #include "kernel_operator.h"
17: #if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
18: #include "ffn_high_performence.h"
19: #include "ffn_quant.h"
20: #include "ffn_antiquant.h"
21: #include "ffn_antiquant_msd.h"
22: #include "ffn_glu.cpp"
```

**ffn/op_kernel/ffn.cpp:75** — `__NPU_ARCH__==3003`

```cpp
73:     MMType<aType<T, cubeFormatA>, bType<T, CubeFormat::ND>, cType<cT, tPositionC>, biasType<biasT>, MM_CFG>;
74: 
75: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
76: template <typename T> struct GetC1Type {
77:     using Type = T;
78: };
79: 
80: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
```

**ffn/op_kernel/ffn.cpp:75** — `__NPU_ARCH__==3113`

```cpp
73:     MMType<aType<T, cubeFormatA>, bType<T, CubeFormat::ND>, cType<cT, tPositionC>, biasType<biasT>, MM_CFG>;
74: 
75: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
76: template <typename T> struct GetC1Type {
77:     using Type = T;
78: };
79: 
80: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
```

**ffn/op_kernel/ffn.cpp:80** — `__NPU_ARCH__==3003`

```cpp
78: };
79: 
80: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
81: template <> struct GetC1Type<bfloat16_t> {
82:     using Type = float;
83: };
84: #endif
85: 
```

**ffn/op_kernel/ffn.cpp:80** — `__NPU_ARCH__==3113`

```cpp
78: };
79: 
80: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
81: template <> struct GetC1Type<bfloat16_t> {
82:     using Type = float;
83: };
84: #endif
85: 
```

**ffn/op_kernel/ffn.cpp:110** — `__NPU_ARCH__==3003`

```cpp
108:     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
109: 
110: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
111:     if (TILING_KEY_IS(2000)) { // One matmul, float16
112:         KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_1);
113:         using mt = mmType<half, CubeFormat::ND, CubeFormat::ND>;
114:         mt::MT mm;
115:         const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
```

**ffn/op_kernel/ffn.cpp:110** — `__NPU_ARCH__==3113`

```cpp
108:     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
109: 
110: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
111:     if (TILING_KEY_IS(2000)) { // One matmul, float16
112:         KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_1);
113:         using mt = mmType<half, CubeFormat::ND, CubeFormat::ND>;
114:         mt::MT mm;
115:         const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
```

**ffn/op_kernel/ffn.cpp:168** — `__NPU_ARCH__==3003`

```cpp
166:         op.Process();
167:     } else if (TILING_KEY_IS(6)) { // ANTI_QUANT
168: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
169:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
170:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
171: #else
172:         if constexpr (IsSameType<DTYPE_X, half>::value &&
173:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
```

**ffn/op_kernel/ffn.cpp:168** — `__NPU_ARCH__==3113`

```cpp
166:         op.Process();
167:     } else if (TILING_KEY_IS(6)) { // ANTI_QUANT
168: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
169:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
170:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
171: #else
172:         if constexpr (IsSameType<DTYPE_X, half>::value &&
173:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
```

**ffn/op_kernel/ffn.cpp:185** — `__NPU_ARCH__==3003`

```cpp
183:         }
184:     } else if (TILING_KEY_IS(12)) { // ANTI_QUANT_PERGROUP
185: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
186:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
187:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
188: #else
189:         if constexpr (IsSameType<DTYPE_X, half>::value &&
190:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
```

**ffn/op_kernel/ffn.cpp:185** — `__NPU_ARCH__==3113`

```cpp
183:         }
184:     } else if (TILING_KEY_IS(12)) { // ANTI_QUANT_PERGROUP
185: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
186:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
187:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
188: #else
189:         if constexpr (IsSameType<DTYPE_X, half>::value &&
190:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
```

**ffn/op_kernel/ffn.cpp:203** — `__NPU_ARCH__==3003`

```cpp
201:         }
202:     } else if (TILING_KEY_IS(7)) { // high precision
203: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
204:         using mt1 = mmWNDType<bfloat16_t, float, float>;
205:         mt1::MT mm1;
206:         using mt2 = mmWNDType<bfloat16_t, bfloat16_t, float>;
207:         mt2::MT mm2;
208:         REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
```

**ffn/op_kernel/ffn.cpp:203** — `__NPU_ARCH__==3113`

```cpp
201:         }
202:     } else if (TILING_KEY_IS(7)) { // high precision
203: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
204:         using mt1 = mmWNDType<bfloat16_t, float, float>;
205:         mt1::MT mm1;
206:         using mt2 = mmWNDType<bfloat16_t, bfloat16_t, float>;
207:         mt2::MT mm2;
208:         REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
```

**ffn/op_kernel/ffn.cpp:214** — `__NPU_ARCH__==3003`

```cpp
212: #endif
213:     } else if (TILING_KEY_IS(11)) { // QUANT with output_type is bf16
214: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
215:         mmQuantType<int8_t, int32_t, int32_t>::MT mm1;
216:         mmQuantType<int8_t, int32_t, int32_t>::MT mm2;
217:         REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
218:         FFNQuant<int8_t, decltype(mm1), decltype(mm2), int32_t, bfloat16_t, int32_t, float, bfloat16_t> op(mm1, mm2);
219:         op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
```

**ffn/op_kernel/ffn.cpp:214** — `__NPU_ARCH__==3113`

```cpp
212: #endif
213:     } else if (TILING_KEY_IS(11)) { // QUANT with output_type is bf16
214: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
215:         mmQuantType<int8_t, int32_t, int32_t>::MT mm1;
216:         mmQuantType<int8_t, int32_t, int32_t>::MT mm2;
217:         REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
218:         FFNQuant<int8_t, decltype(mm1), decltype(mm2), int32_t, bfloat16_t, int32_t, float, bfloat16_t> op(mm1, mm2);
219:         op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
```

**ffn/op_kernel/ffn.cpp:225** — `__NPU_ARCH__==3003`

```cpp
223:     } else if (TILING_KEY_IS(15)) { // ANTI_QUANT_MSD
224:         KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_1);
225: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
226:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
227:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
228: #else
229:         if constexpr ((IsSameType<DTYPE_X, half>::value) && (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
230: #endif
```

**ffn/op_kernel/ffn.cpp:225** — `__NPU_ARCH__==3113`

```cpp
223:     } else if (TILING_KEY_IS(15)) { // ANTI_QUANT_MSD
224:         KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_1);
225: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
226:         if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
227:                       (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
228: #else
229:         if constexpr ((IsSameType<DTYPE_X, half>::value) && (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
230: #endif
```

**ffn/op_kernel/ffn.cpp:250** — `__NPU_ARCH__==3003`

```cpp
248:         op.Process();
249:     } else if (TILING_KEY_IS(14)) { // QUANT SCALE PER-CHANNEL
250: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
251:         using c1TQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, int32_t, half>;
252:         using actTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, float, half>;
253:         using deqTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, bfloat16_t, uint64_t>;
254:         mmQuantType<int8_t, c1TQuant, int32_t>::MT mm1;
255:         mmQuantType<int8_t, c1TQuant, int32_t>::MT mm2;
```

**ffn/op_kernel/ffn.cpp:250** — `__NPU_ARCH__==3113`

```cpp
248:         op.Process();
249:     } else if (TILING_KEY_IS(14)) { // QUANT SCALE PER-CHANNEL
250: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
251:         using c1TQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, int32_t, half>;
252:         using actTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, float, half>;
253:         using deqTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, bfloat16_t, uint64_t>;
254:         mmQuantType<int8_t, c1TQuant, int32_t>::MT mm1;
255:         mmQuantType<int8_t, c1TQuant, int32_t>::MT mm2;
```

**ffn/op_kernel/ffn.cpp:17** — `__CCE_AICORE__==220`

```cpp
15: 
16: #include "kernel_operator.h"
17: #if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
18: #include "ffn_high_performence.h"
19: #include "ffn_quant.h"
20: #include "ffn_antiquant.h"
21: #include "ffn_antiquant_msd.h"
22: #include "ffn_glu.cpp"
```

**ffn/op_kernel/ffn.cpp:75** — `__CCE_AICORE__==220`

```cpp
73:     MMType<aType<T, cubeFormatA>, bType<T, CubeFormat::ND>, cType<cT, tPositionC>, biasType<biasT>, MM_CFG>;
74: 
75: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
76: template <typename T> struct GetC1Type {
77:     using Type = T;
78: };
79: 
80: #if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
```

**ffn/op_kernel/ffn.cpp:110** — `__CCE_AICORE__==220`

```cpp
108:     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
109: 
110: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
111:     if (TILING_KEY_IS(2000)) { // One matmul, float16
112:         KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_1);
113:         using mt = mmType<half, CubeFormat::ND, CubeFormat::ND>;
114:         mt::MT mm;
115:         const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
```

**ffn/op_kernel/ffn_nonquant_nz.h:324** — `__CCE_AICORE__==200`

```cpp
322:     }
323: 
324: #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
325:     GlobalTensor<uint64_t> global;
326:     global.SetGlobalBuffer((__gm__ uint64_t *)0);
327:     DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);
328: #endif
329: 
```

## 未解析项

以下引用未能自动解析，需手动检查：

- #include "fallback/fallback_comm.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_graph/fallback_ffn.cpp) — file not found
- #include "fallback/fallback.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_graph/fallback_ffn.cpp) — file not found
- #include "log/log.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_infershape.cpp) — file not found
- #include "op_host/tiling_templates_registry.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "err/ops_err.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "log/log.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "op_host/tiling_base.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "platform/platform_info.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "tiling/tiling_api.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.h) — file not found
- #include "kernel_tiling/kernel_tiling.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_kernel/ffn.h) — file not found
- #include "lib/matmul_intf.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_kernel/ffn.h) — file not found
- OP_TYPE_REGISTER(FFN) (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/op_api/ffn.cpp) — no operator directory found

## 搜索范围

- 算子目录文件数: 22
- 追踪到的外部文件数: 0

