# FAG Cube侧与Vector侧内存分配一览

---

## 1. CV共享Buffer（kernel_base.h `InitCVCommonBuffer()`，L293-313）

| Buffer | 类型 | 大小表达式 | 条件 | 用途 |
|--------|------|-----------|------|------|
| l1BufferManager | L1管理器 | L1_MAX_SIZE | Always | 管理整个L1空间 |
| dSL1Buf | L1 | CUBE_BASEM × CUBE_BASEN × sizeof(INPUT_TYPE) × 2 | DETER_OLD | ds L1 buffer（确定性双倍pingpong） |
| dSL1Buf | L1 | CUBE_BASEM × CUBE_BASEN × sizeof(INPUT_TYPE) | IS_FP8_INPUT && !DETER_OLD | ds L1 buffer（FP8） |
| dSTransL1Buf | L1 | CUBE_BASEM × CUBE_BASEN × sizeof(INPUT_TYPE) | IS_FP8_INPUT && !DETER_OLD | ds转置L1 buffer（FP8 quant专用） |
| dSL1Buf | L1 | CUBE_BASEM × CUBE_BASEN × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && !DETER_OLD | ds L1 buffer（通用） |
| pL1Buf | L1 | CUBE_BASEM × CUBE_BASEN × sizeof(OUTDTYPE) | Always | p L1 buffer（CV共享） |
| vL1Buf | L1 | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(OUTDTYPE) | IS_FP8_INPUT | V L1 buffer（FP8 quant路径） |
| mm1ResBuf[0] | UB(TBuf) | VECTOR_BASEM × VECTOR_BASEN × sizeof(CALC_TYPE) | Always | mm1结果 PingPong buffer 0 |
| mm1ResBuf[1] | UB(TBuf) | VECTOR_BASEM × VECTOR_BASEN × sizeof(CALC_TYPE) | Always | mm1结果 PingPong buffer 1 |
| mm2ResBuf[0] | UB(TBuf) | VECTOR_BASEM × VECTOR_BASEN × sizeof(CALC_TYPE) | Always | mm2结果 PingPong buffer 0 |
| mm2ResBuf[1] | UB(TBuf) | VECTOR_BASEM × VECTOR_BASEN × sizeof(CALC_TYPE) | Always | mm2结果 PingPong buffer 1 |

> 注：VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO = 128/2 = 64，VECTOR_BASEN = CUBE_BASEN = 128
> mm1ResBuf/mm2ResBuf大小 = 64×128×4 = 32768B = 32KB（FP32）/ 64×128×2 = 16384B = 16KB（FP16）

---

## 2. Cube侧 L1 Buffer（block_cube.h `InitCubeBuffer()`，L275-313）

### 2.1 L1 Buffer分配

| Buffer | 大小表达式 | 条件 | 用途 |
|--------|-----------|------|------|
| dYL1Buf | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(OUTDTYPE) | IS_FP8_INPUT && (IS_L1_REUSE || IS_L1_PRELOAD) | mm1左矩阵dy（FP8用OUTDTYPE） |
| qL1Buf | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | IS_FP8_INPUT && (IS_L1_REUSE || IS_L1_PRELOAD) | mm2左矩阵Q |
| kL1Buf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | IS_FP8_INPUT && (IS_L1_REUSE || IS_L1_PRELOAD) | mm2右矩阵K |
| commonL1Buf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(OUTDTYPE) | IS_FP8_INPUT && !IS_L1_REUSE && !IS_L1_PRELOAD | 共享L1 buffer（FP8非复用模式） |
| dYL1Buf | CUBE_BASEM × ROPE_D_128 × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD && IS_ROPE | mm1左矩阵dy（Rope固定D=128） |
| vL1Buf | CUBE_BASEN × ROPE_D_128 × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD && IS_ROPE | mm1右矩阵V（Rope固定D=128） |
| dYL1Buf | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD && !IS_ROPE | mm1左矩阵dy（通用） |
| vL1Buf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD && !IS_ROPE | mm1右矩阵V（通用） |
| qL1Buf | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD | mm2左矩阵Q |
| kL1Buf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | !IS_FP8_INPUT && IS_L1_REUSE/IS_L1_PRELOAD | mm2/mm3/mm4右矩阵K |
| fp32L1Buf1 | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | IS_FP32_D_EXCEED_256 | FP32 D>256 左矩阵buffer |
| fp32L1Buf2 | CUBE_BASEN × l1BaseD × sizeof(INPUT_TYPE) | IS_FP32_D_EXCEED_256 | FP32 D>256 右矩阵buffer（l1BaseD=512或256） |
| commonL1Buf | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) | 普通模式 | 共享L1 buffer（所有mm轮流使用） |

> 注：IS_L1_PRELOAD模式下额外分配vL1BufMutexId（Mutex ID），用于保护V buffer并发访问。

### 2.2 L1 Buffer典型大小计算（CUBE_BASEM=128, HEAD_DIM_ALIGN=128, INPUT_TYPE=fp16）

| Buffer | 条件 | 大小计算 | 实际大小 |
|--------|------|---------|---------|
| dYL1Buf | IS_L1_REUSE, fp16, D=128 | 128×128×2 | 32KB |
| qL1Buf | IS_L1_REUSE, fp16, D=128 | 128×128×2 | 32KB |
| kL1Buf | IS_L1_REUSE, fp16, D=128 | 128×128×2 | 32KB |
| vL1Buf | IS_L1_REUSE, fp16, D=128 | 128×128×2 | 32KB |
| dYL1Buf | IS_L1_REUSE, IS_ROPE, fp16 | 128×128×2 (ROPE_D_128=128) | 32KB |
| vL1Buf | IS_L1_REUSE, IS_ROPE, fp16 | 128×128×2 (ROPE_D_128=128) | 32KB |
| commonL1Buf | 普通模式, fp16, D=128 | 128×128×2 | 32KB |
| fp32L1Buf1 | IS_FP32_D_EXCEED_256, fp32, D=256 | 128×256×4 | 128KB |
| fp32L1Buf2 | IS_FP32_D_EXCEED_256, fp32 | 128×512×4 | 256KB |

### 2.3 L0A/L0B Buffer分配

| Buffer | 类型 | 大小表达式 | 用途 |
|--------|------|-----------|------|
| l0aBufferManager | L0A管理器 | L0_MAX_SIZE | 管理整个L0A空间 |
| l0bBufferManager | L0B管理器 | L0_MAX_SIZE | 管理整个L0B空间 |
| l0aBuf | L0A | L0_SINGLE_BUFFER_SIZE = 32KB | L0A双buffer（MatmulBase使用） |
| l0bBuf | L0B | L0_SINGLE_BUFFER_SIZE = 32KB | L0B双buffer（MatmulBase使用） |

### 2.4 L0C Buffer分配

| Buffer | 大小表达式 | 条件 | 用途 |
|--------|-----------|------|------|
| l0cBufferManager | L0C_MAX_SIZE = 256KB | Always | 管理整个L0C空间 |
| mm1Mm2Mm3L0CBuf | CUBE_BASEN>HEAD_DIM_ALIGN ? CUBE_BASEM×CUBE_BASEN×4 : CUBE_BASEM×HEAD_DIM_ALIGN×4 | IS_DKV_RESIDENT_L0C | mm1/mm2/mm3/dq共享L0C |
| dkL0CBuf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(float) | IS_DKV_RESIDENT_L0C | dk L0C常驻（跨s2o累加） |
| dvL0CBuf | CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(float) | IS_DKV_RESIDENT_L0C | dv L0C常驻（跨s2o累加） |
| mm1Mm2Mm3L0CSpecialBuf | CUBE_BASEM × Aligned192 × sizeof(float) = 128×192×4 | isDkvL0CResidentForD192Dv128 | D=192,Dv=128特殊mm1/mm2/mm3 L0C |
| dkL0CSpecialBuf | CUBE_BASEN × Aligned192 × sizeof(float) = 128×192×4 | isDkvL0CResidentForD192Dv128 | D=192特殊dk L0C |
| dvL0CSpecialBuf | CUBE_BASEN × Aligned128 × sizeof(float) = 128×128×4 | isDkvL0CResidentForD192Dv128 | Dv=128特殊dv L0C |
| commonl0CBuf | L0C_MAX_SIZE / 2 = 128KB | HEAD_DIM_ALIGN ≤ 256 | 共享L0C（双buffer交替） |
| commonl0CBuf | L0C_MAX_SIZE = 256KB | HEAD_DIM_ALIGN > 256 | 共享L0C（单buffer全量） |

> `isDkvL0CResidentForD192Dv128` = `(BN2GS1S2或BN2S2) && (HEAD_DIM_ALIGN==Aligned192 && dSizeV≤Aligned128) || IS_ROPE`

### 2.5 L0C典型大小（CUBE_BASEM=128, CUBE_BASEN=128, HEAD_DIM_ALIGN=128）

| 条件 | Buffer | 大小计算 | 实际大小 |
|------|--------|---------|---------|
| IS_DKV_RESIDENT_L0C, D=128 | mm1Mm2Mm3L0CBuf | min(128×128,128×128)×4 | 65,536B = 64KB |
| IS_DKV_RESIDENT_L0C, D=128 | dkL0CBuf | 128×128×4 | 65,536B = 64KB |
| IS_DKV_RESIDENT_L0C, D=128 | dvL0CBuf | 128×128×4 | 65,536B = 64KB |
| IS_DKV_RESIDENT_L0C总计 | 上述三项 | | 192KB ≤ 256KB ✓ |
| D192/Dv128 | mm1Mm2Mm3L0CSpecialBuf | 128×192×4 | 98,304B ≈ 96KB |
| D192/Dv128 | dkL0CSpecialBuf | 128×192×4 | 98,304B ≈ 96KB |
| D192/Dv128 | dvL0CSpecialBuf | 128×128×4 | 65,536B = 64KB |
| D192/Dv128总计 | 上述三项 | | 256KB ≤ 256KB ✓ |
| D=128, 普通模式 | commonl0CBuf | 256KB/2 | 128KB |
| D=256 | commonl0CBuf | 256KB | 256KB |

---

## 3. Vector侧 UB Buffer（block_vec.h `InitUbBuffer()`，L260-309）

### 3.1 UB Buffer分配

| Buffer | Queue类型 | 大小表达式 | 条件 | 用途 |
|--------|----------|-----------|------|------|
| attenMaskOrYInQue | VECIN | VECTOR_BASEM × VECTOR_BASEN × sizeof(CALC_TYPE) | Always | y / attenMask输入（VF1/VF2/VF3共用） |
| pseOrDyInQue | VECIN | VECTOR_BASEM × VECTOR_BASEN × sizeof(OUTDTYPE) | Always | dy / pse输入（VF1/VF2共用） |
| softmaxGradResBuf | TBuf | VECTOR_BASEM × sizeof(CALC_TYPE) | Always | softmaxGrad结果（VF1输出→VF3输入） |
| maxSumQue[0] | VECIN | VECTOR_BASEM × MAX_SUM_REDUCE_AXIS_SIZE × 2 | Always | softmax max/sum双buffer 0（VF2输入） |
| maxSumQue[1] | VECIN | VECTOR_BASEM × MAX_SUM_REDUCE_AXIS_SIZE × 2 | Always | softmax max/sum双buffer 1（VF2输入） |
| dropMaskBuf | TBuf | VECTOR_BASEM × VECTOR_BASEN × sizeof(uint8_t) / BIT_MASK_NUM | IS_DROP | dropout mask（VF3输入） |
| dropmaskIndexVecBuf | TBuf | VECTOR_BASEM × VECTOR_BASEN / (2×BIT_MASK_NUM) × sizeof(int32_t) | IS_DROP | dropout mask索引向量 |
| deterInOutQue | QueBind | DETER_DQ_UB_SIZE | IS_DETER_OLD | 确定性dq中间数据 |
| deterOffsetBuf | TBuf | DETER_OFFSET_UB_SIZE = 3KB | IS_DETER_OLD | 确定性offset数据 |
| vselrIndexesBuf | TBuf | VECTOR_BASEN = 128B | IS_FP8_INPUT | FP8 SELR索引表 |
| dsAmaxOutQue | VECOUT | VREG_SIZE / 2 | IS_FP8_INPUT | ds绝对值amax输出 |
| dsinkResOutQue[0] | VECOUT | BIT32_ALIGN_NUM × sizeof(float) = 32B | sinkOptional(运行时) | dsink结果buffer 0 |
| dsinkResOutQue[1] | VECOUT | BIT32_ALIGN_NUM × sizeof(float) = 32B | sinkOptional(运行时) | dsink结果buffer 1 |
| dSOutQue | VECOUT | (VECTOR_BASEM+1) × VECTOR_BASEN × sizeof(OUTDTYPE) | !IS_FP32_INPUT | ds输出队列（VF3→CopyUB2L1） |
| pOutQue | VECOUT | (VECTOR_BASEM+1) × VECTOR_BASEN × sizeof(OUTDTYPE) | !IS_FP32_INPUT | p输出队列（VF4→CopyUB2L1） |
| dSOutQue | VECOUT | VECTOR_BASEM × VECTOR_BASEN × sizeof(INPUT_TYPE) + VECTOR_BASEN × sizeof(INPUT_TYPE) | IS_FP32_INPUT | FP32 ds输出队列（+1行尾块对齐） |
| pOutQue | — | (别名=dSOutQue) | IS_FP32_INPUT | FP32 p输出=ds输出（共用buffer） |

### 3.2 DETER_DQ_UB_SIZE条件取值

| 条件 | DETER_DQ_UB_SIZE值 | 说明 |
|------|-------------------|------|
| IS_FP32_INPUT && HEAD_DIM_ALIGN > 256 | 96KB (DETER_DQ_UB_SIZE_FP32_D768) | FP32大D场景 |
| IS_FP32_INPUT && HEAD_DIM_ALIGN ≤ 256 | 16KB (DETER_DQ_UB_SIZE_FP32_D256) | FP32小D场景 |
| !IS_FP32_INPUT (fp16/bf16) | 12KB (DETER_DQ_UB_SIZE_FP16) | FP16/BF16场景 |

### 3.3 UB Buffer典型大小计算（VECTOR_BASEM=64, VECTOR_BASEN=128, fp16场景）

| Buffer | 大小计算 | 实际大小 | 用途简述 |
|--------|---------|---------|---------|
| attenMaskOrYInQue | 64×128×4=32768 | 32KB | y/mask输入 |
| pseOrDyInQue | 64×128×2=16384 | 16KB | dy/pse输入 |
| softmaxGradResBuf | 64×4=256 | 256B | softmaxGrad结果 |
| maxSumQue[0] | 64×MAX_SUM_REDUCE_AXIS_SIZE×2 | ~数KB | max/sum双buf |
| maxSumQue[1] | 同上 | ~数KB | max/sum双buf |
| dropMaskBuf | 64×128/8=1024 | 1KB | dropout mask |
| dropmaskIndexVecBuf | 64×128/(2×8)×4=2048 | 2KB | dropout索引 |
| deterInOutQue | 12KB | 12KB | 确定性dq |
| deterOffsetBuf | 3KB | 3KB | 确定性offset |
| dSOutQue | (64+1)×128×2=16640 | ~16.25KB | ds输出 |
| pOutQue | (64+1)×128×2=16640 | ~16.25KB | p输出 |
| mm1ResBuf[0/1] | 64×128×4=32768 | 32KB×2 | mm1 pingpong |
| mm2ResBuf[0/1] | 64×128×4=32768 | 32KB×2 | mm2 pingpong |

### 3.4 FP32场景UB大小差异

| Buffer | fp16大小 | fp32大小 | 说明 |
|--------|---------|---------|------|
| attenMaskOrYInQue | 32KB | 32KB | CALC_TYPE始终float |
| pseOrDyInQue | 16KB | 32KB | OUTDTYPE fp16→fp32 |
| dSOutQue | 16.25KB | 64×128×4+128×4=33024 | FP32 dSOutQue多1行对齐 |
| pOutQue | 16.25KB | =dSOutQue | FP32共用 |
| mm1ResBuf | 32KB×2 | 32KB×2 | CALC_TYPE始终float |

---

## 4. TSCM Buffer（entry_regbase.h/fag_flag_data.h，OLD DETER路径）

### 4.1 入口TSCM分配（entry_regbase.h宏内）

| Buffer | 大小表达式 | 条件 | 用途 |
|--------|-----------|------|------|
| dsScm | s1Template × s2Template × sizeof(INPUT_TYPE) × 2 | IS_FP8 || IS_DETER_OLD | ds pingpong TSCM（确定性双倍） |
| dsScm | s1Template × s2Template × sizeof(INPUT_TYPE) | !IS_FP8 && !IS_DETER_OLD | ds TSCM（单buffer） |
| pScm | s1Template × s2Template × sizeof(INPUT_TYPE) | Always | p TSCM buffer |

### 4.2 InitTSCMBuffer详细分配（fag_flag_data.h `InitTSCMBuffer()`，IS_TSCM_REUSE路径）

| Buffer | 大小表达式 | 用途 |
|--------|-----------|------|
| tscmArray[0] | s1Template × dTemplate × sizeof(T) | dx ping |
| tscmArray[1] | s2Template × dTemplate × sizeof(T) | v ping |
| tscmArray[2] | s1Template × dTemplate × sizeof(T) | dx pong |
| tscmArray[3] | s1Template × dTemplate × sizeof(T) | q ping |
| tscmArray[4] | s2Template × dTemplate × sizeof(T) | k ping |
| tscmArray[5] | s1Template × dTemplate × sizeof(T) | q pong |
| tscmArray[6] | s2Template × dTemplate × sizeof(T) | k pong |
| tscmArray[7] (IS_DETER_OLD+IS_ROPE) | s1Template × (dTemplate/3×2) × sizeof(T) | dx next（Rope D=2/3） |
| tscmArray[7] (其他preload) | s1Template × dTemplate × sizeof(T) | dx next（preload） |
| tscmArray[8] (preload) | s1Template × dTemplate × sizeof(T) | q next（preload） |

### 4.3 InitL0CBuffer详细分配（fag_flag_data.h `InitL0CBuffer()`，IS_L0C_REUSE路径）

| Buffer | 大小表达式 | 用途 |
|--------|-----------|------|
| l0cArray[0..N-3] | GET_MM1_MM2_MM3_MAX_L0C_SIZE(s1,s2,d) | mm1/mm2/mm3/dq共享L0C |
| l0cArray[N-2] | s2Template × dTemplate × sizeof(float) | dk L0C buffer |
| l0cArray[N-1] | s2Template × dTemplate × sizeof(float) | dv L0C buffer |

---

## 5. Quant场景专用Buffer（block_cube_quant.h / kernel_quant.h）

### 5.1 Quant Cube L1 Buffer（block_cube_quant.h `InitCubeBuffer()`，L138-168）

| Buffer | 大小表达式 | 用途 |
|--------|-----------|------|
| commonL1Buf[0..3] | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) × 4组 | 共享L1（4组pingpong） |
| vL1Buf[0..3] | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) × 4组 | V L1 buffer（4组复用） |
| kL1Buf[0..3] | CUBE_BASEM × HEAD_DIM_ALIGN × sizeof(INPUT_TYPE) × 4组 | K L1 buffer（4组复用） |

### 5.2 Quant Cube L0 Buffer

| Buffer | 大小 | 用途 |
|--------|------|------|
| l0aBuf[0], l0aBuf[1] | L0_SINGLE_BUFFER_SIZE = 32KB | L0A双buffer |
| l0bBuf[0], l0bBuf[1] | L0_SINGLE_BUFFER_SIZE = 32KB | L0B双buffer |
| mm1L0CBuf | L0C_SINGLE_BUFFER_SIZE = 64KB | mm1专用L0C |
| mm2L0CBuf | L0C_SINGLE_BUFFER_SIZE = 64KB | mm2专用L0C |
| dqkvL0CBuf[0] | L0C_SINGLE_BUFFER_SIZE = 64KB | dqkv L0C ping |
| dqkvL0CBuf[1] | L0C_SINGLE_BUFFER_SIZE = 64KB | dqkv L0C pong |

### 5.3 Quant Kernel CV Buffer（kernel_quant.h `InitCVCommonBuffer()`，L227-254）

| Buffer | 大小表达式 | 用途 |
|--------|-----------|------|
| pL1Buf | 128×128×sizeof(INPUT_TYPE) = 32KB(fp16) | p L1 buffer |
| dsL1Buf[4×4] | 128×128×sizeof(INPUT_TYPE) × 16组 | ds L1 16组buffer（quant多loop复用） |
| vecQue | 224KB（总UB空间） | quant共享UB空间 |
| spTensors (sub-alloc) | 64×128×sizeof(CALC_TYPE) × 2 = 32KB×2 | sp/p tensor |
| dpdsTensors (sub-alloc) | 64×128×sizeof(CALC_TYPE) × 2 = 32KB×2 | dp/ds tensor |
| dkvTensors (sub-alloc) | 64×128×sizeof(CALC_TYPE) × 2 = 32KB×2 | dk/dv结果tensor |
| commonTensors (sub-alloc) | vecQue剩余空间 | max/sum/d/perm公共tensor |

---

## 6. 总览：各模式下的L1/L0C/UB空间需求

### 6.1 fp16 + D=128 + BN2GS1S2 + 非确定性（典型配置）

**L1空间需求（IS_L1_REUSE模式）：**
| Buffer | 大小 |
|--------|------|
| dSL1Buf | 128×128×2 = 32KB |
| pL1Buf | 128×128×2 = 32KB |
| dYL1Buf | 128×128×2 = 32KB |
| vL1Buf | 128×128×2 = 32KB |
| qL1Buf | 128×128×2 = 32KB |
| kL1Buf | 128×128×2 = 32KB |
| **L1总计** | **192KB** |

**L1空间需求（普通模式）：**
| Buffer | 大小 |
|--------|------|
| dSL1Buf | 32KB |
| pL1Buf | 32KB |
| commonL1Buf | 32KB |
| **L1总计** | **96KB** |

**L0C空间需求（IS_DKV_RESIDENT_L0C模式，D=128）：**
| Buffer | 大小 |
|--------|------|
| mm1Mm2Mm3L0CBuf | 64KB |
| dkL0CBuf | 64KB |
| dvL0CBuf | 64KB |
| **L0C总计** | **192KB ≤ 256KB ✓** |

**L0C空间需求（普通模式，D=128）：**
| Buffer | 大小 |
|--------|------|
| commonl0CBuf | 128KB |
| **L0C总计** | **128KB** |

**UB空间需求：**
| Buffer | 大小 |
|--------|------|
| attenMaskOrYInQue | 32KB |
| pseOrDyInQue | 16KB |
| softmaxGradResBuf | 256B |
| maxSumQue×2 | ~数KB |
| dSOutQue | 16.25KB |
| pOutQue | 16.25KB |
| mm1ResBuf×2 | 64KB |
| mm2ResBuf×2 | 64KB |
| dropMaskBuf | 1KB |
| dropmaskIndexVecBuf | 2KB |
| **UB总计** | **~193KB** |

### 6.2 fp32 + D=256 + BN2GS1S2（大D场景）

**L1空间需求：**
| Buffer | 大小 |
|--------|------|
| dSL1Buf | 128×128×4 = 64KB |
| pL1Buf | 128×128×4 = 64KB |
| fp32L1Buf1 | 128×256×4 = 128KB |
| fp32L1Buf2 | 128×512×4 = 256KB |
| **L1总计** | **512KB**（可能超限，需l1BaseD切分） |

**L0C空间需求：**
| Buffer | 大小 |
|--------|------|
| commonl0CBuf | 256KB（全量） |
| **L0C总计** | **256KB** |

**UB空间需求：**
| Buffer | 大小 |
|--------|------|
| attenMaskOrYInQue | 32KB |
| pseOrDyInQue | 32KB（fp32 OUTDTYPE） |
| dSOutQue+pOutQue共用 | 64×128×4+128×4 ≈ 33KB |
| mm1ResBuf×2 | 64KB |
| mm2ResBuf×2 | 64KB |
| **UB总计** | **~225KB** |

---

## 7. 条件编译标志对Buffer分配的影响汇总

| 标志 | 影响范围 | 效果 |
|------|---------|------|
| IS_L1_REUSE | Cube L1 | 4个独立L1 buffer（dy/V/Q/K各一个）而非共享commonL1Buf |
| IS_L1_PRELOAD | Cube L1 | 同IS_L1_REUSE + vL1BufMutexId + preload三buffer策略 |
| IS_FP8_INPUT | Cube L1, Vec UB, CV L1 | dy用OUTDTYPE; 新增dSTransL1Buf/vL1Buf(CV); 新增vselrIndexesBuf/dsAmaxOutQue(UB) |
| IS_FP32_INPUT | Vec UB | dSOutQue/pOutQue共用; 大小sizeof(INPUT_TYPE); +1行尾块对齐 |
| IS_FP32_D_EXCEED_256 | Cube L1 | fp32L1Buf1/fp32L1Buf2替代commonL1Buf |
| IS_DKV_RESIDENT_L0C | Cube L0C | mm1Mm2Mm3L0CBuf+dkL0CBuf+dvL0CBuf替代commonl0CBuf |
| isDkvL0CResidentForD192Dv128 | Cube L0C | mm1Mm2Mm3L0CSpecialBuf+dkL0CSpecialBuf+dvL0CSpecialBuf（D192/Dv128特殊） |
| IS_ROPE | Cube L1 | dYL1Buf/vL1Buf使用ROPE_D_128而非HEAD_DIM_ALIGN |
| IS_DETER_OLD | CV L1, Vec UB | dSL1Buf双倍(×2); 新增deterInOutQue/deterOffsetBuf(UB) |
| IS_DROP | Vec UB | 新增dropMaskBuf/dropmaskIndexVecBuf |
| sinkOptional | Vec UB | 新增dsinkResOutQue×2（运行时判断） |
| HEAD_DIM_ALIGN ≤ 256 | Cube L0C | commonl0CBuf = L0C_MAX_SIZE/2 (128KB双buffer) |
| HEAD_DIM_ALIGN > 256 | Cube L0C | commonl0CBuf = L0C_MAX_SIZE (256KB单buffer) |
| Quant(fp8/hifp8) | Cube L1/L0C, CV, UB | 4组L1 pingpong; 4组L0C独立; 16组dsL1Buf; 224KB vecQue sub-alloc |