# GMMSQ MxA8W4 Kernel

## 输入

### x (激活矩阵)
- dtype: float8_e4m3fn
- format: ND
- shape: (m, k)
- 特殊约束: transpose_x == false

### weight (权重矩阵)
- dtype: float4_e2m1
- format: FRACTAL_NZ_C0_32(或FRACTAL_NZ)
- shape: (e, ceil(k,32), ceil(n,16), 16, 32)
- 特殊约束: transpose_w == true

### weightScale (MX weight反量化缩放因子)
- dtype: float8_e8m0
- format: ND
- shape: (e, n, ceil(k/64), 2)

### weightAssistMatrix
- null

### bias
- null

### xScale (MX x反量化缩放因子)
- dtype: float8_e8m0
- format: ND
- shape: (m, ceil(k/64), 2)

### smoothScale
- null

### groupList (分组列表)
- dtype: int64
- format: ND
- shape: (e,)

### dequantMode
- dtype: int64
- 取值: {2}

### dequantDtype
- dtype: int64
- 取值: {0}
- 说明: 中间计算数据类型，固定为0（DT_FLOAT）

### quantMode
- dtype: int64
- 取值: {2}
- 说明: 量化计算类型，固定为2（MX量化）

### groupListType
- dtype: int64
- 取值: {0, 1}
- 说明: 0=cumsum模式, 1=count模式

### tuningConfig
- dtype: int64数组
- 说明: 调优参数，可选，MxA8W4场景传入空指针

## 输出参数

### output (量化后的输出矩阵)
- dtype: float8_e4m3fn
- format: ND
- shape: (m, n/2)

### outputScale (输出量化缩放因子)
- dtype: float8_e8m0
- format: ND
- shape: (m, ceil((n/2)/64), 2)

## UT (共13个)

### Controller层 (4个)
- **UT-C01 Init()**: 验证GM地址初始化正确，`sharedInputWeight=0`（GMMSQ无sharedInput）
- **UT-C02 UpdateGmAddr()**: 验证输出N/2偏移计算正确，`yGm_ += M*(N/2)`；无logits/rowIndex地址更新
- **UT-C03 SplitNByMultiCore()**: 验证nL1Size=128（GMMFR=256→GMMSQ=128），适配SwiGLU双分支
- **UT-C04 Process()**: 验证去除`InitAtomicGm`调用，无AtomicAdd Scatter操作

### BasicBlock层 (2个)
- **UT-BB01 ComputeBasicBlock()**: 验证AIC/AIV路径正确分发，nL1Size=128
- **UT-BB02 End()**: 验证后处理调用SwiGLU+Quant，而非GMMFR的MulLogits+Routing

### CubeCompute层 (2个)
- **UT-CUBE01 LaunchMatmul()**: 验证MMA计算与GMMFR一致，L0C误差<0.01%
- **UT-CUBE02 GetTensorC()**: 验证Fixpipe输出到`yF32Buffer_`（复用weightHighBit空间）

### VecCompute层 (4个)
- **UT-VEC01 AntiQuantProcessNzMxA8W4()**: 验证4bit反量化参数正确（loopKNum=4），与GMMFR一致
- **UT-VEC02 SwiGLU()**: 验证`Swish(Y[:,0:128]) * Gate(Y[:,128:256])`计算正确，输出N/2
- **UT-VEC03 ComputeMaxExp()+ComputeScale()**: 验证MX量化shared_exp计算，每32元素一组
- **UT-VEC04 UB Buffer复用**: 验证`yF32Buffer_`复用`weightHighBit_`，`yFp8Buffer_`复用`weightLowBit_`，总UB≤248KB

### Sync机制 (1个)
- **UT-SYNC01 CrossCore同步序列**: 验证AIC↔AIV同步信号和时序

## ST (共~70条)

### 白盒/小组合 (40条)
- **dtype×format全组合**: float8_e4m3fn × float4_e2m1 × float8_e8m0
- **维度覆盖**: M∈[32,16384], K∈[64,32768], N∈[128,65536], E∈[1,256]
- **Kernel内部路径**: AIV反量化、AIC矩阵乘(MX Scale融合)、AIV SwiGLU、AIV MX量化
- **同步机制**: AIC→AIV(SYNC_AIC_AIV_FLAG)、AIV→AIC(SYNC_AIV_AIC_FLAG)、fixpipe同步

### 泛化/大空间 (20条)
- **不同规模组合**: 小规模(32,128,128,1)、中规模(1024,2048,4096,16)、大规模(8192,16384,32768,128)、超大规模(16384,32768,65536,256)
- **groupListType两种模式**: cumsum模式(0)、count模式(1)
- **边界值测试**: K=32/32768, N=128/65536, E=1/1024

### 异常/负向 (~12条)
- **invalid_k_alignment**: K%32!=0 → tiling报错GRAPH_FAILED
- **invalid_n_alignment**: N%128!=0 → tiling报错GRAPH_FAILED
- **invalid_k_minimum**: K<32 → tiling报错GRAPH_FAILED
- **invalid_n_minimum**: N<128 → tiling报错GRAPH_FAILED
- **invalid_transpose_x**: transpose_x==true → 属性校验失败
- **invalid_transpose_w**: transpose_w==false → 属性校验失败
- **invalid_weight_format**: weight.format!=FRACTAL_NZ_C0_32 → 场景识别失败
- **invalid_expert_num**: E>1024 → tiling报错GRAPH_FAILED
- **invalid_scale_shape**: weightScale.shape不匹配 → shape校验失败
- **invalid_xscale_shape**: xScale.shape不匹配 → shape校验失败
- **invalid_groupList_sum**: cumsum模式下sum(groupList)!=M → 运行时错误
- **invalid_dtype_combination**: x.dtype!=float8_e4m3fn → 场景识别失败

### 性能/兼容/通路
- **基线性能**: 记录不同shape执行时间，建立性能基线
- **aclnn通路**: 调用aclnnGroupedMatmulSwigluQuantWeightNzV2接口
- **PTA通路**: 通过PTA框架调用