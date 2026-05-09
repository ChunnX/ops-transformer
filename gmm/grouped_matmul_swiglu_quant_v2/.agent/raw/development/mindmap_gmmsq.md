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

## UT

### Tiling UT
- 场景识别验证
- 属性校验验证
- Shape校验验证

### Kernel计算UT
- AIV反量化流程验证
- AIC矩阵乘验证
- SwiGLU激活验证
- MX量化输出验证

## ST

### 功能性用例
- 基础功能验证
- 边界条件验证
- 异常输入验证

### 性能用例
- 吞吐量测试
- 延迟测试
- 资源利用率测试

### 网络用例

### 兼容性测试