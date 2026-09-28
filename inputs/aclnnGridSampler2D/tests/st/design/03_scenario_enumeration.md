# aclnnGridSampler2D 场景枚举

## 约束关系清单

| ID | 约束描述 | 涉及参数 | 关键词 | 需要枚举 | @solves target |
|----|---------|---------|--------|---------|---------------|
| C1 | grid 数据类型与 input 一致（R1） | input.dtype, grid.dtype | 一致 | 否 | grid.dtype |
| C2 | out 数据类型与 input 一致（R2） | input.dtype, out.dtype | 一致 | 否 | out.dtype |
| C3 | grid.shape=(N,H_out,W_out,2) 且 N 与 input 一致、末维=2；out.shape=(N,C,H_out,W_out)（R3/R4） | input.shape, grid.shape, out.shape | 满足公式 | **是** | input.shape, grid.shape, out.shape |
| C4 | interpolationMode ∈ {0,1,2}；paddingMode ∈ {0,1,2}（R5） | interpolationMode.value, paddingMode.value | 枚举值域 | 否 | factor-domain（YAML .value 枚举） |
| C5 | bicubic(2) 时 dtype 须为 FLOAT32/FLOAT16（R6） | interpolationMode.value → input.dtype | 仅...时支持（条件化） | **是** | input.dtype |
| C6 | input 的 H_in×W_in < INT32 最大值（R7） | input.shape | 每个元素值域 | 否 | solve_input_shape 内 assert |
| C7 | input 最后两维 ≥1、N/C 可为 0、grid/out 各维 ≥0（R8 空tensor边界） | input.shape, grid.shape, out.shape | 边界 | 是（ET-S1~S4 增强场景） | ET 场景（03 手动定义） |
| C8 | grid 归一化坐标 [-1,1]，越界按 paddingMode 处理（R9） | grid.value_range, paddingMode.value | 功能语义值域 | 否（值域分段在 YAML 定义，L1 数据生成按段采样覆盖场景A/B；grid.value_range 为引擎内置推导，无 @solves tag 点） | grid.value_range（YAML 分段） |

## 子场景枚举

### C3: shape 公式（grid.shape=(N,H_out,W_out,2)、out.shape=(N,C,H_out,W_out)）

| 子场景 ID | 描述 | @solves 应生成的值示例 |
|----------|------|---------------------|
| C3-S1 | 常规场景：全维度 ≥1 | input.shape=[2,3,8,8], grid.shape=[2,4,4,2], out.shape=[2,3,4,4] |
| C3-S2 | batch 空（N=0）：input/grid/out 的 N 轴均为 0 | input.shape=[0,3,8,8], grid.shape=[0,4,4,2], out.shape=[0,3,4,4] |
| C3-S3 | 通道空（C=0）：input/out 的 C 轴为 0 | input.shape=[2,0,8,8], grid.shape=[2,4,4,2], out.shape=[2,0,4,4] |
| C3-S4 | 输出空（H_out=0 或 W_out=0）：grid/out 空输出 | grid.shape=[2,0,4,2], out.shape=[2,3,0,4] |

### C5: bicubic 条件化 dtype（interpolationMode=2 时仅 FLOAT32/FLOAT16）

| 子场景 ID | 描述 | @solves 应生成的值示例 |
|----------|------|---------------------|
| C5-S1 | 非 bicubic（mode ∈ {0,1}）：dtype 全域 {float16, float32, float64, bfloat16} | mode=0, input.dtype=bfloat16 |
| C5-S2 | bicubic（mode=2）：dtype 收窄为 {float32, float16} | mode=2, input.dtype=float16 |

## 空tensor维度场景

| 子场景 ID | locked因子 | 预期输出 | 测试重点 |
|----------|-----------|---------|---------|
|ET-S1|_n.value=0|out 的 N 轴为 0，input/grid/out 全空|batch 维空 tensor，空到空|
|ET-S2|_c.value=0|out 的 C 轴为 0，input 与 out 通道维空|通道维空 tensor|
|ET-S3|_out_h.value=0|out 的输出高为 0，grid 输出高同步为空|输出高为空（空输出）|
|ET-S4|_out_w.value=0|out 的输出宽为 0，grid 输出宽同步为空|输出宽为空（空输出）|

> 一致性说明：ET-S1~S4 均满足 R8（input 最后两维 ≥1 保持、N/C/输出维可为 0），代入 validate_constraints 返回空列表，为合法空 tensor 场景。input 的 H_in/W_in 不允许为 0（R8），故无对应 ET 场景。

## 边界值场景

| 子场景 ID | locked因子 | 约束来源 | 测试重点 |
|----------|-----------|---------|---------|
|BD-S1|_h_in.value=1,_w_in.value=1|01 R8 最后两维不能为 0（下边界 1）|输入空间维下边界|
|BD-S2|_h_in.value=46340,_w_in.value=46340,_n.value=1,_c.value=1|01 R7 H×W 上界（46340×46340 逼近 INT32 最大值）|H×W 上边界|
|BD-S3|_out_h.value=1,_out_w.value=1|01 R8 输出维度下边界（取最小非空值 1）|最小输出尺寸|
|BD-S4|grid.value_range=[-1.0,-1.0]|01 R9 归一化坐标下边界（取 -1）|全部坐标取下边界|
|BD-S5|grid.value_range=[1.0,1.0]|01 R9 归一化坐标上边界（取 +1）|全部坐标取上边界|

> 上边界一致性：BD-S2 锁定 H_in 与 W_in 均为 46340 时同步锁定 _n 与 _c 为 1（H×W 为 2147395600，小于 INT32 最大值 2147483647，tensor 总乘积不超过 MAX_SHAPE_PRODUCT），100% 求解成功。

## 异常场景

| 子场景 ID | locked因子 | 预期错误码 | 测试重点 |
|----------|-----------|-----------|---------|
|EX-S1|interpolationMode.value=3|ACL_ERROR_INVALID_PARAM|插值模式枚举外（aclnn 错误码 161002）|
|EX-S2|paddingMode.value=3|ACL_ERROR_INVALID_PARAM|填充模式枚举外（161002）|
|EX-S3|interpolationMode.value=2,input.dtype=float64|ACL_ERROR_INVALID_PARAM|bicubic 不支持 DOUBLE（R6 违反）|
|EX-S4|input.dtype=float32,grid.dtype=float16|ACL_ERROR_INVALID_PARAM|grid 与 input 数据类型不一致（R1 违反）|

> 维度类异常（grid.dimensions=3 下溢 / =5 上溢、不支持的 dtype/format 等）由 L2 自动生成（域外异常 + 维度边界异常），不在此重复定义。

## 场景覆盖验证

（步骤 4 阶段A/阶段B 执行后回填）

| 子场景 ID | 是否出现 | 出现次数 | 备注 |
|----------|---------|---------|------|
| C3-S1 | 是 | L0/L1 多条 | solve_out_shape tag 标记（常规全维 ≥1 场景，随机生成即覆盖） |
| C3-S2 | 是 | L0/L1 多条 | tag 标记；ET-S1（_n=0）目标用例同步覆盖 |
| C3-S3 | 是 | L0/L1 多条 | tag 标记；ET-S2（_c=0）目标用例同步覆盖 |
| C3-S4 | 是 | L0/L1 多条 | tag 标记；ET-S3/S4（输出空）目标用例同步覆盖 |
| C5-S1 | 是 | L0/L1 多条 | solve_input_dtype tag 标记（mode ∈ {0,1} 全 dtype 域） |
| C5-S2 | 是 | L0/L1 多条 | tag 标记（mode=2 时 dtype 收窄 float32/float16） |
| ET-S1~S4 | 是 | 各≥1 | L0/L1 [SCENARIO-COVERAGE] 报告确认已覆盖 |
| BD-S1~S5 | 是 | 各≥1 | L0/L1 [SCENARIO-COVERAGE] 报告确认已覆盖（BD-S2 为 input.shape=[1,1,46340,46340] 上边界用例） |
| EX-S1~S4 | 是 | 各≥1 | L2 sidecar 校验确认（EX-S3 含 3 条同类） |

> 注：L0 生成日志中的"[WARN] L0 未覆盖场景: C3-S*/C5-S*"为校验器仅支持 ET/BD 锁定值匹配产生的结构性噪声——C 子场景通过 @solves 内 tag() 标记验证（每条用例均携带 shape_scenario/interp_mode_scenario 标签），实际已全覆盖。
