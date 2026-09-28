# aclnnGridSampler3D 场景枚举

## 约束关系清单

| ID | 约束描述 | 涉及参数 | 关键词 | 需要枚举 | @solves target |
|----|---------|---------|--------|---------|---------------|
| C1 | grid 数据类型与 input 一致（R1） | input.dtype, grid.dtype | 一致 | 否 | grid.dtype |
| C2 | out 数据类型与 input 一致（R2） | input.dtype, out.dtype | 一致 | 否 | out.dtype |
| C3 | out 数据格式与 input 一致（R3） | input.format, out.format | 一致 | 否 | out.format |
| C4 | grid.shape=(N,D_out,H_out,W_out,3) 且 N 与 input 一致、末维=3；out.shape=(N,C,D_out,H_out,W_out)（R4/R5） | input.shape, grid.shape, out.shape | 满足公式 | **是** | input.shape, grid.shape, out.shape |
| C5 | interpolationMode ∈ {0,1}；paddingMode ∈ {0,1,2}（R6） | interpolationMode.value, paddingMode.value | 枚举值域 | 否 | factor-domain（YAML .value 枚举） |
| C6 | input 的 D_in×H_in×W_in < INT32 最大值（R7） | input.shape | 每个元素值域 | 否 | solve_input_shape 内 assert |
| C7 | input 空间三维 ≥1、N/C 可为 0、grid/out 各维 ≥0（R8 空tensor边界） | input.shape, grid.shape, out.shape | 边界 | 是（ET-S1~S5 增强场景） | ET 场景（03 手动定义） |
| C8 | NDHWC 条件格式：f32 + mode=0 + C%4==0（R9） | input.format → input.dtype, interpolationMode.value, input.shape | 仅...时支持（条件化） | **是** | input.dtype, interpolationMode.value, input.shape |
| C9 | grid 归一化坐标 [-1,1]，越界按 paddingMode 处理（R10） | grid.value_range, paddingMode.value | 功能语义值域 | 否（值域分段在 YAML 定义，L1 数据生成按段采样覆盖场景A/B；grid.value_range 为引擎内置推导，无 @solves tag 点） | grid.value_range（YAML 分段） |

## 子场景枚举

### C4: shape 公式（grid.shape=(N,D_out,H_out,W_out,3)、out.shape=(N,C,D_out,H_out,W_out)）

| 子场景 ID | 描述 | @solves 应生成的值示例 |
|----------|------|---------------------|
| C4-S1 | 常规场景：全维度 ≥1 | input.shape=[2,3,4,8,8], grid.shape=[2,2,4,4,3], out.shape=[2,3,2,4,4] |
| C4-S2 | batch 空（N=0）：input/grid/out 的 N 轴均为 0 | input.shape=[0,3,4,8,8], grid.shape=[0,2,4,4,3], out.shape=[0,3,2,4,4] |
| C4-S3 | 通道空（C=0）：input/out 的 C 轴为 0 | input.shape=[2,0,4,8,8], grid.shape=[2,2,4,4,3], out.shape=[2,0,2,4,4] |
| C4-S4 | 输出空（D_out/H_out/W_out=0）：grid/out 空输出 | grid.shape=[2,0,4,4,3], out.shape=[2,3,0,4,4] |

### C8: NDHWC 条件化格式（f32 + mode=0 + C%4==0）

| 子场景 ID | 描述 | @solves 应生成的值示例 |
|----------|------|---------------------|
| C8-S1 | NCDHW/ND 通用格式：dtype 全域、mode ∈ {0,1}、C 任意 | input.format=NCDHW, input.dtype=bfloat16, mode=1, C=5 |
| C8-S2 | NDHWC 条件格式：dtype=float32、mode=0、C 向上对齐 4 的倍数 | input.format=NDHWC, input.dtype=float32, mode=0, C=4（_c=1~4 对齐到 4） |

## 空tensor维度场景

| 子场景 ID | locked因子 | 预期输出 | 测试重点 |
|----------|-----------|---------|---------|
|ET-S1|_n.value=0|out 的 N 轴为 0，input/grid/out 全空|batch 维空 tensor，空到空|
|ET-S2|_c.value=0|out 的 C 轴为 0，input 与 out 通道维空|通道维空 tensor|
|ET-S3|_out_d.value=0|out 的输出深度为 0，grid 输出深度同步为空|输出深度为空（空输出）|
|ET-S4|_out_h.value=0|out 的输出高为 0，grid 输出高同步为空|输出高为空（空输出）|
|ET-S5|_out_w.value=0|out 的输出宽为 0，grid 输出宽同步为空|输出宽为空（空输出）|

> 一致性说明：ET-S1~S5 均满足 R8（input 空间三维 ≥1 保持、N/C/输出维可为 0），代入 validate_constraints 返回空列表，为合法空 tensor 场景。input 的 D_in/H_in/W_in 按 R8 推断约束不允许为 0，故无对应 ET 场景（待确认项见 01）。

## 边界值场景

| 子场景 ID | locked因子 | 约束来源 | 测试重点 |
|----------|-----------|---------|---------|
|BD-S1|_d_in.value=1,_h_in.value=1,_w_in.value=1|01 R8 空间三维下边界 1（推断）|输入空间维下边界|
|BD-S2|_d_in.value=1290,_h_in.value=1290,_w_in.value=1290,_n.value=1,_c.value=1|01 R7 D×H×W 上界（1290×1290×1290 逼近 INT32 最大值）|D×H×W 上边界|
|BD-S3|_out_d.value=1,_out_h.value=1,_out_w.value=1|01 R8 输出维度下边界（取最小非空值 1）|最小输出尺寸|
|BD-S4|grid.value_range=[-1.0,-1.0]|01 R10 归一化坐标下边界（取 -1）|全部坐标取下边界|
|BD-S5|grid.value_range=[1.0,1.0]|01 R10 归一化坐标上边界（取 +1）|全部坐标取上边界|
|BD-S6|input.format=NDHWC,_c.value=4|01 R9 C 轴 4 的倍数边界（最小正倍数）|NDHWC 条件格式的 C 对齐下边界|

> 上边界一致性：BD-S2 锁定 D/H/W 均为 1290 时同步锁定 _n 与 _c 为 1（D×H×W 为 2146689000，小于 INT32 最大值 2147483647，tensor 总乘积不超过 MAX_SHAPE_PRODUCT），100% 求解成功。

## 异常场景

| 子场景 ID | locked因子 | 预期错误码 | 测试重点 |
|----------|-----------|-----------|---------|
|EX-S1|interpolationMode.value=2|ACL_ERROR_INVALID_PARAM|3D 不支持 bicubic，插值模式枚举外（aclnn 错误码 161002）|
|EX-S2|paddingMode.value=3|ACL_ERROR_INVALID_PARAM|填充模式枚举外（161002）|
|EX-S3|input.format=NDHWC,input.dtype=float16|ACL_ERROR_INVALID_PARAM|NDHWC 不支持 FLOAT16（R9 违反）|
|EX-S4|input.dtype=float32,grid.dtype=float16|ACL_ERROR_INVALID_PARAM|grid 与 input 数据类型不一致（R1 违反）|

> 维度类异常（grid.dimensions=4 下溢 / =6 上溢、不支持的 dtype/format 等）由 L2 自动生成（域外异常 + 维度边界异常），不在此重复定义。

## 场景覆盖验证

（步骤 4 阶段A/阶段B 执行后回填）

| 子场景 ID | 是否出现 | 出现次数 | 备注 |
|----------|---------|---------|------|
| C4-S1 | 是 | L0/L1 多条 | solve_out_shape tag 标记（常规全维 ≥1 场景，随机生成即覆盖） |
| C4-S2 | 是 | L0/L1 多条 | tag 标记；ET-S1（_n=0）目标用例同步覆盖 |
| C4-S3 | 是 | L0/L1 多条 | tag 标记；ET-S2（_c=0）目标用例同步覆盖 |
| C4-S4 | 是 | L0/L1 多条 | tag 标记；ET-S3/S4/S5（输出空）目标用例同步覆盖 |
| C8-S1 | 是 | L0/L1 多条 | solve_input_dtype tag 标记（NCDHW/ND 通用格式） |
| C8-S2 | 是 | L0/L1 多条 | tag 标记（NDHWC：f32+mode=0+C%4==0，含 BD-S6 目标用例） |
| ET-S1~S5 | 是 | 各≥1 | L0/L1 [SCENARIO-COVERAGE] 报告确认已覆盖 |
| BD-S1~S6 | 是 | 各≥1 | L0/L1 [SCENARIO-COVERAGE] 报告确认已覆盖（BD-S2 为 D×H×W 上边界 1290³ 用例） |
| EX-S1~S4 | 是 | 各≥1 | 阶段 C Q15b_ex_e2e 端到端校验确认 |

> 注：L0 生成日志中的"[WARN] L0 未覆盖场景: C4-S*/C8-S*"为校验器仅支持 ET/BD 锁定值匹配产生的结构性噪声——C 子场景通过 @solves 内 tag() 标记验证（每条用例均携带 shape_scenario/format_scenario 标签），实际已全覆盖。
