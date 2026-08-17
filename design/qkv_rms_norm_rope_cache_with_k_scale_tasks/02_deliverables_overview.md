# 交付件纵览设计

## 1. 目标

本文描述 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 交付需要新增或修改的文件集合，以及每类文件的职责边界。具体计算公式见 `01_operator_proto_and_formula.md`，kernel 框架见 `05_kernel_design.md`。

## 2. 建议目录

算子属于 attention 前处理和 KV cache 更新融合算子。当前实现落在 `posembedding` 目录：

```text
posembedding/qkv_rms_norm_rope_cache_with_k_scale/
```

当前结构：

```text
posembedding/qkv_rms_norm_rope_cache_with_k_scale/
  op_host/
    op_api/
    op_tiling/arch35/
  op_kernel/
    arch35/
    qkv_rms_norm_rope_cache_with_k_scale_apt.cpp
  tests/
    ut/op_api/
    ut/op_host/
  examples/
  docs/
```

本文只定义交付件类型和职责；具体路径以当前仓库实现为准。

## 3. API 交付件

API 同时承载 RoPE 与 M-RoPE 两种场景。完整签名和固定参数顺序以 9.1 节及
`01_operator_proto_and_formula.md` 第 2 节为准。

交付文件：

```text
op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h
op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp
op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale.h
op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale.cpp
op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h
```

API 职责：

1. 暴露 ACLNN workspace size/executor 接口。
2. ACLNN 形参命名使用 `rotationOptional`、`vScaleOptional`，分别对应语义输入 `rotation`、`v_scale`。
3. 在两种场景中调用 checker 强制 `rotationOptional != nullptr`、`vScaleOptional != nullptr`。
4. 传递必选 `head_nums` ListInt 属性，ACLNN 侧建议用 `const aclIntArray *headNums`。
5. 首期 checker/tiling 对 RoPE 与 M-RoPE 统一只接受 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`。
6. 传递可选 `layout_qkv` 和 `layout_q_out` string 属性，二者位于 `headNums` 后、`epsilon` 前，顺序为 `layoutQkv`、`layoutQOut`；默认值分别为 `TND` 和 `NTD`，空串或空指针按默认值处理，非空时仅支持大小写敏感的 `NTD` 和 `TND`。ACLNN checker 在 RoPE 场景允许 `NTD->NTD`、`TND->TND` 和 `TND->NTD`，在 M-RoPE 场景只允许 `TND->TND`。
7. 传递 `epsilon` 属性。
8. 处理 in-place 输出 tensor：`k_cache`、`v_cache`、`k_scale_cache`。
9. 以固定位置承载 optional `queryStartLoc/seqLens/mropePosition`、`mropeSection`、`qQuantMode` 和
   optional `qScale`，由完整场景合同决定各 optional 参数是否必须存在。

## 4. op_host 交付件

本节描述 op_host 的公共职责；M-RoPE 的 OpDef、InferDtype 与 checker 要求见 9.1/9.2 节。

交付文件：

```text
op_host/qkv_rms_norm_rope_cache_with_k_scale_def.cpp
op_host/qkv_rms_norm_rope_cache_with_k_scale_infershape.cpp
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.h
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.cpp
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h
op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp
```

职责：

1. OpDef 定义两种场景的输入输出、dtype、format、attr，其中 `head_nums` 为 REQUIRED `ListInt`。
2. InferShape 保持最小路径：只读取输出推导所需的 `qkv`、`head_nums[0]`、`layout_qkv/layout_q_out` 和三个 cache 输入 shape，并做安全读取及有效输出推导所需的最小指针、rank、ListInt 长度、正维度与布局检查；其中保留 `layout_qkv=NTD,layout_q_out=TND` 拒绝和逻辑 `qkvN >= Nq`。五个 required output 均按固定 logical index 取得并判空；`layout_q_out=NTD` 时输出 `q_out [Nq,T,D]`、`q_scale [Nq,T]`，`layout_q_out=TND` 时输出 `q_out [T,Nq,D]`、`q_scale [T,Nq]`，cache 输出与输入同 shape。InferDtype 按属性 6 设置 q_out dtype，并始终将 q_scale 设为 FP32，详见 9.2 节。
3. checker/tiling：承担权威的完整 shape/dtype/layout 约束、`head_nums` 三项一致性、`0 < Nq <= 64`、
   `Nq == 8 * Nk`、`Nk == Nv`、D/资源边界以及 `rotation`、`v_scale` 必须非空等校验；除 InferShape
   保留的 `NTD->TND` 拒绝和 `qkvN >= Nq` 外，其余完整契约不在 InferShape 重复实现。
4. tiling：提供 D128 token-only tiling、NTD/TND tiling key、cache stride、core token range 和资源边界校验。
5. 不交付 `op_host/config` 配置文件；参考 `grouped_matmul_finalize_routing` 的 arch35 方案，通过 op_host 代码路径和 tiling key 约束到 arch35。

RoPE 场景 checker 关键项：

```text
qkv dtype bf16, rank 3
q_gamma/k_gamma dtype fp32, shape [D]
cos_sin dtype fp32, shape [MaxSeqLen,D]
slot_mapping dtype int32, shape [T]
query_start_loc dtype int32, shape [Batch + 1]
seq_lens dtype int32, shape [Batch]
head_nums required ListInt, length 3, values [Nq,Nk,Nv], all positive
Nq <= 64 in first arch35 implementation
Nq == 8 * Nk in first arch35 implementation
Nk == Nv in first arch35 implementation
layout_qkv in {NTD,TND}
layout_q_out in {NTD,TND}; allow NTD->NTD, TND->TND, TND->NTD; reject NTD->TND
qkv shape [Nq+Nk+Nv,T,D] if layout_qkv == NTD else [T,Nq+Nk+Nv,D]
k_cache shape [BlockNum,Nk,BlockSize,D]
v_cache shape [BlockNum,Nv,BlockSize,D]
k_scale_cache shape [BlockNum,Nk,BlockSize,1]
rotation dtype bf16, shape [D,D], non-null
v_scale dtype fp32, non-null; RoPE shape [Nv], M-RoPE shape [Nv,D]
k_cache/v_cache/q_out dtype float8_e4m3fn
k_scale_cache/q_scale dtype fp32
```

## 5. op_kernel 交付件

交付文件：

```text
op_kernel/qkv_rms_norm_rope_cache_with_k_scale_apt.cpp
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_common.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_controller.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_basic_block.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_cube.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_vec.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_vf.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h
op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h
```

kernel 框架先参考 `grouped_matmul_finalize_routing` 的 arch35 `weight_quant_basic_block`：

```text
qkv_rms_norm_rope_cache_with_k_scale_apt.cpp
  -> REGISTER_TILING_DEFAULT
  -> GET_TILING_DATA_WITH_STRUCT
  -> KERNEL_TYPE_MIX_AIC_1_2
  -> Controller.Process()

controller
  -> 解析 tiling 和 GM 地址
  -> 分发 basic block

basic_block
  -> AIV 前处理
  -> AIC Cube matmul
  -> AIV 后处理

vec_compute
  -> RMSNorm/RoPE/V quant/QK quant/cache write
  -> V-first order in Vec::ComputeTile: QuantVToFp8 -> BuildRopeBf16NzUb -> ScatterVCache -> Q/K AL1 copy -> A_READY
  -> V-first means V vector quant starts before Q/K preprocess; it does not mean V cache write precedes Q/K preprocess

cube_compute
  -> rotation ND2NZ/L1 resident/matmul/Fixpipe
```

## 6. 测试交付件

交付文件：

```text
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_k_scale_infershape.cpp
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_k_scale_tiling.cpp
tests/ut/op_host/test_qkv_rms_norm_rope_cache_with_k_scale_tiling.csv
tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp
tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.csv
examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp
```

测试职责：

1. InferShape UT 只覆盖默认/合法 layout、最小读取防御、`NTD->TND` 拒绝、`qkvN >= Nq` 和五个输出 shape/dtype 推导；op_host checker/tiling UT 覆盖其余完整输入契约、拒绝路径、tiling data 与资源边界。
2. op_api UT 覆盖 ACLNN workspace/executor 参数检查。
3. ACLNN example 覆盖 workspace/executor 调用链。
4. torch_extension schema 新增可选 `layout_qkv`、`layout_q_out` keyword-only 属性，默认值分别为 `TND`、`NTD`；torch wrapper 将其直接透传给 ACLNN。
5. 必须覆盖缺失 `rotation` 和缺失 `v_scale` 报错。
6. kernel 数值、layout、tail、多 tile、cache stride 和端到端无死锁验证使用 DiscussionForum 下的 NPU golden/TTK 资产：`assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py`、`assets/qkv_rms_norm_rope_cache_with_k_scale_precision.py` 和 `GEN_CASE/qkv_rms_norm_rope_cache_with_k_scale/test_cases/*.csv`。
7. `ops-transformer` 内不要求新增 `tests/ut/op_kernel` 或 ST target。内部 Set/Wait 数量、顺序和 event 生命周期由静态设计审计保证；DiscussionForum NPU 用例验证输出结果，并确认执行无超时、无死锁。验证入口存在不等于当前 worktree 已执行通过。

## 7. 文档交付件

交付文件：

```text
docs/aclnnQkvRmsNormRopeCacheWithKScale.md
```

文档包含：

1. API 说明。
2. 输入输出 shape/dtype。
3. `rotation`、`v_scale` 当前必须非空的说明。
4. `query_start_loc` local 前缀和与 `seq_lens` 历史偏移语义。
5. `head_nums=[Nq,Nk,Nv]` 必选属性及 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv` 约束说明。
6. `layout_qkv` 支持 `NTD/TND`、`layout_q_out` 支持 `NTD/TND`，当前支持 `NTD->NTD`、`TND->TND`、`TND->NTD`；`qkv` 由 `layout_qkv` 解释，`q_out/q_scale` 由 `layout_q_out` 解释。
7. `epsilon` 属性说明。
8. 支持平台 arch35。

## 8. 不交付范围

首期不交付：

1. arch34/910B/910_93 kernel。
2. `rotation == nullptr` fallback。
3. `v_scale == nullptr` fallback。
4. FWHT/Hadamard butterfly 优化。
5. `query_start_loc` 长度模式或双模式 tiling key；当前固定 local 前缀和，并通过 `seq_lens` 支持任意 batch 起始 position。
6. `rotaryDim < D`。
7. `Nq > 64`、`Nq != 8 * Nk` 或 `Nk != Nv` 的正常 kernel 路径。

## 9. M-RoPE 组合模式交付件

M-RoPE 组合模式不新增算子名，在现有目录上兼容既有 logical index 和参数相对顺序，并按完整合同设置
input/output anchor 的 `ParamType`。详细合同见
`11_mrope_bf16_int8_extension_design.md`。

### 9.1 API 与 OpDef

API 与 OpDef 要求：

1. 将既有 input index 8/9 的 `query_start_loc/seq_lens` OpDef `ParamType` 改为 OPTIONAL；ACLNN/L0
   相对顺序不动并命名为 `queryStartLocOptional/seqLensOptional`，Torch schema 使用
   无默认值的 `Tensor?`，M-RoPE 在固定位置显式传 `None`，避免其后的必选 `head_nums` 造成 schema
   默认参数顺序非法。
2. 在已有 12 个输入后追加 optional `mrope_position: int32[T,3]`，第 0 维按 token 排列、列 0/1/2
   依次为 T/H/W 位置；OpDef 已有 4 个属性后追加
   `mrope_section/q_quant_mode/q_out_dtype`；其中 `q_out_dtype` 在 OpDef/L0 中仍是标准 GE enum 对应的
   `Int`。`q_quant_mode` 为 String，默认 `PerTokenPerHead`，只支持 `PerTokenPerHead` 和 `NoQuant`。
3. OpDef 五个 output 均为 REQUIRED。Torch schema/返回注解将 `q_scale` 声明为 `Tensor`：
   RoPE 返回 FP32 scale，M-RoPE 运行值为 `None`。ACLNN 的 `qScaleOptional` 仍为可选，M-RoPE 传
   `nullptr`。
4. 保持 input index 8/9、q_scale output index 1、cache output index 2/3/4 以及既有参数的相对顺序。
   新增 input 12 插入 flat kernel input/output 边界后，`q_out` 及后续 flat GM 形参序号会顺延 1，不能
   把“logical index 不漂移”误写成“所有 flat GM 参数位不变”。
5. OpDef 对每个 tensor 的 `.DataType/.Format/.UnknownShapeFormat` 都使用等长两项相关向量；dtype
   和 ND format 不变的 tensor 也重复两项，包含 optional input
   `query_start_loc/seq_lens/mrope_position` 与 required output `q_scale`。
6. ACLNN/checker 和 graph direct base tiling 先读取 optional effective presence，其中 `M` 表示非空
   `mrope_section`，空列表与未传属性等价：`Q/S/P/M=1100` 匹配 RoPE
   候选，`0011` 匹配 M-RoPE 候选，其他组合拒绝；再校验量化模式、dtype/output/cache/layout 的完整
   合同。ACLNN 不接收独立 `qOutDtype`，从 `qOut->GetDataType()` 读取 dtype 并映射为内部 GE enum 属性；
   graph direct/L0 仍使用并校验属性 6 的 `q_out_dtype`。
7. ACLNN 只处理实际存在的 optional tensor，M-RoPE 的 `qScaleOptional` 保持为 `nullptr`。
8. Torch 两套 schema 的 `query_start_loc/seq_lens` 均改为无默认值的 `Tensor?`，并增加
   `mrope_position/mrope_section/q_quant_mode/q_out_dtype`；public API 的 `q_out_dtype` 接收 `torch.dtype`
   且默认 `torch.float8_e4m3fn`，schema 使用 `ScalarType? q_out_dtype=None` 并由 wrapper 归一化。
   Python/C++ bridge 使用 `Optional[Tensor]`/`c10::optional<at::Tensor>`，在 `.value()` 和 aclTensor
   转换前判 presence；bridge 根据归一化 dtype 创建 `qOut`，不向 ACLNN 传 dtype 整数。
9. inplace/functional 的 q_scale 固定返回位置分别声明为 `(Tensor, Tensor)` 和
   `(Tensor, Tensor, Tensor, Tensor, Tensor)`，Python 返回注解也将第 2 项声明为 required `Tensor`。
   Meta/FakeTensor、pybind 和 wrapper 在 `mrope_position` 存在或 `mrope_section` 非空的 M-RoPE
   候选路径均在该位置返回运行值 `None`，tuple 长度不变。
   Torch 只做输出构造所需的最低读取，不校验 shape/dtype/presence/scene 完整合同；完整合同由 ACLNN checker
   校验，包括 M-RoPE 候选必须满足 `NoQuant + BF16` 且两个 M-RoPE 参数同时存在。
10. graph contract 为 optional input 分别记录 `present/shapePresent`；scene resolver 先于
    `FillShapeDerivedFields/ValidateScalarInputs`，M-RoPE 不从 query shape 推导 batch。
11. checker/base tiling 在 scene resolver 之后校验 `v_scale` 精确 shape：RoPE 只接受 `[Nv]`，M-RoPE
    只接受 `[Nv,D]`；Torch wrapper 不做该 shape 校验，仍由 ACLNN checker 统一报错。

API 文档仍使用既有文件名 `docs/aclnnQkvRmsNormRopeCacheWithKScale.md`，不得重新引入历史漂移名。

### 9.2 InferShape/InferDtype

InferShape 不读取 `query_start_loc/seq_lens/mrope_position/mrope_section/q_quant_mode/q_out_dtype` 的内容或 presence 做模式校验；
它按固定 logical index 取得五个 required output，并始终推导 q_scale shape。
原有 `NTD->TND` 和 `qkvN >= Nq` 保留。

InferDtype 根据属性 6 的 `q_out_dtype` 设置 q_out，并始终将 required q_scale 设置为 FP32。量化行为由
`q_quant_mode` 合同决定，不由 dtype 单独推断。cache output
继续继承输入 dtype。

### 9.3 Tiling 与 kernel

1. tiling key 增加独立的 `ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE` 枚举模板参数；host 只放行
   `(ROPE,FP8,PER_TOKEN_PER_HEAD)` 和 `(MROPE,INT8,NO_QUANT)` 两组相关组合，不写 runtime scene flag。
2. RoPE 场景保留 3 个现有 layout key；M-RoPE 只实例化一个 TND->TND compact direct-Q key，
   总注册选择数为 4，不生成 scene/layout 的完整笛卡尔积，也不按输出大小生成 L2 cache-hint 变体。
3. tiling data 尾部只增加 `mropeSectionH/W`；`mrope_section=[t,h,w]` 由 host 校验三项非负、H/W lane
   容量及三项之和不超过 `D/2`，第 0 项不做独立 lane 上限校验，随后丢弃且不下发 kernel。
4. GlobalTensors/launch 的 input 尾部增加 `mrope_position`，保留 index 8/9 的 nullable
   `query_start_loc/seq_lens`；M-RoPE 下 ACLNN 的 `qScaleOptional` 为 `nullptr`。
   RoPE 场景绑定/读取序列位置与真实 qScale，M-RoPE 只绑定/读取 `mrope_position` 且不绑定 qScale，均由
   对应 mode 的编译期分支隔离。M-RoPE 连续 GM 偏移固定为
   `3 * tokenIndex + axisColumn`，不下发独立 stride；尾 tile 使用 `DataCopyPad` 或等价安全搬运按
   `12 * tileT` 字节精确读取，不得把 128 B 单 slot UB 容量当作 GM 读取长度。
5. RoPE 场景从 query shape 推导 batch；M-RoPE tiling 写 `batch=0`。BasicBlock 的 `InitAic/InitAiv`
   直接透传 tiling data 指针，Cube/Vec 复制本层字段；Vec 用编译期场景条件令 M-RoPE 的 `batch_` 为 0，
   后续热循环不保存 tiling data 指针。
6. 普通 RoPE 共享布局的其他固定 buffers 增加 T/H/W raw cos_sin、256 B mrope_position DB 和 `HALF_D` 项
   uint32 Gather index，静态容量合计 35.5 KiB，总 UB 为 248 KiB。M-RoPE 只使用 compact：80 KiB input
   pool、80 KiB `outputDbPoolUb_`、14 KiB K rotation DB、40 KiB reserve 和 12 KiB position window，
   总 UB 为 226 KiB；其中 `[Nv,D]` FP32 `v_scale` 常驻区按最大 `8*128*4=4 KiB` 分配。两者都不改变
   框架强制预留的 16 MiB workspace。
7. VF 用 RegBase Gather 选出交织 cos/sin，K 后处理使用 `/127` INT8 quant。M-RoPE V scale VF 按
   `[head,D]` 读取 128 个 FP32 通道 scale，与 V row 逐元素相乘，不得沿 D 广播单个 head scale。M-RoPE Q 不进入 Vec BF16
   cast 后处理，而由 Cube Fixpipe 执行 F32->BF16 并直接写 `q_out` GM；旧 Vec QCast 路径已删除。
8. V-first、核内 event、rolling pipeline 和 drain 沿用既有规则；RoPE/M-RoPE 共享 flag 1 的
   `FIX_OUTPUT_READY`，M-RoPE direct-Q 使用 flag 4 的 `MROPE_K_CONSUMED`，compact 的 K credit 为
   K rotation 双槽计数。AIV 在 `StageMropeK(lastTile)` 后、`ScatterMropeK(lastTile)` 前返还 rotation
   credit，并在同一控制迭代立即发起 scatter；K quant 双槽由各自的 `MTE3 -> V` reusable event 保护。
9. common 只描述物理 pool/slot 的 bytes、offset 和 elements，不为 TileParam、A_rot、input/output、
   metadata、V output、K rotation、K quant 分别暴露全局 `BUFFER_NUM`。Controller/BasicBlock/Vec owner
   根据 pool 总容量与单 slot 容量局部确定槽数；每个 physical slot 仍由本资源独立 use counter 取模选择。
   M-RoPE input/raw 各为 4 槽且计数器独立，empty slice 不推进 K quant counter。

### 9.4 验证资产

Host/Tiling/API UT 位于 `ops-transformer`。Kernel 数值、layout、tail、多 tile、cache stride 和
无死锁验证直接使用 DiscussionForum golden/TTK，不新增 kernel UT 或 ST target。
