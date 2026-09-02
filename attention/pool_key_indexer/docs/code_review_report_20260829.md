# PoolKeyIndexer 算子综合检视报告（规格 / 蓝图 / torch 接口交叉比对）

> 检视日期：2026-08-29
> 检视范围：
> - 算子代码：`op_host`（def / infershape / tiling）、`op_kernel`（common / template_tiling_key / arch22 / arch35）、`op_graph`
> - torch 接口：`torch_extension/cann_ops_transformer/ops/pool_key_indexer.py`、`ops/csrc/pool_key_indexer.cpp`、`ops/graph_convert/graph_convert_pool_key_indexer.py`
> - 文档基准：`docs/aclnnKpoolIndexer.md`（接口规格）、`docs/development_blueprint.md`（开发蓝图）
> - 检视方法：以规格文档的参数表 / 约束说明为基准逐条比对 op_host 校验与 kernel 行为；以蓝图记录的实施状态比对代码现状；对 torch 接口逐参数比对规格语义与 csrc/py 两层防护。

## 总览

| 严重度 | 数量 | 说明 |
|--------|------|------|
| P0（静默错算风险） | 2 | eager 路径非连续输入防护缺口 |
| P1（规格-实现矛盾） | 4 | 规格声明的能力实现不支持（或反之） |
| P2（一致性 / 防御性） | 7 | 默认值不一致、校验局部性、文档缺口 |

---

## P0 — 静默错算风险（建议优先处理）

### [P0-1] eager 路径 BSND/TND 布局下 0 轴非连续 pool_key 静默错算（✅ 已修复 2026-08-31）

> **修复记录**：双层拦截 + 回归用例。
> - csrc：非 PA 布局遇到非连续 pool_key 直接 `TORCH_CHECK` 拒绝（报错指明"仅 PA_BBND 支持 0 轴非连续"）；
> - tiling：BSND/TND 分支对显式指定的 `key_stride0` 属性（≥0）报错拒绝（覆盖图模式属性透传/直调 aclnn 绕过 csrc 的场景）；
> - 回归用例：`test_quant.py::test_quant_error_nonpa_noncontig_key`（NPU 上构造非连续视图验证 BSND/TND 双布局拒绝，注明"CPU→NPU 搬运会拍平视图"的测试陷阱）；
> - 验证：量化 23 + smoke + precision 133 + GE 图 68 + aclgraph 5 全过，无回归；ascend950 编译通过。

- **位置**：`torch_extension/.../csrc/pool_key_indexer.cpp:263-280`（csrc 只拦非 0 轴非连续）；`op_host/pool_key_indexer_tiling.cpp:453-475`（BSND/TND 分支无条件 `keyStride0 = 0`，不读属性）；`op_kernel/arch35/pool_key_indexer_service_cube.h` `KeyNd2Nz`（`tensorKeyOffset + s2GmOffset * headDim` 连续寻址）
- **链路**：用户在 eager 传入 NPU 上已存在的 0 轴非连续 pool_key 视图（如 `big[::2]`，BSND 布局）：
  1. csrc：轴 1+ 全连续 → `TORCH_CHECK` 通过；`keyStride0 = stride(0)` 作为属性下传；
  2. tiling：**BSND/TND 分支完全不读 key_stride0 属性**（该属性仅在 PA 分支 `tiling.cpp:424` 消费），且 eager 下 `GetInputStride` 返回 nullptr（`CheckKeyContiguous` 无运行时 stride 可查）；
  3. kernel：BSND `KeyNd2Nz` 连续寻址 → **读到错误数据，无任何报错**。
- **对照规格**：规格约束节将 keyStride0/非连续支持限定在 PA_BBND（"当 layoutKeyOptional 为 PA_BBND 且 poolKey 为 0 轴非连续输入时，keyStride0 应传入…"），但参数表 poolKey 行的"非连续Tensor"列写的是"支持0轴非连续"**未限定 PA**，而 csrc 的实现（只查轴 1+，不查布局）恰好放行非 PA 的 0 轴非连续——三方（规格/csrc/tiling）口径不一，最终落到"静默错算"。
- **建议修复**（任选其一，最好双管齐下）：
  - csrc：非 PA 布局时 `TORCH_CHECK(poolKey.is_contiguous(), "non-PA layout requires contiguous pool_key")`；
  - tiling：非 PA 分支若 `key_stride0` 属性 ≥ 0（显式指定）则报错拒绝（eager 的唯一信号源）。
- **备注**：图模式无此问题（框架上报运行时 stride，`CheckKeyContiguous` 从轴 0 校验会拦截）。

### [P0-2] eager 路径非连续输入防护：实测修正（2026-08-31 复核，原判断部分有误；2026-08-31 二次复核补图模式结论）

- **机制澄清**：`op_def` 的 `AutoContiguous()` 声明**确实会在 aclnn 层生效**（不仅限于图模式）。反汇编 `libcust_opapi.so` 证实生成代码含完整自动连续化链路：
  - `NnopbaseAddInput`（query/weights/q_descale 等 AutoContiguous 声明的输入）→ executor 标记参与连续性检查；
  - `NnopbaseAddIgnoreContinuesInput`（pool_key/k_descale，IgnoreContiguous 声明）→ 跳过检查；
  - `NnopbaseGetUnContiguousTensors(executor, &unContTensors)` 收集非连续 tensor → 有则调 `aclnnContiguousGetWorkspaceSize/aclnnContiguous` 在 workspace 内完成连续化拷贝后执行。
- **实测结论**（NPU 上构造真实非连续视图验证）：
  - **query / weights / q_descale（AutoContiguous）**：非连续输入**自动连续化后正确计算**，与连续参照结果一致 ✅——原报告"零防护/静默错算"的判断**有误**，aclnn 层已有防护；
  - **k_descale（IgnoreContiguous，BSND/TND 布局）**：非连续输入**不自动连续化**（IgnoreContiguous 本意服务 PA 0 轴非连续），且 BSND/TND 的 tiling 不消费 `keyDequantScaleStride0`（非 PA 置 0，kernel `GetKeyScale` 非 PA 分支连续寻址）→ **实测与连续参照结果不一致，静默错算确认** ⚠️；
  - **pool_key（IgnoreContiguous）非 PA 布局**：即 P0-1（已修复）。
- **图模式二次复核（2026-08-31）**：对 query/weights/q_descale/pool_key（BSND 非连续，含 3 倍 stride 数据交错强对照）在 torchair GE 图模式下逐一实测，结果均与连续参照一致。机制定位（GE 图 dump + 运行时调用拦截）：
  1. dynamo 图**不插** contiguous 节点（export 验证），converter 的 `_pool_key_stride0` 对 torchair ge.Tensor 返回 **-1**（无 stride accessor 可用，GE Build 图 `key_stride0=-1` 证实），Build 图**无 TransData/Contiguous 节点**；
  2. 但运行时拦截 `torch.Tensor.contiguous` 捕获到 **torchair 执行期对非连续输入（shape 与 pool_key 一致、stride 非连续）主动做了 contiguous 拷贝**——即 GE 图模式对非连续输入的保护发生在 **torchair runtime 的图输入喂入层**（而非图编译层），对全部输入（含 IgnoreContiguous 的 pool_key/k_descale）一律先连续化。
  3. 结论：**图模式（torchair）下所有输入的非连续均被运行时 contiguous 兜底，无静默错算风险**（代价是隐性拷贝，PA 0 轴非连续优化在该路径实际被绕过——`key_stride0=-1` + 数据已连续，走连续分支）。
- **aclgraph（npugraph_ex）模式专项验证（2026-08-31）**：poolKey 非连续输入**同样被自动转连续**，且**回放语义正确**。机制（`acl_graph.py` 源码 + 运行时拦截定位）：
  1. aclgraph 底层复用 GE 图的输入处理链——`_DiscontiguousTensorInput`（`fx2ge_converter.py:1297`：dynamo placeholder 的 meta `is_contiguous()` 为 False 时启用，`__call__` 即 `args[idx].contiguous()`，并被 codegen 进图执行的前置输入处理函数 `_gen_input_process`）；
  2. 捕获期（`acl_graph.py:1168`）：`clone_input=1`（默认）时每个 user input 被换成 `torch.empty_like(..., memory_format=torch.contiguous_format)` 的**连续静态 buffer**，捕获记录该 buffer 地址；非连续视图经 `_DiscontiguousTensorInput` 拷入；
  3. 回放期（`acl_graph.py:1275-1319`）：每次 replay 执行 `_foreach_copy_(dst=捕获期连续 buffer, src=当前非连续视图参数)`——`copy_` 按 stride 语义 gather，**每次回放都重新拷贝当前视图数据**（实测：修改视图源数据后回放，结果正确跟随新数据）。
  4. 实测结论：捕获/回放两轮结果均与同期 eager 连续参照一致（数据交错强对照），**无静默错算**；代价是每次回放多一次隐性 D2D 拷贝（clone_input 机制本身的固有开销，与是否非连续无关；非连续视图的 copy 为 stride gather）。
- **残余问题（P0-2 最终版，2026-08-31 三次复核修正范围）**：eager/aclnn 直调路径的 **k_descale 非连续输入（全布局）**是真缺口，范围比此前认定的"非 PA 布局"更宽：
  - 非 PA 布局：kernel `GetKeyScale` 连续寻址，tiling 置 `keyDequantScaleStride0=0` → 静默错算（实测确认）；
  - **PA 布局 + 0 轴非连续**：k_descale **没有 pool_key 的 `key_stride0` 属性机制**（def.cpp 无属性、csrc 无直读下传），tiling 在 eager 下 `GetDynamicInputStride` 返回 nullptr → 由 shape 推导 stride，与实际非连续布局不符 → 同样静默错算（此前残余描述漏掉了此半区）；
  - 图模式不受影响（框架上报运行时 stride 经 CheckKeyContiguous 校验，或 torchair runtime 连续化兜底）。
- **✅ 修复（2026-08-31 实施两版，终版为"支持"而非"拒绝"）**：
  - 第一版（已废弃）：csrc 全布局统一拒绝非连续 k_descale；
  - **终版（按需求改为支持 PA 0 轴非连续）**：仿 `key_stride0` 机制新增 **`k_descale_stride0` 属性**——① def.cpp/proto 注册属性（默认 -1）；② csrc 直读 `k_descale.stride(0)` 下传（非 PA 布局与非 0 轴非连续仍拒绝，报错口径同 pool_key）；③ tiling 属性优先级：属性 > 运行时 stride > shape 推导，含属性-运行时冲突检测、属性值 ≥ 连续 stride 下界校验（mode=0/mode=1 分别按 blockSize×N2 / blockSize×N2×(D/32)）、非 PA 布局属性显式指定拒绝（BSND/TND 分支）；④ graph_convert converter 透传属性。
  - 实测：PA 0 轴非连续 k_descale（mode=0 float + mode=1 E8M0 视图）结果均与连续参照一致；BSND 轴 1 / PA 轴 1 非连续正确拒绝。
  - 回归用例：`test_quant.py::test_quant_error_noncontig_kdescale`（双拒绝 + mode=0/1 双支持验证）。全量回归：量化 24 + 泛化/smoke/GE 235 + precision/aclgraph 138 全过；ascend950/910b 编译通过。
- **规格影响**：参数表"非连续Tensor"列的"√"对 AutoContiguous 输入在 eager/aclnn 路径同样成立（aclnnContiguous 兜底）；图模式由 torchair runtime 连续化兜底（全输入）。修订原 P2-7 中"仅图模式支持"的表述。

---

## P1 — 规格-实现矛盾

### [P1-1] BSND 布局下 actual_seq_q / actual_seq_k：规格声明可选支持，实现直接拒绝

- **位置**：`op_host/pool_key_indexer_tiling.cpp:140-157`——BSND（含 PA+BSND）下 `actual_seq_q != nullptr` 或 `actual_seq_k != nullptr` 一律报错（"BSND scenario, actual_seq_q must be null"）。
- **对照规格**：`aclnnKpoolIndexer.md` actualSeqLengthsQueryOptional 条目明确描述了 BSND 用法："该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0，支持长度为B的一维tensor"（非前缀和的每 batch 有效数）；actualSeqLengthsKeyOptional 同样显式列出"（BSND/TND）"两种布局。
- **对照标杆**：`tests/pytest/pool_key_indexer_reference.py:754`（`if layout_q == "BSND" and actual_seq_q is not None:`）——CPU/NPU 金标均实现了 BSND 可选 actual_seq 语义，模块 docstring 也声明"BSND 可选 actual_seq"为能力项。
- **结论**：规格与金标都认为 BSND 可选 actual_seq 是既定能力，tiling 一刀切拒绝属实现缺口（或规格超前）。**需要决策**：要么补 kernel 支持（BSND 每 batch 有效行/池数，涉及 CleanInvalidOutput 与 scoreGm 布局），要么修订规格删除 BSND 相关描述并同步金标。
- **影响面**：TND 前缀和语义不受影响；当前测试套件 BSND 用例均未传 actual_seq（与算子行为一致、与规格不一致）。

### [P1-2] def.cpp 950 量化组合的 weights 只声明 FP16，规格允许 BF16（✅ 已修复 2026-09-02）

- **位置**：`op_host/pool_key_indexer_def.cpp:123-127`——950 config 的 weights `DataType({FP16, BF16, FP16, FP16})`，4 个组合按 query{FP16, BF16, FP8, FP8} 对位：量化组合（3/4）weights 仅声明 **FP16**。代码注释自认"tiling still accepts BF16 weights at runtime"。
- **问题**：tiling（`tiling.cpp:328/358`）确实接受 FP16/BF16；但 **GE 图模式的 op-select 按 dtype 组合表匹配**——`FP8 query + BF16 weights` 无匹配组合，图编译期被拒。规格约束节明确写"量化场景（quantMode为0或1）下，参数weights为FLOAT16或BFLOAT16"。同一输入在 eager 通过、图模式被拒，行为分叉。
- **✅ 修复（2026-09-02）**：BF16 weights 完整支持——
  1. **def.cpp**：组合表扩至 6 组合（FP8×{FP16,BF16}×quant{0,1}），weights 显式枚举 FP16/BF16；
  2. **PkiType**（common.h）：新增 `WEIGHTS_T` 模板参数（默认 void=沿用 PkiTypeTraits 推导），量化分支显式携带 half/bfloat16_t；
  3. **kernel entry**（pool_key_indexer.cpp）：FP8 分支按 `ORIG_DTYPE_WEIGHTS` 分发 WEIGHTS_T（参考 quant_lightning_indexer 的 `ORIG_DTYPE_WEIGHTS == DT_BF16` 双 dtype 模式）；
  4. **arch35/arch22 service_vector + kernel.h**：W_T 推导改为"PkiType 显式 weightsType 优先，PkiTypeTraits 推导兜底"；
  5. **arch35 标量预乘修复**：`static_cast<float>(weightUB_.GetValue(...))` 在 W_T=bfloat16_t 时触发 bisheng "not support bf16 type cast"（bf16 标量语义限制，与 fp8 同类）——**最终方案改为 SIMD 向量预乘**（参考 QLIV2 weightTemp 模式）：独立 `__simd_vf__` helper `MulWeightQScaleSIMD`（`LoadAlign<W_T> + Cast<float,W_T> + Mul + StoreAlign`，每 s1 行 4 条向量指令替代 gSize 次标量），输出落新增的 `weightScaleFp32UB_`（fp32，行距 128 bank 对齐）。bf16 的 Cast 在 `__simd_vf__` 函数内有硬件展开（内联到普通 `__aicore__` 函数会报 "Do not know how to split the result of this operator"，同为 bisheng 限制）。同时删除原标量路径的三组跨 pipe 同步（MTE2ToSSync/VToSSync/SToVSync → 同 pipe PipeBarrier），UB 缓冲重命名 weightScaleUB_ → weightScaleFp32UB_ 语义对齐。
- **验证**：mode=0 BF16 weights 结果与 FP16 weights 一致 + 与 CPU golden 一致；mode=1（E8M0）BF16 weights 同样通过；全量回归 186（量化三层）+ 78（smoke/GE/aclgraph）+ 133（precision）全过；ascend950/910b 编译通过。

### [P1-3] 规格未记载 E8M0=0xFF 的 NaN 传播语义

- **位置**：kernel mode=1 路径（LoadData-Mx 硬件反量化）对 E8M0=0xFF 按 NaN 传播（与 QLIv2 一致，蓝图 §5.2 已拍板并记录）；`tests/pytest/test_quant.py` 有 NaN 专项用例。
- **对照规格**：`aclnnKpoolIndexer.md` 全文无 0xFF/NaN 行为描述（用户无法从规格得知 scale 编码 255 会产生 NaN 分数且不报错）。
- **建议**：规格 mxFP8 描述处补一句："quantMode=1 时，k_descale/q_descale 的 E8M0 编码 0xFF 按约定解释为 NaN，算子不拦截，NaN 参与分数计算与 TopK 排序。"

### [P1-4] Tensor 变体（值输入在 NPU）跳过 pool_tail_k 值域校验，非法值静默产出错误结果

- **位置**：`csrc/pool_key_indexer.cpp:301-376`（Tensor 变体注释自述"tiling 跳过这些输入的 host 值校验"）；`tiling.cpp:901-915`（`IsHostVisibleValue` 为 false 时跳过 `[0, pool_size-1]` 检查）。
- **问题**：`pool_tail_k` 元素 ≥ pool_size 时，kernel `GetS1S2ActualSeqLen` 的 `actS2SizeOrig = actS2Pool * poolSize + tailK` 被抬高 → 因果可见池数计算错误 → **无报错的错误输出**。规格约束节对 poolTailK 的值域要求（[0, pool_size-1]）在 Tensor 变体下无任何一层执行。
- **建议**：kernel 侧 `GetS1S2ActualSeqLen` 对 `tailK >= poolSize` 做 clamp（防御性，语义上等价非法输入截断）或在 csrc Tensor 变体分支强制 D2H 校验（牺牲该路径的无同步收益，可做成可选）。

---

## P2 — 一致性 / 防御性改进

### [P2-1] torch 前端默认值与 op_def 不一致，且 schema docstring 声明失实

- **位置**：`ops/pool_key_indexer.py:46-50`（schema：`topk=128, pool_size=1, mask_mode=0`）vs `op_host/pool_key_indexer_def.cpp:77-81`（`topk=2048, pool_size=16, mask_mode=3`）；`graph_convert_pool_key_indexer.py:83-87` 同样是 128/1/0。
- **问题**：`pool_key_indexer.py` schema docstring 写着 "Attr defaults align with op_host/pool_key_indexer_def.cpp"——**与事实不符**（三处默认值两两不一致：aclnn 生成层 `PoolKeyIndexer.py` 用 2048/16/3，torch 前端用 128/1/0）。
- **影响**：用户经 torch 接口与经 aclnn 直接调用的"缺省语义"不同；`mask_mode` 默认值不同还直接影响是否带因果掩码。
- **建议**：统一为一组（建议对齐 def.cpp），修正 docstring。

### [P2-2] csrc 缺 topk 合法档位检查（错误局部性弱）

- **位置**：`csrc/pool_key_indexer.cpp:177-180` 仅查 `topk>0` 与整除；档位约束（[1,2048]∪{3072,4096,5120,6144,7168,8192}）在 infershape（`infershape.cpp:104-110`）与 tiling（`tiling.cpp:57-65`）都有。
- **影响**：功能上无漏（eager 最终走 tiling 拦截），但报错形态是 ACLNN EZ1008 深层错误而非清晰的 TORCH_CHECK；与 csrc 已有的 topk/整除检查风格不一致。
- **建议**：csrc 补齐档位检查（一行 `TORCH_CHECK`），保持错误在最近层抛出。

### [P2-3] infershape 缺 N1/N2/D/headDim 维度校验（仅 tiling 有）

- **位置**：`infershape.cpp` 只校验 layout/topk/pool_size/mask/quant 与 query 维数；N1≤64、N2==1、D==128、q/k dtype 一致等在 tiling（`tiling.cpp:359-396`）。
- **影响**：图编译期错误暴露晚（需跑到 tiling）；错误信息定位到 tiling 而非 infershape。功能无漏（tiling 必然执行）。
- **建议**：维持现状可接受；若追求错误尽早暴露，可在 infershape 补 N2/D 的快速检查（二者是纯 shape 推导可得）。

### [P2-4] graph_convert 输出名与 op_def 注册名不一致（信息项）

- **位置**：`graph_convert_pool_key_indexer.py:115`（`outputs=["sparse_indices_out", "sparse_values_out"]`）vs `def.cpp:69-73`（`sparse_indices` / `sparse_values`）。
- **影响**：GE 图测试 68 例通过，说明 torchair `custom_op` 的 outputs 名仅作返回值标签（按位置映射），无功能影响；但与 op_def 命名不一致，维护时易混淆。
- **建议**：对齐命名（或注释说明这是 converter 层标签）。

### [P2-5] Meta 实现不校验 layout_q 合法性

- **位置**:`ops/pool_key_indexer.py:76-83`——`layout_q == "BSND"` 之外的任意字符串（含拼写错误如 "bsnd"）都走 TND 分支推导 shape，torch.compile FakeTensor 阶段不报错。
- **建议**：Meta 补 `assert layout_q in ("BSND", "TND")`（或将非法 layout 的报错前移）。

### [P2-6] 测试覆盖缺口：量化一层泛化的 n1 档位缺 48

- **位置**：`test_quant_general.py:88`（`N1_CHOICES_QUANT = [1,2,4,8,12,16,32,64]`，无 48）；48 档在二层（`test_quant_general2.py:81`）才引入并借此抓出了 mBaseSize 半区错位 bug（蓝图 ⑦）。
- **建议**：一层补 48（及 40），保持两层包络连续。

### [P2-7] 规格参数表"非连续Tensor"列语义模糊（文档问题；2026-08-31 按 P0-2 复核结论修订）

- **位置**：`aclnnKpoolIndexer.md` 参数表——query/weights/poolTailK/actualSeq*/blockTable/q_descale/k_descale 均标"√"，poolKey 标"支持0轴非连续"，输出标"x"。
- **实际能力矩阵**（eager 反汇编 + 图模式运行时拦截双重核实，2026-08-31）：
  | 输入 | eager/aclnn | 图模式（torchair） |
  |------|-------------|------------------|
  | query/weights/q_descale 等 AutoContiguous 声明项 | ✅ aclnnContiguous 自动连续化 | ✅ torchair runtime 喂图前 contiguous 兜底 |
  | pool_key（PA_BBND） | ✅ 0 轴非连续（keyStride0 属性） | ✅ runtime 连续化兜底（注：converter 读不到 stride，`key_stride0=-1`，非连续优化被绕过走连续分支） |
  | pool_key（BSND/TND） | ❌ 已拒绝（P0-1 修复） | ✅ runtime 连续化兜底（无错算，但 P0-1 的 tiling 属性拒绝在图模式不触发） |
  | k_descale（PA_BBND） | ✅ 0 轴非连续（keyDequantScaleStride0） | ✅ 同上 |
  | k_descale（BSND/TND） | ❌ 已拒绝（非 PA 布局无属性通道） | ✅ runtime 连续化兜底（图模式 tiling 的 stride 校验实际不会遇到非连续输入） |
| k_descale（PA_BBND，0 轴非连续） | ✅ **已支持（`k_descale_stride0` 属性，2026-08-31 终版修复；mode=0/1 双模式实测）** | ✅ 同上 |
- **建议**：参数表按上表口径修订"k_descale 非连续支持"的布局限定；并落实 k_descale 非 PA 布局的 eager 拒绝（同 P0-1 模式）。

---

## 蓝图（development_blueprint.md）一致性核对

| 蓝图声明 | 代码现状 | 结论 |
|---------|---------|------|
| 步骤1-3 + 2补/8补/8补2 已实施 | 与代码一致（ExpandAndAppendIndices、keyStride0、workspace 16MB 等） | ✅ |
| 步骤4 Phase 0-5 完成，修正项 ①-⑩ | 逐条核对：splitM 禁用（kernel.h isQuant 分支）、kScale `%16` 乒乓、GetKeyScale PA 段偏移重写、weightScaleUB 预乘 + BroadcastLane、mBaseSize=gSize×偶数 s1PerL0 钳制、topk 采样合法性——均落地 | ✅ |
| 量化 22+91+71 用例、非量化 206 回归 | 与 tests/pytest 现状一致（184 量化全过复核过两轮） | ✅ |
| §5.2 规格冲突已修订（mode=1 尾维 [D/64,2]） | 规格已含尾维描述 | ✅ |
| 蓝图 §6.3.2 FixPkiAclDtypes 预留项 | csrc 已实现（E8M0 dtype 校验 + ACL_FLOAT8_E8M0 + 2 元素打包对齐） | ✅ |
| 蓝图未登记项 | 本报告 P0-1/P0-2（eager 非连续缺口）、P1-1（BSND actual_seq 矛盾）为蓝图盲区 | ⚠️ 建议蓝图补录 |

---

## 处置建议优先级（2026-08-31 更新）

1. ~~**立即**：P0-1（csrc 非 PA 布局拒绝非连续 pool_key，一行检查）；P0-2（csrc 补 query/weights/descale 连续性检查）~~
   **→ 均已完成**：P0-1 已修复（csrc 双层拦截 + tiling 属性防御 + 回归用例，2026-08-31）；P0-2 已按三次复核后的正确范围修复（**仅 k_descale 需要检查**——query/weights/q_descale 由 aclnnContiguous 兜底无需检查；k_descale 全布局非连续 csrc 拒绝 + 回归用例，2026-08-31）。
2. **本迭代**：P1-1 需产品决策（补能力或改规格）；P1-2 def.cpp 组合表扩展；P1-3/P2-7 规格文档修订（含 k_descale eager 连续性要求的表述）。
3. **跟进**：P1-4（kernel 防御 clamp）、P2-1（默认值统一）、P2-2/P2-5/P2-6（校验前移与测试补档）。
   注：原"可选增强：`k_descale_stride0` 属性机制"**已在 P0-2 终版修复中落地**（2026-08-31，PA 0 轴非连续 eager 支持含 mode=0/1）。规格文档需同步补充 `k_descale_stride0` 属性条目（与 key_stride0 同格式）。

---

## 附：已核对无问题的高风险点（避免重复排查）

- tiling 架构门禁（quantMode 0/1 仅 DAV_3510）✓；descale 空/非空与 dtype 匹配 ✓；mode=1 尾维 shape 校验 ✓；topk 档位/整除（tiling+infershape 双层）✓；N1≤64/N2==1/D==128 ✓；block_table 形状与 dim(1)≥maxBlockNumPerSeq ✓；TND 前缀和单调性/末值==T1（host 可见时）✓；key_stride0 属性优先级与冲突检测（PA）✓；输出 shape/dtype（INT32/FLOAT/(0,)）三层一致 ✓；ACLNN_CMD 属性顺序与 def.cpp 注册序一致 ✓；E8M0 storage_offset 2 元素对齐 ✓；ValueDepend 输入 host/Tensor 双变体分流与 aclgraph 兼容注释 ✓。
