---
slug: mla-align-flash-attn
status: drafting
intent: clear
review_required: true
pending-action: write .omo/plans/mla-align-flash-attn.md
approach: 以 flash_attn 为基准，把 attention/flash_mla_with_kvcache 的代码结构（kernel 模板聚合/入口风格、TPipe 接线、host 层组织、命名与注释规范、测试基建）对齐到 flash_attn 结构；保持功能/行为不变、保持 4 组合模板实例化面；交付 build_mla.sh（-j20）并保证编译通过
---

# Draft: mla-align-flash-attn

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- K1 | kernel 入口与模板参数化对齐（FAType 聚合、layout 推导函数、tiling key 声明风格） | active | attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp
- K2 | kernel/block 类组织对齐（文件拆分、命名、TPipe 归属、Dummy/Real 机制、ConstInfo 归属、seq-lens 工具） | active | attention/flash_mla_with_kvcache/op_kernel/arch35/*.h
- H1 | op_host def/checkers 对齐（input 声明风格、checker 集、命名） | active | attention/flash_mla_with_kvcache/op_host/*
- H2 | op_host tiling 层对齐（tiling impl/info/parser/adjust 命名与方法签名） | active | attention/flash_mla_with_kvcache/op_host/{arch35,checkers,*}/*
- T1 | tests 基建对齐（pytests utils/backends/assets/test_cases 文件骨架，不迁用例资产） | active | attention/flash_mla_with_kvcache/tests/*
- B1 | 构建与改名爆炸半径（CMake 文件清单、仓内引用、examples） | active | attention/flash_mla_with_kvcache/CMakeLists.txt + 仓内 grep
- B2 | 交付 build_mla.sh（当前目录、-j20、编译 MLA 算子、断言返回码、失败打日志尾部） | active | /workspace/Code/flash_mla/ops-transformer/build_mla.sh

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
- 行为零变化：对齐只改结构不改行为（kernel 计算路径、tiling 数值、模板实例化集合 4×2dtype 均不动）——可逆性：改代码可回滚
- 语义命名保留：FlashMlaWithKvcache* 类名不改成 FANoQuantGqa*（mla 语义就是 MLA，不冒充 GQA）——对齐的是"组织模式"不是"语义命名"——可逆
- 不扩功能面：tiling key 仍只 SEL 实例化 {TND, Config9, hasMask, PA_NZ, isFd} 4 组合，不向 flash_attn 的 160 组合看齐（那是功能扩展）
- 内部编号注释清理：改造时移除 m0070/R1/Todo14/D14 等过程性注释，改写为 flash_attn 式规范注释
- examples/test_aclnn_flash_mla_with_kvcache.cpp 保留（flash_attn 无 examples 不构成删除理由；是否删属 owner 决策，暂保留）

## Findings (cited - path:lines)
- 相同骨架：单 __global__ 入口 + REGISTER_TILING_DEFAULT + KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2) + PipeBarrier 收尾；TPL_ARGS_DECL/TPL_SEL tiling key；四件套 Normal+Dummy；host 三层 + FiaTilingBase/TilingInfo + checker 链；op_api 6 文件同构
- 差异① 模板参数化：flash_attn 用 FAType 聚合（flash_attn/op_kernel/utils/flash_attn_type.h:113-120，entry 传 11 个模板参数），block/kernel 单 FA_T 参数；mla 平铺参数列表（flash_mla_with_kvcache_block_*_mla.h 的 <INPUT_T,T,OUTPUT_T,layout,s1,s2,d,dv,hasAttenMask,KvLayoutType,isFd>）
- 差异② 入口布局推导：flash_attn.cpp:30-100 有 GetQueryLayout/GetOutLayout/GetKvLayout/IsPageAttention 编译期函数；mla 入口无推导（layout 由 PARSE_PARAMS_NoQuant 宏 + InOutLayoutTypeValue[][] 表解析，template_tiling_key.h:56-63,110-117）
- 差异③ TPipe 归属：flash_attn Init 15 参无 TPipe（kernel 自持，构造传 qSeqLensTool_/kvSeqLensTool_，flash_attn_kernel_nd.h:105-115）；mla Init 14 参末尾显式 AscendC::TPipe *tPipe（kernel_noquant_mla.h:127-132，entry 里 TPipe tPipe 外部建）
- 差异④ AIC/AIV 选择：flash_attn 编译期 #ifdef __DAV_C310_CUBE__（flash_attn.cpp:150-169）+ Init 内 ASCEND_IS_AIC/AIV 分叉；mla 运行时 std::conditional<g_coreType==AIC/AIV>（flash_mla_with_kvcache.cpp:72,83,94）
- 差异⑤ tiling data/metadata：flash_attn 的 TilingData=baseTiling+内嵌 flashAttnMetaData[36][16]+[72][16]（flash_attn_tiling_data.h:117-121）；mla 只 baseTiling，metadata 走 GM 输入（flash_mla_with_kvcache_tiling_data.h:179-182，D14）
- 差异⑥ dtype 分支：flash_attn 用 ORIG_DTYPE_Q==DT_BF16/DT_FLOAT16 两路（flash_attn.cpp:110-116）；mla 自 define MLA_DT_FLOAT16=1/MLA_DT_BF16=27 并要求 Q/K_CACHE/ATTN_OUT 三宏一致否则 #error（flash_mla_with_kvcache.cpp:132-144）
- 差异⑦ def 输入/Attr：mla 8 输入（无 v/sinks/cu_seqlens_kv/seqused_kv，有 cache_seqlens；head_dim_v attr；默认 layout TND/NZ/NTD）vs flash_attn 11 输入（def.cpp 逐行可对）
- 差异⑧ checker 集：flash_attn 9 个（含 sinks/fa），mla 8 个（无 sinks/fa，名 flash_mla_with_kvcache_checker）
- 差异⑨ 测试基建：flash_attn 有 gpu backend/gpu_impl、precision_visual、perf_runner/parser、xlsx_to_testcase、assets/testcase/*.csv、gen_cases/spec/ttk_ops/graph.py、functional redline 三件套；mla 均无
- 差异⑩ 注释风格：mla 带 m0070/R1/Todo14/D14/AIC263 过程性编号（如 flash_mla_with_kvcache.cpp:11-17,119,240），flash_attn 注释干净规范

### 构建接线（探索报告Ⅱ，bg_384a07b1）
- attention/CMakeLists.txt 是 glob 自动发现子目录（:12-19），重命名算子目录**无需改父 CMake**
- op_host/CMakeLists.txt 硬编码引用：:12-13 SKIP_HEADER 变量名、:17 flash_mla_with_kvcache_def.cpp、:23 depends 变量、:25/:37/:44 OP_NAME FlashMlaWithKvcache、:52 flash_mla_with_kvcache_tiling_info_parser.cpp、:60 OPTYPE flash_mla_with_kvcache（op_api 绑定）
- kernel 源码绑定在 def.cpp:123 `ExtendCfgInfo("opFile.value", "flash_mla_with_kvcache")`；两 op 均无 op_kernel/op_api/op_graph 的 CMakeLists
- **改名爆炸半径（op 目录之外）仅 torch_extension 5 文件 + 1 文档**：
  - torch_extension/cann_ops_transformer/ops/__init__.py:16（`from .flash_mla_with_kvcache import flash_mla_with_kvcache, flash_mla_metadata`——同一模块还导出 mlá_metadata 伴生算子！）
  - ops/flash_mla_with_kvcache.py（13 处，64-335：OpBuilder/flake/schema/register_fake/impl）
  - ops/csrc/flash_mla_with_kvcache.cpp（7 处：ACLNN_CMD(aclnnFlashMlaWithKvcache,:167)、m.def,:178）
  - ops/graph_convert/graph_convert_flash_mla_with_kvcache.py（3 处）
  - docs/zh/flash_mla_with_kvcache.md（14 处，用户文档）
  - ⚠️ fused_infer_attention_score/tests/st/.../executor_aclnnFusedInferAttentionScoreV4.py:9021-9044 引用的是**外部 flash_mla pip 包**，不可改名，且无需改
  - common/ 共享头**零引用**；flash_attn 的 torch_extension 对应文件已存在（flash_attn.py/csrc/graph_convert + __init__.py:15），命名范式以 flash_attn 为准
- mla tests/CMakeLists.txt:11 有复制粘贴残留（message 里写的是 flash_attn）——对齐时可顺手修

### host/op_api 层（探索报告Ⅲ，bg_473d96d9）
- tiling 顶层入口两者结构完全一致（Parse→Checker→`FiaTilingRegistry::DoTilingImpl`，注册宏 `IMPL_OP_OPTILING(X).Tiling(...).TilingParse<CompileInfo>(...)`，:88-92）；arch35 impl 均 : public FiaTilingBase，注册 `REGISTER_TILING_TEMPLATE_FIA(X, Impl, DAV_3510, 1)`（FA:446-450 / MLA:543-547）
- infershape 两者都是**纯函数** + IMPL_OP_INFERSHAPE（无类）——原假设不成立，结构一致
- checker 集：FA 注册 7 个（含 SinksChecker），MLA 6 个（无 SinksChecker）；复合模式 4 阶段 RunCheck 一致；命名空间 arch35FA vs arch35MLA
- `fa_adjust_sinner_souter.h`：FA 是**头文件内联自由函数**，被 flash_attn tiling 和 flash_attn_metadata_aicpu.cpp / flash_mla_metadata_aicpu.cpp（共享！）使用；MLA 同名副本 `flash_mla_with_kvcache_adjust_sinner_souter.h` **全仓无 include，是孤儿文件**——MLA 实际用 impl 成员方法 AdjustSinnerAndSouter() 硬编码 SOUTER_32/SINNER_128
- impl 类差异：FA 4 字段 FaTilingKeyInfo vs MLA 5 字段（含 isFd，字段序还不同）；FA 声明不定义 CalcMaxWorkspaceSize（死声明）vs MLA 有定义；FA 有 UpdateTilingKeyKvLayout/CheckNeedInitOutput vs MLA 有 3 个 IsCapable*Mla 子检查 + SetAttenMask/SetStartIdx/SetPageAttentionLayout；MLA DoOpTiling 有 memset 零初始化
- IsCapable：FA 恒 true（:40-43）；MLA 真 gating（:108-129 数据结构+布局+maskMode 检查）
- GenTilingKey：FA 4 参 GET_TPL_TILING_KEY（:187-197）；MLA 5 参且顺序不同（:302-315）
- op_api 六文件两侧同构：l0op（OP_TYPE_REGISTER→INFER_SHAPE→ADD_TO_LAUNCHER_LIST_AICORE）、aclnn 外层（FA 多 FlashAttnProcessSinks）、inner 只含 ProcessSoftmaxLse 辅助（aclnnInner* 符号由 CANN 框架提供，仓内无定义）——**op_api 层无需结构对齐**
- op_graph 两 op 的 proto 都是空壳（REG_OP 被注释），结构一致
- TilingInfo：FA 有 sinksFlag/keyNonContigDim；MLA 有 ropeHeadDim/actualLenQDims/actualLenKvDims/preTokens/nextTokens/attenMaskFlag/kernelOutputLayout

### kernel 类级（探索报告Ⅰ，bg_76e12608）
- **FAType 聚合 vs 平铺参数**：flash_attn 的 block/kernel 全部 `template <typename FA_T>`（FAType 定义于 utils/flash_attn_type.h:113-132，聚合 11 个编译期参数）；mla 平铺：cube `<INPUT_T,T,layout,s1,s2,d,dv,KvLayoutType>`（block_cube_noquant_mla.h:30-35/376-381）、vec 12 参（block_vec_noquant_mla.h:41-47/808-814）、fd 11 参（block_vec_flashdecode_mla.h:40-45/541-546）；kernel `FlashAttentionNoQuantMlaKernel<CubBlock,VecFaBlock,VecFdBlock>`（kernel_noquant_mla.h:35-36）vs FA `FlashAttentionNoQuantGqaKernelNd<FA_T,CubeBlock,VecFaBlock,VecFdBlock>`（kernel_nd.h:41-42）
- **TPipe**：flash_attn 全目录 grep 零 TPipe（block 自行管理 buffer）；mla kernel 持 `AscendC::TPipe *pipe_`（kernel_noquant_mla.h:82，Init 时 this->pipe_=tPipe :134），buffer manager 挂 kernel 层（l1BufferManager_/ubBufferManager_/bmm1Buffers_，:61-68,195-207），entry cpp:107 `TPipe tPipe` + :112-113 传入
- **SeqLensTool**：flash_attn kernel 持 q/kvSeqLensTool_（SeqLensTool 组合 ACCUM+BY_BATCH 双 parser，flash_attn_type.h:134-155），构造时按引用传入 block（kernel_nd.h:105-108）；mla kernel 持 qActSeqLensParser_/kvActSeqLensParser_（kernel_noquant_mla.h:111-114）但 **block 各自再绑定自己的 parser**（vec block 成员 qActSeqLensParser，block_vec_noquant_mla.h:88；cube block 绑裸 GM tensor）——parser 所有权分散
- **ConstInfo**：flash_attn 在 utils/flash_attn_common_def.h:401-412，NO_QUANT 特化继承全部 9 个基础结构；mla 在 arch35/flash_mla_with_kvcache_public_define_arch35.h:211-215，NO_QUANT 只继承 CommonConstInfo+PAConstInfo+LseConstInfo（3 个）——继承面不对称
- **flashdecode Dummy**：flash_attn 的 flashdecode block **无 Dummy 类**（只有 FiaBlockVecFlashDecode，flash_attn_block_vec_flashdecode.h:48-49）；mla 有 Normal+Dummy（block_vec_flashdecode_mla.h:40-45/541-546）
- **include 依赖映射**：mla cpp 直接 include FIA 公共集（util_regbase.h→S1/S2/DTemplateType、infer_flash_attention_comm.h、vf_mul_sel_softmaxflashv2_cast_nz{,_dn}/vf_flashupdate_new/vf_div_cast/vf_flash_decode、flash_attention_score_tiling_regbase.h、common attenmask_gs1.h、自定义 gm_format.h→GetQueryGmFormat/GetKVGmFormat、memory_copy_arch35.h→CopyQueryGmToL1/CopyKvGmToL1）；flash_attn cpp 只直接 include template_tiling_key + utils 三件 + 共享 flash_attention_score_common_regbase.h（其余经 block/type 头传递），**attenmask_gs1.h 在 FA 侧是 utils/ 里自带的本地副本**——mla 依赖 common/ 更多、FA 更自包含
- Process 主循环两者同构：`while (shouldDispatchTask || validTaskCount/ExecuteTask)` 环行缓冲流水；ExecuteTask 内 `if ASCEND_IS_AIC → ComputeMm1/Mm2` / `if ASCEND_IS_AIV → ComputeVec1/2`；section 循环 + FlashDecode（AIV）

## Decisions (with rationale)
- **D1 对齐深度 = 模式对齐（用户 Q1 选定）**：保留 MLA 单 kernel 文件路径与现有目录结构，对齐 flash_attn 的**组织模式**：① FAType 式模板参数聚合（可将 block/kernel 收窄为单 FA_T 参数，或等价地在 mla 文件内引入 own FAType 别名）；② 入口显式 layout 推导（GetQueryLayout/GetOutLayout/GetKvLayout/IsPageAttention 风格编译期函数替代散落宏/表解析）；③ TPipe 归属内部化（去掉 Init 显式 tPipe 参数与 entry 的 `TPipe tPipe` 外部接线，kernel 自持 pipe，对齐 FA 15 参 Init 风格——注意 buffer manager 移到 kernel 层是 FA 结构，mla 已是 kernel 层，需确认仅去 tPipe 参数）；④ Dummy/Real 机制对齐 FA 编译期 `__DAV_C310_CUBE__` 或保留运行时 conditional（两者都是模式差异，选择以 FA 为准——见计划 T1 具体化）；⑤ 命名规范：文件/类名保持 FlashMlaWithKvcache* 语义（不冒充 GQA），但块文件命名风格、时序、注释规范对齐；⑥ 清理孤儿 flash_mla_with_kvcache_adjust_sinner_souter.h 与 FA 侧死声明（FA 死声明保持不动——以 FA 为准）
- **D2 测试 = 建骨架不迁用例（用户 Q2 选定）**：t1 组件 active；把 flash_attn tests/pytests 的**文件骨架**建到 MLA 侧：utils/{perf_runner,perf_parser,precision_visual}.py、core/backends/{gpu,gpu_impl}.py、tools/xlsx_to_testcase.py、assets/{spec.py,gen_cases.py,test_cases 模块文件、impl/graph.py} 的结构位置占位或最小可用实现（以 MLA 接口适配，如 GPU backend 可瘦身为三路径骨架）；**不迁移** flash_attn 的 redline CSV/性能用例资产（那需要真实用例数据，属功能开发）；同时修 tests/CMakeLists.txt:11 复制粘贴残留
- **D3 examples 保留（默认，未 veto）**：examples/test_aclnn_flash_mla_with_kvcache.cpp 保留
- **D4 交付 build_mla.sh + 编译通过（用户 m0034/m0037 追加要求）**：计划完成后在仓根（当前目录 /workspace/Code/flash_mla/ops-transformer）交付可执行脚本 `build_mla.sh`；固定 `-j20`；编译命令以仓根标准入口 `build.sh` 为基底，已查证：`bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache -j20`（--ops 用 snake 名，build.sh help:112/133/359；-j[n] 线程默认 8，help:115/341；soc 列表含 ascend950，:53；CI 同型命令编译.sh:134）。脚本职责：① 定位自身所在目录为仓根（readlink -f $0）；② 若 BISHENG 不可用先 source CANN setenv.bash（优先 $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash，兜底 ${HOME}/Ascend/ascend-toolkit/latest/bin/setenv.bash 与 /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash，参考 build.sh set_env() :416-430）；③ 执行上述编译命令；④ 断言返回码=0，失败时打印 build/failed_ops.log 尾部与 build 目录提示；⑤ 支持透传少量参数（--incremental/--soc 覆盖，默认 ascend950）；**验收 = 编译通过（退出码 0，产物 run 包/内核 bin 生成）**。torch_extension whl 不在脚本范围（build.sh --torch_extension_only 是独立目标，用户未要求）
- 其余 announced defaults 均被用户接受（行为零变化/语义命名保留/不扩功能面/注释清理）

## Scope IN
- 对齐 flash_mla_with_kvcache → flash_attn 的代码结构（文件组织、模板聚合方式、入口解析函数、TPipe 归属、Dummy/Real 机制、tiling data 组织、命名规范、注释清理）
- 保持行为不变（4 组合实例化面、tiling 数值、计算路径）
- 交付 `build_mla.sh`（仓根当前目录，-j20，编译 flash_mla_with_kvcache 算子，断言通过）
- **验收硬性条件：对齐改造后编译必须通过**（build_mla.sh 退出码 0）

## Scope OUT (Must NOT have)
- 不扩展功能（不新增 DN 路径、不扩到 160 组合、不改 def 输入集合的行为语义）
- 不把 mla 类名改成 GQA 语义命名
- 不删除 examples/（除非 owner 决定）
- 不做性能优化/精度调整

## Open questions
- ~~Q1 对齐深度~~ → 已答：**模式对齐**（单 kernel 文件 + FAType 式聚合 + 入口推导 + TPipe 内部化，不建 DN/utils 双文件）
- ~~Q2 测试目录~~ → 已答：**建骨架不迁用例**（tests/pytests 文件骨架对齐，redline CSV/性能资产不迁移）
- ~~Q3 examples~~ → 已答：**保留**（未 veto）

## Approval gate
status: awaiting-approval
approach: 以 flash_attn 为基准做"模式对齐"改造：K1 入口模板聚合与 layout 推导、K2 kernel/block 组织（TPipe 归属、Dummy/Real、孤儿清理）、H1/H2 host 层命名与结构、T1 tests 骨架、B1 构建接线核对；行为零变化，4 组合实例化面不变；tests 只建骨架不迁用例；examples 保留；交付 build_mla.sh（-j20）且**编译必须通过**（build_mla.sh 退出码 0）。
next-workflow-action: 用户批准后 → 运行 scaffold 生成 .omo/plans/mla-align-flash-attn.md → 按模板填 Todos/Final verification wave → 交付。
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->