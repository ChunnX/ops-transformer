# Learnings — mla-align-flash-attn

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## [2026-08-22 06:02 UTC] Todo2: 基线回归快照（UT/pytests/examples + before 产物）

1. **UT 入口与范围**：仓根 `tests/ut/` 不含 mla 用例；mla 的 UT 在 `attention/flash_mla_with_kvcache/tests/ut/op_host/`（tiling arch35 + dtype/shape infershape）。正确入口 = `bash build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20`（CI ut_make.sh 同型，仅限一个 soc 以省时）。`--ops` 用算子目录名（小写），cmake 端以 `ASCEND_OP_NAME` 过滤子目录与 UT 源。
2. **UT 失败非玄学，全部有数字**：7/43 失败统一根因 = **host prefill 路径 tilingKey 恒为 3672322（bit21=1），CSV/UT 固化期望 1575170（bit21=0）**；key 派生 workspace 也随 key 走（3672322↔34078720，1575170↔16777216）。decode 路径与期望一致。这是"基线漂移"证据，Todo 14 key 断言必须以 host 实测为准。
3. **examples 依赖两算子**：test_aclnn_flash_mla_with_kvcache.cpp 还链接 flash_mla_metadata 的 aclnn API（metadata 分核前置）。Todo 1 的包只含 mla → 必须重建双算子包：`bash build.sh --ops=flash_mla_with_kvcache,flash_mla_metadata --soc=ascend950 --pkg -j20`，再 `--install-path=... --force` 安装。安装用 `.run --extract=/tmp/xxx` 可零污染解包检查符号。
4. **build.sh `--run_example ... cust` 有 include 缺陷**：编译命令缺 `-I ${ACLNN_INCLUDE_PATH}`，导致 `bfloat16.h` 找不到 `opdev/fp16_t.h`。规避 = 手工 g++（补该 include + rpath cust lib），命令可从 build.sh build_example cust 分支抄。
5. **build 目录被 `--run_example` 的 clean 弄脏后必须 `rm -rf build build_out` 再重建**，否则 autogen `aic-*-ops-info.ini` 缺失报 prepare_build 失败。
6. **torch cpp_extension 锁教训（重要）**：`/root/.cache/torch_extensions/py312_cpu/*/lock` 是 FileBaton 跨进程编译锁（`torch/utils/file_baton.py`，wait = 文件存在即无限 sleep）。进程被 kill -9 会残留 0 字节 lock → **之后任何 import cann_ops_transformer 的进程永久卡死**（faulthandler 栈：builder.py:112 → cpp_extension.load → _jit_compile → baton.wait）。定位法：`PYTHONFAULTHANDLER=1` + `kill -ABRT` dump Python 栈。修复：删残留 lock 文件（本次是 moe_init_routing/lock）。别再 `pkill -f <pattern>` 杀含自身命令行字样的进程（会自伤）。
7. **pytests 产物**：harness 出口 exit code = 失败数>0?1:0；`--result_csv` 能落逐用例数据（attn/lse/meta_section/nc 各维），是行为门对比的良好 before/after 载体。
8. **bf16 上板精度基线差**：examples bf16 fail=2.002%，pytests bf16 1.67-1.85%（阈 0.5%）——改动前后对比时 bf16 失败属基线现状，不是回归。

## [2026-08-22 14:12 UTC] Todo3: K1a 入口 layout 推导函数化（FA 风格 + static_assert 互锁）

1. **改动面**：仅 `attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp`（+139/-28）；`template_tiling_key.h` 零改动（推导函数放 cpp，FA 的 flash_attn.cpp 也是同款做法；DECL/SEL/PARSE_PARAMS 宏一字未动）。
2. **四个推导函数**（`__aicore__ inline constexpr`，模板参数形式，命名 `GetQueryLayoutMla/GetOutLayoutMla/GetKvLayoutMla/IsPageAttentionMla`）：q/out 用 if constexpr 镜像 `InOutLayoutTypeValue[6][2]` 表；kv 侧 `GetKvLayoutMla` 薄封装 `GetKVGmFormat`（PA_NZ→GmFormat::PA_NZ，单一权威防复制；四件套 KV_FORMAT 同源），`IsPageAttentionMla` 语义 = `(kvLayoutType != KvLayoutType_NO_PA)` 即四件套 `PAGE_ATTENTION=(KvLayoutType>0)`。
3. **三层互锁**：① 12 条顶层 static_assert（全 6 行 × q/out vs 表值，防表漂移）；② 5 条 R1 组合语义断言（TND_TND→TND/TND、PA_NZ→PA/NZ）；③ run 函数内 3 条模板实例化断言（推导结果 vs `CubBlockNormal::LAYOUT/PAGE_ATTENTION/KV_FORMAT`）。实际验证：注入错误值 LAYOUT_NTD 后 device 编译立即炸（① 的 `GetQueryLayoutMla<2>() == InOutLayoutTypeValue[2][0]` 与 ② 的 `== LAYOUT_TND` 都触发），恢复后 exit 0。
4. **模板参数取值等价性**：`inputLayoutType/outputLayoutType` 使用处全部改为推导函数调用，`KvLayoutType`（uint8_t 透传四件套）与 config/hasAttenMask/isFd/S1/S2/D/DV 一律不动；宏展开的 `inputLayoutType/outputLayoutType` constexpr 变量变为 unused（无警告）。
5. **任务字面 vs 代码真相**：任务描述 "InOutLayoutType=2→outLayout=NTD" 与代码表不符（表实为 `{TND,TND}`，即 out=TND）——**以代码表为准**，static_assert 把函数锁死在表值上，等价性由编译期保证。
6. **工程教训**：`build_mla.sh --incremental` 在 build 目录被 `bash build.sh -u`（UT）构建污染后增量会诡异重编 tests/ut/framework_special 并报 `OP_LOGD 未声明`（UT 宏缺失）——**验证 static_assert 这类"预期失败"的试验必须 `rm -rf build build_out` 走干净全量**，否则是被污染的增量先炸、kernel 编译根本跑不到。ccache 开着全量也就 ~40s，别省。
7. **编译门/回归门结果**：build_mla.sh --incremental exit 0（kernel .o 2 个真实产物）；UT `build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20` = 36P/7F 与基线全同（7F 仍是 KEY-1 tilingKey 漂移 3672322 vs 1575170）。
8. **cann 工具链正则消费面（Oracle 契约）**：本 todo 零接触 template_tiling_key.h，一次性规避 ASCENDC_TPL_PRE→template_tiling.py→asc_opc 正则风险；Todo 4 若扩张实例化面，优先只动该头 SEL 的列表常量，推导函数/断言会自动跟随（互锁生效点）。

## [2026-08-22 14:30 UTC] Todo15: T1 tests 骨架（pytests/ assets/ 结构占位 + CMakeLists 残留修复）

1. **assets 目录位（关键决策）**：任务字面写 "全部在 tests/pytests/ 下"，但 FA 的 assets 在 `tests/assets/`（pytests 的兄弟目录），且 mla 既有 harness 已有 `tests/assets/impl/{golden,inputs,compare}.py`——spec.py 的 `Path(__file__).with_name("impl")` lazy loader 要求 spec.py 与 impl/ 同目录，若把 spec.py/gen_cases.py 塞进 pytests/assets/ 会与既有 tests/assets/impl 树分裂、lazy loader 找不到模块。**按 FA 文件结构放 `tests/assets/`**。提交含：utils/perf 三件、core/backends/{gpu,gpu_impl}.py 占位、tools/xlsx_to_testcase.py（FA 复用+MLA 列映射 cache_seqlens/head_dim_v(DV)）、test_cases/ 5 个 redline 空壳（TestCases={}，docstring 注明 auto-generated from xlsx; MLA 用例数据待补充）、assets/{spec.py,gen_cases.py,impl/graph.py,testcase/.gitkeep}、tests/CMakeLists.txt:11 message 修复。
2. **gpu 占位安全设计**：gpu.py 只 import `.base`（Backend），不 import torch/gpu_impl；gpu_impl.py 定义 FLASH_ATTN_AVAILABLE=False + 三函数（保持 FA 签名）函数体 raise NotImplementedError。`core/backends/__init__.py` 零改动（本来就是纯 license 头）；orchestrator 显式 `from core.backends.base import Backend`，无自动发现——占位不会被默认加载选中。
3. **precision_visual 的 matplotlib 懒加载**：matplotlib 在绘图函数内 import（`matplotlib.use("Agg")` 也移入），numpy 模块级（torch 传递依赖必装）——保证骨架在任何环境（含无 matplotlib）都可 import 冒烟。compute_metrics 阈值与 utils/compare.py 一致（max(abs*0.5%, 2.5e-5)）。
4. **perf_runner MLA 适配**：生成的 msprof 脚本走 mla 系调用链 `flash_mla_with_kvcache_inputs.generate(c, device, layout=...)` → `build_flash_mla_with_kvcache_params(c, device, inputs)` → `flash_mla_metadata(**meta_kwargs)` → `flash_mla_with_kvcache(q, k_cache, metadata=..., **kernel_kwargs)`（与 backends/npu.py 同源）。_pytests_dir 用 runner 自身 `Path(__file__).resolve().parent.parent`（FA 用生成脚本位置推导是错的——生成脚本在 output_dir 下）。
5. **pytests 复跑确认基线（10P/4F 逐值吻合）**：bf16 三挂（NC_DECODE_BF16 1.6724%、DECODE_CAUSAL_BF16 1.6724%、PREFILL_BF16 1.8467%）+ DECODE_LSE 0.5127% 贴线；fp16 全过（PREFILL_NO_MASK 0.2949% 等与快照 §3.2 完全一致）；SECTION_64B/16B_2K sectionNum=3/2 断言过。**注意：首次全量复跑曾全 14 例崩（aclnnFlashMlaMetadata EZ9903 maxSeqlenQ=-1133462568 垃圾值）**——当时正是并发 kernel worker 编译/安装窗口，单例 PREFILL_NO_MASK 独立跑却 PASS（0.2949%）；等编译窗口过去后全量复跑即恢复 10P/4F。教训：并发编译窗口内跑上板用例结果不可信，遇全挂先重跑一次确认（不能把瞬态当回归）。
6. **torch 环境噪声**：`import torch` 后任何未捕获异常退出时必打印一行 `[ERROR] ... ERR99999 UNKNOWN applicaiton exception`（torch 注册的异常钩子，Device:-1）——环境固有现象，与算子无关，别被吓到。
7. **harness 默认 result_csv 路径坑**：test_flash_mla_with_kvcache.py 不带 --result_csv 时输出相对路径拼接在 CWD 上（在 pytests 目录跑会生成 `pytests/attention/flash_mla_with_kvcache/tests/pytests/result.csv` 嵌套目录）——跑完记得清理，QA 一律显式 `--result_csv /tmp/...`。
8. **pre-commit 与并发 worker 共存**：ruff-check/ruff-format 只改 staged 文件（worktree stash/restore 机制自动隔离未暂存文件），OAT 有跨实例 done-marker 去重——并发提交场景可直接 `git add <自己的文件>` + `pre-commit run`，不会碰对方文件。提交范围验证：`git status` 中 op_kernel/*.h/.cpp + flash_mla_with_kvcache_type.h 是 Todo 4 worker 的（未暂存未提交），本 todo 提交仅 16 个 tests 文件。
9. **gen_cases.py 可用性验证**：`python3 gen_cases.py -o /tmp/mla_stc.csv` 对 12 个 functional_stc 用例成 CSV：q [T,N1,576]、k_cache [num_blocks,1,36,128,16]、bt [B,max_blocks]、attrs 含 *_values 小整数张量约定（与 impl/inputs.py customize_inputs 的 kwargs 读取一致）。

## [2026-08-22 14:38 UTC] Todo4: K1b 模板参数聚合（FAType 式，三聚合类型）

1. **聚合策略落地**：cube/vec/fd 参数集确实不一致（cube 8 参无 mask/out，vec 12 参含 flashDecode，fd 11 参无 isFd）→ 建三聚合 `FATypeCube/FATypeVec/FATypeFd`，镜像 FA `utils/flash_attn_type.h` 组织（成员命名 inputType/mmType/outputType/qLayout/attnOutLayout/mBaseSize/s2BaseSize/dBaseSize/dVBaseSize/hasMask/kvLayoutType/pageAttention/flashDecode），放新头 `op_kernel/arch35/flash_mla_with_kvcache_type.h`（不建 utils/）。块类签名全部收敛为 `template <typename FA_T>` + 类内别名（`using INPUT_T = typename FA_T::inputType` 等，FA 同款）；**body 保持零改动**（布局/hasAtten 别名沿用原名：vec/fd 类内 `static constexpr LayOutTypeEnum layout = FA_T::qLayout;`、`bool hasAtten = FA_T::hasMask;`——原名即模板参名，if constexpr 体不动）。
2. **isFd 铁律落实**：isFd 仍在 `__global__ run_...<..., isFd>` kernel 级位（tiling key 第 5 字段），只进 FATypeVec（FLASH_DECODE 成员），FATypeFd 故意无 flashDecode；FD 块内 `isPa = KvLayoutType > 0` 改 `FA_T::kvLayoutType > 0`。kernel `FlashAttentionNoQuantMlaKernel` 仍三 block 类型参数，零改动。
3. **等价性证据链**：① kernel .o hash 文件名（d11cc332.../e76bc2b1...）改动前后一致（编译签名未变 → 8 实例面保持）；② examples 3 用例 before/after `diff` 零差异（fail 计数/maxAbs/meanAbs/maxRel 逐值位级一致）；③ pytests 14 行 before/after csv diff 零差异（10P/4F 同格局，bf16 1.6724%/1.8467%、DECODE_LSE 0.5127% 全同）。
4. **增量构建假阳性陷阱（新增）**：`build_mla.sh --incremental` 在本 todo 上**静默跳过 kernel 重编**——build 流程 exit 0、CPack 正常产出，但 kernel .o mtime 早于源文件修改（host .o 重编了，kernel .o 没动）。**判定法：`ls -la --time-style=full-iso build/binary/ascend950/bin/<op>/` 对比 .o mtime vs 源文件 mtime**；修复 = `rm -rf build build_out` 全量。此陷阱与 learnings Todo3-6 的污染不同：无报错无异常，纯静默跳过。
5. **include 面**：新类型头依赖 `util_regbase.h` + `infer_flash_attention_comm.h`（S1/S2/DTemplateType + LayOutTypeEnum 经 util.h 传递），三块头 + 入口 cpp 各显式 include；`flash_mla_with_kvcache_template_tiling_key.h` 零触碰（PARSE_PARAMS 宏展开的 s1TemplateType 等 constexpr 变量继续被入口聚合消费，DECL/SEL/PARSE 宏体一字未动）。
6. **编译/UT/行为门汇总**：全量 `bash build_mla.sh`（rm 后）exit 0，kernel .o 2 个新产物；UT 36P/7F 与基线全同（7F 逐条 = KEY-1 漂移清单）；双算子包 `--ops=flash_mla_with_kvcache,flash_mla_metadata --soc=ascend950 --pkg -j20` + `.run --install-path=/home/developer/Ascend/custom_opp --force --quiet` + 手工 g++（补 `-I ${ACLNN_INCLUDE_PATH}`，源文件在 `attention/flash_mla_with_kvcache/examples/` 下非仓根 examples/）行为门全过。提交 `8cbf95c26`（5 文件 +172/-86）。

## [2026-08-22 14:47 UTC] Todo5+8+9: TPipe 内部化 + SeqLensTool 工具化 + ConstInfo 分层（三合一原子提交 777cdc922）

1. **三合一提交面**：7 文件全在 `op_kernel/`（cpp + kernel 头 + 3 block 头 + public_define + type.h），零越界（flash_attn/common/op_host/tests/template_tiling_key.h 零改动）。提交消息按计划注明 (T5+T8+T9)。
2. **TPipe 自持的具体落地**：入口 cpp 删 `TPipe tPipe;` + `&tPipe` 实参；kernel 成员 `AscendC::TPipe *pipe_ = nullptr` 改为 **对象** `AscendC::TPipe pipe_`；Init 内不再赋值（对象直接构造），buffer manager（`l1/ubBufferManager_.Init(&pipe_,...)`）与 block Init（`InitVecBlock(&pipe_,...)`/`InitCubeBlock(&pipe_,...)`/`InitBuffers(&pipe_)`）统一传 `&pipe_`。plan 字面 "this->pipe_ = &pipe_" 同名目标+源不可行——以自持对象+取址接线实现。GetTPipePtr() 依赖被 fused_causal_conv1d/masked_causal_conv1d（成员 TPipe pipe_ + GetTPipePtr）先例证明可行。
3. **FlashMlaSeqLensTool 设计**（type.h，BaseApi）：`template<Q_MODE,KV_MODE>` 含 `actualSeqLengthsGmQ/Gm`（GM 张量，替代 block 各自副本）+ `qActSeqLensParser`（ACCUM+WZH=true）+ `kvActSeqLensParser`；`InitQ/InitKv` 两个入口。**Oracle 双分支铁律**：`InitQ` 内 `if (sequsedQAddr==nullptr) Init(实际SeqLengthsGmQ,dims,s1Size) else Init(gmAddr,dims,sequsedAddr,dims-1)` 逐字保留（不压平）。block 的 SeqLensToolType 由各自的 Q_MODE/KV_MODE 计算，与 kernel 类型必须一致（编译期强制，若某实例化组合漂移则类型不匹配编译失败）。
4. **vec/cube/fd 三 block 的 parser 收敛**：删除各自 `qActSeqLensParser/kvActSeqLensParser/actualSeqLengthsGmQ/actualSeqLengthsGm` 成员，改 `SeqLensToolType &seqLensTool_` 引用（构造收引用，镜像 FA `block_(constInfo_, qSeqLensTool_, kvSeqLensTool_)`）；调用点改 `seqLensTool_.xxxParser.GetActualSeqLength/GetTBase/GetTSize`、`seqLensTool_.actualSeqLengthsGmQ`。`InitVecBlock/InitCubeBlock/InitGlobalTensor` 签名收窄（丢 cuSeqlensQ/sequsedQ/cacheSeqlens 实参——双分支移到 kernel Init 的 tool InitQ）。dummy 三块的 ctor 用 `template<typename FlashMlaSeqLensToolT>` 抹平类型。
5. **两个坑**：① python replace 子串自碰撞（`actualSeqLengthsGm` 是 `actualSeqLengthsGmQ` 前缀，先替换 Q 再替换裸名会双前缀）——用 regex `\b...\b(?!Q)` 负向断言；② cube InitKVBuffer 原参名 `actualSeqLengthsGmQ` 但 body 用的是成员 `actualSeqLengthsGm`（潜伏隐式成员依赖），删成员后 `undeclared identifier`——修法是参数改名对齐（顺带修了潜伏命名不一致）。
6. **ConstInfo 分层**：Common 收窄为 FA 同构公共字段（移出 realG/realN2/5 组 strides/isRowInvalidOpen/enableS1OutSplit/totalSize → 新 MlaExtConstInfo），NO_QUANT 继承链 `Common→MlaExt→PA→LSE`，FULL_QUANT 附加 TensorList。不继承 Sink/Pse/PostQuant/LeftPadding/SysPrefix（mla 无输入对应，防布局漂移）。成员访问零遮蔽歧义（MlaExt 字段名与 Common/PA/LSE 全不重叠）。
7. **行为门证据（位级）**：examples 3 用例输出 diff 与 before 逐字节一致；pytests 14 行 CSV diff 与 before 位级一致（10P/4F）。**分支覆盖澄清**：examples 传 seqused=nullptr → 双分支分支1；pytests 经 normalize_params（setdefault seqused=cu 差分）**恒传非空 seqused** → 分支2。两分支都被实跑覆盖。
8. **边角用例（Todo8 验收）**：新建 throwaway 脚本（/tmp，复用 harness run_case+normalize_params，不入库）跑 3 例对 golden 全 PASS：EDGE_SEQUSED_ZERO_RECORD（seqused=[0,32,64]，0 槽位输出正确置零，0.1396% 正常 margin）、EDGE_EMPTY_CU_BATCH（cu=[0,0,64]，0.2693%）、EDGE_SEQUSED_ALLZERO（seqused=[0,0] 输出与 golden 位级一致 0/262144 fail）。k==v 由结构保证（k_cache 576 宽 nope==v，无 v 输入）+ examples multisection 覆盖。
9. **pre-commit clang-format 会动 ctor init-list**（单行→多行换行），format 后需 re-add + 二跑 pre-commit 确认全绿；OAT 有跨实例 done-marker 去重不影响。
10. **UT/pytest/examples 的 mtime 与产物验证**：kernel .o hash 文件名 d11cc332/e76bc2b1 跟踪 tiling key 签名而非源码内容——本轮代码改动后文件名不变属预期（实例化面 8 不变），验证 kernel 确实重编要看 .o mtime 晚于源码 mtime。

## [2026-08-22 07:00 UTC] Todo6+7: 编译期 core-type 选型 #ifdef 化 + FD Dummy 删除

1. **宏语义实测（T6 硬前置，方法可复用）**：在入口 cpp 插临时 `#pragma message`（`#ifdef __DAV_C310_CUBE__ / #else` 两路各自打印），`rm -rf build build_out` 后全量 `build_mla.sh`，日志 grep：8×"NOT_DEFINED"（AIV pass，两个 hash 名 d11cc332/e76bc2b1 交替 = 2 dtype × AIV TU）+ 8×"DEFINED"（AIC pass）——**宏在 AIC TU 定义、AIV TU 不定义，与 FA flash_attn.cpp:150-169 语义一致**，与 `g_coreType==AscendC::AIC/AIV`（sys_macros.h: `constexpr int32_t g_coreType = AIC/AIV`，由 `__DAV_CUBE__/__DAV_VEC__` 每 TU 注入）等价。探针撤回后用注释记录等价性（计划验收允许"static_assert 或注释记录"）。勿用 #error 探针（会把 build 目录弄脏且必须全量重建兜底，pragma 零副作用 EXIT=0）。
2. **ASCEND_IS_AIV 真相**：`#define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)` —— 即 `if ASCEND_IS_AIV {...}` 展开为 **if constexpr**，死分支不实例化。这是 FD 槽能用哑块/复用哑块的安全前提：AIC TU 中 `vecFdBlock_.InitParams()` 等调用全部在 `if constexpr(FLASH_DECODE){ if constexpr(ASCEND_IS_AIV) }` 弃枝内，只构造不调用。
3. **FA 无 FD 专属 Dummy 的实况**：flash_attn.cpp:165-169 里 AIC TU 的 FD 槽（KernelNd 第 3 模板参）= **VecDummy（与 vec 槽共用）**，即 `KernelNd<FA_T, CubeBlock, VecDummy, VecDummy>`；mla 对齐 = AIC 分支 `using VecFdBlock = VecFaBlockDummy`（FD block 头内 Dummy 类整体删除，仅剩 Normal）。VecFaBlockDummy 的 ctor `(ConstInfoNoQuant&, FlashMlaSeqLensToolT&)` 与 kernel 构造 `vecFdBlock_(constInfo_, seqLensTool_)` 天然匹配。
4. **#ifdef 块内的 using 别名必须引用先声明的符号**：第一版把 `#ifdef` 选型块放在 VecFa/VecFd Normal 别名声明之前 → `#else` 分支 `using VecFaBlock = VecFaBlockNormal` 引用未声明名 = 编译失败（AIV TU 炸）。结构调整为：三组 Normal/Dummy 别名全部先声明，选型 `#ifdef` 块后置，与 FA 的"声明全部别名 → #ifdef 选 Kernel"布局对齐。
5. **门结果**：T6 编译门 EXIT=0（kernel .o mtime > 源 mtime，确认真重编）+ UT 36P/7F（7F 全为 KEY-1 漂移清单，零新增）→ 提交 92c0d72eb（仅 cpp，+13/-3）。T7 编译门 EXIT=0（无回退需要）+ UT 36P/7F 零新增 + 行为门：pytests CSV 与 before **逐字节 diff 零差异**（14 行含 DECODE_FD 0.2991%）、examples 输出与 before **逐字节 diff 零差异**（fp16-causal-b2 PASS 0.1831% / bf16-causal-b2 FAIL 2.002% 基线 / fp16-causal-b8-multisection PASS 0.0000% sectionNum=8）→ 提交 bc98cad6f（cpp+FD 头，+5/-13）。两提交各自独立，无文件混入。
6. **行为门复跑命令沉淀**：pytests `cd attention/flash_mla_with_kvcache/tests/pytests && python3 test_flash_mla_with_kvcache.py --result_csv /tmp/x.csv`（先 source custom set_env.bash，装双算子包 `--ops=flash_mla_with_kvcache,flash_mla_metadata --soc=ascend950 --pkg -j20` 后 `.run --install-path=/home/developer/Ascend/custom_opp --force --quiet`）；examples 手工 g++ 命令（cust 分支 + 补 `-I ${ASCEND_HOME_PATH}/include/aclnn`，源在 `attention/flash_mla_with_kvcache/examples/`，-lcust_opapi 链自定义包符号，`-Wl,-rpath` 穿 cust lib）本次一次成功，命令已可用。

## [2026-08-22 15:20 UTC] Todo10+11+12: 孤儿头删除 / include 面收拢 / 编号注释清理（三独立提交）

1. **T10 删孤儿头**：`op_host/flash_mla_with_kvcache_adjust_sinner_souter.h` 与 FA 的 `fa_adjust_sinner_souter.h` 仅 namespace/include-guard 差异（逐字副本），全仓零 include → `git rm` 删除，全量编译过。提交 9bfb6dfe8（1 文件 -88）。
2. **T11 include 下沉的依赖真相（关键，两次全量编译失败后的结论）**：
   - `flash_mla_with_kvcache_gm_format.h` **不自包含**：只 include kernel_operator_list_tensor_intf.h，`GmFormat` 由 `common/op_kernel/memcopy/gm_layout.h` 定义——原来靠 cpp 先 include memory_copy（其链拉 gm_layout）才编译。下沉到 type.h 后必须在 gm_format.h 自己补 `#include "../..//memcopy/gm_layout.h"`。
   - `using namespace optiling;` **必须保持在 kernel 头 include 之前**：kernel 头 static_assert 用未限定的 `FLASH_ATTN_METADATA_SIZE`（optiling 域，tiling_data.h），原布局就是"using 在 kernel 头前"（FA 亦同款）。下沉 vf 头后 `using namespace FaVectorApi;` 反而必须移到 kernel 头 **之后**（FaVectorApi 声明只在 vf 头里，经 kernel 头才可见）。
   - launch 顺序修正：`using namespace optiling;`（tiling_regbase 留在 cpp）→ kernel 头 → `using namespace FaVectorApi;` → AscendC/AttentionCommon。
   - 其余 8 个头落地：util_regbase/infer_comm → type.h 已自带（cpp 删即可）；cast_nz/flashupdate_new/div_cast/flash_decode/memory_copy → block_vec 等已自带；attenmask_gs1（MaskInfo/IsSkipAttentionmask/AttentionmaskCopyIn 唯一消费者 = block_vec_noquant 的 AttenMaskCopyIn）与 cast_nz_dn（MLA 零符号消费者，纯 FA 组织对齐）→ 加入 block_vec_noquant；memory_copy → kernel/cube/vec/fd 全已含。
   - 门结果：全量编译 exit 0（.o mtime>源）+ UT 36P/7F（7F 全为 KEY-1 漂移清单，零新增）。提交 9fdb7ab7f（4 文件 +12/-15，entry cpp include 从 14 头收窄到 template_tiling_key+common_regbase+softmax+tiling_regbase+type+kernel 6 头）。
3. **T12 编号注释清理**：
   - 验收正则 `m0070|\bR1\b|todo14|D14|D20|D31|AIC ?263` 在 op_kernel/ + op_host/ **零残留**；examples/ 也清了（4 处）；**tests/ut/ 有 3 处 "m0070" 注释残留是 MUST NOT 锁死（T15 已提交）——验收 grep 与 MUST NOT 冲突时以 MUST NOT 为准，learnings 备案**。
   - **template_tiling_key.h 实际有 12 处命中（R1×10 + Todo14×2，:17/:130/:203/:212-217/:227-229/:240）**——验收行"它本来就没有这些编号，需确认"的"需确认"结论=有编号，但 T14 专管（ASCENDC_TPL_PRE 正则契约），T12 一字未动（22 文件清单不含它）。
   - 家族扩扫清掉了 R2/R5/R7/R9/D4/D7/D15/D26/D30/B1/E14/Todo 5/6/8/9/Oracle BLOCKER 等同类编号；error 消息字符串里的 "MLA R1:" 也算文本清理（验收 grep 会命中字符串）。D0/D1（NZ 分块维度名）、D512（head dim）、d0/d1（代码变量）是误报保留。
   - **CRLF 陷阱（重要）**：`flash_mla_with_kvcache_tiling_info_parser.{h,cpp}` 是 CRLF 行尾！python `open(p).read()` 文本模式会把 CRLF→LF 翻译，write 回写即全文件 1570 行变 LF → diff 全文件漂移。修法 = `git checkout --` 恢复 → 二进制模式（rb/wb）读改写，写回前把 '\n' 重新转 '\r\n'。改动后仅 15+1 行真实 diff。
   - **历史尾随空格**：op_host 侧多文件（def/infershape/checker 等）行尾带空格（前几波提交遗留），T12 pre-commit 时 trim-trailing-whitespace/clang-format/eof 三 hook 全文件清理（311 行空白 vs 142 行真改动）；`git diff -w` 可判别注释-only。这是 pre-commit 自动合规修复，随提交可接受（task 允许）。
   - 门结果：全量编译 exit 0 + UT 36P/7F 零新增。提交 c156e5754（22 文件 +453/-452，其中空白清理 ~311 行）。
4. **提交复核**：git log 3 提交连续（9bfb6dfe8→9fdb7ab7f→c156e5754），每提交 git show --stat 文件各自独立无混入；工作区仅 .omo/ 未跟踪。

## [2026-08-22 15:30 UTC] Todo13+14: host def/checker 组织对齐 + tiling-key 头注释对齐（两独立提交）

1. **T13 实际改动面极小（checkers 上波已同构）**：checker 文件组织/命名空间（optiling::flash_mla_with_kvcache + arch35MLA）/注册顺序（Common→ActualSeqLen→PagedAttention→Mask→Metadata→SoftmaxLSE，6 个无 Sinks）此前已与 FA 同构，逐文件 diff（去 CRLF/尾随空格归一后）只剩类型名/接口语义差异。def.cpp 唯一真实风格差 = 文件头 `\brief` 多了 "接口基准 flash_mla_with_kvcache" 冗余 → 删除对齐 FA 的 `（训练推理归一，仅非量化）` 句式。**FA def.cpp 的单行 ctor 无法照搬**：仓库 .clang-format `PackConstructorInitializers: Never`（+ColumnLimit:0）强制两行 ctor，pre-commit clang-format 钩子（v18.1.8）会把单行改回两行——"对齐"以本仓格式化器能落地的形态为准。结构验证法：`diff <(clang-format fa_def.cpp | sed 's/\r$//') <(clang-format mla_def.cpp)` 后只剩语义差异（k_cache 替代 k/v、cache_seqlens 替代 cu_seqlens_kv/seqused_kv/sinks、head_dim_v 增、win_left/right 无、layout 默认值不同）。
2. **OAT 陷阱：def.cpp 的 License 与 FA 同源非合规文本**。FA def.cpp 第 5 行是 "You **can not** use"（CRLF 遗产，从不上 OAT 扫描），MLA def.cpp 初始化时照抄 → 一旦 staged 触发 OAT `License Header Invalid`。修复 = 改为与其余 host 文件一致的 "You **may not** use"（references/oat-issues.md 模板）。这就是 T13 提交里前 2 处改动的另一处（共 2 处：brief 删冗余 + license 修正）。
3. **T14 只动注释，工具链契约验证三件套**（可复用）：
   - **token 面 diff**：`git show HEAD:<file> | grep -oE "ASCENDC_TPL_[A-Z0-9_]+" | sort | uniq -c` vs 改动后同式 → TOKENS IDENTICAL（本次严格零变化，含 `ASCENDC_TPL_5_BW/10_BW/2_BW/8_BW`）。
   - **key 等价探针（真实 FastEncodeTilingKeyDirect）**：/tmp 独立 g++ 工程，`#include "ascendc/host_api/tiling/template_argument.h"`（路径 `/home/developer/Ascend/cann-9.1.0/x86_64-linux/include/`，无 ascendc/ 目录直觉但实际在此）+ `#include "flash_mla_with_kvcache_template_tiling_key.h"`（-I 指 op_kernel/arch35），host 非 ASCENDC_DEBUG 路径下 ASCENDC_TPL_ARGS_DECL 展开为 `g_tilingDeclareParams`、`GET_TPL_TILING_KEY(a,b,c,d,e)` = `FastEncodeTilingKeyDirect(g_tilingDeclareParams, {a,b,c,d,e})`。4 组合输出 **1575170/1837314/3672322/3934466**（与 CSV 8 项/头内 :238 注释/EncodeMlaTilingKey 全一致）。改动前后 + pre-commit 格式化后三跑 diff 逐字节同。
   - **完整重编译**：`rm -rf build build_out` + `bash build_mla.sh` exit 0 覆盖 ASCENDC_TPL_PRE→template_tiling.py→asc_opc 正则消费路径；kernel .o mtime 晚于源码且 hash 名 d11cc332/e76bc2b1 不变（实例化面 8 未变）。
4. **clang-format 钩子对 DECL token 行也会插空白**（T14 必须注意）：本头 :235 原文 `...,TRUE),);` 是非格式化状态（C1 提交时未过钩子），v18 钩子自动改成 `...,TRUE), );`（逗号后空格）+ 末尾 `#endif` 补 `\n`。**这不是 token 变化**（宏名/参数 token 一个没动），但改到了 DECL 宏调用的结尾括号——经验证 asc_opc 容忍（全量编译过），且探针值不变。结论：模板头的格式让钩子统一即可，别手撸 `,);` 硬扛。
5. **门结果**：T13 编译 exit 0 + UT 36P/7F（7F=KEY-1 漂移全名一致，零新增）→ 提交 `489b35a00`（def.cpp +2/-2）。T14 编译 exit 0（两次全量：钩子改动前后各一次）+ UT 36P/7F + key 探针逐字节等价 → 提交 `09e319eb1`（tiling_key.h +22/-22，全部注释/空白）。**32 个 todo 全部完成**，工作区仅 .omo/ 未跟踪，oat_reports/build/build_out 均在 .gitignore。
6. **"Todo 14"/R1 全清**：header 原 R1×10 + Todo14×2 全部改写为中性规范表述（工具链事实保留、编号去掉），`grep -nEi "m0070|\bR1\b|todo14|D14|D20|D31|AIC ?263"` 该头零命中；验收 grep "Todo 14" 零命中。

## [2026-08-22 15:33 UTC] Todo16: B1 构建接线终核 + 全量编译终检（验证型，零代码改动）

1. **复核结论（逐项核对表，全部一致）**：
   | CMakeLists 位置 | MLA 实际值 | 与实际目录/文件匹配 |
   |---|---|---|
   | :12-13 SKIP_HEADER | FLASH_MLA_WITH_KVCACHE_SKIP_HEADER / _INNER_SKIP_HEADER | ✓（FA=FLASH_ATTN_ 同构） |
   | :17 | flash_mla_with_kvcache_def.cpp | ✓（op_host/ 下存在） |
   | :23 depends | flash_mla_with_kvcache_depends attention/common | ✓（attention/common/op_host 存在） |
   | :25/:37/:44 OP_NAME | FlashMlaWithKvcache | ✓ ×3 |
   | :52 parser | flash_mla_with_kvcache_tiling_info_parser.cpp | ✓（31777B 存在） |
   | :60 OPTYPE | flash_mla_with_kvcache | ✓（op_api/ 目录同名前缀 6 文件） |
   | def.cpp:119 | ExtendCfgInfo("opFile.value", "flash_mla_with_kvcache") | ✓（与 op_kernel/ + kernel bin 目录同名） |
   - **备注**：任务描述写 def.cpp:123，实际行号为 **119**（OpAICoreConfig 块结尾），内容一致，仅行号漂移。checkers glob `../op_host/checkers/*checker*.cpp` 命中 8 个 checker 源（base/common/flash_mla_with_kvcache/mask/metadata/paged_attention/seq_len/softmax_lse × .cpp）。
   - attention/CMakeLists.txt 用 `file(GLOB CURRENT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)` 自动发现子目录，无需任何显式登记 → 模式对齐不改名 0 文件依赖 glob，印证零改动正确。
2. **全量编译终检**：`rm -rf build build_out` + `bash build_mla.sh` → **EXIT=0**（[build_mla] SUCCESS）。产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`（2956084B，含 CRC/SHA256 校验行）；内核 bin `build/binary/ascend950/bin/flash_mla_with_kvcache/FlashMlaWithKvcache_{d11cc332,e76bc2b1}*.o` **2 个**（8 实例化面合并为 2 个 hash 名，与全计划逐 todo 记录一致——任务字面 "≥8 或 flash_mla 相关 .o" 落"或"分支）；**failed_ops.log 零命中**（build/ 与全仓 maxdepth3 均无）。
3. **工作区净度**：git status 仅 `?? .omo/`，无任何产品代码/构建残留（build/build_out 均 .gitignore）；无代码改动 → **不提交**，本计划 T1-T16 无新提交，HEAD 仍 09e319eb1。
4. **收尾判断**：B1 接线 + 全量终检双门通过，mla-align-flash-attn 计划 16 todo 全部完成，可交接 F1-F4 最终验证波（F3 将复用同一 `rm -rf build build_out && bash build_mla.sh` 全量 QA 门命令）。
