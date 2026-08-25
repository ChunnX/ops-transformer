# Learnings — mla-kvcache-checkers-rewrite

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## [2026-08-23 11:00 UTC] T6+T7: 键生成映射 + SEL 72 实例 + 7F 漂移修复（联合波，提交 5e14d553a）

1. **T6 键映射落地**（tiling.cpp UpdateTilingKeyLayout/UpdateTilingKeyInfo 仅两函数）：qLayout→InOutLayoutType 用 **MLA 自身枚举**（TND→2/BSND→1/BNSD→0，头文件 :119-121），kvLayout→KvLayoutType（NZ/PA_NZ→3、PA_BNBD→2、PA_BBND→1）。**勿照抄 flash_attn 枚举数值**（FA 是 0=BSND/1=BNSD/2=TND，与 MLA 完全不同）；template_tiling_key.h 只实例化 BY_BATCH,false / ACCUM,true——q 侧 WITH_ZERO_HEAD 必须随 Q_MODE 推导。
2. **7F 漂移两路径决策（选了"校准+重固化"组合）**：读 flash_attn_tiling.cpp:115-130 发现 FA 的 SplitPolicy **无条件 flashDecodeFlag_=true**（:129），MLA SplitPolicy 早已同款 → isFd 派生**零代码改动即对齐**；host prefill 恒 3672322 → 按 learnings "key 断言以 host 实测为准" 刷新 CSV/UT 期望。CSV 4 条 prefill 行（TND_Prefill/…LSE/…BF16：1575170→3672322，TND_Prefill_MaskCausal：1837314→3934466；workspace 16777216→34078720），cpp 两处 EncodeMlaTilingKey(false,false)→(false,true)（MetadataPassthroughConstantFill:123、SeqLensDecodeFlagAndWorkspace:167 + 注释同步）。边界遵守：未动 out=NTD 列、cpp:72/:139（TND_Decode 系 4 红行留 T11）。
3. **本波 UT 的真实状态（与任务验收模型的偏差，必须记录）**：HEAD 起全部 11 红都是 **result-gate 失败**（tilingRet=4294967295 GRAPH_FAILED vs SUCCESS）——T1/T4/T5 门（parser/checker/IsCapable 全部要求 out==q）在键生成**之前**就把 out=NTD 行拒掉，历史"KEY-1 漂移"被门掩盖、不以 key 失败形式暴露。所以刷新 key 期望**不会**让这些行当场变绿（T11 翻 out=NTD→TND + 形状才绿）——但**漂移本身已消除**：用临时 out=TND+shape 探针实测（改后 CSV 单行翻转，跑完还原），TND_Prefill 在 result gate ✓、**key 检查 ✓ 3672322**、**workspace 检查 ✓ 34078720**，只剩 tiling-data hash 不同（5fe56a7f533a0c90 vs c50bf2709568e6cf——out 布局运行时字段变化导致，T11 需重采 hash）。验收口径：漂移=0，剩余红全部 out=NTD gate（T11 scope）。
4. **T7 SEL 扩面 + 工具链契约**：SEL InOutLayoutType 扩 {0,1,2}、KvLayoutType 扩 {1,2,3}（3×3×2×2=36×2 dtype=72 实例）；**DECL 块/位宽 token/PARSE_PARAMS 零改动**——token 面验证法（grep -oE "ASCENDC_TPL_[A-Z0-9_]+" | sort | uniq -c，改动前后逐 token 相同）通过；KvLayoutType_PA_BBND(1)/PA_BNBD(2) define 补在头文件 :205-207（DECL 外，允许）。
5. **T7 暴露的 kernel 编译缺口（T8 ① 报错即修，并入本提交）**：SEL 扩到 q=BSND/BNSD 后 `FlashMlaSeqLensTool<BY_BATCH,...>` 的 qActSeqLensParser 硬编码 `ActualSeqLensParser<Q_MODE,SEQLEN_T,true>` 无 BY_BATCH,true 特化 → 18 编译错；修复 = type.h 加 `Q_WITH_ZERO_HEAD=(Q_MODE==ACCUM)` + InitQ 双分支 if constexpr（BY_BATCH 走 seqused 3 参 Init）。第二层：cube block `FaGmTensor<...true>` queryGm 对 BNGSD/BSNGD（GM_Q_OUT_BNGSD 是 3 参实现，无 WITH_ZERO_HEAD）落到空 primary → 改 `(LAYOUT==LAYOUT_TND)`（FA 的 IS_TND 同构）；InitQBuffer 形参同步。**教训：SEL 扩面必须连带跑全量 kernel 编译，规格外组合会暴露 format 特化盲区**。
6. **构建/实例证据**：`rm -rf build build_out && bash build_mla.sh` EXIT=0；kernel .o 仍 2 个 hash 名文件（d11cc332/e76bc2b1，按 dtype 合并），但 `.ascend.meta.*` 段 36 unique keys × aic/aiv = 72 实例/.o；旧 4 key meta 段都在（1575170/1837314/3672322/3934466）。key 探针（/tmp/tiling_key_probe，FastEncodeTilingKeyDirect）：改前 4 值 PASS；改后 4 旧值不变 + 36 组合与公式 `InOutLayoutType|9<<8|mask<<18|kv<<19|isFd<<21` 全等。
7. **提交面**：6 文件（tiling.cpp 键映射 + template_tiling_key.h SEL/注释 + type.h/block_cube_noquant_mla.h 编译修复 + UT csv/cpp 漂移重固化）；不 stage docs/zh/flash_mla_with_kvcache.md（授权改写已存在）、.omo/、meta.txt。

## [2026-08-23] T8: PA_BBND 路径只读核验（已存在，非补齐）— 结论：5/5 CONFIRMED，零代码改动

核验对象：KvLayoutType=1 (PA_BBND→PA_BnBsND) kernel KV-load 路径端到端存在性。逐链读码 + 构建 meta JSON 实例证据：

1. **CONFIRMED** `flash_mla_with_kvcache_gm_format.h` GetKVGmFormat(:59-60)：`KvLayoutType == 1 → return GmFormat::PA_BnBsND`（:62 `KvLayoutType == 2 → GmFormat::PA_BnNBsD`；:63 else → PA_NZ）。kvcache KV format 派生完整。
2. **CONFIRMED** `offset_calculator_v2.h` GmLayoutParams(:101-109)：`PA_BnBsND` 与 `PA_BnNBsD` 均 `CATEGORY = FormatCategory::GM_KV_PA_BNBD` —— **同一既有类别**，无 GM_KV_PA_BBND 类别（正确，不必新增）。
3. **CONFIRMED** `flash_mla_with_kvcache_block_cube_noquant_mla.h` InitKVBuffer(:176-178)：首个分支 `if constexpr (GM_KV_PA_BNBD)` 调用 `offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm, maxBlockNumPerBatch, bnStride, n2Stride)`，PA_BnBsND 经类别分派落入此分支 → **PA_BBND 已被覆盖**（按 CATEGORY 分派而非按 FORMAT，天然涵盖）。
4. **CONFIRMED** `gm_layout.h` GmLayout<PA_BnBsND> MakeLayout(:245-262)：shape=[n,blockSize,d]，`dStride=1, nStride=d, bsStride=N*d, bnStride=Bs*N*d`（bn2stride 可选覆盖），row-major [Bn,Bs,N,D]；host `flash_mla_with_kvcache_tiling_info_parser.cpp` GetKvIsContiguous(:351-359) PA_BBND 分支 `keyBnStride_=stride0, keyN2Stride_=stride2` 与 N 为中间维度一致。
5. **CONFIRMED** 实例证据：`build/binary/ascend950/bin/flash_mla_with_kvcache/FlashMlaWithKvcache_{d11cc332,e76bc2b1}*.json`（2 文件=2 dtype，kernelList len=36 各）：kv<<19 字段解码 kv=1(=PA_BBND) 12 key / kv=2(=PA_BNBD) 12 key / kv=3(=PA_NZ) 12 key。PA_BBND 证据：526592(=9<<8|1<<19)、2623744、788736；PA_BNBD 证据：1050880(=9<<8|2<<19)、3148032、1313024。decode 526592: kv=1 layout9 mask0 isFd0 ✓；1050880: kv=2 layout9 mask0 isFd0 ✓。4 个旧 legacy key（1575170/1837314/3672322/3934466）均在 kv=3 组中保留。

**结论：PA_BBND kernel KV-load 路径端到端存在（format 派生→category 归类→InitKVBuffer 分派→宿主 stride 一致→36 实例 meta 含 12 个 PA_BBND key）。未发现缺陷；未增加任何 GM_KV_PA_BBND 类别/分支；T8-①编译修复已于提交 5e14d553a 完成。Zero diff（相对本波起点，仅 learnings.md 追加）。**

## [2026-08-23] T10: infershape out==q 校验收紧 + UT csv 迁移（改 2 文件，未跑 UT）

1. **infershape.cpp 两处改动**：
   - 新增 layout_out==layout_q 严格一致检查（:145-151）：紧跟 q 布局分支之后、last-dim 检查之前，"fail fast"；用文件既有宏 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_out", ..., "The value of layout_out must be equal to layout_q")`，返回 GRAPH_FAILED。放前面意味着 head_dim_v/last-dim 之类的行即使 layout 不一致也会先因布局失败——对 FAILED 断言无影响。
   - 删除 NTD 输出分支（原 :184-189）+ 错误信息 "BSND/BNSD/TND/NTD"→"BSND/BNSD/TND"（现 :192-196）。`grep -n '"NTD"'` 零命中（唯一残留是 :145 注释里的"NTD"字样，非引号串，满足验收）。
2. **csv 迁移逻辑（关键决策）**：现有 FAILED 负例（q_last_dim/k_cache_last_dim/head_dim_v/q_rank/invalid）的 layout_out 也从 NTD→TND 迁移——否则新 out==q 检查会抢在这些行的"本意故障"之前触发，负例名义与实际失败原因错位。**原则：每个 FAILED 行必须因它的 case_name 声明的理由失败，而不是布局副作用**。新加 3 行 out≠q 负例：TND/BNSD、BNSD/TND、BSND/TND（CSV :13-15），含任务要求的两例（q=BNSD/out=TND、q=TND/out=BNSD）。
3. **UT 未跑**：任务明确"op_host UT 是 orchestrator 的 serial gate，prefer reporting 不抢跑"——与本波另一个 worker（T11 pytests 迁移，可能碰 build 目录/共享 UT 产物）并发风险权衡后，选择报告不动 build.sh。infershape 逻辑面：4 正例（TND out 形状 (T,N,512)=64 32 512 与 q 同序）+ 11 负例全部按上述分析裁决正确。

## [2026-08-23 11:30 UTC] T9: torch 层 layout_out 可空化 + out=layout_q + examples 迁移（未提交，等待编排者）

1. **改动面（3 文件）**：`torch_extension/cann_ops_transformer/ops/flash_mla_with_kvcache.py`（schema L82 `str? layout_out=None`、register_fake L129、metadata impl L191、主 impl L287 全部 None 化；L209/L304-306 两处 None→layout_q 回退；fake 分支 L153-165 重写为 out==q）；`ops/csrc/flash_mla_with_kvcache.cpp`（L138-147 NTD 分支删除，仅留 TND/BNSD/BSND）；`attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp`（L17/L25-26 头注释、outShape (T,N,512)、layoutOut "TND"、golden [N][T][D]→[T][N][D] 重索引）。函数名/参数序零改动，仅默认值+回退+分支文本。
2. **python 断言的环境坑（重要，可复用）**：本机装了 cann_ops_transformer 1.0.0 wheel（site-packages，41 ops），而仓内代码在 `torch_extension/cann_ops_transformer/` 且**无 torch_extension/__init__.py**（纯 namespace）。直接 `from torch_extension.cann_ops_transformer.ops... import` 时，模块内 `from cann_ops_transformer.op_builder.builder import ...` 解析到 **wheel 版** builder → 同一 Library("cann_ops_transformer","DEF") 重复 define → `RuntimeError: Tried to register an operator ... multiple times`。**修法：workdir=torch_extension/ 下 `import cann_ops_transformer.ops.flash_mla_with_kvcache`**，使顶层包名解析到本地（本地有 __init__.py 的 regular package）。
3. **torch 2.7.1 断言 API 变化**：`torch.ops.cann_ops_transformer.flash_mla_with_kvcache` 是 OpOverloadPacket，`._schema` 直接取 → `AttributeError: has no overload name '_schema'`（走 __getattr__）；`op.default._schema` 返回 `torch._C.FunctionSchema`，**无 .defaults 属性**（2.7 已移除）；用 `{a.name: a.default_value for a in op.default._schema.arguments}`（Argument.default_value 存在）。另：`import cann_ops_transformer.ops.flash_mla_with_kvcache as m` 会被 `from .ops import *` 的星号绑定遮蔽（m 拿到 torch.ops 包对象）——取子模块用 `sys.modules[...]`。
4. **动态验证假实现（无 NPU 可跑，强烈推荐）**：register_fake 注册的假实现经 meta 张量可直接触发真代码：裸调 layout_q="TND" → out (T,N,512)；BSND → (B,S,N,512)；BNSD → (B,N,S,512)；显式 layout_out="NTD" → RuntimeError "layout_out must equal layout_q (TND), but got NTD"（可读报错链路在 fake 层先于 op checker 拦截）。这是 T9 验收最简单可靠的行为门。
5. **examples g++ -c 编译门命令（K2 机制的 compile-only 变体）**：`g++ -c attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp -o /tmp/x.o -I /home/developer/Ascend/cann-9.1.0/include/aclnn -I /home/developer/Ascend/cann-9.1.0/include -std=c++17` → EXIT=0。需要 **两个** -I：`<ASCEND_HOME>/include/aclnn`（aclnn/opdev/fp16_t.h 等）+ `<ASCEND_HOME>/include`（acl/acl.h，aclnn/ 下没有 acl/ 副本）；`../op_api/...` 与 metadata `op_host/op_api/...` 相对 include 由 g++ 按 TU 目录自行解析，无需额外 -I。compile-only 不需要 -lcust_opapi/-Wl,-rpath（那是链接步骤的事）。
6. **golden 重索引细节**：device 输出 (T,N,D) 展平，golden 内部仍是 [N][T][D]（`g.o[(n*T+t)*D]`），比较前必须重索引 expectO[(t*N+n)*D] = golden.o[(n*T+t)*D]；softmax_lse 保持 [N][T]（wrapper softmaxOutSize={n,t} 不变），lse 比较逻辑零改动。三个用例（fp16/bf16/b8-multisection）共用同一 outShape/golden 路径，一次性迁移。
7. **提交面预期**：仅上述 3 文件 + 本 learnings 追加；工作区里的 infershape.cpp/csv 与 docs/zh/*.md 改动是其他 worker（T10 等）的，本波不碰。验证输出：python 断言（schema default None 主+metadata 双 op）、裸调 TND/BSND/BNSD 各形状、（TND,TND）显式接受、NTD 显式拒绝、g++ -c EXIT=0。

## [2026-08-23] T11: tiling UT 门翻转 + 布局矩阵用例 + pytests NTD 迁移（提交 5ce82caa2）

### Part A/B — tiling csv/cpp（UT 54/54 全绿，EXIT=0）

1. **csv 翻转面**：8 条旧 SUCCESS（TND_Prefill×4/TND_Decode×3/MetadataMulti）+ 24 条 FAIL 行的 layout_out NTD→TND、attn_out_shape 由 [N,T,D]→[T,N,D]（按 q_shape dim0/dim1 生成，EmptyTensor 行 [0,8,512]）；OutLayoutInvalid_TND 更名 OutLayoutTND 翻转 SUCCESS（几何与 TND_Prefill 完全一致）；KvLayoutNotPaNz(BBH) 保持 FAIL（"BBH" 不在 layoutMap，parser 拒绝——新旧矩阵均如此）。FAIL 行翻转 out 时 attn_out_shape 也同步翻转，避免"非本意理由失败"。
2. **expectTilingDataHash 真机重采（15 个）**：写占位 CAPTURE_ME → 单跑直接从 gtest 失败输出取实际 FNV-1a hash（tiling_case_executor.cpp:368 的 "Which is:" 第一个 16 位 hex 即产生值）→ 回填 → 二跑全绿。共享几何的行 hash 相同：TND_Prefill/BF16/MetadataMulti/OutLayoutTND=5fe56a7f533a0c90（=T6 探针值）；TND_Prefill_LSE=80f53a96cdb82d75；MaskCausal=ca5c3e4060a395ae；Decode 系 2411a0e06f8c1a09/90f1ebda0528af8a/ff5b82b89156c606；BNSD×PA_NZ=170d732bff3d7368，BNSD×PA_BNBD=PA_BBND=6034258085730e0a（4-D k_cache 只影响 paLayoutType 不进 tiling data），BSND×PA_NZ=50ae6054f590f159，BSND×PA_BNBD=PA_BBND=ac86ab738809977b。
3. **新矩阵 6 正例 + 2 负例全绿 → T1 的 CompareShape 轴表声明 PROVEN**：PA_BNBD [Bn,1,Bs,D]=`4 1 64 576`、PA_BBND [Bn,Bs,N,D]=`4 64 1 576` 的 k_cache 4-D shape 直接通过 CheckKVShapeForPageAttention（fia_tiling_shape.cpp:311-316 GetExpectedShape 按轴表生成期望 shape + CompareShape dim 对齐），**无需 checker 修复**。key 按 `InOutLayoutType|9<<8|mask<<18|kv<<19|1<<21`：BNSD(0)×PA_NZ(3)=3672320、×PA_BNBD(2)=3148032、×PA_BBND(1)=2623744；BSND(1)×PA_NZ=3672321、×PA_BNBD=3148033、×PA_BBND=2623745——与 T8 的 kv=1/2 实例 key 反推一致。负例：q=BNSD/out=TND（parser `layout_out must equal layout_q` 先于 shape 检查拒绝）、layout_kv="TND"（连续 KV 拒绝）。BNSD/BSND 行 cu_seqlens_q 三列全空（seq_len_checker "非 TND 必空"），workspace 全部 34078720（isFd=1 + 全核 numBlocks 与布局无关）。
4. **SeqLensDecodeFlagAndWorkspace 的 delta 断言被 T6 打破**：T6 后 prefill/decode 均 isFd=1 且 numBlocks 相同 → workspace 恒等（34078720），旧 `decode==prefill+expectedDelta` 前提（prefill 无 FD 增量）失效 → 改为组成校验：`prefillWs-prefillDelta == decodeWs-expectedDelta == defaultSysWorkspaceSize` + `decodeWs==prefillWs`（公式保留在测试中）。T6 提交时该测试因 out=NTD gate 红，未暴露此点——T11 先暴露后修复。
5. **cpp 其余**：:72 layout_out "TND"、attn_out StorageShape 翻 (T,N,512)（否则 CheckAttnOutShape 拒）、:139 EXPECT_EQ(outputLayout,3)=KERNEL_LAYOUT_TND、EncodeMlaTilingKey 参数化（inOutLayoutType 默认 2=TND、kvLayoutType 默认 3=PA_NZ，旧调用点零改动）；:32 m0070 注释字节级原样。
6. **csv 格式坑（重复教训，两处）**：C++ loader（csv_case_load_utils.h）按裸逗号 split，**无引号支持**——case_annotation 内不能有逗号（第一次生成 OutLayoutTND 注解带逗号直接列错位报错）；空字段规则 = 三列全空才不实例化（GetTensorGE 判 shape+dtype+format 全空）。

### Part C — pytests NTD 迁移（NPU 上板 10P/4F，与 baseline 逐字节一致）

7. **import 路径核实（关键发现）**：harness 顶层 `from cann_ops_transformer.ops import ...`（npu.py:20），无任何 sys.path/PYTHONPATH 操纵 → 解析到 site-packages 的**已安装 wheel**（1.0.0，torch_extension/dist 08-22 18:14 构建），**完全遮蔽仓内源码**。该 wheel 是 T9 前旧版（layout_out="NTD" 硬默认 + `"NTD" if None` 回退）→ 若直接跑会把 NTD 传给新 gate → 全量失败。**修复 = 从仓重建 wheel + 重装**：`cd torch_extension && python3 setup.py bdist_wheel --dist-dir /tmp/…`（py3-none-any 纯 py+csrc 源码，JIT 编译）→ `pip install --force-reinstall --no-deps <wheel>` → 断言 schema layout_out default=None 通过。**结论：pytests 正确运行方式 = 安装当前仓 torch wheel + 安装 build_out .run 自定义算子包**（两件都要新）。
8. **迁移面**：data.py:16/46、case_loader.py:42/50、npu.py:271、functional_nc.py:20/27、functional_stc.py:18/32 的 "NTD" → "TND"；cpu_golden mla_forward 输出 [N1,T,D]→[T,N1,D]（`outu.permute(1,0,2)` 自 [N1,q_len,D]）；cpu.py CPUBackend.compute 增加 golden 形状断言 `(T,N1,D)`（每 case 生效）；readme.md PA 路线行 out=NTD→TND。`grep -rn '"NTD"'` 零命中；残余 "TND/NZ/NTD（R1）" 为历史注释按任务规则保留。redline_*.py 是空 TestCases 外壳（xlsx 生成占位），无需迁移。
9. **pytests 对比（2 次运行均在 ruff 格式化前后）**：v1（迁移后直接跑）与 v2（pre-commit ruff-format 后重跑）均与 .omo/start-work/before/pytests_result.csv **14 行逐字段一致**：10P/4F，bf16 三条 1.6724%/1.6724%/1.8467% + DECODE_LSE 0.5127%（LSE PASS）为基线已知失败，无新增失败。meta_section/NC 自洽列同 baseline。
10. **pre-commit 注意**：ruff-format 会对**整个文件**重排（包括未改行，如 `float('inf')`→`float("inf")` 引号风格、签名多行化）——属 CI 要求，接受并重跑确认即可（本波 5 个 py 文件被 reflow，逻辑零变化）。
11. **提交面**：10 文件 = tiling csv + tiling cpp + pytests 8 文件（core/backends/{cpu,cpu_golden,npu}.py、core/{case_loader,data}.py、test_cases/{functional_nc,functional_stc}.py、readme.md）；未碰 torch_extension/docs/zh md（他人 T9 遗留工作区）、.omo/、meta.txt。UT 命令 `bash build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20` 54/54 EXIT=0。

## [2026-08-23] T12: PA_BBND/PA_BNBD 上板冒烟实测 — 结论：验证通过（三布局输出位级一致，0 缺陷）

脚本 `/tmp/opencode/t12_pa_layout_smoke.py`（不入仓），Ascend 950PR fp16：BATCH=2、num_heads_q=8、seq/batch=128、blockSize=64、bn=每 batch 2（block_table 线性 0..3）、q/cache 最后维 576、head_dim_v=512、scale=1/sqrt(512)、mask=0、q=TND/out=TND。同一份逻辑 KV（fp16 位相同）经纯索引填三种 paged 布局 → 每布局独立 metadata + 独立 kernel 实例（TND×PA tiling key：PA_NZ=3672322[kv3] / PA_BNBD=3148034[kv2] / PA_BBND=2623746[kv1]）：

| layout_kv | k_cache shape | attn_out shape | maxAbs vs PA_NZ | meanAbs/meanRef vs PA_NZ | 判定 |
|---|---|---|---|---|---|
| PA_NZ | (4,1,36,64,16) | (256,8,512) | — (参照) | — | — |
| PA_BNBD | (4,1,64,576) | (256,8,512) | 0.000e+00 | 0.000e+00 | PASS |
| PA_BBND | (4,64,1,576) | (256,8,512) | 0.000e+00 | 0.000e+00 | PASS |
| PA_BBND vs PA_BNBD 互比 | — | — | 0.000e+00 | 0.000e+00 | PASS |

要点：① **位级一致而非容差内**——三布局输出逐元素全等（三个不同 KvLayoutType kernel 实例跑同一逻辑数据，仅 KV-load stride 推导不同，结果 0 差异 → stride 语义被实测坐实）；② 填数正确性自证：三布局 decode 回逻辑 KV 逐元素全等（含 PA_NZ 的 D1=36×D0=16 分块序：k[bn,0,dg,i,dl]=KV[bi,bsIdx*64+i,dg*16+dl]）；③ 信息性 CPU fp64 golden vs PA_NZ maxAbs=2.6e-3/meanAbs=3.7e-4（fp16 I/O 正常量级，参照本身数值合理）；④ 判定标准写死在脚本（maxAbs<1e-2 且均值比<1e-2），实测远优；⑤ 运行环境：`source /home/developer/Ascend/custom_opp/vendors/custom_transformer/bin/set_env.bash && source /home/developer/Ascend/cann-9.1.0/set_env.sh`（不 source custom opp 的 set_env.bash → aclnnFlashMlaWithKvcacheMetadata not in libopapi.so，libcust_opapi.so 在 custom_opp op_api/lib，由该脚本注入 LD_LIBRARY_PATH + ASCEND_CUSTOM_OPP_PATH）。

**stride 语义书面结论（T8 读码 → T12 实测确认）**：PA_BBND（GmFormat::PA_BnBsND）为 [Bn,Bs,N,D] row-major（dStride=1, nStride=D, bsStride=N*D, bnStride=Bs*N*D，gm_layout.h:245-262），经 GM_KV_PA_BNBD 类目与 host GetKvIsContiguous(PA_BBND 分支) 一致——实测三布局输出与 PA_NZ 参照位级全等，**验证通过，无真实缺陷，kernel 零改动**（MUST NOT 守约：未跑任何 build，未 git 写操作，脚本仅 /tmp/opencode）。
