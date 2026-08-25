---
slug: mla-kvcache-checkers-rewrite
status: plan-finalized
intent: clear
review_required: false
pending-action: ✅ 用户 m0050 批准 → `.omo/plans/mla-kvcache-checkers-rewrite.md` 已生成（Todos 1-13 + F1-F4 完整）→ Metis 缺口评审已完成（13 项发现全部折叠，见下文"Metis 折叠记录"）→ 待用户选择：直接 start-work 或先跑高精度评审（review_required=false，默认直接开工）
approach: 以 flash_attn/flash_attn_metadata 的拦截组织为基准，把 attention/flash_mla_with_kvcache（op_host/checkers/*）与 attention/flash_mla_with_kvcache_metadata（flash_mla_with_kvcache_metadata_check.h）的参数校验按新约束矩阵重写：layout_q∈{TND,BNSD,BSND}、layout_kv∈{PA_NZ,PA_BBND,PA_BNBD}（仅PA分页）、layout_out=layout_q（不支持转置）、mask∈{0,3}（与 flash_attn 约束一致）；q/k_cache D=576、head_dim_v=512、num_heads_kv=1 等 MLA 硬约束保留。
---

# Draft: mla-kvcache-checkers-rewrite

## 说明（环境偏差）
- 本环境无 shell 工具，无法运行 `scaffold-plan.mjs`；按技能模板手建 draft/plan 骨架，模板头与脚本输出一致。

## Intent verdict
- intent: **CLEAR**，review_required: **false**（用户给出明确目标与新约束矩阵，未要求高精度评审）
- 分类：Standard（跨 2 个算子包 + parser/tiling 门 + UT，约 10-15 文件；非架构级）

## 用户约束矩阵（来自本会话文档改写，唯一权威）
1. layout_q ∈ {TND, BNSD, BSND}
2. layout_kv ∈ {PA_NZ("NZ"), PA_BBND, PA_BNBD} —— **仅 Paged Attention 分页布局，不支持连续 KV**
3. layout_out 必须与 layout_q 一致，**不支持转置**（NTD 不合法）
4. mask_mode 仅 {0, 3}；具体约束与 flash_attn 一致：mask_mode=0 → attn_mask 必须为空；mask_mode=3 → attn_mask 必须传入（2048×2048 下三角、对角线全0）
5. 保留既有硬约束：q/k_cache 最后维 D=576（nope 512 + rope 64）；head_dim_v=512；num_heads_kv=1；softmax_lse shape、metadata 容量公式等

## Findings（已核实，path:lines）
- **checker 组织已与 flash_attn 同构**（上一计划 mla-align-flash-attn 已对齐）：Composite + 4 阶段（SinglePara/Existence/Feature/MultiPara），7 文件 6 checker（Common/ActualSeqLen/PagedAttention/Mask/Metadata/SoftmaxLSE），调用点 op_host/flash_mla_with_kvcache_tiling.cpp:60-63（Parse→Checker→DoTilingImpl）
- **flash_attn 拦截参考（探索报告 bg_f9af3ff4）**：layQ∈{BNSD,BSND,TND}、layKv∈{BNSD,BSND,TND,PA_BBND,PA_BNBD,PA_NZ}、layOut∈{BNSD,BSND,TND}，LAYOUT_CONSTRAINT_TABLE 3 行；mask_mode∈{0,3,4}+win_left/right 规则；PA_BBND 非连续 dim0、PA_BNBD/PA_NZ dim0|dim1；flash_attn_metadata 用 header-only CheckBaseAttr/CheckMask/CheckExistency/CheckConsistency（CHECK_COND 宏），aicpu 内核二段内容校验（单调性等）
- **当前 MLA 拦截=旧 R1 路由**：
  - common_checker.cpp:57-61 supportedQLayouts={TND}、supportedKvLayouts={NZ,PA_NZ}、supportedOutLayouts={NTD}；LAYOUT_CONSTRAINT_TABLE:320 仅 TND 一行
  - parser flash_mla_with_kvcache_tiling_info_parser.cpp:494-502 GetInAndOutLayout 拒绝非 TND/NTD/NZ（**在 checker 之前**：Parse 顺序 GetInAndOutLayout(708)→ParseAxisInfo(718)）
  - arch35 tiling:96-104 IsCapableSparseLayoutCheckMla 要求 qLayout==TND、outLayout==NTD、kvLayout==NZ；UpdateTilingKeyLayout:289 硬编码 InOutLayoutType_TND_TND
  - template_tiling_key.h:209-249 ASCENDC_TPL_SEL 只实例化 {TND_TND, Config9, mask 2, PA_NZ, isFd 2}={2|9|m|3|f} 4 组合×2 dtype；注释明言"不实例化 BSND/BNSD query、非-NZ KV、非-NTD 输出"
  - infershape（flash_mla_with_kvcache_infershape.cpp:109-192）已接受 BSND/BNSD/TND q 与 BSND/BNSD/TND/NTD out（较宽），OUT layout "NTD" 分支:184-185 已存在（head-major）
  - metadata check（flash_mla_with_kvcache_metadata_check.h:111-116）CheckBaseAttr 限制 layoutQ=="TND"、layoutKv∈{NZ,PA_NZ}、layoutOut=="NTD"；mask check {0,3} 已到位（:121-130）；KV==1、headDim==512 已到位
  - MLA 的 MaskChecker 已与 flash_attn 语义一致（{0,3}，mask_mode=0→attrs_mask 空、=3→必传）——mask 部分**无需改动**
- **layoutMap/枚举已全量存在**（tiling_info.h:110-117 含 BSND/BNSD/TND/NZ/NTD/PA_BBND/PA_BNBD/PA_NZ；FiaLayout 复用）
- **kernel 侧对新布局的支持（探索报告 bg_4fb16d52，进行中）**：
  - cube 读 q 走 GetQueryGmFormat<qLayout>（BNSD→GM_Q_OUT_BNGSD 类别，代码路径存在）
  - **cube InitKVBuffer 只处理 GM_KV_PA_BNBD/GM_KV_PA_NZ/GM_KV_BNSD/GM_KV_TND，不含 GM_KV_PA_BBND**（PA_BBND 无 kernel KV 加载路径 → 运行期崩溃）
  - paLayoutType 在 MLA kernel 中仅写入不消费（死字段）
  - kernelOutputLayout 为运行时字段，vec CopyAttentionOut 已按 BSH/BNSD/TND/NTD/NBSD 运行时 switch
  - template SEL 只实例化 TND_TND → BNSD/BSND/PA_BBND/PA_BNBD **无编译产物实例**
- **测试**：tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.csv 现有 11 个 FAIL 用例（KvLayoutNotPaNz(BBH)、OutLayoutInvalid_TND、HeadDimVInvalid_256、QLastDimInvalid_512 等）——**其中 OutLayoutInvalid_TND（q=TND/out=TND）在新矩阵下变为合法**；infershape csv 有 layout_q/layout_out INVALID 用例

## Open owner-decision（探索无法定论；简报后征求）
- **Q1 范围**：新布局（BNSD/BSND q、PA_BBND/PA_BNBD kv、out=q）下游（parser 门 + arch35 tiling IsCapable/UpdateTilingKeyLayout + template SEL 实例化面 + kernel cube KV 加载）**当前不支持**。若只重写拦截层，新布局会通过 checker 但在 parser/tiling-key 阶段被拒（行为=仍拒绝、拒绝点从 checker 移到 parser/调度）。两条路：
  - A. **仅拦截层对齐**：重写 checker + metadata check.h 的新矩阵 + parser 门同步放宽？(若 parser 不放宽则 checker 改动不可达) —— 需明确 parser 门是否属于"拦截"
  - B. **全链路布局支持**：checker + parser 门 + arch35 tiling 门 + SEL 实例化面 + kernel（含 PA_BBND cube 加载补齐）——含 kernel 功能改动，工程量显著
  默认倾向：**B 中的"checker+parser+arch35 tiling 门+infershape 输出形状"**（把"拦截/门"统一到 flash_attn 的程度），kernel/SEL 属于"功能实现"建议单独立计划或由 FIA 团队跟进（在交付说明中如实标注"新布局 checker 放行但 kernel 实例缺失"的风险）——但这是 owner 决策，需用户拍板。

## Decisions（已定）
- D1 mask 部分不改：MLA MaskChecker 已与 flash_attn 语义一致；metadata check mask {0,3} 已就位
- D2 MLA 独有硬约束保留：kv==1、head_dim_v==512、D==576、q/cache 最后维 576（这些不是要放宽的项）
- D3 flash_attn 的 D∈{64,128,256} 对 MLA 不适用（MLA 恒 576/512），照 flash_attn 组织风格但值集按 MLA
- D4 非连续 KV 约束按 flash_attn：PA_BBND 仅 dim0；PA_BNBD/PA_NZ dim0|dim1（mla paged_attention_checker 已部分实现，需扩 PA_BBND/PA_BNBD）

## Components (topology ledger)
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- C1 | flash_mla_with_kvcache/op_host/checkers/* 按新矩阵重写（CommonChecker 布局集合+约束表、PagedAttentionChecker PA 布局规则、seq_len_checker cu_seqlens 规则核对） | active | attention/flash_mla_with_kvcache/op_host/checkers/
- C2 | flash_mla_with_kvcache_metadata_check.h 布局门放宽（layoutQ/Kv/Out） | active | attention/flash_mla_with_kvcache_metadata/op_host/flash_mla_with_kvcache_metadata_check.h:90-119
- C3 | parser 门 GetInAndOutLayout 放宽（先于 checker 执行，不改则新布局不可达） | active | tiling_info_parser.cpp:463-508,708-718
- C4 | arch35 tiling 门 IsCapableSparseLayoutCheckMla + UpdateTilingKeyLayout/UpdateTilingKeyInfo 键生成映射 | active | arch35/flash_mla_with_kvcache_tiling.cpp:62-106,286-315
- C5 | template SEL 实例化面扩展（新 key 组合需有编译实例，否则 asc_opc 拒绝） | active | template_tiling_key.h:209-249
- C6 | UT 翻转/新增（tiling.csv FAIL 门用例 OutLayoutInvalid_TND 等 + 新组合正例） | active | tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.{cpp,csv}
- C7 | runtime 验证（新组合上板/定向用例；PA_BBND/PA_BNBD offset 语义零运行证据） | deferred（取决于 Q1） | 见探索报告 bg_4fb16d52

## Open questions（interview，已答）
- ~~Q1 范围/门链深度~~ → **全门链对齐（含 SEL）**：C1~C6 全部纳入；C7 按硬件可用性
- ~~Q2 测试策略~~ → **tests-after + 门翻转**

## Decisions（已定）
- D1 mask 部分不改：MLA MaskChecker 已与 flash_attn 语义一致；metadata check mask {0,3} 已就位
- D2 MLA 独有硬约束保留：kv==1、head_dim_v==512、D==576、q/cache 最后维 576（不是要放宽的项）
- D3 flash_attn 的 D∈{64,128,256} 对 MLA 不适用（MLA 恒 576/512）：照 flash_attn 组织风格，值集按 MLA
- D4 非连续 KV 约束按 flash_attn：PA_BBND 仅 dim0；PA_BNBD/PA_NZ dim0|dim1
- **D5（Q1）全门链对齐含 SEL**：C1 checker + C2 metadata check + C3 parser 门 + C4 arch35 tiling 门/键生成 + C5 template SEL 实例化面 + C6 UT 翻转/新增 全部纳入本计划
- **D6（Q2）tests-after + 门翻转**：host UT 基建沿用；Old FAIL 门用例按新矩阵翻转；新增布局组合正/负例；无 NPU 如实标注
- **D7 out=NTD 淘汰连锁**：新矩阵 out 必须=layout_q → 现有 TND+NTD 的 examples/pytests/UT 正例、infershape 的 NTD out 分支、outputLayout(KERNEL_LAYOUT_NTD) 路由全部迁到 out=TND；torch schema 默认值不动（以校验强制，见已改文档）
- **D8 SEL 实例化面规模**：InOutLayoutType{0 BNSD_BNSD, 1 BSH_BSH, 2 TND_TND} × KvLayoutType{1 PA_BBND(键名 PA_BBH), 2 PA_BNBD, 3 PA_NZ} × mask{0,1} × isFd{0,1} × Config9 = 36 组合 × 2 dtype = 72 kernel 实例（当前 8）；out=same-as-q 映射：BNSD→key0、BSND→key1、TND→key2
- D9 kv 仅 PA_*：KvLayoutType_NO_PA(0) 不实例化（连续 KV 永不受支持，符合"cache 只支持 PA"）
- D10 PA_BBND offset/stride 语义风险（探索标注 MakeLayout 可疑）：计划内置定向 UT+上板冒烟，失败如实记录结论，不静默放行

## Dirty worktree
- `torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md`：本会话已改写（新约束矩阵文档）——跟踪文件已修改，勿被 worker 覆盖；相关但非本计划产物

## Approval gate
status: ✅ **approved**（用户 m0050 "写计划"）→ plan 已定稿
approach: 全门链对齐改写（D5~D10）；示例/UT 的 TND+NTD 迁移为 out=TND；SEL 扩到 36 组合×2 dtype；PA offset 定向验证
next-workflow-action: ✅ `.omo/plans/mla-kvcache-checkers-rewrite.md` 已生成（Todos 1-13 + F1-F4）→ Metis 缺口评审完成（13 项全折叠）→ 待用户选择开工或高精度评审

## Metis 缺口评审折叠记录（m0055，13 项发现）
- **BLOCKER-1 7F 漂移** → T6 增加"修复 prefill tilingKey 漂移"（isFd 派生对齐 FA 或按 host 实测重固化 CSV）；成功标准 #3 增加"漂移已清零"
- **BLOCKER-2 torch D7 对 BNSD/BSND 裸调用是坏的** → T9 改为 layout_out 全链路可空化（schema L82/register_fake L129/两 impl 默认全 None）+ L220 回退改 `layout_out = layout_q if None else`；枚举 L153-173 分支、L296-298 主 impl 默认、`ops/csrc/flash_mla_with_kvcache.cpp:99,138-144` wrapper 分支；QA 加裸调用 TND/BNSD 断言
- **BLOCKER-3 Scope OUT"不改签名"与 T9 矛盾** → OUT 行改为"签名形态不变，仅默认值与 impl 回退可改"
- **MAJOR-4 outputLayout 锚点在 parser 不在 arch35** → 验证策略/T6 改为"T6 只改键映射；out=TND 翻转随 T4 抬门自动生效（GenerateLayoutInfo 已 TND→3）"
- **MAJOR-5 波次间红是预期** → 验证策略显式声明 W1-W3 允许红，全绿判据在 T13/F1-F4；T1 验收改为"不要求边界全绿"
- **MAJOR-6 PA_BBND 放行但不可运行 vs 成功标准#1** → T8 扩展为"kernel 兼容编译 + PA_BBND KV 加载路径补齐（InitKVBuffer 加 GM_KV_PA_BBND 分支 + gm_layout.h PA_BnBsND 按需修正）"；成功标准 #1 收严"必须可运行或如实标注"
- **MAJOR-7 m0070 静态门误报** → 静态门改为 `git diff -U0` 只查**新增**残留（既有 3 处 m0070 MUST NOT 清理，沿用 prior 裁决）
- **MAJOR-8 kvcake 拼写 + doc 路径错** → 全计划 s/kvcake/kvcache/；doc 路径改 `torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md`
- **MAJOR-9 T9 QA 不可执行** → 验收改为具体 `python3 -c` schema 断言 + 裸调用断言 + K2 双算子联合编译
- **MINOR-10 expectTilingDataHash 无降级路径** → 降级条款加"hash 单元格标『需真机重采』跳过断言"
- **MINOR-11 learnings 无路径** → 验证策略落点名 `.omo/notepads/mla-align-flash-attn/learnings.md`
- **MINOR-12 pytests NTD 面清单不全** → T11 枚举 8 个文件 + golden 后端形状逻辑验证
- **MINOR-13 T9 需双算子联合编译** → 并入 MINOR-9
- **D7 修正（Metis #2 联动）**：draft 原 D7"torch schema 默认值不动"→ **改为 "torch layout_out 默认值 None + q 跟随回退"**（doc 原型默认 NTD 属文档展示层，不改已写好的 doc；代码层以可空默认与 q 跟随为保证裸调用合法的最小改法）

## 高精度评审记录（m0069-m0071，用户 m0064 选择先评审；mommus bg_184d555f + oracle bg_fe57dad5 双轨，结论均已独立复核折叠）
- **ORACLE-BLOCKER：T8 前提有误 — PA_BBND kernel 路径已存在**。复核证据：`GetKVGmFormat`（flash_mla_with_kvcache_gm_format.h:44-66）KvLayoutType 1→`GmFormat::PA_BnBsND`；`GmLayoutParams<PA_BnBsND>::CATEGORY == GM_KV_PA_BNBD`（offset_calculator_v2.h:102-104）；InitKVBuffer 首个分支 `if constexpr (...==GM_KV_PA_BNBD)`（block_cube_noquant_mla.h:172-174）按 CATEGORY 分发命中；PA_BnBsND GmLayout（gm_layout.h:245-263）stride 推导 [Bn,Bs,N,D] row-major 正确；host `GetKvIsContiguous`（tiling_info_parser.cpp:351-359）已含 PA_BBND stride(2) 分支。**探索报告 bg_4fb16d52 的"无 GM_KV_PA_BBND 分支→运行期崩溃"为误读（分支按 CATEGORY 非 FORMAT）**——T8 改为读码核验，不新增 FormatCategory；仅 T12 实测失败才修。折叠进：IN-8、OUT common/ 行、验证策略约束行、T8、T12、Success#1、TL;DR、approach。
- **MOMUS-MAJOR：T1 漏 CheckKVShape**。复核证据：common_checker.cpp:397-414 现硬拒非 PA_NZ（"only PA_NZ KV"）——新矩阵 PA_BBND/PA_BNBD 正例会在 checker 级被截。折叠：T1 动作/验收新增 CheckKVShape 的 4D layout 分支（[Bn,Bs,N,D]/[Bn,N,Bs,D]，无 D0 split，D=576）。
- **MOMUS-MINOR：tiling UT cpp 迁移锚点**。复核：test_flash_mla_with_kvcache_tiling.cpp:72 layout_out="NTD"、:139 EXPECT_EQ(outputLayout,5)（KERNEL_LAYOUT_NTD）。折叠：T11 显式写"迁移后 outputLayout 5→3"。
- **MOMUS-MINOR：'BBH' 行措辞**。复核：tiling_info.h:110-117 layoutMap 仅 8 布局，"BBH" 字符串不合法（新旧矩阵均 parser 拒）→ 保持 FAIL 正确；与 KvLayoutType_PA_BBH 枚举名（PA_BBND 键值 1，会实例化）区分。折叠：T11 措辞锐化。
- **ORACLE-MINOR：InOutLayoutType 枚举防呆**。MLA 枚举（0=BNSD_BNSD/1=BSH_BSH/2=TND_TND）与 flash_attn（0=BSND/1=BNSD/2=TND）编号不同。折叠：T6 补"取 MLA 自身常量，勿照抄 FA 数值"。
- **其余全部锚点两份评审均验证通过**：parser 门 463-508/611-640、common_checker 52-87/318-345/361/416、paged_attention 42-82/105-144、metadata check 90-130/186-190、arch35 70-106/286-315、SEL 242-250、entry static_assert 111-142/199-206、KERNEL_LAYOUT 值（BSH=0/BNSD=1/TND=3/NTD=5）、torch py L75/82/129/202/220/296-298、csrc wrapper 99/138-144、examples 451/479-481/597、infershape 109-194、pytests 8 文件行号、key 公式与 4 值核算（1575170=2|9<<8|3<<19；3672322=+1<<21；1837314/3934466=+mask）、seq_len_checker 已含 TND/非 TND 双分支（休眠、措辞即可）、paLayoutType 死字段确认。
- **评审后状态**：review_required=false 路径下双轨评审已按用户指示完成，无残留 blocker；计划已更新，可交付开工。