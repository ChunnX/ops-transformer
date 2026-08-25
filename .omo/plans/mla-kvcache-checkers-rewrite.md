---
slug: mla-kvcache-checkers-rewrite
status: plan
intent: clear
review_required: false
predecessors: [.omo/drafts/mla-kvcache-checkers-rewrite.md]
approach: 以 flash_attn/flash_attn_metadata 的拦截组织为基准，对 attention/flash_mla_with_kvcache（op_host/checkers/*、parser 门、arch35 tiling 门、template SEL）与 attention/flash_mla_with_kvcache_metadata（flash_mla_with_kvcache_metadata_check.h）做**全门链一致性改写**：layout_q∈{TND,BNSD,BSND}、layout_kv∈{PA_NZ,PA_BBND,PA_BNBD}（仅 PA 分页）、layout_out=layout_q（不支持转置）、mask∈{0,3}（现状已一致，不改）；SEL 实例化面 8→72；示例/UT/torch 的 TND+NTD 迁移为 out=layout_q（PA_BBND kernel 路径经高精度评审确认已存在，仅核验）；tests-after + 门翻转；修复既有 host UT 7F 漂移基线。
---

# mla-kvcache-checkers-rewrite - Work Plan

## TL;DR (For humans)

把 `flash_mla_with_kvcache` 与 `flash_mla_with_kvcache_metadata` 的"拦截"（参数校验）从旧的"只支持 q=TND / kv=PA_NZ / out=NTD"改写为文档新矩阵：**q 支持 TND/BNSD/BSND，kv 只支持 Paged Attention 三种布局（PA_NZ/PA_BBND/PA_BNBD），输出必须与 q 同布局（不支持转置），mask 维持 0/3**。照 flash_attn 的方式做**全门链一致**改写：checker（含 metadata check）→ parser 门 → arch35 tiling 门与键生成映射 → kernel 模板实例化面（SEL 从 4 组合扩到 36 组合×2 dtype=72 实例）；伴随后果是旧默认输出 NTD 被淘汰、examples/infershape/UT/torch 迁移到 out=layout_q、PA_BBND 的 kernel 路径经高精度评审确认**已存在**（仅核验+实测，无需补分支），并顺带修复既有 host UT 的 7F tilingKey 漂移基线。MLA 独有硬约束（kv=1、最后维 576、head_dim_v=512）与 mask 语义保持不动；flash_attn/common/docs 零改动。

## Scope

**IN（要做）**
1. C1 `op_host/checkers/common_checker.{h,cpp}`：布局集合与 LAYOUT_CONSTRAINT_TABLE 按新矩阵重写（照 flash_attn `flash_attn/op_host/checkers/common_checker.cpp:43-45,265-345` 组织），新增 BNSD/BSND 的 shape 校验路径；MLA 硬约束（kv==1、D=576、head_dim_v=512、rope 64、q/cache 最后维 576）原样保留。
2. C1 `op_host/checkers/paged_attention_checker.{h,cpp}`：PA_BBND/PA_BNBD 非连续维规则（PA_BBND→仅 dim0；PA_BNBD/PA_NZ→dim0|dim1，照 flash_attn paged_attention_checker）。
3. C1 `op_host/checkers/seq_len_checker.{h,cpp}`：核验 cu_seqlens_q 存在性规则（TND→必传 / 非 TND→必空）在新矩阵下仍正确；不改语义。
4. C2 `attention/flash_mla_with_kvcache_metadata/op_host/flash_mla_with_kvcache_metadata_check.h`：CheckBaseAttr 布局门放宽（layoutQ∈{TND,BNSD,BSND}、layoutKv∈{PA_NZ,NZ,PA_BBND,PA_BNBD}、layoutOut==layoutQ）；mask {0,3}、kv==1、headDim==512 保留；CheckExistency 的 cu_seqlens_q 规则同步（layoutQ==TND 必传 / 非 TND 禁传）。
5. C3 `op_host/flash_mla_with_kvcache_tiling_info_parser.cpp`：`GetInAndOutLayout`（463-508）去掉硬编码门，改为 layoutMap 校验 + 新矩阵裁决（q∈{TND,BNSD,BSND}、kv∈{PA_NZ,NZ,PA_BBND,PA_BNBD}、out==q；PA_NZ→NZ 归一保留）。
6. C4 `op_host/arch35/flash_mla_with_kvcache_tiling.cpp`：`IsCapableSparseLayoutCheckMla`（70-106）放宽到新矩阵；`UpdateTilingKeyLayout`（286-290）/`UpdateTilingKeyInfo`（292-300）键生成改为按 faInfo_->qLayout/kvLayout 映射（照 `flash_attn/op_host/arch35/flash_attn_tiling.cpp:153-177`）：TND→InOutLayoutType_TND_TND(2)、BSND→BSH_BSH(1)、BNSD→BNSD_BNSD(0)；PA_NZ→KvLayoutType 3、PA_BNBD→2、PA_BBND→1；**执行时一并修复 prefill tilingKey 漂移（见 T6 验收）**。
7. C5 `op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h`：`ASCENDC_TPL_SEL`（242-250）扩到 InOutLayoutType{0,1,2} × KvLayoutType{1,2,3} × HasAttenMask{0,1} × IsFd{0,1}（Config 恒 9）；同步头注释（实例面 8→72、key 公式注释扩到新枚举）；**DECL 块、位宽 token（ASCENDC_TPL_8_BW/10_BW/2_BW/1_BW）、PARSE_PARAMS 一字不动**；entry static_assert（`op_kernel/flash_mla_with_kvcache.cpp:111-142,199-206`）保持（已证实对新增组合为通用等式）；KvLayoutType 0（连续 KV）不实例化。
8. C5+kernel：kernel 兼容与 PA_BBND 路径**核验**（高精度评审纠偏：不存在功能缺口）——T7 后编译错误一律修（修法保持与 SEL 一致的实例面，不得收窄）；**PA_BBND 不新增 FormatCategory/分支**：`GmFormat::PA_BnBsND`（KvLayoutType 1）经 `attention/common/op_kernel/memcopy/offset_calculator_v2.h:102-104`（GmLayoutParams → GM_KV_PA_BNBD）已命中 `InitKVBuffer` 首个 `if constexpr` 分支（`op_kernel/arch35/flash_mla_with_kvcache_block_cube_noquant_mla.h:172-174`），PA_BnBsND GmLayout（gm_layout.h:245-263）stride 推导为 [Bn,Bs,N,D] row-major 正确，host `GetKvIsContiguous`（`op_host/flash_mla_with_kvcache_tiling_info_parser.cpp:351-359`）已含 PA_BBND stride 分支——T8 上仅"读码核验路径完整 + T12 实测确认"，**仅当 T12 实测出现真实失败才改**（改前先探索确认根因，不得凭空加分支）。
9. D7 迁移：torch 层 + examples + infershape——见 T9/T10（含 `ops/csrc/flash_mla_with_kvcache.cpp` wrapper 的 NTD 分支核验）。
10. C6 UT：`tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.{cpp,csv}`（门翻转 + 新增矩阵用例 + 7F 漂移固化）；`tests/ut/op_host/test_flash_mla_with_kvcache_shape_infershape.csv`（out=TND 迁移 + out≠q 负例）。
11. C6 迁移：`attention/flash_mla_with_kvcache/tests/pytests/**` 全部 NTD 面迁移（含 golden 后端形状逻辑）。
12. C7 验证：PA_BBND/PA_BNBD 定向验证（UT 级 shape/stride 断言 + 硬件可用时上板冒烟，T8 补齐后必须"可运行"）。
13. 全量回归终检（见 T13 与 Final verification wave）。

**OUT（Must NOT have）**
- 不改 mask 语义：mask_mode∈{0,3} 及其 attn_mask 存在性规则现状已正确，保持。
- 不改 MLA 硬约束：kv heads==1、head_dim_v==512、q/k_cache 最后维 576、rope 64、metadata 容量公式、softmax_lse 形状规则。
- 不引入连续 KV：KvLayoutType_NO_PA(0) 不实例化、checker/parser 不接受非 PA layout_kv（BNSD/BSND/TND 作为 kv 均拒绝）。
- 不改 tiling key 位编码布局：DECL 顺序、位宽 token、FastEncodeTilingKeyDirect 编码规则不变；仅扩展 SEL 值表（既有组合 key 数值不变：1575170/1837314/3672322/3934466）。
- 不改 `attention/flash_attn/`、`attention/common/`（**除非 T12 实测发现 PA_BBND 真实缺陷才触及 gm_layout.h PA_BnBsND 相关函数；预期零改动**）、`torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md`（已改写完成，勿覆盖）任何文件——只读参考。
- 不改 torch 公共函数签名形态（参数名/位置不变），仅默认值与 impl 回退可改（D7 必需）；`ops/csrc/flash_mla_with_kvcache.cpp` wrapper 仅按需核验/清理 NTD 分支，不改属性序。
- 不做性能/精度调优；上板验证只判定"能跑/数值合理"，不调参。
- 不重命名算子、不改 build 基建（build_mla.sh 已存在）。
- 不改 metadata AICPU kernel（已用 load_balance::ConvertToLayout 通用路径，与布局无关；Metis 已复证）。

## Verification strategy

**每 todo 的 QA 门**（命令沿用已固化的 `.omo/notepads/mla-align-flash-attn/learnings.md`，下称 learnings）：
- 编译门：`rm -rf build build_out`（必要时）后 `bash build_mla.sh`（等价的 `bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata -j20`，**双算子联合包**，learnings K2）退出码 0；**所有编译命令一律 `-j20`**；kernel .o mtime 晚于源文件（防增量假阳性）。
- 回归门：Host UT `bash build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20`；pytests `cd attention/flash_mla_with_kvcache/tests/pytests && python3 <用例脚本> --result_csv /tmp/xxx.csv` 与迁移前 baseline 对比（learnings 已记录：bf16 三条 1.67-1.85% 挂饰为基线现状、DECODE_LSE 0.5127% 贴线；迁移后对比须逐项吻合，**禁止用"基线就挂"掩盖新增失败**）。
- **波次间红是预期**：W1-W3 内各 todo 的回归门允许阶段性红——T1 后既有 out=NTD SUCCESS 行预期变 FAIL（催 T11 迁移）；BNSD/BSND 正例在 T4/T5 抬门后才可绿；**全绿判据只在 W4 结束后的 T13 与 F1-F4**。执行者不得因中途红而回退 W1-W3 改动。
- **key 等价探针（T6/T7/T11 强制）**：复用 learnings 的 /tmp 隔离 g++ 探针法（include `op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h` + FastEncodeTilingKeyDirect），断言：既有 4 组合 key 值改动前后不变（1575170/1837314/3672322/3934466），新组合 key 按 `InOutLayoutType | Config(9)<<8 | HasAttenMask<<18 | KvLayoutType<<19 | IsFd<<21` 公式枚举（TND→2、BSND→1、BNSD→0 作 InOutLayoutType；PA_BBND→1、PA_BNBD→2、PA_NZ→3 作 KvLayoutType）。
- 静态门（最终波）：`git diff -U0 -- <改动文件清单> | grep -nEi "m0070|\bR1\b|todo14|D14"` 零**新增**残留（`tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.cpp:32` 既有 m0070 注释属 MUST NOT 清理项，沿用 prior 裁决）；`git diff --stat` 确认 flash_attn/、docs/zh 所述 doc 零改动。
- **无 CANN 环境降级**：编译门降级为 `bash -n` 静态对照并如实标注"脚本未在真机验证"；`expectTilingDataHash` 单元格标记"需真机重采"并跳过断言（F3 记录为未验证）；**禁止伪造验证结果**。

**关键工程约束**：
- `template_tiling_key.h` 是 host 与 kernel 共享、被 CANN 工具链（ASCENDC_TPL_PRE→template_tiling.py→asc_opc）正则消费的文件：只扩 SEL 值表与注释，DECL 宏体/位宽 token/PARSE_PARAMS 一字不动（learnings T14 经验）。
- T6 与 T7 耦合（键生成映射 ↔ SEL 值表必须一致），联合验证、可同提交。
- **outputLayout 运行时字段锚点在 parser**：`kernelOutputLayout` 由 `op_host/flash_mla_with_kvcache_tiling_info_parser.cpp:611-640 GenerateLayoutInfo` 赋值且已通用（TND→KERNEL_LAYOUT_TND(3)、BSND→BSH(0)、BNSD→1）；`arch35/flash_mla_with_kvcache_tiling.cpp:423` 仅拷贝。故 out=TND 的运行时翻转随 T4 抬门自动生效——**不在 T6 改 outputLayout，T6 只改键映射**；T4 验收附一条核验（parser 确认 TND→3，已是则不改）。
- torch 层 `layout_out` 必须改为可空 + q 跟随（见 T9），否则 BNSD/BSND 裸调用伪造 out="TND" 被新校验拒绝——**不允许固定字符串默认值**。
- PA_BBND 的 kernel 路径**已存在**（KvLayoutType 1→GmFormat::PA_BnBsND→GM_KV_PA_BNBD 类别，InitKVBuffer 首个 if constexpr 命中）——T8 只做读码核验，无功能缺口可补；仅当 T12 实测出现真实失败才动 kernel 代码（先探索根因）。
- 交付说明必须如实标注：上板冒烟是否执行、PA_BBND/PA_BNBD 运行验证结论、无 CANN 时未验证项清单。

## Execution strategy

- **顺序**：T1→T2→T3（拦截面，可并行分批）→ T4→T5→T6（host 门，T6 与 T7 联合波）→ T7→T8（SEL+kernel 兼容+PA_BBND 路径）→ T9→T10→T11（迁移与用例）→ T12→T13（验证）→ F1~F4。
- **只读参考**：`attention/flash_attn/op_host/checkers/common_checker.cpp:43-45,265-345`（约束表组织）、`attention/flash_attn/op_host/checkers/paged_attention_checker.cpp`（PA 规则）、`attention/flash_attn/op_host/arch35/flash_attn_tiling.cpp:40-43,153-177`（IsCapable 恒 true + 键生成映射）、`attention/flash_attn/op_kernel/arch35/flash_attn_template_tiling_key.h:68-73`（SEL 矩阵）、`attention/flash_attn_metadata/op_host/flash_attn_metadata_check.h`（metadata check 组织）。
- **硬约束**：mask 语义/MLA 硬约束零改动；SEL 扩不带动 key 位编码结构变化；所有编译 `-j20`；用例翻转按用户约束矩阵为唯一真值。
- **提交**：见 Commit strategy。

## Todos

### W1 拦截面（checker + metadata check）
- [x] 1. common_checker 布局集合与约束表重写（新矩阵）
    - 参考：`op_host/checkers/common_checker.{h,cpp}`（现 52-87 单元素集合 + 318-345 单行表；CheckQueryShape 361、CheckAttnOutShape 416）；`attention/flash_attn/op_host/checkers/common_checker.cpp:43-45,265-345`
    - 动作：布局集合改为 supportedQLayouts={TND,BNSD,BSND}、supportedKvLayouts={PA_NZ,NZ,PA_BBND,PA_BNBD}、supportedOutLayouts={TND,BNSD,BSND}；`LAYOUT_CONSTRAINT_TABLE` 扩为 3 行（BNSD→kv{PA_NZ,PA_BBND,PA_BNBD} out{BNSD}；BSND→同 kv 集 out{BSND}；TND→同 kv 集 out{TND}——out 必须=q，无转置行）；CheckQueryShape/CheckAttnOutShape 增加 BNSD/BSND 分支（shapeParams.B/N/S/D 按 FiaTilingShapeCompare 语义；qTSize=0 为 BNSD/BSND 正常行为，照 flash_attn）；**CheckKVShape（common_checker.cpp:397-414，现硬拒非 PA_NZ）增加 PA_BBND / PA_BNBD 分支**：4D layout 的 shape 参数（PA_BBND=[Bn,Bs,N,D]、PA_BNBD=[Bn,N,Bs,D]，无 PA_NZ 的 D0 split；最后维 D=576 规则同 PA_NZ），pageAttentionFlag/block_table 存在性检查复用；MLA 硬约束检查（kv==1、qkHeadDim 512、vHeadDim 512、head_dim_v 512+576、最后维 576）**逐行保留不动**；错误信息与注释同步新矩阵
    - 验收：编译通过；TND/PA_NZ 单例 checker 行为不变（对照迁移前）；PA_BBND/PA_BNBD 的 k_cache shape 正/负例在 checker 单元级正确裁决（4D shape、block_table 存在性）；报错信息反映新集合；**不要求 T1 边界 UT 全绿（既有 out=NTD SUCCESS 行必然红，预期）**
    - QA：编译门 + 新矩阵正/负例的 checker 单元断言（直接构造 Checker 单例调用，不依赖 parser 门）
    - 提交：`refactor(flash_mla_with_kvcache): widen layout matrix in common_checker`
- [x] 2. paged_attention_checker 扩展 PA_BBND/PA_BNBD
    - 参考：`op_host/checkers/paged_attention_checker.{h,cpp}`（现 42-82 仅 PA_NZ 语义 + 105-144 非连续规则）；`attention/flash_attn/op_host/checkers/paged_attention_checker.cpp`
    - 动作：非连续规则按 kvLayout 分支：PA_BBND→keyNonContigDim<=0（仅 dim0）；PA_BNBD/PA_NZ→keyNonContigDim<=1（dim0|dim1）；block_size/block_table 规则保持；注释更新（去掉"唯一 PA_NZ"措辞）
    - 验收：编译通过；PA_BBND dim1 非连续 → FAIL、PA_BNBD dim1 非连续 → SUCCESS（若其他约束满足）等规则断言正确（checker 单元级）
    - QA：编译门 + 单元级正/负例
    - 提交：`refactor(flash_mla_with_kvcache): extend PA layout non-contig rules`
- [x] 3. metadata check.h 布局门放宽 + seq_len_checker 核验
    - 参考：`attention/flash_mla_with_kvcache_metadata/op_host/flash_mla_with_kvcache_metadata_check.h`（CheckBaseAttr 90-119 硬编码 TND/NZ/NTD；CheckMask 121-130 已 {0,3}；IsPA 186-190 已含三 PA）；`attention/flash_attn_metadata/op_host/flash_attn_metadata_check.h`；`op_host/checkers/seq_len_checker.cpp`（104-204）
    - 动作：CheckBaseAttr：layoutQ∈{TND,BNSD,BSND}、layoutKv∈{PA_NZ,NZ,PA_BBND,PA_BNBD}、新增 layoutOut==layoutQ 一致性校验；CheckExistency 改"layoutQ==TND 时 cuSeqlensQ 必传 / 非 TND 禁传"（与主算子一致）；mask {0,3}、kv==1、headDim==512 保留；seq_len_checker 逐项核验判定不变（仅措辞）
    - 验收：op_api 编译通过；check.h 无旧 R1 措辞残留（新矩阵集合）；参数序不变
    - QA：编译门（`bash build.sh -u --opapi --ops=flash_mla_with_kvcache_metadata` 或等价双算子 joint 编译）+ 单元级正/负例（非 TND 传 cuSeqlensQ → FAIL）
    - 提交：`refactor(flash_mla_with_kvcache_metadata): widen layout gates in metadata check`

### W2 host 门（parser + arch35 tiling）
- [x] 4. parser GetInAndOutLayout 放宽
    - 参考：`op_host/flash_mla_with_kvcache_tiling_info_parser.cpp:463-508`；`op_host/flash_mla_with_kvcache_tiling_info_parser.cpp:611-640`（GenerateLayoutInfo，outputLayout 已通用，TND→3）
    - 动作：删除 GetInAndOutLayout 内硬编码门；改为 layoutMap 校验（已有 466-491）+ 新矩阵裁决（q∈{TND,BNSD,BSND}、kv∈{PA_NZ,NZ,PA_BBND,PA_BNBD}、out 必须==q）；`PA_NZ→NZ` 归一保留（FiaLayout::NZ 即 kernel 侧 PA_NZ）；**核验 GenerateLayoutInfo：out=TND 时 kernelOutputLayout=3（TND）——已是则不改；BSND→BSH(0)、BNSD→BNSD(1) 已存在**
    - 验收：编译通过；BNSD/BSND q parse 成功并正确填 s1/n1/b/size（qTSize=0 属预期）；非 PA kv（如 "TND" 作 kv）FAIL；out≠q FAIL
    - QA：编译门 + parse 层定向断言（正/负例）
    - 提交：`refactor(flash_mla_with_kvcache): lift parser layout gate`
- [x] 5. arch35 IsCapable 门放宽
    - 参考：`op_host/arch35/flash_mla_with_kvcache_tiling.cpp:62-106`；`attention/flash_attn/op_host/arch35/flash_attn_tiling.cpp:40-43`
    - 动作：`IsCapableSparseLayoutCheckMla` 放宽：qLayout∈{TND,BNSD,BSND}、outLayout==qLayout、kvLayout∈{NZ}(PA_NZ 归一后)∪{PA_BNBD,PA_BBND}；`IsCapableFeatureCheckMla`（PA/MLA 特性）保留；mask/HeadDimV/N2/N1 既有检查保留
    - 验收：编译通过；新布局组合 IsCapable true；旧负例语义保持（head_dim_v 错→false 等）
    - QA：编译门 + IsCapable 定向断言
    - 提交：`refactor(flash_mla_with_kvcache): widen arch35 IsCapable layout gate`
- [x] 6. arch35 键生成映射 + prefill tilingKey 漂移修复（与 T7 联合验证）
    - 参考：`op_host/arch35/flash_mla_with_kvcache_tiling.cpp:286-315`（UpdateTilingKeyLayout/UpdateTilingKeyInfo/GenTilingKey）；`attention/flash_attn/op_host/arch35/flash_attn_tiling.cpp:153-177`；learnings L12/L28（漂移说明）
    - 动作：`UpdateTilingKeyLayout` 删硬编码，按 faInfo_->qLayout 映射：TND→InOutLayoutType_TND_TND(2)、BSND→BSH_BSH(1)、BNSD→BNSD_BNSD(0)；**映射值取 MLA 自身枚举常量（flash_mla_with_kvcache_template_tiling_key.h:119-124：0=BNSD_BNSD、1=BSH_BSH、2=TND_TND），勿照抄 flash_attn 的枚举数值（FA 的 InOutLayoutType 编号不同：0=BSND/1=BNSD/2=TND）**；`UpdateTilingKeyInfo` 按 kvLayout 映射 kvLayoutType：NZ/PA_NZ→3、PA_BNBD→2、PA_BBND→1；**修复既有漂移**：host prefill 路径 tilingKey 恒为 3672322(bit21=1) 而 CSV 固化 1575170(bit21=0)（7F 根因）——将 isFd 派生对齐 flash_attn 参考实现后再跑 host UT；若按参考校准后 host 输出仍为 3672322（即 FA 同款行为），则按 learnings "key 断言以 host 实测为准" 将 CSV/UT 期望刷新为 host 真值（两种路径二选一，验收都是 host UT 无漂移失败）
    - 验收：key 探针断言既有 4 组合 key 值不变（1575170/1837314/3672322/3934466）、新组合按公式枚举；host UT KEY-1 漂移清零（36P/7F → 36P/0F 或按刷新后期望全绿）
    - QA：key 等价探针（强制）+ 编译门 + 回归门（漂移项专项）
    - 提交：与 T7 合并或紧随：`refactor(flash_mla_with_kvcache): map layouts to tiling keys like flash_attn`

### W3 kernel 实例化面（SEL + 兼容 + PA_BBND 路径）
- [x] 7. template SEL 扩到 3 q 键 × 3 PA kv × mask × isFd
    - 参考：`op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h:209-250`；`attention/flash_attn/op_kernel/arch35/flash_attn_template_tiling_key.h:68-73`
    - 动作：`ASCENDC_TPL_SEL` 的 InOutLayoutType 列表扩为 {BNSD_BNSD(0), BSH_BSH(1), TND_TND(2)}、KvLayoutType 列表扩为 {PA_BBH(1), PA_BNBD(2), PA_NZ(3)}；Config 恒 9、HasAttenMask/IsFd 保持 {0,1}（3×3×2×2=36 组合×2 dtype=72 实例）；**DECL 块、位宽 token、PARSE_PARAMS 一字不动**；头注释同步（实例面 8→72、key 公式注释扩到新枚举）；entry static_assert（`op_kernel/flash_mla_with_kvcache.cpp:111-142,199-206`）保持——编译验证其通过
    - 验收：`rm -rf build build_out && bash build_mla.sh` 全量编译 exit 0；kernel .o 产物数按 72 实例对应增长并记录；key 探针既有 4 值不变
    - QA：编译门 + key 等价探针 + 实例面计数记录
    - 提交：`refactor(flash_mla_with_kvcache): instantiate full layout matrix in tiling-key SEL`
- [x] 8. kernel 兼容编译 + PA_BBND 路径核验（已存在，非补齐）
    - 参考：`op_kernel/arch35/flash_mla_with_kvcache_block_cube_noquant_mla.h:164-185`（InitKVBuffer）；`attention/common/op_kernel/memcopy/offset_calculator_v2.h:101-114,569-634`（GM_KV_PA_BNBD 类目）；`attention/common/op_kernel/memcopy/gm_layout.h:245-285`（PA_BnBsND MakeLayout——**仅此文件、仅 PA_BBND 相关函数按需改**）
    - 动作：① T7 后全量编译，报错即修（修法保持与 SEL 一致的实例面，不得收窄）；② **PA_BBND 路径核验（不新增 FormatCategory/分支）**：读码确认 `GetKVGmFormat`（flash_mla_with_kvcache_gm_format.h:44-66，KvLayoutType 1→GmFormat::PA_BnBsND）→ `GmLayoutParams<PA_BnBsND>::CATEGORY==GM_KV_PA_BNBD`（offset_calculator_v2.h:102-104）→ InitKVBuffer 首个 if constexpr 分支（block_cube_noquant_mla.h:172-174）逐环命中；PA_BnBsND MakeLayout（gm_layout.h:245-263）stride 推导与 host GetKvIsContiguous（tiling_info_parser.cpp:351-359）PaBBND 分支一致；③ 仅当 T12 实测出现 PA_BBND 真实失败才改（先探索根因，不凭空加分支）；④ 预期 T8 零 kernel 改动（模板路径已 layout-generic + 类别分发已覆盖）
    - 验收：编译 exit 0 且 72 实例完整；`git diff --stat` 记录实际改动（预期零改动或仅 T7 编译修复）；PA_BBND 路径核验结论书面记录（读码链路 + T12 实测）；PA_BBND 用例（T12 的 ①）在 UT/上板级可运行（数值合理比例对标 PA_NZ 基线）
    - QA：编译门 + T12 定向验证联动 + 实例面计数
    - 提交：并入 T7 提交或独立 `fix(flash_mla_with_kvcache): add PA_BBND KV-load path and compile-fix expanded instantiations`

### W4 迁移与用例（D7 连锁 + UT）
- [x] 9. torch 层 layout_out 可空化 + q 跟随 + examples 迁移
    - 参考：`torch_extension/cann_ops_transformer/ops/flash_mla_with_kvcache.py`（L75 metadata schema `str? layout_out=None` 为现成范式；L82 主算子 schema `str layout_out="NTD"`；L129 register_fake 默认 "NTD"；L202 metadata impl 默认 "NTD"；L220 回退 `layout_out = "NTD" if None`；L296-298 主算子 impl 默认 "NTD" 且**无 None 回退**；L153-173 layout_out TND/NTD 分支）；`torch_extension/cann_ops_transformer/ops/csrc/flash_mla_with_kvcache.cpp`（L99/138-144 TND/NTD/BNSD 分支核验）；`attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp`（L451 outShape {N,T,512}、L479-481 layoutQ="TND" layoutOut="NTD"、L597 golden NTD）
    - 动作：主算子 schema L82 `str layout_out="NTD"` → `str? layout_out=None`；register_fake L129 默认 None；L220 回退改 `layout_out = layout_q if layout_out is None else layout_out`；主算子 impl L296-298 默认 None + 同样回退；L153-173 分支清理/核验（layout_out 不再有 NTD 形态，q 侧校验同步）；cpp wrapper L99/138-144 的 NTD 分支核验清理（属性序不动）；examples layoutOut "NTD"→"TND"、outShape {N,T,512}→{T,N,512}、L597 golden 随迁 `(T,N,D)` 展平
    - 验收：`python3 -c "import torch; from torch_extension.cann_ops_transformer.ops.flash_mla_with_kvcache import flash_mla_with_kvcache; s=flash_mla_with_kvcache._schema; assert s.defaults.get('layout_out') is None"` 通过；裸调用 TND → layout_out=TND、裸调用 BNSD → layout_out=BNSD（构造断言/单元级）；显式 layout_out="NTD" → 新增 op checker 校验拒绝（报错信息可读）；examples g++ -c 编译通过（learnings K2 双算子联合包）
    - QA：python 断言 + 裸调用断言 + 编译门（examples 编译-only + 双算子联合包）
    - 提交：`refactor(torch_extension): make output layout follow query layout`
- [x] 10. infershape out 校验收紧 + UT
    - 参考：`op_host/flash_mla_with_kvcache_infershape.cpp:109-194`（现接受 q∈{BSND,BNSD,TND}、out∈{BSND,BNSD,TND,NTD}，NTD 分支 184-185）；`tests/ut/op_host/test_flash_mla_with_kvcache_shape_infershape.csv`
    - 动作：infershape 增加 layout_out==layout_q 一致性检查；删除 NTD 输出分支（q 无 NTD，一律拒绝 NTD out）；infershape csv 现有 NTD out 正例改 TND；新增 out≠q 负例；`kernelOutputLayout` 对 out=TND 产生 (T,N,D)
    - 验收：infershape UT csv 全绿；`grep -n '"NTD"' op_host/flash_mla_with_kvcache_infershape.cpp` 零命中（注释除外）
    - QA：回归门（infershape UT）
    - 提交：`refactor(flash_mla_with_kvcache): enforce out layout == query layout in infershape`
- [x] 11. tiling UT 门翻转与新增 + pytests NTD 面全量迁移
    - 参考：`tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.{cpp,csv}`（现 25 行：FAIL 门用例两行 KvLayoutNotPaNz(BBH) 与 OutLayoutInvalid_TND、key 期望 1575170 系；`test_flash_mla_with_kvcache_tiling.cpp:32` 既有 m0070 注释 MUST NOT 删除）；`attention/flash_mla_with_kvcache/tests/pytests/**`（NTD 面清单：core/backends/npu.py:271、core/case_loader.py:42,50、core/data.py:16,46、test_cases/functional_nc.py:20,27、test_cases/functional_stc.py:18,32、cpu_golden.py/cpu.py/gpu.py/gpu_impl.py 的 NTD 形状逻辑）
    - 动作：tiling.csv——`OutLayoutInvalid_TND` 翻转为 SUCCESS（q=TND/out=TND，key 仍 1575170；tiling data hash 若变化需真机重采，无 CANN 时标"需重采"跳过）；`KvLayoutNotPaNz(BBH)` **保持 FAIL**（"BBH" 字符串不在 layoutMap（tiling_info.h:110-117 仅 8 布局），新旧矩阵下均在 parser 被拒；注意与 `KvLayoutType_PA_BBH` 枚举名区分——后者是 PA_BBND 的键值 1，属会实例化的合法项）；新增正例：BNSD/BSND × PA_NZ/PA_BNBD/PA_BBND 各至少 1（q 最后维 576；k_cache 形状按 PA 布局 [Bn,Bs,N,D]/[Bn,N,Bs,D]/[Bn,N,D1,Bs,D0]；key 按公式枚举并经探针核对）；新增负例：out≠q、layout_kv 用连续值；`.cpp` 的 EncodeMlaTilingKey 辅助扩到新字段映射（TND→2、BSND→1、BNSD→0；kv 1/2/3）；**`.cpp` 迁移联动**：`test_flash_mla_with_kvcache_tiling.cpp:72`（layout_out="NTD"）迁移为 "TND"，:139 `EXPECT_EQ(base.outputLayout, 5)`（KERNEL_LAYOUT_NTD）改为 3（KERNEL_LAYOUT_TND）；pytests：全部 NTD 字面 → TND，golden 后端（cpu_golden/gpu_impl 等）输出形状逻辑改为 (T,N,512) 并跑 golden CPU 用例断言形状；外置参数化（perf 工具链）同步迁移
    - 验收：UT csv 全绿（含 7F 漂移修复后的 KEY-1 项）；新用例与 key 探针数值一致；pytests 迁移后跑 baseline 对比无新增失败
    - QA：回归门（UT + pytests）+ 门翻转用例逐一核对 + golden 形状断言
    - 提交：`test(flash_mla_with_kvcache): flip gate cases, add layout-matrix cases, migrate pytests`
- [x] 12. PA_BBND/PA_BNBD 定向验证（offset/stride + 运行）
    - 参考：`attention/common/op_kernel/memcopy/offset_calculator_v2.h:101-114,569-634`；`attention/common/op_kernel/memcopy/gm_layout.h:245-285`；`op_kernel/arch35/flash_mla_with_kvcache_block_cube_noquant_mla.h:164-185`
    - 动作：① 单元级：PA_BBND/PA_BNBD 的 k_cache 形状/非连续规则用例（T2 已建）覆盖裁定正确；② （硬件可用时）上板冒烟：examples 或 pytests 单例（q=TND kv=PA_BNBD、q=TND kv=PA_BBND 各一）跑通并数值合理（比例对标 TND/PA_NZ 基线）；③ 无论是否可跑，产出 PA_BnBsND（PA_BBND）stride/offset 语义书面结论（读码链路已确认路径存在（T8 ②），此处以实测确认），结论 ∈ {验证通过, 发现真实缺陷（联动 T8 ③ 修复，先探索根因）}；**PA_BBND 路径已存在（T8 ② 核验），本 todo 是确认而非补齐**
    - 验收：①全绿；②③结论书面记录于交付说明；不接受静默放行
    - QA：回归门 + 上板冒烟（若硬件可用）；无硬件时如实标注
    - 提交：并入 T11 提交或独立 `test(flash_mla_with_kvcache): targeted PA_BBND/PA_BNBD verification`

### W5 收尾
- [x] 13. 全量回归终检
    - 动作：`rm -rf build build_out && bash build_mla.sh` 全量 exit 0（72 实例产物记录）；host UT（infershape+tiling）全绿（漂移已清零）；pytests baseline 对比（迁移后 vs 迁移后第一跑）无新增失败；静态门：`git diff --stat` 确认 flash_attn/、`torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md` 零改动（common/ 仅 gm_layout.h PA_BBND 相关函数改动）；`git diff -U0 -- 改动文件 | grep -nEi "m0070|\bR1\b|todo14|D14"` 零新增残留
    - 验收：上列全绿；交付说明齐备（未验证项清单、PA_BBND/PA_BNBD 结论、漂移修复说明）
    - QA：全部门
    - 提交：无（验证型；若发现修复并入最后提交）

## Final verification wave

- [x] F1. Plan compliance audit
    - 动作：对照本计划逐条核验 Todos 1–13 完成；Scope OUT 无越界（grep 确认 flash_attn/、docs 零改动；common/ 仅 gm_layout.h PA_BBND 相关函数；mask 语义与 MLA 硬约束零改动；KvLayoutType 0 未实例化；torch 签名形态未变）
    - 验收：检查清单全绿
- [x] F2. Code quality review
    - 动作：全量 diff 审查——key 探针（既有 4 值 + 新 32 组合枚举）与 SEL/键生成/UT 三方数值一致；template_tiling_key.h DECL/位宽 token 未动（`git diff` 复核仅 SEL 列表与注释）；checker 报错信息措辞与新矩阵一致；torch 侧 `grep -n '"NTD"' torch_extension/cann_ops_transformer/` 为 0 或仅文档句（main op/metadata impl/wrapper 全清）；7F 漂移修复说明与 host 实测一致
    - 验收：评审意见清零或闭环
- [x] F3. Real manual QA
    - 动作：真实环境（或环境门降级）执行 T13 全量门 + 上板冒烟（若硬件可用）；记录退出码/产物/UT 计数/新 key 枚举/hash 重采项
    - 验收：编译 exit 0 + UT 全绿 + pytests 对比无新增失败（bf16 基线差除外，述明）；无硬件时交付说明显式列"未跑项"
- [x] F4. Scope fidelity
    - 动作：确认拦截五级（checker/parser/IsCapable/键生成/SEL）对新矩阵裁决一致；既有 TND/PA_NZ 路由行为除 out=NTD→out=TND 与漂移修复外无其它变化；metadata check 与主算子存在性规则一致
    - 验收：逐项比对通过

## Commit strategy

- 按 wave 分组原子提交，消息前缀：`refactor(flash_mla_with_kvcache):`（1-8、10）、`refactor(flash_mla_with_kvcache_metadata):`（3）、`refactor(torch_extension):`（9）、`test(flash_mla_with_kvcache):`（11-12）。
- **T6+T7 联合验证、可合并为一个提交**（键生成↔SEL 耦合）；T5 若与 T6 文件重叠则合并；T8 可并入 T7 或独立。
- Git 写操作前执行 pre-commit 检查（ops-transformer-skill 规范）。
- 不做 squash 历史改写；全部提交留在当前分支。
- T13 验证型不提交。

## Success criteria

1. 拦截五级门（CommonChecker/PagedAttentionChecker→parser→IsCapable→键生成→SEL）对新矩阵（q∈{TND,BNSD,BSND} × kv∈{PA_NZ,PA_BBND,PA_BNBD} × out=q，mask{0,3}）裁决一致；PA_BBND/PA_BNBD 的 kernel 路径**已存在**（T8 ② 核验链路：KvLayoutType 1→PA_BnBsND→GM_KV_PA_BNBD 类别分支），T12 实测确认可运行（或交付说明如实标注未上板验证项，不接受静默）。
2. `bash build_mla.sh` 全量编译 exit 0；kernel 实例面 8→72；既有 4 组合 tiling key 数值不变。
3. host UT 全绿：门翻转（OutLayoutInvalid_TND→SUCCESS）+ 新增 BNSD/BSND×PA 正例与负例全部通过，**既有 7F prefill tilingKey 漂移已清零**；pytests 迁移 out=TND 后与 baseline 对比无新增失败。
4. metadata check.h 与主算子拦截对新矩阵一致；mask 语义与 MLA 硬约束（kv==1、D=576、head_dim_v=512）零改动。
5. 静态门：flash_attn/、docs 零改动；common/ 仅 gm_layout.h PA_BBND 相关函数改动；连续 KV（KvLayoutType 0）未放行未实例化；torch 公共签名形态未变（仅默认值与 impl 回退）。
6. 交付说明如实标注：上板冒烟是否执行、PA_BBND stride 语义结论、漂移修复方式（校准或重固化）、无 CANN 时未验证项清单。