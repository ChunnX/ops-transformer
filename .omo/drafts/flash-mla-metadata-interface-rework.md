# Draft: flash-mla-metadata-interface-rework

- slug: flash-mla-metadata-interface-rework
- intent: clear
- review_required: false
- classification: Standard（单算子接口重适配，受限于 attention/flash_mla_metadata + torch_extension，~10 文件）
- created: 2026-08-22
- status: awaiting-approval
- interview: 已完成（1 轮，2 问）
  - Q1 重命名范围 → **全量改名（含 torch）**：目录、op 类型、kernel/json 注册名、aclnn 文件名、torch 公开接口全部改为 `flash_mla_with_kvcache_metadata`；aclnn 导出 C 函数名保持 `aclnnFlashMlaMetadata*`（doc 明文）；同步更新 pytests/__init__.py/文档。
  - Q2 测试环境 → **编译通过即可**（无上板）：QA 限定为编译 + 静态校验。
- next action: 用户批准后 → 手写 plan 骨架（无 shell，无法跑 scaffold，draft 同源等价物）→ 派发 Metis gap 分析 → 追加 Todos → 最后填 TL;DR → 交付

> 说明：本环境无 shell 执行能力，无法运行 scaffold-plan.mjs，此 draft 为手写等价物（模板/字段对齐 ulw-plan 约定）。

## 用户请求（原文要点）

- "attention/flash_mla_metadata 这个接口设计的有问题，现在这个是新的接口文档 meta.txt 按照这个重新适配，给我出一个计划"
- 唯一输入：`/workspace/Code/flash_mla/ops-transformer/meta.txt`（新接口文档）

## meta.txt 接口契约（权威）

op 名 `flash_mla_with_kvcache_metadata`；aclnn 函数名保持 `aclnnFlashMlaMetadata`。

- inputs: cu_seqlens_q (optional int32 1D (B+1,), layout_q=TND 必传)；cache_seqlens (optional int32 1D (B,), layout_kv 为 NZ/PA_NZ 必传)；seqused_q (optional int32 1D (B,))
- 参数（顺序即目标 aclnn 签名顺序）: batch_size (>0；传 seqused_q 时可省)、max_seqlen_q (>=0 或 -1)、max_seqlen_kv (>=0 或 -1；与 cache_seqlens 至少提供其一)、num_heads_q (>0)、num_heads_kv (=1 MLA 硬约束)、head_dim (=512)、mask_mode ({0,3}，3=causal)、layout_q (=TND)、layout_kv (∈{NZ,PA_NZ})、layout_out (=NTD)
- output: metadata (tensor)

## 现状 vs 文档差异（已核实的发现）

1. **aclnn 公共签名顺序不符**（最大差异）：现为 `(cuSeqlensQ, cacheSeqlens, sequsedQ, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv, maskMode, layoutQ, layoutKv, layoutOut, metaData, ws, exec)`；doc 顺序为 `(…, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, …)`。
2. **op 类型名不符**：doc 名 `flash_mla_with_kvcache_metadata`；现状 op 类型/目录/kernel/json 全为 `FlashMlaMetadata`/`flash_mla_metadata`。
3. **内部 l0 复用 flash_attn 4 输入槽位 hack**（l0_flash_mla_metadata.cpp 注释自述）：slot1=cuSeqlensKv 恒 nullptr、slot3=sequsedKv←cacheSeqlens；且 op def 带恒 -1 的 win_left/win_right。doc 只定义 3 个输入、无 win_*。
4. **torch 层顺序/默认值不一致**：schema 位置参数 (num_heads_q, num_heads_kv, head_dim) 镜像旧 aclnn 顺序；csrc wrapper ACLNN_CMD 同旧顺序；torch python None→mask_mode=1（host 校验{0,3}会拒 1，潜在 bug）；torch 文档/ml示例用 mask_mode=1（doc 语义为 3）。
5. **示例/文档链**：examples/test_aclnn_flash_mla_metadata.cpp 有 3 处 GetWorkspaceSize 调用点需改；torch_extension docs/zh/flash_mla_with_kvcache.md 示例 mask_mode=1→3（若沿用 fused_infer_attention_score 的 RIGHT_DOWN=3 约定）；docs/zh/op_list.md 等文档列表当前未列 flash_mla_metadata（无需改，除 README news 视情况）。

## 已查明的常量/事实（executor 参考）

- 容量上界：((36+72)*batch*1+1)*16 int32 words → 对齐 4096（csrc 与 example 同源公式，kv=1 硬约束）。
- MLA 硬约束：num_heads_kv==1；head_dim==512；layout_q=TND；layout_kv∈{NZ,PA_NZ}；layout_out=NTD；mask∈{0,3}（host CheckMask 已实现，无需改语义）。
- AICPU kernel：Prepare 读 4 输入(ParamId 0..3) + attrs(含 win_left/win_right/soc_version/aic/aiv)；BASE_SIZE 头字段强制 64/128（HEAD_M_BASE_SIZE_MLA/HEAD_S2_BASE_SIZE_MLA）。
- 构建：attention/<dir>/op_host/CMakeLists.txt `add_modules_sources(OPTYPE ${dir名} ACLNNTYPE aclnn)`；rty_obj_func.cmake 编译过滤 ASCEND_OP_NAME 与 COMPILED_OPS 用**目录名**→ 改 op 类型名须同步改目录名，否则按 op 名单编会静默跳过。aclnn 头文件安装被 `FLASH_MLA_METADATA_SKIP_HEADER TRUE` 跳过。
- 调用面（仓库内全部）：torch_extension/cann_ops_transformer/ops/csrc/flash_mla_with_kvcache.cpp（1 处 ACLNN_CMD）；examples/test_aclnn_flash_mla_metadata.cpp（3 处）；pytests 走 torch op `flash_mla_metadata`（perf_runner.py/functional_stc.py/test 主流程）；无其他 aclnn 直调者。OAT.xml 无该 op 条目。
- 样式参照仓：attention/sparse_flash_mla_metadata/（同构：op_kernel_aicpu + op_host/op_api + check.h，无 infershape/op_graph；soc/aic/aiv 由 aclnn 层派生传入 l0 内部 attr）。

## 决策与默认（已采纳）

- D1: aclnn 公共函数名保持 `aclnnFlashMlaMetadata`（doc 明文）。
- D2: 公共 aclnn 签名标量顺序按 doc 重排为 (batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, layout_q, layout_kv, layout_out)。
- D3: l0/op def 精简为 3 输入 (cu_seqlens_q, cache_seqlens, seqused_q)，删除 cuSeqlensKv/sequsedKv/winLeft/winRight；AICPU kernel 输入映射改为 slot0=cu_seqlens_q、slot1=cache_seqlens、slot2=seqused_q，cache_seqlens 直接按 per-batch KV 长度语义读取；preToken/nextToken 恒 unlimited（mask∈{0,3} 无需窗口）。
- D4: soc_version/aic_core_num/aiv_core_num 保持 aclnn 内部派生、作为 l0 attr 传入（不进公共接口；与 sparse 族一致）。
- D5: mask_mode 修复：torch python None→1 改为 doc 合法值（默认 0，与主算子 flash_mla_with_kvcache 默认一致）；torch 文档与示例 mask_mode=1→3（RIGHT_DOWN）。
- D6（用户已答）: **全量改名（含 torch）** — 目录/op 类型/kernel/json/aclnn 文件名/torch 公开接口全改 `flash_mla_with_kvcache_metadata`；aclnn 导出函数名保持 `aclnnFlashMlaMetadata*`；pytests、__init__.py、文档同步。
- D7（用户已答）: QA 标准 = **编译通过即可**（无上板）：CANN op 编译出包 + example 编译 + torch_extension wheel 构建 + ruff/pre-commit + 残留标识符 grep。

## 方案（approval brief 摘要）

1. **C1+C2 全量改名 + 内部清洗**：`git mv attention/flash_mla_metadata attention/flash_mla_with_kvcache_metadata`；文件改名 aclnn_flash_mla_with_kvcache_metadata.h/.cpp、l0_flash_mla_with_kvcache_metadata.h/.cpp、flash_mla_with_kvcache_metadata_check.h、aicpu .h/.cpp/.json；全部内部标识符 FlashMlaMetadata→FlashMlaWithKvcacheMetadata；json key、REGISTER_CPU_KERNEL、OP_TYPE_REGISTER 同步；OP_TYPE 由新目录名自动派生。aclnn 签名按 D2 重排；l0/kernel 按 D3 清洗；soc/aic/aiv 按 D4。
2. **C3 torch 层**：csrc wrapper 参数顺序按新签名、m.def 名改 flash_mla_with_kvcache_metadata；python MLA_METADATA_OP_NAME/schema/meta/impl/fallback 全改；__init__.py 导出改；mask_mode 默认按 D5。
3. **C4 示例/tests**：example 文件改名 + 3 处调用点签名重排 + include 路径；pytests 引用改名。
4. **C5 文档**：torch_extension docs/zh/flash_mla_with_kvcache.md 函数名/参数表/示例更新 + mask_mode 1→3。
5. **QA（D7）**：`bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache_metadata -j20`（aicpu 需时加 --opkernel_aicpu）编译通过；example 编译；`python3 -m build --wheel -n`（torch_extension）编译通过；ruff/pre-commit 通过；grep 无残留旧标识符（aclnnFlashMlaMetadata 导出名除外，doc 保留）。

## Scope OUT（Must-NOT-Have）

- 不改调度算法 / metadata 布局不变量（16 words/core、HEAD_M/S2_BASE_SIZE 64/128、容量公式 ((36+72)*B*1+1)*16 对齐 4096）。
- 不改 flash_mla_with_kvcache 主算子 kernel/op_host（仅 torch 层改名）。
- 不动 sparse/quant/mixed/flash_attn 等其它 metadata 族算子。
- 不做上板精度/性能验证（用户选编译级 QA）。
- 不新增图模式支持；不改 meta.txt。