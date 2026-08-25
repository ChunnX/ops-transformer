# flash-mla-metadata-interface-rework - Work Plan

## TL;DR (For humans)
`attention/flash_mla_metadata` 接口标量顺序与 meta.txt 新契约不符。本计划按 meta.txt 全量适配：目录/文件/op 类型/torch 接口全部改名为 `flash_mla_with_kvcache_metadata`（aclnn 导出函数名 `aclnnFlashMlaMetadata*` 按 doc 保留），公共签名 10 个参数重排为 batch_size 先行，l0/AICPU kernel 从 4 输入精简为 3 输入（删 cu_seqlens_kv/seqused_kv/win_left/win_right），torch 侧修复 mask_mode 默认值，并同步两个 example（含主算子 example 的 2 处直调）、pytests、文档。QA 按用户选定为编译级（CANN 出包 + example 编译-only + torch wheel + lint + 残留标识符静态门），无上板。关键风险已被三重审查消除（Metis + Momus + Oracle）：example 编译门改用可执行的 g++ -c 机制；静态门改 token 边界 grep（flash_mla_metadata 是独立算子，sparse/quant 族同名子串不算命中，用户确认）；csrc wrapper 形参列表与 python 桥三序对齐（pybind11 位置绑定，避免 6/10 标量静默错位）；主算子 kernel 5 行注释补"旧 producer 名"标注并豁免于 diff=0 审计；check.h 形参序（实际为旧序）与 aclnn 同步重排。6 个 wave 15 个实现步骤 + 4 项最终验证。

## Scope

### In scope
- **目标接口契约**：`/workspace/Code/flash_mla/ops-transformer/meta.txt` 唯一权威（op 名 `flash_mla_with_kvcache_metadata`；aclnn 导出 C 函数名保持 `aclnnFlashMlaMetadata` / `aclnnFlashMlaMetadataGetWorkspaceSize`；3 输入 cu_seqlens_q / cache_seqlens / seqused_q；10 参数按 doc 顺序 batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, layout_q, layout_kv, layout_out；1 输出 metadata）。
- **C1 aclnn 公共签名**：`attention/flash_mla_with_kvcache_metadata/op_host/op_api/aclnn_flash_mla_with_kvcache_metadata.h/.cpp` 按 doc 重排标量顺序（现在顺序 `(…, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv, maskMode, …)` 与 doc 不符）。
- **C2 全量改名 + 内部清洗**（目录 git mv + 精确文件清单，含 Metis finding 5/7）：
  - 目录 `attention/flash_mla_metadata/` → `attention/flash_mla_with_kvcache_metadata/`；
  - op_host/op_api/：`aclnn_flash_mla_metadata.{h,cpp}`、`l0_flash_mla_metadata.{h,cpp}` 一并改名；
  - op_host/：`flash_mla_metadata_check.h` 改名，`CMakeLists.txt` 内变量 `FLASH_MLA_METADATA_SKIP_HEADER` → `FLASH_MLA_WITH_KVCACHE_METADATA_SKIP_HEADER`（custom_build.cmake 按 `${op_name_upper}_SKIP_HEADER` 动态查，漏改会静默改变头部打包行为）；
  - op_kernel_aicpu/：**4 个文件**全部改名 —— `flash_mla_metadata_aicpu.cpp`、`flash_mla_metadata_aicpu.h`、`flash_mla_metadata_aicpu.json`、`flash_mla_metadata.h`（纯布局头，146 行，含全部布局不变量，被 aicpu cpp 与 example 相对 include）；include guard（如 `FLASH_MLA_METADATA_H`、`ACLNN_FLASH_MLA_METADATA_H`）同步改名；
  - l0 与 kernel 精简为 3 输入（slot0=cu_seqlens_q / slot1=cache_seqlens / slot2=seqused_q），删除 cu_seqlens_kv / seqused_kv / win_left / win_right，preToken/nextToken 恒 unlimited；soc_version / aic_core_num / aiv_core_num 保持 aclnn 内部派生、l0 attr 传入（**13 项** attr 列表：10 个 doc 参数 + soc_version/aic_core_num/aiv_core_num；现 TOP_ATTR 15 项，删 win_left/win_right 后为 13 项）。
- **C3 torch_extension**：`ops/csrc/flash_mla_with_kvcache.cpp` 内 wrapper 类与 `m.def` 改名 `flash_mla_with_kvcache_metadata`，ACLNN_CMD 实参按新签名顺序（见 Key Decisions K1）；`ops/flash_mla_with_kvcache.py` 的 MLA_METADATA_OP_NAME / schema 名 / impl / fallback / 导出别名改名；`ops/__init__.py` 导出改名；mask_mode None 默认 1→0 修复（D5）。
- **C4 示例与 tests（含 Metis finding 1/4/8）**：
  - metadata example：`examples/test_aclnn_flash_mla_metadata.cpp` → `test_aclnn_flash_mla_with_kvcache_metadata.cpp`，include 更新（`aclnnop/` 头 + 相对 include 布局头），3 处 `aclnnFlashMlaMetadataGetWorkspaceSize` 调用点（RunMetadataOnce / RunNegativeRuntimeOnly / main smoke）与对应 exec 调用按新签名重排，`execl("/proc/self/exe", "test_aclnn_flash_mla_metadata", …)` argv[0] 字符串同步；
  - **主算子 example（漏网直调者）**：`attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp` 的 L41 include `../../flash_mla_metadata/op_host/op_api/aclnn_flash_mla_metadata.h` 与 L499/L514 两处 aclnn 调用点按新路径/新签名适配；除此之外不改主算子任何其它文件；
  - tests：`attention/flash_mla_with_kvcache/tests/**`（含 pytests/ 与 `assets/impl/graph.py` L19）全部 torch op 名引用更新。
- **C5 文档**：`torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md` 函数原型/参数表/调用示例改名 + `mask_mode=1`→`3`（L255/L270 附近）。
- **QA（D7，用户选定：编译通过即可）**：CANN op 编译出包、example 编译-only、torch_extension wheel 构建、ruff/pre-commit、残留旧标识符静态门。

### Out of scope（Must-NOT-Have）
- 不改调度算法与 metadata 布局不变量：16 words/core stride、HEAD_M_BASE_SIZE_MLA=64 / HEAD_S2_BASE_SIZE_MLA=128 强制写、容量公式 `((36+72)*batch*1+1)*16` words 对齐 4096、FA/FD/HEAD section 布局。
- 不改 `flash_mla_with_kvcache` 主算子 op_host / op_kernel / tiling（唯一例外：C4 主算子 example 的 metadata 调用点；主算子 kernel 内解释性注释**允许**保留旧名提及并明确标注"producer 旧名，历史原因"）。
- 不动 sparse / quant / mixed_quant / flash_attn 等其它 metadata 族算子。
- 不做上板精度/性能验证（编译级为准）；不新增图模式 GE 支持；不改 `meta.txt`；不改 git 历史（git mv 保留 rename 记录）。

## Key Decisions（已定，执行者不得更改）
- **K1（Metis finding 6 + Oracle BLOCKER）torch 公共签名形态保持、csrc wrapper 形参**重排：python 公共函数签名保持 `(num_heads_q, num_heads_kv, head_dim, *, Tensor? cu_seqlens_q=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, int? batch_size=None, …)` 不变（pytests 全部 keyword 调用，不受影响）；python impl 内传给 `op_module.flash_mla_with_kvcache_metadata(...)` 的位置实参重排为新 aclnn 顺序 `(cu_seqlens_q, cache_seqlens, seqused_q, effective_batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, layout_q, layout_kv, layout_out, output)`；**csrc wrapper 形参列表必须同步重排为同一新序** `(cuSeqlensQ, cacheSeqlens, sequsedQ, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, layoutQ, layoutKv, layoutOut, output)`，使 python 位置实参序 == wrapper 形参序 == ACLNN_CMD 实参序（pybind11 按位置绑定，三序不一致会导致 6/10 标量静默错位、且编译级 QA 无法发现）。
- **K2（Metis finding 2/9）example 编译-only 机制**：禁止 `--run_example`（会运行二进制、无 NPU 不可行）。执行方式 = 复制 `build.sh` 中 `build_example()` 的 g++ 编译命令（非 pkg 分支，含 `-I ${INCLUDE_PATH} / ${ACLNN_INCLUDE_PATH} / ${EAGER_INCLUDE_OPP_ACLNNOP_PATH}` 与 `-lopapi_transformer` 等链接库），**去掉** `./test_aclnn_*` 执行与 `mpirun/msvp` 步骤，改用 `-c` 只产出 `.o`；预期 exit 0 且 `.o` 生成。`aclnnop/aclnn_flash_mla_with_kvcache_metadata.h` 从构建产物 include 路径解析（SKIP_HEADER 只跳过 run 包打包，构建目录头仍生成；若解析失败，将 op_api 源码目录追加 -I，不得改 example 的 include 行）。
- **K3（Metis finding 3）静态门统一为仓库级 + 显式白名单**（见 Verification strategy），与 Scope OUT（主算子 kernel 注释保留）通过白名单自洽。
- **K4（Metis minor）**: 静态门豁免 `.omo/` 目录（skeleton/notepad 文档为计划产物，非代码）。

## Verification strategy
- **编译门（每 wave 后）**：
  - `bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache_metadata -j20` 必须成功产出 run 包（含 op_api 库与 aicpu kernel）；若 aicpu 包需单独开关，按 `bash build.sh -h` 的 `--opkernel_aicpu` 用法附加；
  - UT 对象编译（可选加固）：`bash build.sh -u --ophost --ops=flash_mla_with_kvcache_metadata` 与 `--opapi`（CI 方式，脚本 scripts/ci/ut_make.sh）；
  - 主算子回归（**建议跑**，环境允许时）：`bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache -j20` 仍成功（metadata 改名不破坏主算子打包）。
- **example 门（W4 后，K2 机制）**：g++ `-c` 编译两个 example 翻译单元（metadata example + 主算子 example）成功，exit 0。
- **torch 门**：`cd torch_extension && python3 -m build --wheel -n` 成功（内部 ninja 编译 csrc）；`.pre-commit-config.yaml` 规定的 ruff-check / ruff-format / clang-format / codespell / trailing-whitespace / check-json 通过。
- **静态门（最终 wave，K3/K4；Momus+Oracle MAJOR 修正）**：仓库级 token 边界 grep（**排除** `.omo/`、`build/`）—— 模式 `grep -rEn '(^|[^A-Za-z0-9_])(FlashMlaMetadata|flash_mla_metadata)([^A-Za-z0-9_]|$)'`（下划线属 `\w`，词边界不会穿透 `sparse_flash_mla_metadata` 等**独立算子**名 —— 该族独立算子本就不在改名范围，经用户确认）—— 仅允许命中：
  1. aclnn 导出函数名，用显式模式 `(^|[^A-Za-z0-9_])aclnnFlashMlaMetadata(GetWorkspaceSize)?([^A-Za-z0-9_]|$)`（`\bFlashMlaMetadata\b` 在 `aclnn` 前缀后不触发，必须整名匹配）；
  2. 主算子 kernel 内明确标注"旧 producer 名/历史原因"的解释性注释（见 Todo 15：执行者须给 `flash_mla_with_kvcache_kernel_noquant_mla.h` L79/L81/L83、`flash_mla_with_kvcache_tiling_data.h` L30/L33 共 5 行注释补标注），以及主算子 example L503/L516 的 LOG_PRINT 字符串字面量与 `kernel_noquant_mla.h` L83 static_assert 字符串字面量（因 aclnn 导出名保留，这些字面量仍为真，属合法豁免）；
  3. git mv 历史与 `.omo/` 计划产物。
  其余一律为 0。
- 因无 NPU（用户选定），不做真机 run；example 的 byte-golden / 容量负例断言逻辑保留为未来上板回归资产（改名不动其语义）。

## Execution strategy
Wave 划分与依赖（严格顺序；改名原子性要求 rename-then-edit 在单 wave 内完成，禁止双路径并存）：
- **W1 改名基底**（Todo 1）：git mv 目录 + 全部文件改名 + include guard / CMakeLists 变量同步。先于一切编辑。
- **W2 接口重排**（Todo 2-3）：aclnn 签名重排 + check.h 重排。
- **W3 内部清洗**（Todo 4-6）：l0 3 输入去 win_* / AICPU kernel 输入映射与 attr / json。
- **W4 示例**（Todo 7-8）：metadata example + 主算子 example 调用点适配。
- **W5 torch 桥与 tests**（Todo 9-11）：csrc / python / __init__ / tests。
- **W6 文档 + 全量 QA**（Todo 12-15）：文档 / CANN 编译门 / example 编译-only + wheel + lint / 静态门。
- **最终验证**：F1-F4。

依赖：1 → 2,3 → 4,5,6 → 7,8 → 9,10,11 → 12 → 13,14,15 → F1-F4。
完成标准 = 全部 todo 完成 + 最终验证 wave 全 approve。

## Todos

### W1 改名基底
- [x] 1. attention/flash_mla_metadata/：git mv 目录为 flash_mla_with_kvcache_metadata，并改名全部文件（op_host/op_api/aclnn_flash_mla_metadata.{h,cpp}、l0_flash_mla_metadata.{h,cpp}、op_host/flash_mla_metadata_check.h、op_kernel_aicpu/flash_mla_metadata_aicpu.{cpp,h,json}、op_kernel_aicpu/flash_mla_metadata.h 布局头、examples/test_aclnn_flash_mla_metadata.cpp），同步改 include guard（ACLNN_FLASH_MLA_METADATA_H / L0_FLASH_MLA_METADATA_H / FLASH_MLA_METADATA_H 等 → WITH_KVCACHE 变体）与 op_host/CMakeLists.txt（L10-L12 附近）的 FLASH_MLA_METADATA_SKIP_HEADER → FLASH_MLA_WITH_KVCACHE_METADATA_SKIP_HEADER，使目录内改名前缀的标识符残留为 0（body 尚可含 flash_mla_metadata 词，W2-W4 逐步清除）。QA：`git status --short` 全为 rename 对（Rxxx）；`find attention/flash_mla_with_kvcache_metadata -name '*flash_mla_metadata*'` 为空；`grep -rn 'FLASH_MLA_METADATA' attention/flash_mla_with_kvcache_metadata/` 为空。Commit：W1-W3 合并提交 `refactor(attention): rename flash_mla_metadata -> flash_mla_with_kvcache_metadata and clean 3-input op def`。

### W2 接口重排
- [x] 2. op_host/op_api/aclnn_flash_mla_with_kvcache_metadata.h/.cpp：GetWorkspaceSize 与 exec 双函数签名按 doc 重排为 (const aclTensor* cuSeqlensQOptional, const aclTensor* cacheSeqlensOptional, const aclTensor* sequsedQOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t maskMode, const char* layoutQ, const char* layoutKv, const char* layoutOut, const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor)；ACLNN_DFX 宏、头文件注释、.cpp 内 l0 调用实参按新序（l0 形参序 (batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, layoutQ, layoutKv, layoutOut, socVersion, aicCoreNum, aivCoreNum) 不变）。QA：grep 确认 .h/.cpp 无旧参数序残片；`bash build.sh -u --opapi --ops=flash_mla_with_kvcache_metadata` 编译通过。
- [x] 3. op_host/flash_mla_with_kvcache_metadata_check.h：ParamsCheck 形参序**当前为旧序**（`(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv, maskMode, layoutQ, layoutKv, layoutOut, metaData)`，与现 aclnn 签名一致；L36 注释"m0070 参数序"指与现 aclnn 一致、非 meta.txt）→ 重排为 batchSize 先行（与新 aclnn 签名同序）；校验语义保持现状（已核对 doc：batchSize>0 或 sequsedQ 存在、maxSeqlenQ/Kv 为 -1 或 >=0、numHeadsKv==1、headDim==512、layoutQ==TND、layoutKv∈{NZ,PA_NZ}、layoutOut==NTD、mask∈{0,3}、TND 时 cuSeqlensQ 必传、PA_NZ 时 cacheSeqlens 必传、形状 (B+1,)/(B,)）。**警示：aclnn 层 Forwarding 实参与 check.h 形参序必须同步改，只改一处会造成参数错位（全 int64，静态无感）。** QA：grep 确认 .h 无旧形参序残片；op_api 编译通过。

### W3 内部清洗
- [x] 4. op_host/op_api/l0_flash_mla_with_kvcache_metadata.h/.cpp：l0op 类改名 FlashMlaWithKvcacheMetadata；OP_TYPE_REGISTER 同步；输入槽 4→3（删 cuSeqlensKvOptional / sequsedKvOptional，输入改为 cuSeqlensQOptional / cacheSeqlensOptional / sequsedQOptional）；删 winLeft / winRight attr 及 OP_ATTR_NAMES 对应项，attr 列表收敛为 {batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, layout_q, layout_kv, layout_out, soc_version, aic_core_num, aiv_core_num}；OP_INPUT / OP_OUTPUT 宏、类注释同步；保留 socVersion / aicCoreNum / aivCoreNum 形参（aclnn 层 GetCurrentPlatformInfo 派生）。QA：`grep -rn 'CuSeqlensKv\|SequsedKv\|winLeft\|winRight'`（l0 目录内）为 0；op_api 编译通过。
- [x] 5. op_kernel_aicpu/flash_mla_with_kvcache_metadata_aicpu.cpp/.h：类名与 REGISTER_CPU_KERNEL 同步；ParamId 输入枚举改为 cuSeqlensQ=0 / cacheSeqlens=1 / sequsedQ=2；Prepare 从 slot1 读 cacheSeqlens 作为非累积 KV 长度（原 CheckActualKvSeq 的 sequsedKv 路径），删除 cuSeqlensKv 输入处理与 win_left / win_right attr 读取及成员；preToken / nextToken 恒 UINT32_MAX（unlimited）；maskMode_ 默认值改 0（与 torch None 默认对齐；capability 检查 {0,3} 不变）；GetAttr 列表同步（必读 num_heads_q / num_heads_kv / head_dim / soc_version / aic_core_num / aiv_core_num；可选 batch_size / max_seqlen_q / max_seqlen_kv / mask_mode / layout_q / layout_kv / layout_out）；调度与 metadata 写入逻辑零改动。QA：`grep -rn -i 'win_\|seqlenskv\|sequsedkv' op_kernel_aicpu/` 为 0；kernel 编译通过。
- [x] 6. op_kernel_aicpu/flash_mla_with_kvcache_metadata_aicpu.json：顶层 key FlashMlaMetadata → FlashMlaWithKvcacheMetadata；engine=DNN_VM_AICPU、kernelSo=libtransformer_aicpu_kernels.so、opKernelLib=CUSTAICPUKernel、functionName=RunCpuKernel 不变（**json 无输入/输出列表，仅 opInfo 段，无需改输入数**）。QA：`python3 -c "import json; json.load(open('op_kernel_aicpu/flash_mla_with_kvcache_metadata_aicpu.json'))"` 通过；key 与 OP_TYPE 一致。

### W4 示例
- [x] 7. attention/flash_mla_with_kvcache_metadata/examples/test_aclnn_flash_mla_with_kvcache_metadata.cpp：include 更新（L39 `aclnnop/aclnn_flash_mla_with_kvcache_metadata.h`、L41 相对 include 布局头新名）；3 处 GetWorkspaceSize 调用点（RunMetadataOnce / RunNegativeRuntimeOnly child / main smoke）按新签名重排，并同步其后的 aclnnFlashMlaMetadata（exec）实参；L347 `execl("/proc/self/exe", "test_aclnn_flash_mla_metadata", …)` argv[0] → `test_aclnn_flash_mla_with_kvcache_metadata`。QA：K2 机制 g++ -c 编译通过（exit 0，产出 .o）。
- [x] 8. attention/flash_mla_with_kvcache/examples/test_aclnn_flash_mla_with_kvcache.cpp（主算子 example，仅此文件）：L41 include 路径 → `../../flash_mla_with_kvcache_metadata/op_host/op_api/aclnn_flash_mla_with_kvcache_metadata.h`；L499/L514 两处 aclnnFlashMlaMetadataGetWorkspaceSize / aclnnFlashMlaMetadata 调用点的实参按新签名重排（语义不变）。QA：K2 机制 g++ -c 编译通过；`git diff --stat` 确认仅此 example 文件被改。

### W5 torch 桥与 tests
- [x] 9. torch_extension/cann_ops_transformer/ops/csrc/flash_mla_with_kvcache.cpp：wrapper 名 FlashMlaMetadata → FlashMlaWithKvcacheMetadata；**wrapper 形参列表重排为新 aclnn 序**（`(cuSeqlensQ, cacheSeqlens, sequsedQ, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, layoutQ, layoutKv, layoutOut, output)`，pybind11 位置绑定，与 python impl 位置实参序一致，见 K1）；L89 ACLNN_CMD 实参按同一新序（cuSeqlensQVal, cacheSeqlensVal, sequsedQVal, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode, layoutQPtr, layoutKvPtr, layoutOutPtr, output）；L179 m.def("flash_mla_with_kvcache_metadata", …)；头文件 include 与注释同步。QA：`grep -n 'flash_mla_metadata\b\|FlashMlaMetadata'` csrc 内为 0（aclnn 导出名除外）；wheel 构建门。
- [x] 10. torch_extension/cann_ops_transformer/ops/flash_mla_with_kvcache.py：MLA_METADATA_OP_NAME → "flash_mla_with_kvcache_metadata"（L16）；schema 名（L72）；impl 内 op_module 调用改 `flash_mla_with_kvcache_metadata` 且位置实参按 K1 新序（cu_seqlens_q, cache_seqlens, seqused_q, effective_batch_size, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, layout_q, layout_kv, layout_out, output）；L215 mask_mode None 默认 1→0；flash_mla_metadata_fallback、register_kernel、`_flash_mla_metadata` 别名、L335-336 公开导出同步改名。QA：`python3 -c "import ast; ast.parse(open('cann_ops_transformer/ops/flash_mla_with_kvcache.py').read())"` 通过；ruff 检查通过。
- [x] 11. torch_extension/cann_ops_transformer/ops/__init__.py（L16）导出改名；attention/flash_mla_with_kvcache/tests/**（含 tests/pytests/utils/perf_runner.py L58/L66/L80/L247/L259、tests/pytests/test_flash_mla_with_kvcache.py L53、tests/pytests/core/data.py L39、tests/pytests/core/backends/gpu_impl.py L31、tests/pytests/core/backends/npu.py L20/L106/L151、tests/assets/impl/graph.py L19）中 flash_mla_metadata → flash_mla_with_kvcache_metadata（纯改名，关键字调用形态不动）。QA：`grep -rn 'flash_mla_metadata'` flash_mla_with_kvcache/tests/ 为 0。

### W6 文档 + 全量 QA
- [x] 12. torch_extension/cann_ops_transformer/docs/zh/flash_mla_with_kvcache.md：函数名 / 原型 / 参数表 / 调用示例中 flash_mla_metadata → flash_mla_with_kvcache_metadata；mask_mode=1 示例 → 3（L255/L270 附近）。QA：`grep -n 'flash_mla_metadata'` 文档内为 0。
- [x] 13. W6 CANN 编译门：`bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache_metadata -j20` 成功产出 run 包（op_api + aicpu kernel + json 注册）；环境允许时追加主算子回归 `bash build.sh --pkg --soc=ascend950 --ops=flash_mla_with_kvcache -j20`。QA：退出码 0 + 产物目录含 flash_mla_with_kvcache_metadata 相关 so/json；失败则定位修复后重跑。
- [x] 14. W6 example + wheel + lint：K2 机制下两个 example（Todo 7/8 产物）g++ `-c` 编译成功；`cd torch_extension && python3 -m build --wheel -n` 成功；`.pre-commit-config.yaml` 全量钩子（ruff / clang-format / codespell / trailing-whitespace / check-json）通过。QA：exit 0；wheel 含新 op 名符号。
- [x] 15. W6 静态门：按 Verification strategy 的 token 边界 grep 白名单执行（词边界模式，见 L44-50；含主算子 kernel 5 行注释补标注 `kernel_noquant_mla.h` L79/L81/L83、`flash_mla_with_kvcache_tiling_data.h` L30/L33 —— 追加"producer 旧名(f: flash_mla_metadata)，历史原因"标注；LOG_PRINT 与 static_assert 字符串字面量豁免），非白名单命中为 0；输出命中清单留档。QA：`grep -rEn '(^|[^A-Za-z0-9_])(FlashMlaMetadata|flash_mla_metadata)([^A-Za-z0-9_]|$)' --exclude-dir=.omo --exclude-dir=build .` 仅白名单命中（用 `grep -rEn '(^|[^A-Za-z0-9_])aclnnFlashMlaMetadata(GetWorkspaceSize)?([^A-Za-z0-9_]|$)'` 单独核对导出名命中集 = {2 个 doc 保留 + 主算子 example/csrc 调用点}）。

## Final verification wave
- [x] F1. 计划合规自检：所有实现行 `- [ ] N. ` 与验证行 `- [ ] F<N>. ` 列零对齐、语法匹配；TL;DR 最后填写；依赖矩阵一致。
- [x] F2. 代码质量审查（Oracle 只读）：全量 diff 审查 —— aclnn/l0/kernel 三处签名与 attr 一致且与 meta.txt 同序；`cuSeqlensKv/sequsedKv/win_*` 残留 grep **限定在 `attention/flash_mla_with_kvcache_metadata/` 目录**（全仓 grep 会误命中 sparse/quant 等独立算子族，非本 op 残留）；Scope OUT 目录 diff=0（主算子 op_host/op_kernel/tiling **豁免 Todo 15 新增的 5 行"旧 producer 名"注释标注**、其它 metadata 族算子、布局常量）。
- [x] F3. 全量 QA 重跑：干净树（git clean 排除 .omo）重跑 Todo 13（CANN 包）+ Todo 14（example 编译-only + wheel + lint）+ Todo 15（词边界静态门）全绿。
- [x] F4. 范围保真审计：`git diff` 中 flash_mla_with_kvcache_metadata 内布局常量（16/36/72/64/128/4096 与容量公式）与调度逻辑零改动；torch 侧 mask_mode 默认 0 与文档示例 3 到位。

## Commit strategy
按 wave 分组提交（单 wave 内不留半成品提交）：
1. W1-W3：`refactor(attention): rename flash_mla_metadata -> flash_mla_with_kvcache_metadata and clean 3-input op def`
2. W4：`test(examples): adapt metadata aclnn call sites to new signature (incl. main-op example)`
3. W5：`refactor(torch_extension): rename flash_mla_metadata torch op to flash_mla_with_kvcache_metadata`
4. W6 文档：`docs(torch_extension): update flash_mla_with_kvcache metadata docs + mask_mode example`
5. QA 修正单独提交（如有）。
提交信息风格先 `git log --oneline -10` 验证仓内惯例；不做 squash 历史改写。

## Success criteria
1. aclnn 公共签名与 `meta.txt` 逐参数一致（顺序、类型、optional 语义、默认语义）；`attention/flash_mla_with_kvcache_metadata/` 下无任何旧名标识符（aclnn 导出函数名除外）。
2. l0/kernel 为 3 输入且无 cuSeqlensKv / sequsedKv / win_* 痕迹；kernel 从 slot1 读 cache_seqlens；preToken/nextToken 恒 unlimited。
3. torch 公共签名形态保持（K1）、op 名 flash_mla_with_kvcache_metadata、mask_mode None 默认 0。
4. 编译门全绿：CANN run 包 + 两个 example 编译-only + torch wheel + lint；主算子回归（若跑）通过。
5. 仓库级 token 边界静态门（词边界模式，非白名单命中为 0；sparse/quant 等独立算子族的同名子串**不算**命中）；metadata 布局/调度代码 diff 为 0；其它 metadata 族算子 diff 为 0。