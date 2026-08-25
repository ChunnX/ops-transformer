---
slug: mla-align-flash-attn
status: plan
intent: clear
review_required: false
predecessors: [.omo/drafts/mla-align-flash-attn.md]
approach: 以 flash_attn 为基准，把 attention/flash_mla_with_kvcache 的代码结构（kernel 模板聚合/入口风格、TPipe 接线、host 层组织、命名与注释规范、测试基建）对齐到 flash_attn 组织模式；保持功能/行为不变、保持 4 组合模板实例化面；交付 build_mla.sh（-j20）并保证编译通过
---

# Plan: mla-align-flash-attn

## TL;DR (For humans)

把 `attention/flash_mla_with_kvcache` 的**代码组织方式**对齐到 `attention/flash_attn`（flash_attn 是 2026/05 新上的"干净参考系"），但**不做功能变化**：

- kernel 层：入口引入编译期 layout 推导；模板参数从"平铺 8~12 个"收敛为"聚合类型"（FAType 式）；TPipe 从外部传入改为 kernel 自持；Dummy/Real 选型统一为 flash_attn 的编译期宏风格；移除孤儿文件与 m0070/D14 等过程性编号注释。
- host 层：def/checker/tiling 的组织与命名风格对齐（注意：**tiling key 位编码不变**，只对齐声明/宏风格）。
- tests：建 flash_attn 式的文件骨架（perf 工具/gpu backend/xlsx 工具/redline 模块占位），**不迁移** redline CSV 性能用例资产。
- 交付：仓根 `build_mla.sh`（`-j20`，编译 flash_mla_with_kvcache 算子并断言成功）；**验收硬性条件：编译必须通过（退出码 0）**。

所有改动被限定在 `attention/flash_mla_with_kvcache/` 目录内 + 仓根新增 `build_mla.sh`；不改 `attention/flash_attn/`、不改 `common/`、不改算子名（torch_extension/文档等 6 处外部引用零影响）。

## Scope

**IN（要做的）**
1. `build_mla.sh`（仓根 `/workspace/Code/flash_mla/ops-transformer/`，`-j20`，编译 `flash_mla_with_kvcache` 算子，断言退出码 0，失败打印日志尾部）。
2. K1 入口：`op_kernel/flash_mla_with_kvcache.cpp` 增加 flash_attn 风格的编译期 layout 推导（GetQueryLayout/GetOutLayout/GetKvLayout/IsPageAttention 等价物）；模板参数聚合收敛。
3. K2 kernel/block 组织：TPipe 内部化；Dummy/Real 选型对齐；SeqLensTool 工具对象化；ConstInfo 组织同构化；删除孤儿文件 `op_host/flash_mla_with_kvcache_adjust_sinner_souter.h`（实测路径，非 arch35/）；include 面收拢；过程性编号注释（m0070/R1/Todo14/D14 等）清理为规范注释。
4. H1/H2 host 层：def/checker/tiling 的组织与命名风格对齐（tiling key 位编码不变）。
5. T1 tests 骨架：建 flash_attn 的 pytests 文件骨架（utils/{perf_runner,perf_parser,precision_visual}.py、core/backends/{gpu,gpu_impl}.py、tools/xlsx_to_testcase.py、assets/testcase/ 结构、test_cases redline 模块骨架、impl/graph.py 骨架）；修 `tests/CMakeLists.txt:11` 复制粘贴残留（message 误写 flash_attn）；不迁移 redline CSV/性能用例资产。
6. 回归与验证：全程每次组件改动跑编译 + 既有测试；最终 `build_mla.sh` 全量编译通过 + UT/ST 全绿。

**OUT（Must NOT have）**
- 不扩展功能面：不新增 DN 路径、不扩到 flash_attn 的 160 组合实例化面（保持 4 组合×2 dtype）、不改 def 输入集合的行为语义、不把 mla 类名改成 GQA 语义命名。
- 不改 `attention/flash_attn/`、`common/`、`torch_extension/`、`docs/`、`fused_infer_attention_score/` 的任何文件（它们是只读参考）。
- 不重命名算子/不删除 `examples/test_aclnn_flash_mla_with_kvcache.cpp`（用户确认保留）。
- 不做性能优化/精度调整；不改 tiling 数值与 tiling key 位编码。
- 不迁移 flash_attn 的 redline CSV/性能用例资产（缺真实用例数据，迁移属功能开发）。
- `build_mla.sh` 不做 torch_extension whl 构建、不做安装（opp_install）——只负责编译并断言通过。
- 不做 `--genop`/`--run_example` 集成；examples 的构建不在脚本范围（可留注释指引）。

## Verification strategy

**每 todo 的 QA 门**（除注明外）：
- 编译门：`bash build_mla.sh --incremental`（或 Todo 16 前的 `build.sh ...` 等价命令）退出码 0。**全局硬约束：所有编译任务（build.sh / cmake --build / UT 构建 / 任何编译类命令）一律带 `-j20`，禁止出现 -j 取其它值或省略 -j。**
- 回归门：`tests/ut` 现有用例全绿；`tests/pytests` 现有 functional_stc/functional_nc 全绿。**硬件依赖注记**：tests/ut（op_host 侧 infershape/tiling）可无 NPU 运行；pytests（functional_nc/stc，上板用例）需要 NPU 硬件——若本机无 NPU，回归门降级为 host 侧 UT 全绿 + 在交付说明如实标注"上板用例未跑"，不得伪造。
- 行为门：涉及模板签名/选型/工具对象化的改动（Todo 4/6/7/8/9），在编译门之外加"产物对比"：改动前后对同一输入跑 `examples/test_aclnn_flash_mla_with_kvcache.cpp`（或 pytests 用例）输出一致——若环境不可跑则退化为编译+UT 全绿并如实记录。

**关键工程约束（双审增补）**：
- tiling-key 头（`op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h`）是 **host 与 kernel 共用、且被 CANN 工具链（ASCENDC_TPL_PRE → template_tiling.py → asc_opc）正则消费**的文件——Todo 3/4/14 凡是触碰此头都按"工具链契约面"处理：只改含义保持的代码（推导函数/断言/注释），DECL/SEL 宏体与位宽 token 一字不动。
- Todo 5（TPipe）+ 8（SeqLensTool）+ 9（ConstInfo）同改 kernel 构造函数/Init/解析路径，为联合行为承载——**合并为一个原子提交**，同一验证波（含边角用例），禁止各自独立提交（见 Commit strategy）。
- FD 的 isFd 是 kernel 级模板参数（tiling key 第 5 字段），FD block 类无 isFd 模板参——聚合与选型改动不得把它挪动/删除（Todo 4/6/7 均适用）。

**最终验证波（F1–F4）**：全部 todo 完成后统一执行（见 Final verification wave）。

**环境门（Todo 1 前置）**：若本机无 CANN/ascend950 编译环境（`npu-smi` 无输出、`bisheng` 不在 PATH、无 toolkit/setenv.bash），worker 必须：脚本用 `bash -n` + 对照 build.sh 参数逐项静态校验，并在交付说明中如实标记"脚本未在真机验证"，不得伪造通过。

## Execution strategy

- **顺序**：1（环境+build_mla.sh+基线编译）→ 2（基线回归快照）→ 3~14（K1/K2/H1/H2，可并行分批，每批后编译+回归）→ 15（tests 骨架）→ 16（全量编译终检）→ F1~F4。
- **只读参考**：所有对齐动作以 `attention/flash_attn/` 下对应文件为参照（flash_attn.cpp、utils/flash_attn_type.h、utils/flash_attn_common_def.h、kernel_nd.h、block_cube_nd.h、block_vec_nd.h、block_vec_flashdecode.h、template_tiling_key.h、def.cpp、checkers、arch35 tiling 头）。
- **硬约束**：语义命名保留 `FlashMlaWithKvcache*`；实例化面 4 组合×2 dtype；tiling key 位编码不变；**所有编译命令统一 -j20**（用户 m0068 追加要求，覆盖每个 todo 的编译门）。
- **提交**：见 Commit strategy。

## Todos

- [x] 1. 环境探测并创建 `build_mla.sh`，跑通基线编译
    - 参考：`build.sh`（:13,:35,:53,:112-115,:341,:359,:416-430 set_env）；`.gitcode/scripts/compile.sh:134`（CI 同型命令 `bash build.sh --pkg --jit --soc=ascend950 -j16`）；`scripts/util/soc_validator.sh`
    - 动作：① 探测 CANN 环境（`which bisheng`、`npu-smi`、`$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash`、`~{/usr/local}/Ascend/ascend-toolkit/latest`）；② 写 `build_mla.sh`：`#!/bin/bash` + `set -euo pipefail`；`ROOT=$(dirname $(readlink -f "$0"))`；BISHENG 不可用时按优先级 source setenv.bash（`$ASCEND_CANN_PACKAGE_PATH` → `~/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/ascend-toolkit/latest`），仍缺则报错退出；默认 `SOC=ascend950`、`OPS=flash_mla_with_kvcache`；**从透传参数中剥离用户可能传入的 `--ops=`/`--soc=`/`-j`/`-jN`（防止双标志冲突；`-j20` 由脚本强制统一、不允许外部覆盖，Oracle 审校）后再把其余参数原样透传**；执行 `bash build.sh --pkg --soc=$SOC --ops=$OPS -j20 "$@"`（不残留未用变量）；`trap` 打印失败位置；失败时 `tail -50 build/failed_ops.log`（若存在）并退出 1；成功打印产物路径；③ `chmod +x`；④ 以 `bash build_mla.sh`（非 incremental 首次全量）跑通基线编译，退出码 0（若环境缺失按 Verification strategy 的环境门处理）
    - 验收：`bash -n build_mla.sh` 通过；`bash build_mla.sh` 退出码 0；产物存在（build/ 内 run 包或 kernel bin）；失败路径打印日志尾部
    - QA：编译门（本 todo 即编译门本身）
    - 提交：`build: add build_mla.sh for flash_mla_with_kvcache (-j20)`
- [x] 2. 基线回归快照（UT + pytests 现状）
    - 参考：`tests/ut/op_host/arch35/test_flash_mla_with_kvcache_tiling.{cpp,csv}` 等现有用例
    - 动作：运行 `tests/ut` 现有用例 + `tests/pytests/test_flash_mla_with_kvcache.py`，记录通过/失败清单到计划附录（作为后续所有回归门的基线；若基线有失败项，逐条记录原因，禁止用"基线就挂"掩盖新引入的失败）；**同时为行为门采集"改动前"产物**：若环境可跑，保存 `examples/test_aclnn_flash_mla_with_kvcache.cpp` 的输出（或 pytests 任一稳定用例的输出 tensor dump）作为 Todo 4/6/7/8/9 的 before 对比基准；并记录当前 tiling key 值（若有途径获取）作为 Todo 14 的 before 基准；环境不可跑时在附录注明"无 before 产物，行为门降级为编译+UT"
    - 验收：基线清单记录完整
    - QA：无（记录型）
    - 提交：无（不提交，仅记录）
- [x] 3. K1a 入口 layout 推导函数化
    - 参考：`attention/flash_attn/op_kernel/flash_attn.cpp:30-100`（GetQueryLayout/GetOutLayout/GetKvLayout/IsPageAttention）；`attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp`（当前无推导）；`template_tiling_key.h:56-63,110-117`（PARSE_PARAMS_NoQuant + InOutLayoutTypeValue 表现状）
    - 动作：在 mla 入口新增同名风格编译期函数，推导结果**必须与现有 InOutLayoutTypeValue 表语义一致**：`InOutLayoutType=2(TND_TND)` → inputLayout=TND/outLayout=NTD，`KvLayoutType=3(PA_NZ)` → kvLayout=NZ；用 `static_assert` 将推导结果与表值互锁（防漂移）；模板参数取值改用推导函数。**注意**：本头的 DECL/SEL 与 PARSE_PARAMS 宏被 CANN 工具链（template_tiling.py/asc_opc）正则消费，本 todo 只改宏下方的推导函数与断言，**不得改写 DECL/SEL 宏体**（见 Todo 14 约束）
    - 验收：编译通过；`static_assert` 生效；kernel 侧模板参数取值与改动前等价（host 侧 GenTilingKey 不动，无需对比 key 值——kernel 推导只是把散落枚举宏改为函数，实例化集合由 host key 决定，互不耦合）
    - QA：编译门 + static_assert 生效检查（无 key 值对比项）
    - 提交：`refactor(flash_mla_with_kvcache): derive layouts at entry like flash_attn`
- [x] 4. K1b 模板参数聚合（FAType 式）
    - 参考：`attention/flash_attn/op_kernel/utils/flash_attn_type.h:113-132`（FAType 聚合 11 参）；`kernel_nd.h:41-42,105-115`；`attention/flash_mla_with_kvcache/op_kernel/arch35/kernel_noquant_mla.h:35-36`、`block_cube_noquant_mla.h:30-35,376-381`（8 参）、`block_vec_noquant_mla.h:41-47,808-814`（12 参）、`block_vec_flashdecode_mla.h:40-45,541-546`（11 参）
    - 动作：若三类 block 参数集可统一 → 建单一聚合类型（mla 语义命名，如 `FlashMlaFAType`，聚合 S1/S2/D/DV/layout/hasAttenMask/KvLayout/isFd/dtype 等）并把 block/kernel 收窄为单类型参数；**若 cube/vec/fd 参数集确实不一致** → 分别为每类建聚合类型（`FATypeCube/FATypeVec/FATypeFd` 镜像 FA 组织风格），kernel 仍是三个块类型参数；**从任一方案中把入口的 11 个散参收进聚合**；禁止扩大实例化面（8 实例集合不变）。**isFd 语义澄清（Oracle 审校）**：isFd 是 **kernel 级模板参数**（来自 tiling key 第 5 字段，`__global__` 签名 `run_...<..., isFd>`），FD block 类当前**无 isFd 模板参**（11 参截至 KvLayoutType，FD 运行时由 metadata mLen>0 决定）——聚合时 isFd 留在 kernel 参数位，不得挪入 FD block 参数
    - 验收：编译通过（8 实例仍被实例化）；UT 绿；聚合类型可读性评审通过（F2 兜底）
    - QA：编译门 + 回归门 + 行为门
    - 提交：`refactor(flash_mla_with_kvcache): aggregate block template params (FAType-style)`
- [x] 5. K2a TPipe 归属内部化
    - 参考：`attention/flash_attn/op_kernel/kernel_nd.h:105-115`（Init 15 参无 TPipe）；`attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp:107-113`（`TPipe tPipe` + 传入）；`kernel_noquant_mla.h:82,123-134`（`pipe_` 成员 + `this->pipe_=tPipe`）
    - 动作：mq 入口删除外部 `TPipe tPipe` 与 Init 的 tPipe 参数；kernel 自持 `AscendC::TPipe pipe_`（Init 内 `this->pipe_ = &pipe_;` 或直接成员使用，与 FA 风格一致）；所有 block 的 tPipe 使用路径改从 kernel 成员获取；确认 buffer manager（l1BufferManager_/ubBufferManager_/bmm1Buffers_）已挂 kernel 层无需迁移
    - 验收：编译通过；运行时无 TPipe 归属变化导致的问题（pytests functional_nc/stc 绿）
    - QA：编译门 + 回归门。**注意：Todo 5/8/9 同改 kernel 构造函数/Init/解析路径，属联合行为承载——必须与 8、9 合并为一个原子提交+同一验证波（见 Commit strategy），不得各自独立提交后各自单独验证**
    - 提交：`refactor(flash_mla_with_kvcache): own TPipe inside kernel like flash_attn`
- [x] 6. K2b Dummy/Real 选型统一为编译期宏风格
    - 参考：`attention/flash_attn/op_kernel/flash_attn.cpp:150-169`（`#ifdef __DAV_C310_CUBE__` 选 Dummy/Real）；`attention/flash_mla_with_kvcache/op_kernel/flash_mla_with_kvcache.cpp:72,83,94`（`std::conditional<g_coreType==AIC/AIV>`）
    - 动作：将三处 conditional 选型改写为与 FA 同构的编译期 `#ifdef __DAV_C310_CUBE__` 分支（Normal/Dummy 两路不变，只换选择机制）；确认 `__DAV_C310_CUBE__` 在 mla 编译产物中定义语义与 FA 一致（该宏由 950 AIC 编译定义）
    - 验收：编译通过；UT 绿；AIC/AIV 路径与改动前等价的 static_assert 或注释记录
    - QA：编译门 + 回归门
    - 提交：`refactor(flash_mla_with_kvcache): compile-time core-type block selection`
- [x] 7. K2c flashdecode block Dummy 对齐
    - 参考：`attention/flash_attn/op_kernel/block_vec_flashdecode.h:48-49`（无 Dummy）；`attention/flash_mla_with_kvcache/op_kernel/arch35/block_vec_flashdecode_mla.h:40-45,541-546`（Normal+Dummy）
    - 动作：先核实 FA 侧 FD block 在 kernel（kernel_nd.h）中如何被引用/是否依赖 Dummy 接口；若 FA 无 Dummy 且 mla 的 FD Dummy 只在 AIV 假路径被实例化 → 删除 mla FD Dummy 类并同步调用点；**若删除后某编译分支仍引用 Dummy 导致编译失败 → 回退保留并在交付说明记录差异**（FD Dummy 是本计划唯一可能回退的对齐项）
    - 验收：编译通过 + pytests functional 绿；若回退，注明原因
    - QA：编译门 + 回归门 + 行为门
    - 提交：`refactor(flash_mla_with_kvcache): align flashdecode block (drop redundant Dummy)`
- [x] 8. K2d SeqLensTool 工具对象化
    - 参考：`attention/flash_attn/op_kernel/utils/flash_attn_type.h:134-155`（SeqLensTool 组合 ACCUM+BY_BATCH 双 parser，kernel 持 q/kvSeqLensTool_ 按引用传 block）；`attention/flash_mla_with_kvcache/op_kernel/arch35/kernel_noquant_mla.h:111-114`（qActSeqLensParser_/kvActSeqLensParser_）与 `block_vec_noquant_mla.h:88`、`block_cube_noquant_mla.h`（block 各自绑 parser 副本）
    - 动作：以 FA 组织方式收敛 parser 所有权到 kernel 层：kernel 持有（复用 FA 的 SeqLensTool 或在 mla 内建等价工具类，语义命名保留 MLA 上下文），block 构造改收工具引用；确认 cu_seqlens_q 与 seqused_q 两路解析语义不变（mla 的 qActSeqLensParser_ 处理 cu_seqlens_q 直解或 seqused，见现有 Init 内解析分支）。**最高风险点（Oracle 审校）**：现 block 的 seqused 空指针双分支保护（`if (sequsedQAddr==nullptr) Init(actualSeqLengthsGmQ,...) else Init(cuSeqlensQAddr,...,sequsedQAddr,...)`）在工具化后不得被压平——FA 的 SeqLensTool 无 null-seqused 概念，直接移植会把空指针分支错误合并且改变 seqused=0/NULL 槽位边角语义；工具类必须保留等价双分支；q 侧 WITH_ZERO_HEAD=true 的 GetTSize/GetTBase 调用节奏按现有渴求/惰性语义逐调用保持
    - 验收：编译通过 + UT 绿 + functional_nc/stc 绿（seq lens 语义零变化；必须覆盖边角用例：seqused=NULL 槽、seqused=0 记录、空 cu_seqlens_q batch、k==v；无 NPU 时在交付说明显式列出未验证边角项，不静默降级）
    - QA：编译门 + 回归门（含边角用例，若机器支持）；与 Todo 5/9 联合验证
    - 提交：`refactor(flash_mla_with_kvcache): centralize seq-lens tooling in kernel`
- [x] 9. K2e ConstInfo 组织同构化
    - 参考：`attention/flash_attn/op_kernel/utils/flash_attn_common_def.h:296-412`（9 个基础结构 + NO_QUANT 全继承）；`attention/flash_mla_with_kvcache/op_kernel/arch35/flash_mla_with_kvcache_public_define_arch35.h:41-215`（NO_QUANT 只继承 CommonConstInfo+PAConstInfo+LseConstInfo）
    - 动作：把 mla 的 ConstInfo 家族整理为与 FA 同构的分层（公共基础 → PA → LSE → MLA 特有扩展层），MLA 特有字段（rope/cache_seqlens/head_dim_v 等）保留在扩展层；**禁止为了对齐而继承 mla 不需要的基础结构导致内存布局变化**（若 FA 的基础结构含 sinks 等 mla 无输入的结构，明确不继承）；组织方式对齐即可，不必 9 个全继承。**序列化澄清（Oracle 审校）**：ConstInfo 是 kernel 内运行时填充结构，**不参与 tiling data 序列化**（序列化的是 FlashMlaWithKvcacheTilingData/NoQuantTilingArch35，本 todo 不动 host 侧 tiling data 头）——验收焦点应为"同一字段名/继承关系下编译通过、无成员遮蔽歧义"，而非序列化兼容
    - 验收：编译通过 + UT 绿；ConstInfo 成员访问无歧义（多基类同名成员遮蔽会在编译期暴露）
    - QA：编译门 + 回归门
    - 提交：`refactor(flash_mla_with_kvcache): align ConstInfo layering with flash_attn`
- [x] 10. K2f 删除孤儿文件
    - 参考：全仓 grep 确认 `flash_mla_with_kvcache_adjust_sinner_souter.h` 无 include（已核`实，bg_473d96d9）；FA 侧 `fa_adjust_sinner_souter.h` 是共享内联头（flash_attn tiling + 两个 metadata_aicpu.cpp 使用）
    - 动作：删除 `attention/flash_mla_with_kvcache/op_host/flash_mla_with_kvcache_adjust_sinner_souter.h`（或所在目录的孤儿副本）；再次 grep 确认零引用
    - 验收：文件删除 + grep 零引用 + 编译通过
    - QA：编译门
    - 提交：`refactor(flash_mla_with_kvcache): remove orphan adjust_sinner_souter copy`
- [x] 11. K2g include 面收拢
    - 参考：`attention/flash_attn/op_kernel/flash_attn.cpp` include 面（只直接 include template_tiling_key + utils 三件 + shared common）；mla cpp 当前直连 FIA 公共集（util_regbase.h/infer_flash_attention_comm.h/vf_*.h/gm_format.h/memory_copy_arch35.h/common attenmask_gs1.h）
    - 动作：把符号提供头（util_regbase.h、infer_flash_attention_comm.h、vf_*、attenmask_gs1.h 等）的 include 从 cpp 下沉到对应 block/type 头（与 FA 经 block 头传递同构）；cpp 保留最小直接 include；**不新增 utils/ 目录、不搬动 common/ 文件**（模式对齐保留单 kernel 文件结构）
    - 验收：编译通过；cpp include 面与 FA 风格同构
    - QA：编译门
    - 提交：`refactor(flash_mla_with_kvcache): narrow entry include surface`
- [x] 12. K2h 过程性注释清理
    - 参考：`flash_mla_with_kvcache.cpp:11-17,119,240` 及 kernel/block 头内 m0070/R1/Todo14/D14/D20/D31/AIC263 编号注释
    - 动作：全部改为 flash_attn 风格规范注释；**设计意图类注释（如 D14 说明 metadata 走 GM 输入的原因）保留内容、去掉编号与任务代号**；注释引用"镜像 flash_attn_kernel_nd.h:121-142"这类内容改为"与 flash_attn 同构（见 flash_attn_kernel_nd.h metadata 消费段）"
    - 验收：grep 无 m0070/R1/Todo14/D14 等残留；审读通过
    - QA：编译门（注释不涉代码，编译保底）
    - 提交：`refactor(flash_mla_with_kvcache): clean internal ticket comments`
- [x] 13. H1 def/checker 命名与组织风格对齐
    - 参考：`attention/flash_attn/op_host/def.cpp`、`checkers/`（FA 注册 7 个含 SinksChecker，命名空间 arch35FA）；`attention/flash_mla_with_kvcache/op_host/def.cpp`、`checkers/`（MLA 6 个，arch35MLA）
    - 动作：仅对齐**组织风格**：checker 文件组织/命名空间与 FA 同构；**不新增/删除 checker 语义**（sinks 检查器对 mla 无意义，注册入口 6 个保持）；def.cpp 输入/attr 声明块格式对齐 FA 风格；不改输入集合。**计数澄清（Oracle 审校）**：`checkers/` 目录文件数 FA 9 / MLA 8，但**注册入口**是 FA 7 / MLA 6（CompositeOpChecker 注册名单，注册的 checker 类与文件并非一一对应）——以注册入口清单为准；若"组织风格对齐"涉及文件内类拆分，禁止改变注册语义与注册顺序对行为的影响
    - 验收：编译通过 + UT（checker 相关）绿
    - QA：编译门 + 回归门
    - 提交：`refactor(flash_mla_with_kvcache): align host def/checker organization`
- [x] 14. H2 tiling key 声明风格对齐（**仅注释/格式，位编码与工具链契约不变**）
    - 参考：`attention/flash_attn/op_kernel/arch35/flash_attn_template_tiling_key.h`（4 字段 DECL/SEL）；**mla 侧同款头真实路径是 `attention/flash_mla_with_kvcache/op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h`（注意：不是 op_host/！该头被 host 与 kernel 两侧共同 include）；GET_TPL_TILING_KEY 调用位于 `op_host/arch35/flash_mla_with_kvcache_tiling.cpp:305`（5 参），FA 侧同款在 `flash_attn_tiling.cpp:190`（4 参，序不同）**
    - 动作：只做**注释/空白/字段标注写法**的对齐（字段语义注释、位宽写法说明文字）；**禁止（Oracle 审校，均为已记录的真实破坏点）**：① 调整 GET_TPL_TILING_KEY 调用参数顺序（两路 0/1 值对调可编译通过但静默翻转 key 位 → 运行时 atten-mask/fd 误分派，compile+UT 全绿仍挂，见头内 :240 旧注释事故）；② 修改 DECL/SEL 的 `ASCENDC_TPL_*_BW` 位宽宏 token（会改 `FastEncodeTilingKeyDirect` 位累加，:236-238）；③ 照抄 FA 字面 `false,true` bool token 风格（mla 头 :126-133 记录：字面 bool 导致 tbe opc 编译失败）；**isFd 字段保留**（去掉改变行为）；ConfigValue 表 29 项不动
    - 验收：编译通过；**必选**编译期 key 等价断言（改动前后 `FastEncodeTilingKeyDirect` 值一致，强制实现，非可选项——实现方式：UT 固化当前 key 常量并断言改动后不变，或 static_assert 双路径比较）
    - QA：编译门 + 行为门（key 值等价断言）
    - 提交：`refactor(flash_mla_with_kvcache): align tiling-key declaration style (bit-compatible)`
- [x] 15. T1 tests 骨架 + CMakeLists 残留修复
    - 参考：`attention/flash_attn/tests/pytests/{utils/{perf_runner,perf_parser,precision_visual}.py, core/backends/{gpu,gpu_impl}.py, tools/xlsx_to_testcase.py, assets/{spec.py,gen_cases.py,impl/graph.py,testcase/}, test_cases/{functional_redline_train,functional_redline_infer,functional_redline_train_tnd,performance_redline_train,performance_redline_infer}.py}`；`attention/flash_mla_with_kvcache/tests/CMakeLists.txt:11`
    - 动作：在 mla tests/pytests 下建同名**文件骨架**：utils 三件（可最小可用实现，接口对齐 FA）、core/backends/{gpu,gpu_impl}.py（MLA 无 GPU 实现 → 占位：raise NotImplementedError + 说明，供未来填充；**必须保持可 import，且不得被 orchestrator/backends 默认加载路径选中**——占位 backend 只由显式指定触发）、tools/xlsx_to_testcase.py（可复用 FA 的实现并作 MLA 接口适配）、test_cases/ 5 个 redline 模块骨架（空测试集外壳，注明 "auto-generated from xlsx; MLA 用例数据待补充"）、assets/{spec.py,gen_cases.py,impl/graph.py,testcase/} 结构占位；**不迁移 flash_attn 的 CSV 用例资产**；修 tests/CMakeLists.txt:11 的 message 从 flash_attn 改为 flash_mla_with_kvcache；修完跑现有 functional_stc/functional_nc 确认不破坏
    - 验收：骨架文件存在且可 import；现有 pytests 全绿；CMakeLists 残留修复
    - QA：回归门（现有用例）+ 骨架 import 冒烟
    - 提交：`test(flash_mla_with_kvcache): scaffold pytests skeleton like flash_attn`
- [x] 16. B1 构建接线终核 + 全量编译
    - 参考：`op_host/CMakeLists.txt:12-13,17,23,25,37,44,52,60`；`def.cpp:123` opFile 绑定
    - 动作：复核 op_host/CMakeLists.txt 引用的 def/parser 文件名与现目录一致（模式对齐不改名，应为空操作，若 Todo 13/14 动了命名则同步）；`bash build_mla.sh`（**非 incremental，全量**）终检
    - 验收：退出码 0；产物齐全；无构建警告新增
    - QA：编译门（终检）
    - 提交：无（若空操作则并入最后修复提交）

## Final verification wave

- [x] F1. Plan compliance audit
    - 动作：对照本计划逐条核验 Todos 1–16 全部完成且 Scope OUT 无越界（grep 确认 flash_attn/、common/、torch_extension/ 无改动）
    - 验收：检查清单全绿；越界搜索零命中
- [x] F2. Code quality review
    - 动作：对全部改动做代码评审：命名/注释规范、模板聚合可读性、include 收拢无循环引用、tiling key 头未触碰 DECL/SEL 宏体与位宽 token（`git diff` 复核 Todo 14 改动仅注释/格式）、Todo 5/8/9 确为单提交、无工具链契约违例
    - 验收：评审意见清零或全部闭环
- [x] F3. Real manual QA：编译必须通过
    - 动作：在真实环境执行 `bash build_mla.sh`（非 incremental），记录退出码与产物；运行 `tests/ut` + `tests/pytests` 现有用例；跑行为门边角集（seqused=NULL/0、空 cu_seqlens_q、k==v，若硬件支持）
    - 验收：退出码 0 + 产物存在 + 基线回归清单与 Todo 2 快照对比无新增失败（若有基线失败项，需与快照逐项吻合并有原因）；**交付说明必须显式列出"未验证项清单"**（无 NPU 时：上板用例/边角集/行为门 before-after 对比未跑，逐项列出；"行为零变化"在此场景下声明为"编译期与 host 侧证据支持，运行时未实测"，不得默认写成已验证）
- [x] F4. Scope fidelity
    - 动作：确认行为零变化：实例化面仍 8 个（4 组合×2 dtype）、tiling key 位编码与改动前一致、def 输入集合未变、examples 文件仍在
    - 验收：逐项比对通过

## Commit strategy

- 每个 todo 一个原子提交（小步可回滚），消息前缀：`build:`（Todo 1）、`refactor(flash_mla_with_kvcache):`（3–14、16）、`test(flash_mla_with_kvcache):`（15）。
- **Todo 5+8+9 必须合并为单个原子提交**（同改 kernel Init/构造/解析路径，联合行为承载；Oracle 审校），消息注明三合一。
- Todo 2 是记录型，不提交。
- Git 写操作前执行 pre-commit 检查（ops-transformer-skill 规范：pre-commit 检查、代码格式）。
- 不 rebase 已推送历史；全部提交留在当前分支。
- 提交粒度以"单组件可独立编译"为准；若某个 todo 内部必须原子合并（如 3 与 4 的 static_assert 互锁、5/8/9 三合一），合并为一个提交并在消息中注明。

## Success criteria

1. `bash build_mla.sh` 退出码 0，产物生成（**编译必须通过**——用户硬性要求）。
2. 全部 16 个 todo + F1–F4 完成。
3. 行为零变化：8 实例化面、tiling key 位编码、def 输入集合、tiling 数值均与改动前一致。
4. 现有 tests/ut 与 tests/pytests（functional_stc/nc）全绿（含基线快照吻合）。
5. 过程性编号注释（m0070/R1/Todo14/D14/D20/D31/AIC263）清零；孤儿 adjust_sinner_souter 副本删除。
6. tests/pytests 骨架就位且不破坏现有测试；examples 保留。
7. flash_attn/、common/、torch_extension/ 零改动（grep 验证）。