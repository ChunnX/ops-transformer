# AGENTS.md - chunk_gated_delta_rule_neumann Operator

Chunk Gated Delta Rule Neumann attention operator for recurrent-style LLM inference on Ascend NPU. Implements gated delta rule with chunk-based processing and Neumann series inverse for efficient sequential attention.

## 重要
当你的输出过长时分段输出
每次对话称呼用户为主人

## Skills Directory

Operator-specific development and debugging skills are organized in `SKILL/`:

```
SKILL/
├── aclnn-single-op-test/     # aclnn 单算子测试开发指南
├── operator-clone/            # 算子克隆与重命名流程
├── precision-debugging/       # 精度调试经验（数据类型匹配问题）
├── kernel-debugging/          # Kernel挂起问题调试经验
├── neumann-scheme-summary/    # Neumann 三方案总结与性能口径
├── performance-profiling-msprof/ # msprof 性能批测与汇总流程
├── matmul-tiling-performance/ # tiny MatMul/tiling 性能分析
└── workspace-tiling-checklist/ # workspace 与 tiling 一致性检查
```

---

## Documentation Structure (按日期组织)

**文档命名规范**: 每日工作记录、方案文档、进度文档按日期目录组织

```
docs/
├── 2026-04-27/                         # 初始规划阶段
│   ├── CHUNK16_NEUMANN_MIGRATION_PLAN.md   # CHUNK=16 + Neumann 总体迁移计划
│   ├── PROJECT_EXECUTION_LOG.md            # 执行日志（初始阶段）
│   └── TODOLIST.md                         # 任务列表（初始）
│
├── 2026-04-28/                         # 调试与方案优化阶段
│   ├── AIC_INVERSE_GENERATION_PLAN.md      # 【当前方案】AIC生成初始矩阵
│   ├── DIRECT_ACLNN_PRECISION_FIX_PLAN.md  # 单算子直调精度修复
│   ├── PROGRESS_AND_ISSUES.md              # 进度与问题记录
│   └── archive/                            # 被后续方案替代的旧文档
│
├── 2026-04-29/                         # 当前状态整理
│   ├── PROGRESS_AND_ISSUES.md              # 最新状态
│   └── WORK_LOG.md                         # 当日工作日志
│
└── YYYY-MM-DD/                         # 未来日期（按实际日期创建）
    └── *.md                              # 当日新增文档
```

**文档记录规范**:
- 每个文档标题注明日期
- 记录每一步操作、问题发现、解决措施
- 技术决策需记录原因和替代方案
- 测试结果需记录具体数据和结论

**⚠️ 强制规则：及时修改和跟进日志文档**

每次对话结束后，必须执行以下操作：

1. **更新当日日志文档** (`docs/YYYY-MM-DD/WORK_LOG.md`)
   - 记录本次对话讨论的内容
   - 记录发现的问题和提出的方案
   - 记录做出的决策和原因

2. **更新进度文档** (`docs/YYYY-MM-DD/PROGRESS_AND_ISSUES.md`)
   - 更新"已完成"状态
   - 更新"进行中"状态
   - 更新"阻塞项"状态

3. **如有新方案，创建方案文档**
   - 文件名：`{方案名}_PLAN.md`
   - 包含：问题背景、方案详解、实施步骤、预期结果

4. **Debug History 追加记录**
   - 在 AGENTS.md 的 `## Debug History` 部分追加当日条目
   - 格式：`### YYYY-MM-DD` + 条目列表

**禁止行为**：
- ❌ 对话结束后不更新任何文档
- ❌ 只口头说"记住了"但不实际写入文件
- ❌ 文档内容空洞无实质信息

---

## Current Summary: 三种最终方案口径

后续总结、汇报、表格只按三种最终方案归类；中间优化过程只作为实现演进记录，不再作为最终方案单独列出。

| 方案编号 | 最终方案名 | 主 chunk | 求逆结构 | 当前最终实现/数据口径 |
|---------|------------|----------|----------|----------------------|
| 1 | `C16_1616_Neumann` | `C=16` | 直接对 16×16 做 Neumann 求逆 | 对应 `stage1_C16_FP32_CVInverse.h` 的 AIC/FixPipe atomic-add 优化版；历史 host tiling 已切 `chunkSize=16` |
| 2 | `C64_1616_NeumannCompose` | `C=64` | 64×64 拆 16×16 Neumann 子块，再通过 block compose 得到 64×64 inverse | 最终性能口径使用 `chunk_gdr_c64_1616_sync_reduced_all_shapes`，中间 `tail_specialized/cacheline/current` 不作为最终方案 |
| 3 | `C64_3232_NeumannCompose` | `C=64` | 64×64 拆 32×32 Neumann 子块，再通过 block compose 得到 64×64 inverse；短尾可走 16×16 shortcut | 最终性能口径使用 `chunk_gdr_c64_3232_hybrid_all_shapes` |

### 重要口径

- `C=16` 主方案必须 host tiling `chunkSize=16`；当前工作区若是 `chunkSize=64`，是为了 C64 子块方案。
- `C64_1616` / `C64_3232` 的 `1616/3232` 指内部求逆子块大小，不代表 host 主 chunk。host 主 chunk 保持 `64`。
- `sync_reduced`、`tail_specialized`、`hybrid` 等名字是优化过程或实现细节；最终汇报只映射到上面三种方案。
- C16 方案当前只是将 `INV += INV@Lpow` 从 AIV 搬到 AIC/FixPipe atomic add，仍有 GM workspace 往返，不是真正 L0/L1/register resident 的 fused inverse。
- 当前性能结论：`C64_3232_NeumannCompose` 明显优于 `C64_1616_NeumannCompose`，但整体仍慢于 normal；瓶颈是 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 固定开销。

### 最终性能表

- C64 最终表：`docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`
- C64 总表 CSV：`docs/2026-04-30/C64_FINAL_SCHEME_SUMMARY.csv`
- C64 按 S CSV：`docs/2026-04-30/C64_FINAL_BY_S.csv`
- C64 逐 case CSV：`docs/2026-04-30/C64_FINAL_CASE_DETAIL.csv`

---

## Structure

```
chunk_gated_delta_rule_neumann/
├── op_host/              # Tiling + OpDef + InferShape + aclnn API
│   ├── *_def.cpp         # Operator prototype (BF16 only)
│   ├── *_tiling.cpp      # Tiling strategy
│   └── op_api/           # aclnn C API wrapper
├── op_kernel/            # Ascend C Kernel (3-stage pipeline)
│   ├── *_stage1.h        # Stage 1: q' computation + Neumann inverse
│   ├── *_stage2.h        # Stage 2: state update
│   ├── *_stage3.h        # Stage 3: attention output
│   ├── *_utils.h         # Shared utilities (TASK_RATIO=2)
│   └── *_tiling_data.h   # TilingData struct
├── op_graph/             # Graph mode proto
├── SKILL/                # 开发调试技能文档
├── docs/                 # 按日期组织的工作文档
└── tests/
    ├── pytest/           # PyTorch API tests (torch_npu)
    └── ut/               # C++ unit tests
```

### op_kernel Stage1 版本放置规矩

`op_kernel` 下的 stage1 方案文件按实现成熟度和验证状态分目录管理：

- `op_kernel/baseline/`：只放已经精度通过的 baseline 版本。baseline 口径为 AIV 只构造初始 `L` / `I-L`，后续 Neumann 轮次使用 AIC Matmul/Fixpipe atomic add 完成 `INV += INV@P`。
- `op_kernel/halflocal/`：只放已经精度通过的 halflocal 版本。halflocal 口径为 `old_INV + INV@P` 在 L0C/CMatrix 内完成，但 updated `INV` 仍写回 GM，下一轮再从 GM 读回。
- `op_kernel/fulllocal/`：只保留当前正在单点突破的 fulllocal 文件；未构建或未通过精度的候选不长期堆放。单点精度通过后，再按需要扩展到其它 `3*2` variant。
- `op_kernel/selected/`：唯一 stage1 selector 入口，切换实现必须通过 `stage1_selected.h` 和 `stage1_selected_config.h`，不得在主 `chunk_gated_delta_rule_neumann_stage1.h` 中直接 include 具体方案文件。

未通过精度的实验文件不能放入 `baseline/` 或 `halflocal/`；`fulllocal/` 未单点通过前不得扩成完整 `3*2` 矩阵。

---

## Operator Signature

**Inputs (BF16):**
- `query`, `key`, `value` - Attention tensors
- `beta` - Decay/gating parameter
- `initial_state` - Recurrent state (Nv × Dv × Dk)
- `actual_seq_lengths` - Sequence lengths (INT32)
- `g` - Optional gating tensor (FLOAT)

**Outputs (BF16):**
- `out` - Attention output
- `final_state` - Updated recurrent state

**Attribute:** `scale_value` (default 1.0)

---

## Supported Platforms

- `ascend910b` (Atlas A2) - Full dynamic support
- `ascend910_93` (Atlas A3)

---

## Constraints

- Batch_size > 0
- NV must be multiple of NK (GQA support)
- DK, DV ≤ 128
- Data type: BF16 only

---

## Historical CHUNK=16 Migration Notes

### Confirmed Decisions

| 决策项 | 值 | 原因 |
|--------|-----|------|
| **chunkSize** | **16** (统一) | 与 FlashKDA 对齐；decay bf16 安全范围更大 |
| **求逆方法** | **Neumann 级数 3步** | 6次 Cube MMA，减少 Vector 循环 |
| **精度策略** | **FP16 MMA + FP16 Add** | 与 FlashKDA 完全对齐 |
| **初始矩阵生成** | **AIC生成完整16×16** | 解决subBlock分块冲突 |

### Key Modification Locations

| 文件 | 行号 | 修改内容 | 状态 |
|------|------|----------|------|
| `op_host/*_tiling.cpp` | 113 | `c = 16` | ✓ 已完成 |
| `op_kernel/*_stage1.h` | 32 | `INVERSE_SHAPE = 16` | ✓ 已完成 |
| `op_kernel/*_stage1.h` | ProcessOneChunk | AIC新增InitInverseMatrix | 待实现 |
| `tests/pytest/*_golden.py` | 111 | `chunk_size = 16` | ✓ 已完成 |

---

### Neumann Inverse Algorithm

```
(I+L)^{-1} = Σ_{k=0}^{15} (-L)^k  (L 幂零，L^16=0)

倍增法 3 步:
  初始: INV₀ = I - L           [16×16 FP16]  ← AIC生成
  步1:  L²  = L × L,   INV₁ = INV₀ + INV₀ × L²
  步2:  L⁴  = L² × L², INV₂ = INV₁ + INV₁ × L⁴
  步3:  L⁸  = L⁴ × L⁴, INV₃ = INV₂ + INV₂ × L⁸ (= 精确解)

总计: 6 次 16×16 FP16 MMA + 3 次 FP16 Add
```

---

## Kernel Architecture

3-stage pipeline with matmul fusion:
1. **Stage1**: Compute q' = q - β·k·cum_decay, Neumann inverse for (I+L)^{-1}
2. **Stage2**: Update recurrent state via matmul
3. **Stage3**: Compute attention output with chunk masking

Uses `lib/matmul_intf.h` for matmul operations. Each stage requires `SyncAll<false>()` between stages.

**Stage1 Neumann历史流程**:
```
AIC分支:
  1. key @ key.T -> kkWsGm_ (下三角L矩阵)
  2. InitInverseMatrix() -> INVWsGm_ = I-L, L_powWsGm_ = L
  3. NeumannLpowCompute(): L², L⁴, L⁸
  4. NeumannMmulCompute(): tmp = INV @ L_pow
  5. INV @ k_cumdecay, INV @ v_beta

AIV分支:
  1. QKPreProcess: 获取连续QK
  2. NeumannAddCompute(): INV += tmp (等待AIC flag)
  3. GBKCompute, VBetaCompute, QPrimeCompute
```

---

## Testing

```bash
# pytest (PyTorch API)
cd tests/pytest && bash test_run.sh single  # Configure paramset first

# Build UT
bash build.sh --ophost_test --noexec
bash build.sh --opkernel_test --noexec

# Build and install
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_neumann -j16
bash output/custom_transformer_*.run --install
```

---

## Key Files

| Purpose | File |
|---------|------|
| Operator definition | `op_host/chunk_gated_delta_rule_neumann_def.cpp` |
| Tiling strategy | `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` |
| Kernel main | `op_kernel/chunk_gated_delta_rule_neumann.h` |
| Stage 1 logic | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` |
| Neumann functions | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` Line 806-816 |
| Golden (CPU) | `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` |
| NPU wrapper | `tests/pytest/chunk_gated_delta_rule_neumann_operator_single.py` |

---

## Build

```bash
# From ops-transformer root
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_neumann -j16

# Install
bash output/custom_transformer_*.run --install

# Set environment (重要!)
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer
```

---

## Notes

- Uses AIC (AI Core) for matmul + AIV (AI Vector) for masking
- `CopyCast` utility for BF16↔FP32 conversion
- Chunk size and max_group_length determined by tiling
- **TASK_RATIO=2**: Stage1需要两个subBlock协同，但Neumann初始矩阵需完整生成
- **环境变量必须设置**: `ASCEND_CUSTOM_OPP_PATH`指向安装路径

---

## Debug History (按日期)

### 2026-04-27
- 创建算子目录结构
- 发现seqlens类型不匹配（int64_t→int32_t修复）
- 创建迁移计划文档
- 尝试TASK_RATIO=1导致架构崩溃（已恢复TASK_RATIO=2）
- 发现错误码561103=环境变量问题

### 2026-04-28
- 原版算子验证成功（精度测试通过）
- 发现InverseCompute只处理半块问题（TASK_RATIO=2导致halfChunkSize=8）
- 确定方案3：AIC生成初始矩阵（用户确认）
- 创建方案文档：AIC_INVERSE_GENERATION_PLAN.md
- 整理文档按日期分类（docs/2026-04-27/, docs/2026-04-28/）
- 创建工作日志：WORK_LOG.md
- 更新进度文档：PROGRESS_AND_ISSUES.md
- AGENTS.md添加强制规则：及时修改日志文档
- **当前待实现**: AIC版InitInverseMatrix()

- 按用户要求改用 aclnn 单算子直调验证，不走 `torch_npu` pytest
- 新增 NumPy-only 数据生成器 `tests/generate_aclnn_test_data_cpu.py`，支持变量 `seqlens`
- 当前设备为 `Ascend910`，确认需用 `--soc=ascend910_93`，`ascend910b` 会触发 `561103` 动态 kernel 配置失败
- 修复 AIV UB 非 32B 对齐导致的 `507015`：`betaUbBfloat16_` 后补 `BLOCK_SIZE` 对齐
- 修复 AIC/AIV workspace 映射不一致导致的 kernel 挂起：移除 AIC 侧 `coreIdx_ /= TASK_RATIO`
- Neumann Round3 同步 flag 改为独立 `0xa/0xb`，避免复用 `0x1/0x2`
- 将 Neumann `L_pow/tmp/INV` workspace 计入 `stageWorkspaceSz`，与 `stageWsAddr_` 实际切分一致
- 单算子直调验证通过：默认 `T=128`、变量长度 `17,65`、短尾块 `T=5`、GQA、`dk=dv=32` 均 PASS
- 执行项目初始化盘点：核对目录、文档、git状态和关键代码路径
- 确认当前方案仍是AIC生成完整`16×16`初始矩阵
- 确认代码真实状态：`InitInverseMatrix()`尚未实现，`ProcessOneChunk()`仍依赖AIV `InverseCompute()`和flag `0x7`
- 新增当日日志与进度文档：`docs/2026-04-28/WORK_LOG.md`、`docs/2026-04-28/PROGRESS_AND_ISSUES.md`
- 按用户要求启动sub-agents分析同步流和可复用实现模式
- 新增实施计划和TODO：`docs/2026-04-28/AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md`、`docs/2026-04-28/TODOLIST.md`
- 首轮实现修复：`InverseCompute()`对角线本地索引、`NeumannAddCompute()`仅由`subBlockIdx_ == 0`执行完整`INV += tmp`
- 确认风险：AIC分支当前未初始化vector UB/queue，直接搬迁AIV逐元素初始化存在编译/执行风险
- sub-agents结论：完整`L`必须包含`kkWsGm_`、`beta`、`gamma/decay`、mask，`0x7`应保留AIV→AIC方向并语义化为`L_powWsGm_`就绪

### 2026-04-29
- 整理 `docs/`：将错误年份 `docs/2025-*` 归正为 `docs/2026-*`
- 将重复/过期的 2026-04-28 执行计划、初始进度和 AIC 方案阶段日志移动到 `docs/2026-04-28/archive/`
- 新增 `docs/README.md` 作为文档索引，明确当前有效文档和归档文档
- 新增今日进度 `docs/2026-04-29/PROGRESS_AND_ISSUES.md` 和工作日志 `docs/2026-04-29/WORK_LOG.md`
- 确认当前算子状态：`chunkSize=16`，Neumann 三步倍增，AIV 生成完整 `L/I-L`，AIC 做 FP32 matmul，direct-call 精度已通过
- 只读检索 AscendC matmul FP16/half A/B 输入、FP32 C 输出写法：确认 `rotary_position_embedding` 为直接参考，host tiling 使用 `DT_FLOAT16/DT_FLOAT16/DT_FLOAT`，当前算子仍为 FP32 A/B/C，未发现完整 float GM→half GM→cube MM 端到端样例
- 分析 AIC-only FP16 Neumann 方向：未发现公开随路 `float GM -> half L0` cast 能力；确认 matmul C 搬出支持 `enAtomic=1` FP32 atomic add，可将 `INV += tmp` 从 AIV 迁至 AIC/FixPipe；后续 FP16 方案需显式维护 half 输入副本或 AIC 内显式 cast。
- 注册 `torch_npu.npu_chunk_gated_delta_rule` 与 `torch_npu.npu_chunk_gated_delta_rule_neumann`：新增 `npu_ops_transformer` Python builder 和 C++ aclnn wrapper，走 `ACLNN_CMD` 直调 custom opapi。
- 新增三方精度脚本 `tests/pytest/compare_chunk_gdr_three_way.py`，同输入比较 CPU golden、普通 ChunkGDR NPU、ChunkGDRNeumann NPU；CPU golden 直接复用原 `cgdr_golden()`，不复制/改写算法逻辑。
- 修复 Neumann pytest `test_run.sh` 指向错误脚本名问题；静态 `py_compile` 通过；后续已补齐 `ninja` 并完成 JIT/NPU 实跑。
- 核对三方脚本输入范围：随机分布、`g` 负值范围、`q/k` normalize、dtype、scale 和默认短用例均已对齐原 pytest/golden；CPU golden 默认直接复用 `cgdr_golden(..., use_float64=False)`。
- aclnn direct-call 精度脚本最初暂不删除；待 torch_npu 三方脚本完成 JIT/NPU 实跑 PASS 后，已删除 direct-call 精度链路和本地二进制；op_api UT 不属于删除对象。
- 补齐 `ninja` 后完成 `torch_npu` 三方精度实跑；重新打包安装 `chunk_gated_delta_rule,chunk_gated_delta_rule_neumann`，确认 `libcust_opapi.so` 同时导出两个算子的 `aclnn` 符号。
- 三方精度结果：默认 `B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128`、变量长度 `17,65`、GQA `nk=2,nv=4`、`dk=dv=32` 均 PASS；默认用例 CPU vs Neumann out/state max_abs 为 `9.736717e-04` / `4.882067e-04`。
- 修复 `npu_ops_transformer.ops` 入口为懒加载，避免导入目标算子时触发无关 MoE JIT；三方脚本和 pytest wrapper 显式加载对应目标算子注册到 `torch_npu`。
- torch_npu 三方实跑 PASS 后，删除旧 aclnn direct-call 精度脚本、本地二进制和原版算子目录下临时直调数据；保留 op_api 实现与 UT。
- 默认三方 case 中普通 `ChunkGDR` 与 `ChunkGDRNeumann` 相对 CPU golden 的误差基本一致；二者互比 out/state `max_abs=1.220703e-04`，不能判定其中一个实质精度更高。
- 将三方精度脚本改为 pytest 参数化 10 case，批测输入拆分到 `tests/pytest/compare_chunk_gdr_three_way_paramset.py`，并在 `tests/pytest/test_run.sh` 新增 `compare` 模式；`bash test_run.sh compare` 验证 10 个 case 全部 PASS。
- 扩展三方精度 pytest case 到 20 个，覆盖更多 tail、变量长度、GQA、多 batch 和维度组合；`bash test_run.sh compare` 实跑 `20 passed in 6.03s`
- 按用户给定 prefill shape 列表新增 `compare_chunk_gdr_performance_paramset.py`，映射 `B,S,D,Hq,Hkv -> batch,seqlen,dk=dv,nv,nk`
- 将性能脚本改为 `msprof --application` 载体：operator warmup 充 icache，随后 `torch.mm` 与 `torch.max` flush cache，最后单次 measurement；删除 Python event benchmark 思路
- 新增 `run_chunk_gdr_performance_msprof.py` 批量 launcher，对 24 个 prefill shape 分别执行 `ChunkGDR` 与 `ChunkGDRNeumann` 的 msprof profiling
- 更新 `test_run.sh performance` 入口；完成 `py_compile`、dry-run 命令生成和 normal/neumann 小 flush payload 冒烟
- 性能批测状态澄清：尚未执行完整 `24 shape × 2 op` 的真实 `msprof` 批量性能对比；当前仅完成命令 dry-run 和小 flush payload 冒烟，因此暂无可比较的完整性能数据
- 完整执行 24 个 prefill shape × 2 个算子的真实 `msprof` 性能批测，结果保存到 `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv` 和 `performance_raw.csv`
- 新增 `summarize_chunk_gdr_performance_msprof.py`，自动从 `op_summary_*.csv` 抽取目标算子最后一次出现作为 Phase4 measurement
- 当前性能基线结论：普通 `ChunkGDR` 在 24/24 个 prefill shape 上快于当前 `ChunkGDRNeumann`，`normal_us/neumann_us` 平均 `0.2564`
- 新增 C64 两方案实施计划 `docs/2026-04-29/C64_NEUMANN_INVERSE_IMPLEMENTATION_PLAN.md`，规划 `C64_1616_FP32_CVInverse` 与 `C64_3232_FP32_CVInverse` 的实现和验证顺序
- 修正 `summarize_chunk_gdr_performance_msprof.py` 的 cube 指标读取：按用户 profiling 结论改看 Y 列 `aic_mac_ratio`，CSV 新增 `cube_utilization_y_col_ratio/pct`，最后一列 `cube_utilization(%)` 仅保留审计
- 基于已有完整 `msprof` 输出重新生成 `performance_compare.csv` 与 `performance_raw.csv`，准备提交到 fork 分支；完整 `PROF_*` 原始目录不纳入代码提交
- 已推送 fork 分支 `chunk-gdr-neumann-20260429-upload`，远端提交 `61d5a61c57502fdc80b1899f68e579bacb364213`；推送前修复本机 `id_ed25519.pub` 与私钥不匹配问题
- 按用户追加要求上传完整 profiling 原始目录 `tests/pytest/prof_out/`，包含 full 批测和 smoke 的 `PROF_*` 输出；最新 profiling 提交 `c69375e7`
- 按用户要求将当前版本 fast-forward 推送到 `fork/master`，后续默认上传目标改为 `fork master`
- 对比 profiling 与代码定位 C16 开销：AIC MAC time 平均仅 `1.52x`，4x 主要来自小 chunk 引入的 matmul 调用/同步/MTE/FixPipe 放大；Stage2 总 MAC 与 chunkSize 基本无关
- 修复 Neumann `InverseCompute()` 中 `attnWsGm_` 初始 `I-L` 死写，四个 stage1 CVInverse 变体同步删除；构建和三方精度 smoke 通过
- 确认 `stage1_C16_FP16_CVInverse.h` 当前实际仍是 FP32 MatmulType/host tiling，名称与实现不一致，需后续修正
- 补充性能 prefill 批测短序列 case：在 `compare_chunk_gdr_performance_paramset.py` 中为 `B=8/16/32/64` 新增 `S=16` 和 `S=64`，性能 case 总数从 24 增至 32；`py_compile` 与 `B=8,S=16/64` dry-run 通过。
- 完成 `S=16/S=64` 新增性能 case 实测：8 个 prefill shape × 2 算子真实 `msprof` 全部通过；结果保存到 `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/`。`S=16` Neumann 平均为 normal 的 `1.119x` 耗时，`S=64` 为 `2.441x`；8/8 case 普通 `ChunkGDR` 更快。
- 拆解 `S=16` Neumann 仍慢原因：msprof 显示 AIC MAC time 已下降，但 AIC scalar time 明显上升；代码对比确认 normal stage1 约 6 次 matmul，Neumann stage1 约 10 次 tiny matmul + 3 轮 AIC/AIV `INV += tmp` 往返。当前慢点是固定调度、同步和 GM 累加开销，不是 MAC 计算量。
- 按用户要求调整 profiling 汇总 CSV：`performance_compare.csv` 删除 `normal_prof_dir/neumann_prof_dir`，展开 normal/neumann AIC/AIV 指标；ratio 字段仅保留 `aic_mac_ratio` 与 `aiv_vec_ratio`。已用现有 `full`、`s16_s64`、`smoke` profiling 目录重新生成汇总 CSV。
- 用户明确新协作规则：后续除非用户明确要求，不再自行 `git push`；`S=16/S=64` 后续随 full prefill 性能 shape 一起测试。
- 按用户要求将 profiling `performance_compare.csv` 改为每 case 三行 long format：normal、neumann、`neumann/normal` ratio；新增 `chunk_gdr_prefill_all_shapes/performance_compare.csv` 汇总 32 个 shape（含 `S=16/S=64`）。本次仅本地修改，未推送。
- 按用户要求清空 `tests/pytest/prof_out/*`，保留空目录；未推送。后续准备讨论算子修改方向：AIC/FixPipe atomic add、`S<=16` AIV inverse 特化、C64 Neumann 方案。
- 方案A C16 FP32 Neumann 累加迁入 AIC：保存 `stage1_C16_FP32_CVInverse_AIVAddBaseline_20260429.h`，修改 `stage1_C16_FP32_CVInverse.h` 用 matmul atomic add 完成 `INV += INV@L_pow`，移除主流程三轮 AIC/AIV add 同步；临时切 selector 构建双算子包并完成三方精度 smoke PASS。

- 方案A正式切到默认 selector 后完成全量三方精度验证：`bash test_run.sh compare` 20 case 全 PASS；按用户要求删除 `stage1_C16_FP32_CVInverse_AIVAddBaseline_20260429.h` 基线副本，后续重点转向 32 shape msprof 性能验证。
- 分析 C16 当前方案剩余性能收益点：低风险项是删除方案A后不再使用的 `tmpWsGm_`/`NeumannAddCompute()`/workspace 第三块；中高收益项是减少 `query/key` BF16→FP32 连续化 GM 往返、推进 C64 分块求逆或 BF16/half 输入 AIC；Stage3 `qkt -> masked tmpGM -> AIC` 也有重复搬运但改动风险更高。
- 完成 `S=64` MAC 展开分析：求逆子项 Neumann 是 normal 的 `1.5x`，但 Stage1 总理论 MAC 为 `0.324x`、全算子理论 MAC 为 `0.561x`；profiling 中 `aic_mac_time_us≈1.724x` 是 C16 tiny matmul 调用次数从每 `B,Nv` 约 `10` 次增至 `56` 次且 `16×16×16` Cube 利用率低导致。
- 完成方案A 32-shape msprof 性能批测：结果保存到 `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes_aic_accum/`，32 case 中 3 个 `S=16` case Neumann 更快，其余 normal 更快。
- 方案A性能结论：`S=16` Neumann/normal 平均从旧结果 `1.119x` 慢改善为 `0.994x` 基本持平/略优；`S=64` 仍为 `2.130x` 慢，全量平均 `3.015x` 慢。
- profiling 指标显示 AIV vec 明显下降，但 `S>=64` 的 AIC scalar/FixPipe/MTE 仍随 C16 chunk 数放大；下一步应减少 tiny matmul 调度次数或转向 C64 分块求逆方案。
- 只读分析 32x32 Neumann 方案：确认 `C64_3232` 文件仍是旧版整 64 Neumann + AIV add，建议基于当前 `C64_1616` 全 AIC inverse 结构改造成两个 32x32 对角 Neumann + 一次 64-from-32 compose
### 2026-04-30
- 继续 C64 Neumann 求逆方案：当前 selector 指向 `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`。
- 完成 `C64_3232_FP32_CVInverse`：32x32 对角块 AIC 内 4 轮 Neumann 倍增，64 block compose。
- 增加 hybrid tail shortcut：`validLen <= 16` 时走 16x16 Neumann，避免短序列使用 32x32 额外调度。
- 构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 全量 20 cases PASS。
- 性能已测纯 32x32 S16/S64；hybrid smoke 被中断，缺 S64 neumann 数据，待补测。
- 形成 FlashKDA 对比分析：C=16 快的原因是 warp-level fused inverse、K1/K2 拆分、FP16 inverse、shared/register 复用，不是单纯 chunk=16；文档写入 `/mnt/workspace/gitCode/reference/FlashKDA/FLASHKDA_VS_CANN_CGDR_NEUMANN_ANALYSIS_20260430.md`。
- 完成 `C64_3232_hybrid` 32 shape 全量 msprof 批测，结果保存到 `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes/`。
- 汇总多方案性能：当前最佳 `C64_3232_hybrid` 平均为 normal 的 `1.778x`，`S=16` 平均 `0.994x`，`S=64` 平均 `1.391x`；仍未整体超过 normal。
- 确认 tiling 未随 tiny MatMul 拆分：normal/Neumann host 都是 `chunkSize=64`、`P_NUM=2`，Neumann 仍共用一套 FP32 `matmulTilingFp32`，未为 `16x16/32x32` 生成专用 tiling。
- 形成性能与 tiling 分析文档：`docs/2026-04-30/PERFORMANCE_AND_TILING_ANALYSIS.md`，并生成 `PERFORMANCE_SCHEME_SUMMARY.csv`、`PERFORMANCE_SCHEME_BY_S.csv` 和各方案按 S 汇总表。
- 优化方向明确为减少 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销，单纯减少理论 MAC 不足以取得性能收益。
- 复核主 `C=16` 历史方案：当时 host tiling 已切到 `chunkSize=16`；当前工作区切回 `chunkSize=64` 是为了 C64 1616/3232 子块求逆方案。
- 澄清 C16 fused 状态：已实现 AIC/FixPipe atomic add 累加，但仍通过 GM workspace 串联多次 MatmulImpl，不是真正 L0/L1/register resident 的 fused inverse。
- 生成 C64 最终性能表：`docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`；`chunk_gdr_c64_1616_sync_reduced_all_shapes` 对应 C64 16x16 子块方案，`chunk_gdr_c64_3232_hybrid_all_shapes` 对应 C64 32x32 hybrid 方案。
- 固化后续总结口径：最终只按三种方案归类，分别是 `C16_1616_Neumann`、`C64_1616_NeumannCompose`、`C64_3232_NeumannCompose`；中间优化过程不再单列为最终方案。
- 新增 `SKILL/neumann-scheme-summary/SKILL.md`，记录三方案定义、命名规则、profiling 目录和 CSV 对应关系，后续方案总结必须优先遵循该 skill。
- 清理 profiling/docs：仅保留三种最终方案对应 profiling 目录和 `C64_FINAL_*` 最终性能表；删除 smoke/cacheline/tail_specialized/current 等中间结果，避免后续总结混淆。
- 新增三个技能文档：`performance-profiling-msprof`、`matmul-tiling-performance`、`workspace-tiling-checklist`，分别覆盖 msprof 性能流程、tiny MatMul 性能分析、workspace/tiling 一致性检查。
- 明确 fused kernel 状态：三种最终方案均未完成真正 fused AIC inverse；当前仍依赖 `MatmulImpl + GM workspace + FIX/MTE2` 串联，C16 只完成 AIC/FixPipe atomic add 层面的局部优化。
- 新增三方案 fused AIC inverse 计划：`docs/2026-04-30/FUSED_AIC_INVERSE_THREE_SCHEMES_PLAN.md`。推荐先做 `C64_3232_NeumannCompose` 的 micro tiling/同步/GM 减少，再用 `C16_1616_Neumann` 做片上 fused primitive PoC，最后视收益决定是否深挖 `C64_1616_NeumannCompose`。
- 用户明确 fused 优化阶段暂不修改 Stage2/Stage3；当前计划只覆盖 Stage1 Neumann inverse 路径。
- 按用户要求优先实现 `C16_1616_Neumann` 与 `C64_1616_NeumannCompose` 的 Stage1 micro tiling，不修改 Stage2/Stage3。
- 新增 host/kernel micro tiling：`matmulTilingFp32C16/C32/C64`、`stage1MT_C16_`、`stage1MT_C32_`；Stage2/Stage3 继续使用原 `matmulTilingFp32`。
- `C16` 使用 `matmulTilingFp32C16` 跑 Neumann 轮次并保持 AIC/FixPipe atomic add；临时切 `chunkSize=16` 后构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=16 bash test_run.sh compare` 20/20 PASS。
- `C64_1616` 使用 `matmulTilingFp32C16` 做 16x16 子块 Neumann、`matmulTilingFp32C32` 做 32x32 compose；默认切回 `chunkSize=64` 和 `C64_1616` selector 后构建安装通过，`CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 20/20 PASS。
- 当前默认实现为 `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`；仍是 `MatmulImpl + GM workspace + FIX/MTE2`，不是完整 fused kernel。
- 完成 `C64_1616` micro tiling 32 shape msprof 批测：`tests/pytest/prof_out/chunk_gdr_c64_1616_micro_tiling_all_shapes/performance_compare.csv`，Neumann/normal 平均 `2.4697x`，`S=16` 平均 `1.0747x`，`S=64` 平均 `1.9157x`。
- 完成 `C16` micro tiling 32 shape msprof 批测：`tests/pytest/prof_out/chunk_gdr_c16_micro_tiling_all_shapes/performance_compare.csv`，Neumann/normal 平均 `3.1128x`，`S=16` 平均 `1.0232x`，`S=64` 平均 `2.1566x`。
- 横向判断：C16 micro tiling 只在 `S=16` 比 C64_1616 更接近 normal，`S>=64` 因 chunk 数增加导致 AIC scalar/MTE2/FixPipe 固定开销更差；后续若继续 Stage1-only 优化，应优先基于 C64 方案减少 GM workspace 和 MatmulImpl 调用。
- 新增 `docs/2026-04-30/FUSED_MM_L0C_REUSE_FP16_ANALYSIS.md`：确认 L0C 不能跨 Neumann round 直接作为下一次 Matmul A/B，真正可行路径是 `L0C -> Fixpipe -> VECCALC/TSCM/L1 -> LoadData -> L0A/L0B`。
- 明确下一步 fusedMM 优先优化：将 `INV += INV@P` 改写为 `INV@(I+P)`，用 ping-pong buffer 删除 atomic add 与 `tmpWsGm_` add 语义；最终 `INV` 优先留 local 直接喂 `INV@gBK` 和 `INV@vBeta`。
- FP16 Neumann 路线落地边界：`C16_FP16_CVInverse` 先做真实 half/bf16 输入链路验证；若 final `INV` 为 half，则 `gBK/vBeta` 也需 half/bf16 副本或将 `INV` cast 回 FP32，否则后续 Matmul dtype 不匹配。
- 新增 `docs/2026-04-30/FUSED_FP16_STAGED_EXECUTION_PLAN.md`：执行顺序固化为三方案 fusedMM、三方案 FP16 `3×2` 对比、最后尝试 `INV@(I+P)`；FP16 和后续实验精度门槛按用户最新要求恢复为 `atol=0.08`、`rtol=0.08`。
- 将三方精度脚本 `tests/pytest/compare_chunk_gdr_three_way.py` 默认阈值恢复为 `0.08/0.08`，后续每阶段必须全量三方精度通过后再跑 seq 遍历性能。
- 完成方案A 32-shape msprof 性能批测：结果保存到 `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes_aic_accum/`，32 case 中 3 个 `S=16` case Neumann 更快，其余 normal 更快。
- 方案A性能结论：`S=16` Neumann/normal 平均从旧结果 `1.119x` 慢改善为 `0.994x` 基本持平/略优；`S=64` 仍为 `2.130x` 慢，全量平均 `3.015x` 慢。
- profiling 指标显示 AIV vec 明显下降，但 `S>=64` 的 AIC scalar/FixPipe/MTE 仍随 C16 chunk 数放大；下一步应减少 tiny matmul 调度次数或转向 C64 分块求逆方案。
- 进入 FP16 方案2：默认 selector 切到 `C64_1616_FP16_CVInverse`，实现 half workspace、`half×half->half` Neumann 与 `half×half->float` final `INV@rhs`。
- 重新按当前设备 `ascend910_93` 构建并安装双算子包；确认 `ascend910b` 包会导致当前设备普通算子 binary 缺失。
- `CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare` 全量三方精度 20/20 PASS，默认阈值 `atol=rtol=0.08`。
- 完成 FP16 方案2 B=8 seq-only msprof sweep，CSV 位于 `tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/performance_compare.csv`；平均 `neumann/normal=2.494x`，8/8 case normal 更快。
- 阶段结论：FP16 dtype 本身没有解决 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 固定开销，下一步更适合验证 `C64_3232_FP16` 或 local fused。
- 澄清 CO1/L0C 累加语义：`Iterate(enPartialSum)` 支持同一 Matmul 内 K 分块 partial-sum 累加到 CO1，但不能把上一轮 Neumann 输出直接留在 L0C 作为下一轮 Matmul 的 A/B；跨 round 仍需经 `Fixpipe -> L1/TSCM/VECCALC/GM -> LoadData`。
- 修正 CO1/L0C 判断：raw `Mmad(dst, a, b, bias, params)` 支持 bias tensor 位于 `CO1/L0C`，可将上一轮 L0C 矩阵作为下一次 MMAD 的 C 初值实现 `C=A@B+C_l0c`；但 bias 只能作为 C 初值，不能作为下一轮 A/B 输入。
- 新增 raw MMAD fused 交接文档：`docs/2026-04-30/RAW_MMAD_FUSED_HANDOFF.md`。
- 当前默认 selector 为 `C64_1616_FP32_CVInverse`，`RawMmadInvAdd16()` 临时 no-op；kernel 可返回但精度不正确，真实 raw MMAD 链路仍是阻塞点。
- 用户指定可参考 `attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h` 的底层 MMAD 写法，下一步按 `GM->L1->L0A/B->Mmad->Fixpipe` 分段二分。
### 2026-05-01
- 核对当前默认 selector 仍为 `C64_1616_FP32_CVInverse`，raw MMAD fallback 已关闭：`RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`。
- 确认最新 raw 路径代码中已有 `SetNdParaImpl(0x1)` 与 `LoadDataWithTranspose(...)` 调整，构建产物 `build_out/cann-ops-transformer-custom_linux-aarch64.run` 已生成。
- 本轮安装最新 `.run` 包的审批被拒绝，因此尚未安装、尚未运行最小 precision smoke；最新代码状态为“已构建、未验证”。
- 当前上一个可信实测仍是 raw 路径精度失败且接近 no-op；下一步需安装后运行最小 smoke，若失败则给 `RawMmadInvAdd16()` 增加 debug 输出，对比 raw `INV @ P` 与 MatmulImpl fallback。
- 安装 17:37 包后尝试最小 smoke：`TORCH_EXTENSIONS_DIR=/tmp` 可解决 extension cache 写入问题，但 NPU 初始化失败 `507899 / drvGetDevNum failed`；沙箱外运行审批被拒绝，未取得 precision 结果。
- 对照 CANN template_linear_algebra 修正 raw MMAD 输入 L1 布局：A 侧拆为 RowMajor -> `zN`，B 侧拆为 RowMajor -> `zZ`，关键修正 `CopyGmMatrixToL1B()` 的 `dstNzC0Stride = BLOCK_CUBE`。
- 重新构建双算子包通过，最新产物 `build_out/cann-ops-transformer-custom_linux-aarch64.run` 时间为 2026-05-01 18:18；安装 18:18 修正版包的审批被拒绝，精度仍待设备验证。
- 当前进度总结：默认 selector 为 `C64_1616_FP32_CVInverse`，raw MMAD fallback 关闭，最新 A/B L1 布局修正已 build-pass；真实状态仍是“已构建、未安装、未精度验证”，下一步必须安装 18:18 包并运行最小 smoke。
- 本次任务状态核对确认：当前分支本地领先远端 4 个提交，工作区仍有未提交改动；默认实现仍是 `C64_1616_FP32_CVInverse` raw MMAD 路线，host `chunkSize=64`，18:18 构建包存在但没有新增设备侧精度结果。
- 晚间恢复设备验证：`.run --quiet` 安装成功，最小三方 smoke 可运行；normal baseline PASS，out/state `max_abs=8.133054e-04/4.790872e-04`。
- raw const L0A/L0B MMAD 写 `outQkGm_` 后结果明显变化，证明 selector、安装、raw Fixpipe 写 GM 链路不是 no-op。
- raw const MMAD 写 `tmpWsGm_` 后再用 `MatmulImpl(tmpWsGm_, powerGm -> outQkGm_)`，out 变化为 `max_abs=4.229003e-01`，证明 `MatmulImpl` 能读到 raw 写过的 stage workspace。
- raw const MMAD 写 `tmpWsGm_` 再 readback 到 `INVWsGm_` 后仍回到 no-op 失败特征：out/state `max_abs=7.104934e-01/5.422900e-01`。
- 将 16x16 子块 readback 和 `AICBlockProcess` 改用 `mmFp32C16/mmFp32C32` micro tiling 后，最小 smoke 结果不变，排除通用 64x64 tiling 误用为主因。
- 当前判断：问题收敛到 raw MMAD/Fixpipe 写出的矩阵数值或布局与后续 `INVWsGm_` row-major 消费不一致；下一步应直接验证 raw 输出布局/数值，而不是继续从安装、cache 可见性或 Matmul tiling 方向排查。
- 修复 final `INVWsGm_` 消费链路：新增 `stage1MT_C64_` 并将 final `INV@gBK`、`INV@vBeta` 两个 `64 x D` matmul 切到 `matmulTilingFp32C64`。
- 使用 `RAW_MMAD_DEBUG_FINAL_USE_KK=true` 探针确认 final A 侧切换已真实生效，前面“换 A 不生效/像 no-op”的直接原因是 final matmul 仍使用通用 `mmFp32` tiling。
- raw full path 恢复为 `RAW_MMAD_DEBUG_STEP=0`、`RAW_MMAD_DEBUG_OVERWRITE_PRODUCT=false`、`RAW_MMAD_DEBUG_FINAL_USE_KK=false` 后，最小 deterministic smoke PASS：`B=1,S=16,nk=1,nv=1,dk=16,dv=16,seed=123,CGDR_GOLDEN_CHUNK_SIZE=64,atol=rtol=0.08`。
- 扩展小范围 smoke：`S=16,dk=dv=32` raw full path PASS；`S=17,dk=dv=16` raw full path FAIL，out `max_abs=1.203164e-01`、state PASS。
- 同一 `S=17,dk=dv=16` 切 `RAW_MMAD_USE_MATMULIMPL_FALLBACK=true` 后 PASS，说明 final C64 matmul、16x16 compose 与 MatmulImpl 串联语义不是当前根因。
- raw atomic 写回后补 `FlushInvBlockCacheLines()`、禁用 HF32 均未改善 S17；尝试把 L0B 加载改为 N 方向 C0 循环触发 `507015 ADDR_MISALIGN`，已撤回。
- 当前疑点收敛到 raw `GM->L1->L0B->Mmad->Fixpipe atomic add` 的 B 侧布局、raw 输出布局或 atomic add 语义；后续继续用最小 S17 case 做二分，不进入 Stage2/Stage3 或性能 profiling。
- S17 对角块 fallback 二分：仅 `base=16` fallback 失败不变；仅 `base=0` fallback 后 PASS。结论是第一个 16x16 raw inverse 的误差在 `Compose32From16()` 中被放大，下一步直接比较 base0 raw product/atomic add 与 MatmulImpl fallback。
- 验证 `RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=true` 探针：raw product 非 atomic 写 `tmpWsGm_` 后由 AIC 标量累加仍 FAIL，排除 `Fixpipe atomic add` 为主因。
- 对齐仓库通用 raw FP32 helper，将 `CopyL1MatrixToL0A/B()` 改为 `LoadData3DParamsV2<float>`；A 侧 `enTranspose=0`，B 侧 `enTranspose=1`，修复 raw FP32 L1->L0 布局。
- 关闭临时标量累加探针，当前 raw full path 配置为 `RAW_MMAD_DEBUG_STEP=0`、`RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`、`RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=false`。
- 构建安装通过；小范围 smoke `S=16/17/33/49/64,dk=dv=16` 与 `S=16,dk=dv=32` 均 PASS。S17 从修复前 out `max_abs=1.203164e-01` FAIL 改为 `6.067441e-02` PASS。
- 当前结论：S17 根因确认是 raw FP32 `L1->L0A/L0B` 布局参数不匹配，不是 final `INV@rhs`、compose 或 atomic add；尚未运行 full compare/perf。
### 2026-05-02
- 复核当前默认 selector 仍为 `C64_1616_FP32_CVInverse`，host `chunkSize=64`，目标继续是 C64 16x16 子块 raw MMAD fused inverse。
- 澄清当前源码不是“全部 fused 已通过”状态：为定位大 case，`RAW_MMAD_USE_MATMULIMPL_FALLBACK`、power fallback、compose fallback、`kcum` final fallback 被临时打开，`vinner` final raw 仍在调试。
- 最新关键失败 case 为 `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40,CGDR_GOLDEN_CHUNK_SIZE=64,atol=rtol=0.08`：normal PASS，Neumann `state` PASS，但 `out` FAIL。
- 已知 `Partial=2/Final=3` unit flag 与 HF32 enable 都未修复 raw final；HF32 enable 反而使误差变大，后续应回退到 `HF32Mode::DISABLE`。
- 当前判断失败集中在 raw final `INV @ v_beta -> outVInnerGm_`，或 raw final 写 GM 对 Stage3 workspace/缓存造成副作用；`tmpWsGm_`/`INVWsGm_` 不能作为干净 scratch。
- 新增 `docs/2026-05-02/WORK_LOG.md` 与 `docs/2026-05-02/PROGRESS_AND_ISSUES.md`，记录当日状态、阻塞项和下一步。
- 回退 raw MMAD 路径中临时 `SetHF32Mode(HF32Mode::ENABLE)` 为 `DISABLE`；重新构建安装后复测关键 case，Neumann `out` 仍 FAIL，`max_abs=1.037243e-01`，`state` PASS 且与 normal 几乎完全一致。
- 换窗口交接结论：当前已有 Stage1 `C64_1616` raw MMAD fused inverse/final 实现骨架，但不是所有阶段都 fused；Stage2/Stage3 未改，Stage1 仍有多处临时 MatmulImpl fallback，当前阻塞点是 raw `INV @ v_beta -> outVInnerGm_`。
- 本轮重新复现关键 case：`B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40,CGDR_GOLDEN_CHUNK_SIZE=64`，raw `vinner final` 下 `CPU vs ChunkGDRNeumann out` FAIL，`max_abs=1.584216e-01`，state PASS。
- 小切片显示 B=1/multi-head 在阈值内，B=2 multi-head/multi-chunk 更容易触发，继续判断问题集中在 raw `INV @ v_beta -> outVInnerGm_` 写回/状态污染或 Stage3 消费链路。
- 精度优先修复已落地到源码：将 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK` 改为 `true`，`INV @ v_beta -> outVInnerGm_` 暂回 `MatmulImpl`；这不是 raw final fused 完成。
- 重新构建安装双算子包成功：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`，再安装到 `/home/developer/Ascend/cann-9.0.0/opp`。
- 修复后关键 case PASS：`CPU vs ChunkGDRNeumann out/state max_abs=9.762645e-04/4.881620e-04`，`ChunkGDR vs ChunkGDRNeumann out max_abs=0`。
- 全量三方精度恢复：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 从算子根目录运行 `20 passed in 6.18s`。
- 修复 `tests/pytest/test_run.sh`：自动切到脚本目录并透传 `single/compare/performance` 退出码，避免 pytest 找不到脚本但 shell 仍 `exit 0` 的假通过。
### 2026-05-03
- 继续精度收口：默认 selector 保持 `C64_1616_FP32_CVInverse`，host 主 chunk 保持 `64`。
- 关闭 raw final 调试探针 `RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK`，避免 raw final 写 `attnWsGm_` 后再 fallback 仍污染后续链路。
- 将 final 两路统一回退到 `MatmulImpl`：`RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK=true`、`RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK=true`、`RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=true`。
- 重新构建双算子包通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 重新安装 custom OPP 到 `/home/developer/Ascend/cann-9.0.0/opp` 成功。
- 关键失败 case `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40,CGDR_GOLDEN_CHUNK_SIZE=64` 恢复 PASS：Neumann out/state `max_abs=9.762645e-04/4.881620e-04`，Neumann 与 normal out `max_abs=0`。
- 全量三方精度验证通过：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 结果 `20 passed in 6.19s`。
- 当前真实状态：精度已通过，但 raw `INV @ v_beta -> outVInnerGm_` final fused 仍未完成；当前 PASS 依赖 final MatmulImpl fallback，后续追 fused 性能需新增安全 scratch 后单独定位 raw final。
- 继续 raw final 二分：新增 host `rawFinalScratchSz` 与 Stage1 `rawFinalScratchGm_`，让 raw final 先写安全 scratch 后再 fallback 覆盖真实 `outVInnerGm_`；关键 case 仍 FAIL，排除 `tmpWsGm_` / `attnWsGm_` 真实输出复用污染。
- 增加 Stage2/Stage3 `RestoreCubeState()` 探针，显式恢复 atomic、padding、ND、HF32 和 layout transform；关键 case 仍 FAIL，说明常规 cube 状态恢复不能修复 raw final 副作用。
- 将 raw final 多 K 分段 unitFlag 从 `0/0` 调为 `2/3` 对齐仓库通用写法，关键 case 仍 FAIL，排除单纯 unitFlag 未更新。
- 设置 `RAW_MMAD_DEBUG_SKIP_RAW_FINAL=true` 跳过 raw final 执行但保留 fallback 覆盖链路，关键 case PASS；结论收敛为 `RawMmad64ByN()` 执行本身触发 out 偏差。
- 最终回到精度交付开关：`RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK=true`、`RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK=true`、`RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=true`、关闭 safe scratch/skip 探针；重新构建安装后全量三方精度 `20 passed in 6.24s`。
- 按用户要求再次运行全量三方精度：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，最新复跑结果 `20 passed in 6.22s`；当前精度状态确认通过。
### 2026-05-04
- 状态核对：当前默认 selector 仍为 `C64_1616_FP32_CVInverse`，host 主 chunk 为 `64`。
- 当前源码处于精度交付配置：`RAW_MMAD_USE_MATMULIMPL_FALLBACK=true`，final/kcum/vinner final fallback 均为 `true`，raw final skip/safe scratch/attn/tmp/inv 调试探针均为 `false`。
- 最新可信精度结果沿用 2026-05-03 复跑：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，`20 passed in 6.22s`。
- 当前真实状态：精度已通过，但 raw `RawMmad64ByN(INVWsGm_, vBetaWsGm_, outVInnerGm_, dv_)` final fused 仍未完成；后续若追性能需从 `RawMmadAccumulateL0C()` / `StoreAccumulatedL0C()` 继续二分。
- 澄清 state 侧口径：关键失败定位中 `state` 不是当前 blocker，但当前 `INV @ k_cumdecay` final 仍有 `RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK=true`，不能宣称 state final 已完整 raw MMAD fused；当前最明确的最后 debug 点是 raw `INV @ v_beta -> outVInnerGm_`。
- fused inverse 实现口径：当前默认 `C64_1616_FP32_CVInverse` 是 AIV 生成 `L` 和 `I-L`，AIC 按 16x16 子块执行三轮 Neumann 倍增，再用 `Compose32From16()` / `Compose64From32()` 组合 64x64 inverse；raw 目标路径为 `GM->L1->L0A/B->Mmad->L0C->Fixpipe->GM`，但当前精度配置仍打开 inverse/power/compose/final 的 `MatmulImpl` fallback。
- 验证关闭 inverse fallback：仅将 `RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`，保留 power/compose/final/kcum/vinner fallback，构建安装成功；关键 case `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40` 中 `CPU vs ChunkGDRNeumann state` PASS `max_abs=4.881620e-04`，`ChunkGDR vs ChunkGDRNeumann state` PASS `max_abs=9.313226e-10`，但 out 仍 FAIL `max_abs=1.037243e-01`；结论是关闭 inverse fallback 后 state 能通过，剩余 blocker 仍是 out/vinner。实验后已将源码开关恢复为 `RAW_MMAD_USE_MATMULIMPL_FALLBACK=true`，重新构建安装，并复跑同一关键 case 恢复 PASS。
- L0C 累加口径：raw inverse add 的 `INV += INV@Lpow` 不是跨 round 全程 L0C resident，而是 `Mmad` 结果在 L0C 后通过 Fixpipe atomic add 写回 GM，下一轮再从 GM 读；raw final/blocked 多 K 分段设计可以在 L0C 做 partial sum，但 raw final 当前仍是 blocker，精度交付配置未启用该路径。
- 修正 L0C resident 口径：底层 `Mmad` 可以控制 L0C 作为累加矩阵来源/C 初值，实现 `C=A@B+C_l0c`，这正是应替代当前 Fixpipe atomic add 的方向；当前代码尚未把 Neumann `INV += INV@P` 改成 L0C accumulator 结构，仍是 `Mmad->L0C->Fixpipe atomic add->GM`。剩余难点是更新后的 `INV` 下一轮还要作为 A 输入，仍需搬到可 `LoadData` 的位置，或进一步用 `INV@(I+P)` / ping-pong 合并输入与累加。
- 主目标纠偏：后续开发必须围绕 Neumann inverse 展开中的 `INV += INV @ P` 做 L0C accumulator fused inverse。具体目标是在 `InvAdd16()` raw 分支中把旧 `INV` 作为 L0C C 初值/累加矩阵来源，用底层 `Mmad` 得到 `INV @ P + old_INV` 后写回 updated `INV`。Fixpipe atomic add 只算过渡方案，不是最终目标；raw final `INV @ v_beta` debug 暂时降级为后续事项，不能抢占 inverse 主线。验证顺序固定为：保留 power/compose/final fallback，仅关闭 inverse fallback，先跑关键 state case，再跑全量 compare。
- 主目标纠偏：后续开发必须围绕 Neumann inverse 展开中的 `INV += INV @ P` 做 L0C accumulator fused inverse。具体目标是在 `InvAdd16()` raw 分支中把旧 `INV` 作为 L0C C 初值/累加矩阵来源，用底层 `Mmad` 得到 `INV @ P + old_INV` 后写回 updated `INV`。Fixpipe atomic add 只算过渡方案，不是最终目标；raw final `INV @ v_beta` debug 暂时降级为后续事项，不能抢占 inverse 主线。验证顺序固定为：保留 power/compose/final fallback，仅关闭 inverse fallback，先跑关键 state case，再跑全量 compare。
- 按用户要求复用/开启子代理审计 raw MMAD 路径；新子代理指出 raw Fixpipe 写 GM 后、S 发 DCCI 前需要补 `SyncFixToS()`，并确认当前 16x16/32x32 compose 的 `CopyGmMatrixToL1B()`、`RawMmadBlocked32()`、`Compose32From16()`/`Compose64From32()` 未发现明显布局或公式错误。
- 完成 Neumann inverse 矩阵生成 raw MMAD 化：`RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`、`RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK=false`、`RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK=false`，`RAW_MMAD_DEBUG_BASE0_FALLBACK=false`、`RAW_MMAD_DEBUG_BASE16_FALLBACK=false`、`RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=false`。
- `InvAdd16()` 当前走 `RawMmadInvAdd16()`：先 `INV @ I` 初始化 L0C，再以 `initC=false` 执行 `INV @ P`，在同一 L0C accumulator 内得到 `updated_INV = old_INV + INV @ P`，最后普通 Fixpipe 写回 `INVWsGm_`；不再依赖 Fixpipe atomic add、AIV add 或 GM 标量加法完成该累加。
- `PowerMmadSquare()` 当前走 `RawMmadSquare()`，覆盖 `L2/L4/L8`；`ComposeMmadSquare()` 当前走 `RawMmadSquare()` / `RawMmadBlocked32()`，覆盖 `Compose32From16()` 和 `Compose64From32()`。
- raw 写回可见性修复：`RawMmadInvAdd16()`、`RawMmadMatrix()`、`StoreAccumulatedL0C()` 在 Fixpipe 写 GM 后补 `SyncFixToS()`，再执行 `FlushMatrixCacheLines()`。
- 构建安装通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`，随后安装 `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp` 成功。
- 分段 smoke 覆盖 `S=16/17/33/49` 全部 PASS，分别覆盖单 16x16、`Compose32From16()`、64 分块下半单 16 block、完整 4 个 16 block + `Compose64From32()`。
- 关键 case `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40,CGDR_GOLDEN_CHUNK_SIZE=64` PASS：Neumann out/state `max_abs=7.336114e-03/4.881620e-04`，Neumann vs normal state `max_abs=9.313226e-10`。
- 最终全量三方精度通过：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.33s`。
- 本轮完成范围是 Neumann inverse 矩阵生成相关乘法；final RHS 乘法 `INV @ k_cumdecay` / `INV @ v_beta` 仍保持 `MatmulImpl` fallback，后续若要求 final 也 raw，需要继续定位 `RawMmad64ByN()`，不能混同为本轮 Neumann 矩阵生成验收范围。
- 本轮继续 local/TSCM PoC：`C64_3232_FP16_LocalTscm_CVInverse` 构造函数补齐一个忽略的 `StageOneMT &stage1MT_C64_` 参数后，主 kernel 10 参数构造路径可编译。
- PoC 构建安装通过，但全量三方精度未通过：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 结果 `17 failed, 3 passed in 6.30s`，不能进入 msprof 性能 probe。
- 失败集中在 `C=64` 主路径，短尾/小块 case 部分可过；初步怀疑 64x64 NZ 大矩阵中的 32x32 子块 offset 与 local Matmul compact TSCM/NZ tile 布局不匹配。
- 已将 selector 回退到稳定 `C64_3232_FP32_CVInverse`，重新构建安装后全量三方精度恢复：`20 passed in 6.33s`。
- `C64_3232_FP16_LocalTscm_CVInverse` 文件保留为独立 PoC，后续继续前应先验证 compact 32x32/16x16 TSCM tile 搬运和子块读取，不应作为默认 selector。
- 用户追问当前情况：确认最新状态仍为稳定默认线 `C64_3232_FP32_CVInverse` 可交付，local/TSCM PoC 精度未过；后续继续前先做 TSCM/NZ compact 子块布局验证。
- 用户追问 local/TSCM 累加位置：当前 PoC 的 local Matmul 在 `AICBlockProcessFp16L1()`，但 `INV += INV@Lpow` 的实际累加由 `AddLocalMatrix()` 通过 AIC 标量循环完成，不是依赖 `mm.IterateAll(..., enableAtomicAdd=1)`。
- 用户确认是否“靠标量累加而不是 Matmul 结果矩阵累加”：答案是是的；当前 `AICBlockProcessFp16L1()` 只生成乘积矩阵，`AddLocalMatrix()` 逐元素加回 `inv`，后续应考虑 `INV@(I+P)` 或验证 local Matmul C 输出累加能力。
- 用户明确阶段策略：先用标量循环实现一版 local/TSCM 精度正确版本并对比性能，要求中间结果尽量不出 cube/local；最终仍要用底层 `MmadParams(..., cmatrixSource, cmatrixInitVal)` 的 C 矩阵累加能力，当前 raw helper 中 `initC` 对应 `cmatrixInitVal`，`initC=false` 表示继续在 L0C accumulator 上累加，`cmatrixSource=true` 可从额外 C/bias tensor 作为 C 初值来源。
- 用户追问既然之前做过类似 L0C accumulator 思路为什么还没做好：原因是既有 raw MMAD 只验证了局部 `old_INV + INV@P` 累加，结果仍写回 GM 后下一轮再读，final RHS 仍有 `MatmulImpl` fallback 且 raw final 曾导致 out 精度失败；`C64_1616` 子块调用数多导致性能 probe 极慢，`C64_3232_FP16_LocalTscm` 又有 compact TSCM/NZ 布局阻塞。后续应先打通 local/TSCM compact tile 正确性，再替换 `AddLocalMatrix()` 为底层 C 初值/累加。
- 回答 L1/TSCM 承接问题：累加完可以搬到 L1/TSCM，不必须到 GM；当前 GM 是 `MatmulImpl` 串接和精度二分路径，最终目标仍是 `L0C accumulator -> Fixpipe -> L1/TSCM -> LoadData`，难点是 compact NZ/stride/layout 正确性。
- 按用户要求先实现标量正确基线：selector 切到稳定 `C64_3232_FP32_CVInverse`，新增 `DEBUG_SCALAR_FORWARD_GM_INVERSE=true` 和 `C64InverseByScalarForwardGm()`，用 AIC 标量前代递推从 `L_powWsGm_` 生成 `INVWsGm_`，final 仍走 FP32 GM `MatmulImpl`。
- 标量前代精度验证通过：Neumann-only smoke 覆盖 `S=16/S=17/S=64` PASS；双算子重新打包安装后 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 全量 `20 passed in 6.16s`。
- 标量前代性能 smoke 使用 `msprof --application ... --task-time=on` 完成，结果保存到 `tests/pytest/prof_out/chunk_gdr_c64_3232_scalar_forward_gm_smoke/`；`B=8,S=16` Neumann/normal `2.573x`，`B=8,S=64` `23.856x`，主耗时是 AIC scalar time，证明标量循环只能作为正确性基线。
- 新增方案文档 `docs/2026-05-04/C64_3232_SCALAR_FORWARD_BASELINE_PLAN.md`；后续方向更新为先打通 local/TSCM resident 和 compact tile layout，再把标量累加替换为 `MmadParams(..., cmatrixSource, cmatrixInitVal)` / `initC=false` 的底层 C 矩阵累加。
- 按用户明确要求完成当前默认 `C64_3232_FP32_CVInverse` 的 CMatrix/MMAD 累加主线：`RawMmadInvAddOutputTile()` 先用 `INV @ I` + `initC=true` 初始化 L0C，再用 `INV @ P` + `initC=false` 在同一 L0C/CMatrix 内累加，`StoreRawMmadInvAddRow()` 行级写回避免覆盖旧 `INV`。
- 清理当前默认文件中的标量前代残留和未用 scratch：移除 `DEBUG_SCALAR_FORWARD_GM_INVERSE`、`C64InverseByScalarForwardGm()`、`rawInvAddScratchGm_`、`RAW_LOAD3DV2_NO_FMATRIX_CONFIG`、`CeilDivLocal()`；保留 host `rawFinalScratchSz` 以免影响其它变体。
- 重新构建安装双算子包通过：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`，安装到 `/home/developer/Ascend/cann-9.0.0/opp` 成功。
- 全量三方精度通过：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.22s`；本轮未重新运行 msprof，完成范围是 Neumann inverse 矩阵生成中的 CMatrix/MMAD 累加正确性。
- 固化 CMatrix/MMAD 默认策略：保留 `RawMmadLowerTriangularBlock()` 计算 power 并显式清零上三角；full-tile power 与 `INV@P` k-range 裁剪均已回退，原因是部分 case 出现 `1e-2` 级 out 漂移。
- 语义化 raw helper：新增 `RAW_MMAD_CMATRIX_SOURCE_L0C=false`、`RAW_MMAD_INIT_C=true`、`RAW_MMAD_ACCUMULATE_C=false`，并将 `initC` 改名为 `initCMatrix`，明确对应 `MmadParams(..., cmatrixSource, cmatrixInitVal)` 的 CMatrix 控制参数。
- 最新验证：重新构建双算子包并安装到 `/home/developer/Ascend/cann-9.0.0/opp` 后，`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 全量三方精度 `20 passed in 6.14s`。
### 2026-05-05
- 当前求逆实现口径：默认 selector 为 `C64_3232_FP32_CVInverse`；AIV 生成 `L_powWsGm_=L` 和 `INVWsGm_=I-L`，AIC 在 `C64InverseBy32x32Neumann()` 中按 C64 拆 32x32、32x32 再拆 16x16 对角块做 Neumann inverse 和 compose。
- `INV += INV@P` 继续使用 CMatrix/MMAD 累加：`INV@I` 用 `cmatrixInitVal=true` 初始化 L0C，`INV@P` 用 `cmatrixInitVal=false` 累加到同一 CMatrix，Fixpipe 只负责写回。
- 当前主要耗时判断：latest lower-triangular power smoke 中 S16 Neumann/normal `1.323x`、S64 `1.664x`；主要放大项是 AIC scalar、MTE1/MTE2、Fixpipe、同步和 GM workspace 往返，不是单纯理论 MAC。
- 记录到 `docs/2026-05-05/WORK_LOG.md` 和 `docs/2026-05-05/PROGRESS_AND_ISSUES.md`；本轮未改代码、未重新跑 msprof。
- 纯 cube 核内口径澄清：当前还不是全程 L0/L1/TSCM resident 的纯 cube 融合求逆；已完成的是 `INV += INV@P` 的 L0C/CMatrix 累加，但每轮仍有 `L0C -> Fixpipe -> GM -> LoadData -> L0A/B` 往返，final RHS 仍为 FP32 GM `MatmulImpl`。
- CMatrix 累加与 GM atomic add 区别：旧路线是 `tmp=INV@P` 后由 Fixpipe/Matmul atomic add 写 GM 完成 `INVWsGm += tmp`；当前路线是在 L0C 里先 `INV@I` 再用 `cmatrixInitVal=false` 累加 `INV@P`，Fixpipe 只普通写回 updated `INV`。
- 下一步方向确认：应把 L0C 中 updated `INV` 通过 Fixpipe 写到 L1/TSCM 片上 buffer，再由下一轮 `LoadData` 读入 L0A/B；关键阻塞是 Fixpipe 到 L1/TSCM 的 layout/stride 以及 16x16/32x32 子块 compact NZ offset，不是 Neumann 公式本身。
- 后台任务检查：无仍在运行的 `build.sh`、`pytest`、`msprof`、`opc`、`ninja/gmake/cmake` 或安装任务，`jobs -l` 为空。
- 新增今日完成计划 `docs/2026-05-05/L0C_TO_L1_TSCM_INVERSE_COMPLETION_PLAN.md`：先做 16x16 `L0C -> Fixpipe -> L1/TSCM -> LoadData` 数值探针，再扩到 32x32 compose，全量 compare 通过后才跑 S16/S64 性能 smoke。
- 用户明确 final RHS 可以先不用改；今日实现范围收窄为 Neumann inverse 矩阵生成 resident 链路，`INV @ k_cumdecay` / `INV @ v_beta` 保持 FP32 GM `MatmulImpl` 稳定消费端。
- 按用户要求使用子代理执行 5.05 计划审计：子代理确认仓库没有 `float L0C -> float L1/TSCM -> LoadData<float>` 端到端样例，只有 `int32_t -> half` L0C->L1 iterator 参考；`C64_3232_FP16_LocalTscm` PoC 是 half local Matmul + 标量 add，不能直接作为默认 FP32 RHS 稳定线。
- 尝试在默认 `C64_3232_FP32_CVInverse` 中实现 FP32 resident inverse：增加 `aicL1Inv_`、`CopyL0CMatrixToL1Nz()`、`LoadInvGmToL1Resident()`、`RawMmadInvAddBlockResident()` 等 helper。
- 构建确认硬限制：`ascend910_93` / DAV2201 的 CANN Fixpipe L0C->L1 不支持 `float -> float`，`Fixpipe<float, float, CFG_NZ>(LocalTensor<float>, L0C)` 编译失败；当前只支持 float L0C 到 half/bfloat16/int8/uint8 等目标类型。
- 收敛默认线：移除不可编译的 FP32 L1 resident cache 和相关 helper，`Neumann16InverseBlock()` / `Neumann32InverseBlock()` 恢复使用已验证的 `RawMmadInvAddBlock()`，即 L0C/CMatrix 完成 `old_INV + INV@P` 后普通 Fixpipe 写回 GM。
- 验证通过：单算子包构建通过，双算子包构建通过，安装到 `/home/developer/Ascend/cann-9.0.0/opp` 成功；`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 全量三方精度 `20 passed in 6.22s`。
- 今日结论：默认算子保持可交付 `C64_3232_FP32_CVInverse` GM 中转基线；5.05 FP32 `L0C -> L1/TSCM` resident 闭环在当前架构上不可落地，后续只能转 half/bfloat16 resident PoC 或等待支持 FP32 L1 写回的架构。
- 用户纠正 `0C` 可搬到 L1：结论收窄为 `L0C -> L1` 支持，但 DAV2201 不支持 `float L0C -> float L1/TSCM`；可验证方向是 `float L0C -> half/bfloat16_t L1 NZ`。
- 布局口径补正：`Fixpipe CFG_NZ` 写到 L1 后是 C/output 的 NZ layout，不等于原样保留 L0C 内部物理排布；后续作为 A/B 输入时必须分别用 L0A/L0B 侧 `LoadData` 参数解释，不能共用同一套 transpose/stride。
- 状态复核：默认 selector 仍为 `C64_3232_FP32_CVInverse`，`FP16_L1_RESIDENT_INVERSE` 及相关 probe 开关均为 `false`，当前默认算子仍是稳定 FP32 GM 中转基线。
- 子代理补充确认：CANN 头文件未发现 `CFG_ZN`，`L0C -> half/bfloat16 L1/TSCM` 应使用 `CFG_NZ`；该 NZ 输出作为 A/B 输入时，L0A/L0B 的 `LoadData` transpose/stride 语义需要拆开验证。
- 当前下一步：先做隔离 16x16 `L0C -> Fixpipe CFG_NZ -> L1/TSCM -> LoadData -> MMAD` 数值探针，探针通过前不把 resident 分支切入默认 selector。
- 按用户纠正重新定义 baseline：不是旧远端 GM+AIV-add，而是 AIV 仅构造初始 `L/I-L`，后续 Neumann 轮次用 AIC Matmul/Fixpipe atomic add 的最早精度通过版本。
- 拆分 `op_kernel/baseline`、`op_kernel/halflocal`、`op_kernel/fulllocal`，并通过 `op_kernel/selected` selector 控制当前 stage1 版本；stage1 新文件名已去除 `CVInverse/cvinverse`。
- 修复 C16 halflocal/fulllocal 文件中从 C64_3232 复制残留的 compose 逻辑和重复 `C16InverseBy16x16Neumann()`，保留单 16x16 raw/L0C 累加路径。
- baseline `3*2` 全部构建、安装、三方精度通过：`C16_FP32/FP16`、`C64_1616_FP32/FP16`、`C64_3232_FP32/FP16` 均 `20 passed`。
- halflocal `3*2` 全部构建、安装、三方精度通过：`C16_FP32/FP16`、`C64_1616_FP32/FP16`、`C64_3232_FP32/FP16` 均 `20 passed`。
- 完成指定 `S=16/64/125/2048` 的 baseline/halflocal/original normal msprof 对比，汇总到 `docs/2026-05-05/BASELINE_HALFLOCAL_PERFORMANCE_SUMMARY.md` 和 `tests/pytest/prof_out/stage1_baseline_halflocal_s_selected_summary_20260505.csv`；当前最佳为 `halflocal/C64_3232_FP32`，ratio 为 `1.305x/1.651x/1.873x/2.332x`，normal 仍全点更快。
- `halflocal/C16_FP16/S=2048` 首次 msprof warmup 触发一次 `507015 L0A read/write conflict`，retry 成功并已合入最终性能表。
- 当前 selector 复核为 `halflocal/C64_3232_FP32`，`stage1_selected.h` include `../halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_halflocal.h`；无仍在运行的 build/pytest/msprof/opc 任务。
- 新增 fulllocal 单点突破计划：`docs/2026-05-05/FULLLOCAL_SINGLE_POINT_BREAKTHROUGH_PLAN.md`；首选 `fulllocal/C64_3232_FP16`，先做 16x16 `L0C -> Fixpipe CFG_NZ -> half/bfloat16 L1/TSCM -> LoadData -> MMAD` 布局探针，再接入 32x32/C64 path。
- 解释 `C64_1616` 性能异常：`halflocal/C64_1616_FP32` 四点 ratio 为 `5.240x/31.341x/43.869x/133.229x`，但 AIC MAC ratio 仅 `1.087x/1.609x/1.609x/1.224x`；慢点来自 16x16 子块数量、power/compose MatmulImpl fallback、同步/cache flush/GM workspace 往返和 AIV 等待放大，不是理论 MAC 主导。
- 复核 FP16/FP32 性能口径：`halflocal/C64_3232_FP16` 与 `halflocal/C64_3232_FP32` stage1 除文件名/header guard 外无实际逻辑差异，`FP16_L1_RESIDENT_INVERSE=false`，核心求逆和 final RHS 仍是 FP32 路线；四点中 FP16 仅 `S=16` 较慢，`S=64/125/2048` 基本持平或略快，因此当前不能解释为真实 FP16 计算慢于 FP32。
- 已尝试仅打开 `FP16_L1_RESIDENT_INVERSE` 的最小 FP16 resident 实验：构建与安装通过，但 `compare` 20/20 case 全失败；实验后开关恢复为 `false`。结论是单开关不足以形成可用 FP16 搬运/累加版本，后续必须补齐 resident layout/LoadData 链路。
- 按用户确认删除其它 fulllocal 候选，仅保留 `op_kernel/fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h` 作为单点突破文件；selector 恢复为 `halflocal/C64_3232_FP32`。
- 稳定线重新构建、安装并复跑全量三方精度：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.15s`。
- 按用户追加要求提交历史 `tests/pytest/prof_out` tracked 删除；仅纳入已跟踪原始 profiling 输出删除，不纳入当前未跟踪的新 profiling 目录。
- 澄清 selector 回退口径：`halflocal/C64_3232_FP32` 是编译期固定回退点，切换 `op_kernel/selected/stage1_selected.h` 和 `stage1_selected_config.h` 后必须重新 build/install/compare；不是运行时开关，也不修改 baseline/halflocal 实现文件。
