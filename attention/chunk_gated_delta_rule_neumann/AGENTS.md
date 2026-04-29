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
└── kernel-debugging/          # Kernel挂起问题调试经验
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

## Current Plan: AIC生成初始矩阵

**方案**: 将(I-L)初始矩阵生成从AIV移至AIC，实现纯Cube MMA Neumann求逆

**问题背景**:
- 当前InverseCompute(AIV)只处理16×8半块（TASK_RATIO=2导致）
- NeumannLpowCompute(AIC)需要完整16×16矩阵
- workspace布局冲突：两个subBlock试图拼成完整矩阵

**解决方案**:
```
原流程:
  AIV InverseCompute -> 写INVWsGm_(16×8半块) ❌
  AIC NeumannLpowCompute -> 读完整16×16

新流程:
  AIC InitInverseMatrix -> 写完整INVWsGm_(16×16) ✓
  AIC NeumannLpowCompute -> 读完整16×16
```

**详细方案文档**: `docs/2026-04-28/AIC_INVERSE_GENERATION_PLAN.md`

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

## CHUNK=16 Migration Status

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

**Stage1 Neumann流程** (待实现):
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
