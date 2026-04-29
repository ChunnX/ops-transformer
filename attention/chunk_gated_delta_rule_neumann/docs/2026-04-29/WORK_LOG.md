# 工作日志 - 2026-04-29

**日期**: 2026-04-29  
**主要工作**: 整理 `docs/` 文档结构，确认算子当前实现状态和精度状态

---

## 用户请求

1. 整理 `/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/docs`：修正日期不对和重复文档，并新建今天进度。
2. 说明当前算子状态：chunk、求逆方式、精度情况。

---

## 文档整理动作

- 将错误年份目录：
  - `docs/2025-04-27/` 移动为 `docs/2026-04-27/`。
  - `docs/2025-04-28/` 内容移动到 `docs/2026-04-28/`。
- 将重复或已被替代的 2026-04-28 文档移动到：
  - `docs/2026-04-28/archive/EXECUTION_PLAN_SUPERSEDED.md`
  - `docs/2026-04-28/archive/PROGRESS_AND_ISSUES_INITIAL.md`
  - `docs/2026-04-28/archive/WORK_LOG_AIC_PLAN.md`
- 新增 `docs/README.md` 作为文档索引。
- 新增 `docs/2026-04-29/PROGRESS_AND_ISSUES.md` 记录今天最新状态。

---

## 当前状态确认

- 当前 chunk 为 `16`。
- 当前求逆方式是 Neumann 三步倍增：`INV0=I-L`，累加 `INV@L²`、`INV@L⁴`、`INV@L⁸`。
- 当前初始矩阵仍由 AIV 两个 subBlock 拼完整 `16×16`，不是纯 AIC `InitInverseMatrix()`。
- AIC 当前负责 Neumann 相关矩阵乘，AIV 当前负责 `INV += tmp` 和逐元素预处理。
- 单算子直调精度已通过，详见 `docs/2026-04-29/PROGRESS_AND_ISSUES.md`。

---

## 决策

| 决策 | 说明 |
|---|---|
| 保留历史文档 | 早期计划和日志仍有调试参考价值，不删除，按真实日期和 archive 归档 |
| 以 2026-04-29 进度为最新状态 | 后续状态查询优先看 `docs/2026-04-29/PROGRESS_AND_ISSUES.md` |
| 标记方案差异 | 当前实现可通过精度，但未完成纯 AIC 初始化和 FP16 MMA 目标 |

---

## 下一步

1. 如需交付当前版本，继续做 direct-call 参数矩阵扩展。
2. 如需对齐原始方案，单独推进 AIC `InitInverseMatrix()` 和 FP16 MMA 策略评估。
3. 清理工作区临时二进制和历史调试产物，避免提交污染。

---

## 只读分析：AscendC FP16 A/B + FP32 C Matmul 写法

**用户请求**: 在当前目录和上层 `ops-transformer` 仓库中查找 AscendC matmul 使用 FP16/half A/B 输入、FP32 C 输出的写法和 host tiling 配置方式，只读分析，不修改代码。

**检索结论**:
- 最直接样例位于 `posembedding/rotary_position_embedding`：kernel 中 `MatmulType<GM, ND, half>` + `MatmulType<GM, ND, float>`，host tiling 对应 `DT_FLOAT16`/`DT_FLOAT16`/`DT_FLOAT`。
- 同文件还提供 BF16 A/B + FP32 C 配置，可作为当前 BF16 输入场景的参考。
- `ffn/ffn` 提供模板封装样例：`mmWNDType<half, float>` 将 A/B dtype 固定为 `half`，C dtype 参数化为 `float`。
- 当前算子 `chunk_gated_delta_rule_neumann` 的 Stage1/2/3 仍为 `float` A/B/C，host tiling 也是 `DT_FLOAT`/`DT_FLOAT`/`DT_FLOAT`，若要切到 half A/B 必须同步修改 kernel `MatmulType` 与 host `SetAType/SetBType`。
- 未在仓库中发现完整的“float GM workspace cast 成 half GM workspace 后立即作为 cube matmul 输入”的端到端样例；当前算子只有通用 `CopyCast` 可用于 GM↔GM dtype 转换，其他样例多为 UB 内 Cast 后输出或 half GM 直接作为 matmul 输入。

**关键路径**:
- `posembedding/rotary_position_embedding/op_kernel/rotary_position_embedding.cpp`
- `posembedding/rotary_position_embedding/op_host/rope_rotate_matrix_tiling.cpp`
- `ffn/ffn/op_kernel/ffn.cpp`
- `attention/chunk_gated_delta_rule_neumann/op_kernel/chunk_gated_delta_rule_neumann_stage1.h`
- `attention/chunk_gated_delta_rule_neumann/op_host/chunk_gated_delta_rule_neumann_tiling.cpp`
- `attention/chunk_gated_delta_rule_neumann/op_kernel/chunk_gated_delta_rule_neumann.h`

---

## 设计判断：AIC 全载与随路 Cast

**用户问题**: 希望 Neumann 全部放在 AIC 侧完成，询问是否有“随路 cast”能力，避免 AIV 参与 FP32→FP16 转换和累加。

**接口检索结论**:
- `MatmulType` 的 dtype 是编译期输入 dtype，`SetTensorA/SetTensorB` 接口按该 dtype 解释 GM/LocalTensor 输入；未发现可声明 GM 为 `float`、进入 Cube 时自动转成 `half` 的公开配置。
- `GetL0DataType<T>` 默认保持 `T`，只有 FP8/MX 等特殊类型存在 L0 dtype 映射；未看到 `float -> half` 的 L0 随路转换映射。
- `Matmul::IterateAll/GetTensorC(dst, enAtomic)` 支持搬出时 atomic add，`enAtomic=1` 可让 `INV @ L_pow` 的 FP32 C 结果在 FixPipe 搬出阶段直接累加到 `INVWsGm_`。

**当前判断**:
- FP32 GM workspace 直接作为 FP16 MMA 输入不可行或至少没有公开稳定接口支撑。
- 更稳妥的 AIC-only 方案是：在 AIC 内用 LocalTensor/UB 做 `float -> half` cast，作为 matmul 的 LocalTensor 输入或写入 half GM/L1 工作区后再执行 `half x half -> float` MMA。
- `INV += tmp` 可以从 AIV 移到 AIC：用 `IterateAll(INVWsGm_, 1)` 或 `GetTensorC(INVWsGm_, 1)` 做 FP32 atomic add，删除三轮 AIV `NeumannAddCompute()` 同步。

**待验证点**:
- 当前 `MatmulType<TPosition::GM, ..., half>` 是否可直接改为 `TPosition::A1/B1` 接收 AIC LocalTensor half；若不可行，则需要 AIC 侧 half GM workspace。
- atomic add 后若下一轮 `INV` 仍作为 FP16 MMA 输入，需要同步维护 `INVHalfWsGm_` 或在 AIC 下一轮前重新 cast `INVWsGm_`。

---

## torch_npu 接口注册与三方精度脚本

**用户请求**: 给 `chunk_gated_delta_rule` 和 `chunk_gated_delta_rule_neumann` 注册 `torch_npu` 风格接口，并参考现有 pytest 脚本新增 CPU golden、普通 NPU 算子、Neumann NPU 算子的三方精度对比。

**实现动作**:
- 新增 `torch_extension/npu_ops_transformer/ops/csrc/chunk_gated_delta_rule.cpp`，通过 `ACLNN_CMD(aclnnChunkGatedDeltaRule, ...)` 直调普通算子 aclnn 接口。
- 新增 `torch_extension/npu_ops_transformer/ops/csrc/chunk_gated_delta_rule_neumann.cpp`，通过 `ACLNN_CMD(aclnnChunkGatedDeltaRuleNeumann, ...)` 直调 Neumann 算子 aclnn 接口。
- 新增 Python builder：`torch_extension/npu_ops_transformer/ops/chunk_gated_delta_rule.py` 和 `torch_extension/npu_ops_transformer/ops/chunk_gated_delta_rule_neumann.py`。
- 在 `torch_extension/npu_ops_transformer/ops/__init__.py` 导出两个接口，并将函数挂到 `torch_npu.npu_chunk_gated_delta_rule`、`torch_npu.npu_chunk_gated_delta_rule_neumann`，兼容现有 pytest golden 调用方式。
- 更新两个 pytest operator wrapper，使其在未安装 wheel 时也能从仓库源码 `torch_extension/` 导入 `npu_ops_transformer` 完成注册。
- 新增 `tests/pytest/compare_chunk_gdr_three_way.py`，使用同一组输入比较：CPU golden vs `ChunkGDR`、CPU golden vs `ChunkGDRNeumann`、`ChunkGDR` vs `ChunkGDRNeumann`。
- 修复 Neumann `tests/pytest/test_run.sh` 中单测脚本名，改为 `test_chunk_gated_delta_rule_neumann_single.py`。

**CPU golden 约束**:
- 按用户明确要求，三方脚本不复制、不改写 CPU golden 逻辑。
- 三方脚本直接调用原 `chunk_gated_delta_rule_neumann_golden.py` 中的 `cgdr_golden()`，默认使用其 `use_float64=False` 路径；如需额外排查可传 `--golden-float64`。

**验证状态**:
- 已执行 Python 静态编译检查：`compare_chunk_gdr_three_way.py`、两个 operator wrapper、两个 golden、两个 Python builder 均通过 `py_compile`。
- 已补齐 Python JIT 依赖 `ninja`。
- 已重新构建并安装包含 `chunk_gated_delta_rule` 与 `chunk_gated_delta_rule_neumann` 的 custom OPP 包。
- 已完成 torch_npu 三方 NPU 实跑，默认、变量长度、GQA、`dk=dv=32` 均 PASS。

**已执行验证命令**:
```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/tests/pytest
python3 compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 128 --dv 128
python3 compare_chunk_gdr_three_way.py --batch-size 2 --seqlens 17,65 --nk 4 --nv 4 --dk 64 --dv 64
python3 compare_chunk_gdr_three_way.py --batch-size 2 --seqlen 64 --nk 2 --nv 4 --dk 64 --dv 64
python3 compare_chunk_gdr_three_way.py --batch-size 2 --seqlen 64 --nk 4 --nv 4 --dk 32 --dv 32
```

---

## 三方脚本输入范围核对与 aclnn 直调删除判断

**输入范围核对**:
- `compare_chunk_gdr_three_way.py` 的随机输入生成已经按原 pytest/golden 生成逻辑对齐：`query/key/value/beta/initial_state` 使用 `torch.rand`，取值 `[0, 1)`；`g` 使用 `torch.rand(...) * -1.0`，取值 `(-1, 0]`；`query/key` 同样做 `F.normalize(..., p=2, dim=-1)`；`scale = 1 / sqrt(dk)`；`actual_seq_lengths` 使用 `int32`。
- CPU golden 路径直接调用原 `cgdr_golden()`，默认 `use_float64=False`，与原 `run_chunk_gated_delta_rule_eager()` 一致；可选 `--golden-float64` 仅用于额外排查。
- 三方脚本默认维度已对齐两个原 pytest 的共同短用例：`B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128`。原普通 GDR 的长用例为 `seqlen=16384,chunk=64`，Neumann 长用例为 `seqlen=1024,chunk=16`，三方脚本可用命令行参数单独覆盖。

**aclnn 直调删除判断**:
- `torch_npu.npu_chunk_gated_delta_rule` 与 `torch_npu.npu_chunk_gated_delta_rule_neumann` 均已完成导入/JIT 编译，并通过三方精度实跑。
- 已删除旧 aclnn direct-call 精度链路和本地二进制：`tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp`、`tests/generate_aclnn_test_data_cpu.py`、`tests/test_aclnn_chunk_gated_delta_rule_original.cpp`、`tests/test_neumann_direct`、`tests/test_neumann`、`tests/test_original`，以及原版算子目录下的临时直调文件/数据。
- `op_host/op_api/aclnn_*` 与 `tests/ut/op_host/op_api/*` 未删除；它们属于算子 API 实现和 UT，不是临时精度脚本。

---

## 三方精度实跑结果

**环境与安装**:
- 安装依赖：`python3 -m pip install --user ninja`。
- 重新打包：`bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 安装方式：`bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=${ASCEND_HOME_PATH}/opp`。
- 符号确认：`libcust_opapi.so` 同时导出 `aclnnChunkGatedDeltaRule*` 与 `aclnnChunkGatedDeltaRuleNeumann*`。

**实跑结论**:
- 默认短用例 `B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128`：PASS。
- 变量长度 `B=2,seqlens=17,65,nk=4,nv=4,dk=64,dv=64`：PASS。
- GQA `B=2,seqlen=64,nk=2,nv=4,dk=64,dv=64`：PASS。
- 小维度 `B=2,seqlen=64,nk=4,nv=4,dk=32,dv=32`：PASS。

**默认用例关键误差**:
- CPU vs `ChunkGDR` out/state：max_abs 分别为 `9.736717e-04`、`4.882067e-04`。
- CPU vs `ChunkGDRNeumann` out/state：max_abs 分别为 `9.736717e-04`、`4.882067e-04`。
- `ChunkGDR` vs `ChunkGDRNeumann` out/state：max_abs 均为 `1.220703e-04`。

**过程问题与修复**:
- 初次导入 `npu_ops_transformer` 会加载无关 MoE 算子并因缺 `ninja` 失败；已将 `ops/__init__.py` 改为懒加载，并显式只加载目标算子。
- 初次实跑普通 GDR 失败，原因是已安装 `libcust_opapi.so` 只有 Neumann 符号；已重包安装两个算子后解决。

**精度高低判断**:
- 默认三方 case 中，普通 `ChunkGDR` 与 `ChunkGDRNeumann` 相对 CPU golden 的 out/state `max_abs`、`mean_abs`、`rmse` 基本一致。
- `ChunkGDR` 与 `ChunkGDRNeumann` 之间仅有 BF16 量级差异，out/state `max_abs=1.220703e-04`，不能判定其中一个实质精度更高。

---

## pytest 10 Case 三方精度对比

**实现动作**:
- 将 `tests/pytest/compare_chunk_gdr_three_way.py` 改造成 pytest 参数化脚本。
- 新增独立批测输入脚本 `tests/pytest/compare_chunk_gdr_three_way_paramset.py`，集中维护 `THREE_WAY_CASES` 共 10 个 case。
- 保留原命令行单 case 入口，复用同一套 `run_three_way_case()` 执行逻辑。
- 在 `tests/pytest/test_run.sh` 新增 `compare` 模式，一键执行三方精度对比。

**覆盖 case**:
- default：`B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128`
- varlen：`B=2,seqlens=17,65,nk=4,nv=4,dk=64,dv=64`
- GQA：`B=2,seqlen=64,nk=2,nv=4,dk=64,dv=64`
- dims32：`B=2,seqlen=64,nk=4,nv=4,dk=32,dv=32`
- short tail：`B=1,seqlen=5,nk=4,nv=4,dk=64,dv=64`
- exact chunk：`B=1,seqlen=16,nk=4,nv=4,dk=64,dv=64`
- tail31：`B=1,seqlen=31,nk=4,nv=4,dk=64,dv=64`
- seq128：`B=1,seqlen=128,nk=4,nv=4,dk=64,dv=64`
- mixed dims：`B=2,seqlen=32,nk=4,nv=4,dk=128,dv=64`
- GQA nk1：`B=1,seqlen=64,nk=1,nv=4,dk=64,dv=32`

**验证命令**:
```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/tests/pytest
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}
bash test_run.sh compare
```

**验证结果**:
- `compare_chunk_gdr_three_way.py` 收集 10 个 pytest case。
- 拆分 paramset 后复跑 `bash test_run.sh compare`，结果 `10 passed in 5.74s`。

---

## 扩展精度 Case 与 msprof 性能对比脚本

**用户请求**:
- 三方精度 pytest case 继续扩充，覆盖更全面。
- 性能对比只保留 prefill shapes，参考用户给定 `(B,S,D,Hq,Hkv)` 列表。
- 性能脚本不能用 Python event benchmark，必须作为 `msprof --application` 的载体，并包含 warmup、MatMul flush、ReduceMax flush、真实测量四阶段。

**实现动作**:
- 扩展 `tests/pytest/compare_chunk_gdr_three_way_paramset.py`，三方精度 case 从 10 个增加到 20 个，覆盖 tail、chunk 边界、变量长度、GQA、多 batch、`dk/dv` 不同组合、`nv=16/32` 等场景。
- 新增 `tests/pytest/compare_chunk_gdr_performance_paramset.py`，按用户给定 24 个 prefill shape 维护性能批测输入；映射关系为 `B->batch_size`、`S->seqlen`、`D->dk=dv`、`Hq->nv`、`Hkv->nk`。
- 重写 `tests/pytest/compare_chunk_gdr_performance.py`：作为单次 `msprof` application payload，不在 Python 内部计时；流程为 operator warmup、`torch.mm` cache flush、`torch.max` cache flush、被测算子单次 measurement。
- 新增 `tests/pytest/run_chunk_gdr_performance_msprof.py`：批量生成并执行 `msprof --application=... --task-time=on`，每个 shape 分别跑 `normal` 与 `neumann`，输出目录按 `case/op` 切分。
- 更新 `tests/pytest/test_run.sh`：新增 `performance` 模式，调用 msprof 批测 launcher；保留 `single` 与 `compare`。
- 性能 payload 已调整为按 `--ops` 只加载对应算子，避免一个 profiling 进程内同时 JIT/加载两个目标算子。

**验证动作**:
```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/tests/pytest
python3 -m py_compile compare_chunk_gdr_performance.py compare_chunk_gdr_performance_paramset.py run_chunk_gdr_performance_msprof.py compare_chunk_gdr_three_way.py compare_chunk_gdr_three_way_paramset.py
python3 run_chunk_gdr_performance_msprof.py --dry-run --limit 1 --output ./prof_out_dryrun
python3 compare_chunk_gdr_performance.py --ops normal --B 1 --S 16 --D 32 --Hq 4 --Hkv 4 --flush-mm 16 --flush-reduce-outer 1 --flush-reduce-mid 16 --flush-reduce-inner 16
python3 compare_chunk_gdr_performance.py --ops neumann --B 1 --S 16 --D 32 --Hq 4 --Hkv 4 --flush-mm 16 --flush-reduce-outer 1 --flush-reduce-mid 16 --flush-reduce-inner 16
bash test_run.sh compare
```

**验证结果**:
- Python 静态编译通过。
- `run_chunk_gdr_performance_msprof.py --dry-run --limit 1` 正确生成普通 GDR 与 Neumann 两条 `msprof --application` 命令。
- 小 flush 冒烟通过：`normal` 与 `neumann` payload 均完成 warmup、flush、measurement，输出 shape 正确。
- 三方精度批测扩展到 20 case 后实跑通过：`20 passed in 6.03s`。

**注意事项**:
- `bash test_run.sh performance` 会对 24 个 prefill shape × 2 个算子执行真实 `msprof`，默认 flush 为 `MatMul 10240x10240` 和 `ReduceMax 96x1024x1024`，耗时和磁盘输出较大。
- 若只验证命令，可先执行：`python3 run_chunk_gdr_performance_msprof.py --dry-run --limit 1`。
- 若只跑一个 shape，可执行：`python3 run_chunk_gdr_performance_msprof.py --case prefill_b8_s128_d64_h16_hkv4`。

---

## 性能批测状态澄清

**用户问题**: 是否已经批量测试过性能对比。

**当前真实状态**:
- 尚未执行完整 `24 个 prefill shape × 2 个算子` 的真实 `msprof` 批量性能对比。
- 已完成的是：性能脚本 `py_compile`、批量 msprof 命令 `dry-run`、以及 normal/neumann 两个算子的小 flush payload 冒烟。
- 因此当前还没有可用于判断两算子性能高低的完整 `msprof` 批测结果。

**建议下一步**:
- 先跑单个 representative case 确认 `msprof` 输出解析链路，再跑完整批测。
- 单个 case 建议：`prefill_b8_s128_d64_h16_hkv4`。

---

## 完整 msprof 性能批测与 C64 方案规划

**用户请求**:
1. 先跑一个性能 case；如果通过，继续完整批量跑性能对比。
2. 后续将 chunk 改为 64，并实现两套 FP32 CVInverse 方案：
   - 方案1：`C64_1616_FP32_CVInverse`，用 16×16 Neumann 分块组合出 32×32，再组合 64×64。
   - 方案2：`C64_3232_FP32_CVInverse`，直接 32×32 Neumann 展开求逆。
3. 先写大计划，首先保证性能对比数据完整保存。

**执行结果**:
- 单 case `prefill_b8_s128_d64_h16_hkv4` 已真实执行 normal/neumann 两次 `msprof`，均通过。
- 新增 `tests/pytest/summarize_chunk_gdr_performance_msprof.py`，从 `op_summary_*.csv` 抽取目标算子的最后一次出现作为 Phase4 measurement，并保存 raw/compare CSV。
- 完整执行 24 个 prefill shape × 2 个算子的真实 `msprof` 批测。
- 批测输出已保存：
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_raw.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv`
- 新增 C64 两方案计划：`docs/2026-04-29/C64_NEUMANN_INVERSE_IMPLEMENTATION_PLAN.md`。

**性能结论**:
- 24/24 个 shape 中，当前普通 `ChunkGDR` 都快于当前 `ChunkGDRNeumann`。
- `normal_us / neumann_us` 平均为 `0.2564`，即当前 Neumann 约为普通 GDR 的 `3.90x` 耗时。
- 按序列长度聚合：`S=128` 平均比值 `0.3600`，`S=256` 为 `0.3061`，`S>=512` 约 `0.218`。
- 按维度聚合：`D=64` 平均比值 `0.2922`，`D=128` 平均比值 `0.2206`。

**说明**:
- 以上结果是当前 C16/现状 Neumann 实现相对普通 GDR 的 msprof 基线，不是 C64 新方案结果。
- C64 两方案实现后，需要分别重编并用同一批测脚本生成新目录结果对比。

---

## profiler cube 指标修正与分支上传准备

**用户反馈**: profiling 中 cube 利用率应看 Y 列，不应看最后一列。

**核对结果**:
- 当前 `msprof` 导出的 `op_summary_*.csv` 表头中，Y 列对应 `aic_mac_ratio`。
- 最后一列对应 `cube_utilization(%)`，不再作为本轮性能判断的 cube 指标。

**实现动作**:
- 修改 `tests/pytest/summarize_chunk_gdr_performance_msprof.py`，从 `aic_mac_ratio` 读取 Y 列指标。
- 输出字段改为 `cube_utilization_y_col_ratio` 和 `cube_utilization_y_col_pct`。
- 保留 `msprof_last_cube_utilization_pct` 作为审计字段，避免后续误把最后一列当成本轮判断指标。
- 基于已完成的 `msprof` 原始输出重新生成：
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_raw.csv`

**上传准备**:
- 当前准备提交到独立 fork 分支。
- 提交范围只包含算子代码、torch_npu 注册、pytest/msprof 脚本、文档和小体积 CSV 摘要。
- 不提交完整 `PROF_*` profiler 原始目录，避免把约 55MB 临时 profiling 产物纳入代码分支。

**上传结果**:
- 已创建并推送 fork 分支：`chunk-gdr-neumann-20260429-upload`。
- 本次提交：`61d5a61c Add chunk GDR precision and profiling tooling`。
- fork MR 地址：`https://gitcode.com/chuqihang/ops-transformer/merge_requests/new?source_branch=chunk-gdr-neumann-20260429-upload`。
- 推送前修复本机 SSH 公钥文件与私钥不匹配问题，仅重建 `~/.ssh/id_ed25519.pub`，未修改私钥。

---

## profiling 原始目录上传

**用户请求**: 将完整 profiling 也上传到 fork 分支。

**执行动作**:
- 将 `attention/chunk_gated_delta_rule_neumann/tests/pytest/prof_out/` 下完整 profiling 产物纳入提交。
- 上传内容包含：
  - `chunk_gdr_prefill_full/`：24 个 prefill shape × normal/neumann 的完整 `PROF_*` 输出。
  - `chunk_gdr_prefill_smoke/`：单 case smoke profiling 输出。
  - 已修正 Y 列指标后的 `performance_compare.csv` 与 `performance_raw.csv`。

**上传结果**:
- profiling 提交：`c69375e7 Upload chunk GDR profiling artifacts`。
- 远端分支：`fork/chunk-gdr-neumann-20260429-upload`。

---

## fork master 推送规则更新

**用户要求**: 后续直接推送到 fork 的 `master`，不再只推临时分支。

**执行结果**:
- 已将当前 `chunk-gdr-neumann-20260429-upload` 分支 fast-forward 推送到 `fork/master`。
- 当前 `fork/master` 最新提交：`44eff3448a9a206e3ae1eff22c292eae078aa2d0`。
- 后续默认上传目标：`fork master`。
