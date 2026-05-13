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

---

## C16 profiling 4x 开销分析与死写修复

**用户问题**: chunk 从 64 减到 16 后，为什么 profiling 中 AIC/AIV 计算和搬运相关时间接近 4 倍，是否存在重复计算。

**分析结论**:
- profiling 中 AIC/AIV 总时间约 `4x`，但 AIC MAC time 平均仅 `1.52x`，4x 主要来自 scalar/MTE/FixPipe/同步和小 matmul 调用次数放大。
- Stage2 总 MAC 不随 chunk 从 64 到 16 下降到 1/4；其总量约为 `3*S*Dv*Dk`，基本与 chunkSize 无关。
- Stage3 MAC 理论上约降到 1/4，但 chunk 数和 matmul 调用次数变 4 倍，搬运/同步开销明显放大。
- 当前 `stage1_C16_FP16_CVInverse.h` 实际仍使用 `float/float/float` MatmulType，host tiling 也是 `DT_FLOAT`，不是实际 FP16 MM。
- 当前 C16 Neumann 每 chunk matmul 调用约 14 次，普通 C64 每 chunk约 10 次；考虑 chunk 数 4 倍，matmul 调用次数约 `5.6x`，与 AIC scalar time 约 `5x` 匹配。

**已修复问题**:
- 删除 Neumann `InverseCompute()` 中对 `attnWsGm_` 写入初始 `I-L` 的死写；该 workspace 在 Round1 会被 `L@L` 立即覆盖。
- 同步修复四个 stage1 变体：`C16_FP16`、`C16_FP32`、`C64_1616_FP32`、`C64_3232_FP32`。

**验证**:
- `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16` 通过。
- 重包安装 `chunk_gated_delta_rule,chunk_gated_delta_rule_neumann` 后，三方精度 smoke `B=1,S=64,nk=4,nv=4,dk=dv=64` 通过。
- 单 case Neumann profiling smoke 完成；MTE/FixPipe 小幅下降，但总耗时无明显改善，说明主因不是该死写。

**详细记录**: `docs/2026-04-29/PROFILING_C16_OVERHEAD_ANALYSIS.md`

## 性能批测补充 S=16/S=64 cases

**用户请求**: 在现有测试中加一组 `seq=16` 和 `seq=64`。

**执行动作**:
- 更新 `tests/pytest/compare_chunk_gdr_performance_paramset.py`。
- 在 prefill 性能 shape 网格中为 `B=8/16/32/64` 各补充两档短序列：`S=16` 与 `S=64`。
- 新增 case 名称：
  - `prefill_b8_s16_d64_h16_hkv4`、`prefill_b8_s64_d64_h16_hkv4`
  - `prefill_b16_s16_d64_h16_hkv4`、`prefill_b16_s64_d64_h16_hkv4`
  - `prefill_b32_s16_d64_h16_hkv4`、`prefill_b32_s64_d64_h16_hkv4`
  - `prefill_b64_s16_d64_h16_hkv4`、`prefill_b64_s64_d64_h16_hkv4`

**验证**:
- `PERFORMANCE_CASES` 从 24 个扩展为 32 个。
- `python3 -m py_compile compare_chunk_gdr_performance_paramset.py run_chunk_gdr_performance_msprof.py compare_chunk_gdr_performance.py` 通过。
- `run_chunk_gdr_performance_msprof.py --dry-run` 对 `B=8,S=16/64` 的 normal/neumann 命令生成通过。

**说明**:
- 三方精度 paramset 之前已覆盖 `seqlen=16` 与 `seqlen=64`，本次补的是性能 prefill 批测网格。

## S=16/S=64 性能实测结果

**用户问题**: `S=16`、`S=64` 性能如何。

**执行动作**:
- 对新增 8 个 prefill shape 执行真实 `msprof`：`B=8/16/32/64` × `S=16/64` × normal/neumann。
- 输出目录：`tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/`。
- 汇总文件：
  - `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/performance_compare.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/performance_raw.csv`

**核心结果**:

| S | B | normal_us | neumann_us | neumann/normal | faster | normal_cube_Y% | neumann_cube_Y% |
|---:|---:|---:|---:|---:|---|---:|---:|
| 16 | 8 | 608.28 | 689.54 | 1.134x | normal | 1.1 | 0.7 |
| 16 | 16 | 1246.86 | 1394.16 | 1.118x | normal | 1.0 | 0.7 |
| 16 | 32 | 2387.38 | 2727.76 | 1.143x | normal | 1.1 | 0.7 |
| 16 | 64 | 4853.34 | 5241.96 | 1.080x | normal | 1.1 | 0.7 |
| 64 | 8 | 652.94 | 1550.74 | 2.375x | normal | 1.7 | 1.2 |
| 64 | 16 | 1251.72 | 3131.46 | 2.502x | normal | 1.7 | 1.2 |
| 64 | 32 | 2498.70 | 6133.46 | 2.455x | normal | 1.7 | 1.2 |
| 64 | 64 | 5037.96 | 12246.48 | 2.431x | normal | 1.7 | 1.2 |

**聚合结论**:
- `S=16`：normal 平均 `2273.97 us`，Neumann 平均 `2513.36 us`，Neumann 平均为 normal 的 `1.119x` 耗时。
- `S=64`：normal 平均 `2360.33 us`，Neumann 平均 `5765.53 us`，Neumann 平均为 normal 的 `2.441x` 耗时。
- 8/8 case 都是普通 `ChunkGDR` 更快。

**分析**:
- `S=16` 时只有一个 C16 chunk，Neumann 额外 inverse 开销存在，但还没触发 chunk 数放大，因此只慢约 `8%~14%`。
- `S=64` 时当前 Neumann 为 4 个 C16 chunk，而普通 GDR 为 1 个 C64 chunk；chunk 调度、同步、MTE/FixPipe 与 stage2 state 搬运明显放大，因此慢到约 `2.4x`。
- 这与前面 C16 profiling 分析一致：当前主要瓶颈是 chunk 数和小 matmul 调用次数，不是单个大矩阵 MAC 计算量。

## S=16 Neumann 仍慢的 profiling 拆解

**用户质疑**: `S=16` 时 stage2/stage3 耗时相同甚至更快，stage1 又按预期减少了求逆 MAC，为什么总耗时仍明显慢。

**核对结果**:
- `S=16` 的 msprof 明细确认 Neumann 的 AIC MAC time 确实低于 normal：
  - `B=8`: `6.382us -> 4.565us`
  - `B=16`: `12.759us -> 9.135us`
  - `B=32`: `25.513us -> 18.252us`
  - `B=64`: `51.052us -> 36.500us`
- 但 Neumann 的 AIC scalar time 明显上升：
  - `B=8`: `113.288us -> 139.257us`，增加 `25.969us`
  - `B=16`: `223.471us -> 272.425us`，增加 `48.954us`
  - `B=32`: `436.537us -> 542.528us`，增加 `105.991us`
  - `B=64`: `856.229us -> 1068.133us`，增加 `211.904us`
- Neumann 的 AIV vector/scalar 计算下降，但 AIV MTE2 增加，说明收益没有落到关键路径上。

**代码路径对比**:
- normal stage1 在 `S=16` tail 情况下仍是 `chunkSize=64`，但 `validLen=16`，`subValidRows` 会裁剪并补零；stage1 主要是 6 次 AIC matmul 调用：`K@K`、`Q@K`、2 次 inverse block combine、`attn@k`、`attn@v`。
- 当前 Neumann stage1 为 `chunkSize=16`，MAC 更少，但 matmul 调用变成 10 次：`K@K`、`Q@K`、3 次 `Lpow`、3 次 `INV@Lpow`、`INV@k`、`INV@v`。
- 当前 `INV += tmp` 仍在 AIV 做，每轮需要：AIC 产出 tmp → set flag → AIV 读 tmp/INV → AIV add → AIV 写 INV → set flag → AIC 继续下一轮。三轮 Neumann 带来 3 次 AIC/AIV 往返和额外 GM 读写。

**结论**:
`S=16` 下 Neumann 的算法 FLOPs/MAC 是降低了，但当前实现把一次 AIV 连续求逆换成了 6 个极小 FP32 matmul + 3 轮 AIC/AIV 同步/GM 累加。对 16×16 小矩阵，matmul 固定调度与 scalar 开销、跨核 flag 和 GM 往返比 MAC 更贵，因此总耗时仍慢。

**下一步优化方向**:
1. 优先把 `NeumannAddCompute()` 从 AIV 搬到 AIC/FixPipe atomic add：`tmp = INV @ Lpow` 搬出时直接累加到 `INVWsGm_`，减少 3 轮 AIV MTE2/MTE3 和 AIC/AIV 往返。
2. 为 `S<=16` 或 C16 exact chunk 增加直接 AIV 16×16 inverse 特化路径，避免 6 次 tiny matmul 的调度开销；该路径需要重新精度/性能对比。
3. 真正实现 FP16 A/B matmul 只能降低 MAC 成本，不能解决当前 `S=16` 的主要 scalar/sync 问题，优先级低于减少 matmul 调用和同步次数。

## profiling 汇总 CSV 字段调整

**用户要求**:
1. `performance_compare.csv` 不再输出 `normal_prof_dir` 和 `neumann_prof_dir`。
2. CSV 需要提取 AIC/AIV 关键指标：`aic_mac_time`、`aic_mac_ratio`、`aic_scalar_time`、`aic_mte1_time`、`aic_mte2_time`、`aic_fixpipe_time`、`aic_icache_miss_rate`、`aiv_time`、`aiv_total_cycles`、`aiv_vec_time`、`aiv_vec_ratio`、`aiv_scalar_time`、`aiv_mte2_time`、`aiv_mte3_time`、`aiv_icache_miss_rate`。
3. 只有 `aic_mac` 和 `aiv_vec` 保留 ratio，其他 ratio 不输出。
4. 用现有 profiling 结果按新格式重新汇总。

**实现动作**:
- 修改 `tests/pytest/summarize_chunk_gdr_performance_msprof.py`：
  - 新增统一 `METRIC_COLUMNS` 指标映射。
  - `performance_raw.csv` 保留 `prof_dir/summary_path` 用于追溯原始 profiling。
  - `performance_compare.csv` 移除 `normal_prof_dir/neumann_prof_dir`。
  - `performance_compare.csv` 展开 normal/neumann 两组指标，字段前缀分别为 `normal_` 和 `neumann_`。
  - 只保留 `aic_mac_ratio` 与 `aiv_vec_ratio`，不输出 `aic_scalar_ratio/aic_mte*_ratio/aiv_scalar_ratio/aiv_mte*_ratio`。

**重汇总结果**:
- 已重新生成：
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_raw.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/performance_raw.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/performance_compare.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_smoke/performance_raw.csv`
  - `tests/pytest/prof_out/chunk_gdr_prefill_smoke/performance_compare.csv`

**验证**:
- `python3 -m py_compile tests/pytest/summarize_chunk_gdr_performance_msprof.py` 通过。
- 三个已跟踪 profiling 目录重新 summarize 通过。
- 检查确认所有 `performance_compare.csv` 均不含 `prof_dir` 字段。
- 检查确认 ratio 字段只包含 `aic_mac_ratio` 和 `aiv_vec_ratio`。

## 协作规则更新

**用户要求**:
1. 后续除非用户明确说要推送，否则不再自行 `git push`。
2. 后续性能批测中，`S=16`、`S=64` 要随原有 full prefill shape 一起测试。

**执行记录**:
- 当前仅更新本地文档，不执行 push。
- `S=16/S=64` 已在 `tests/pytest/compare_chunk_gdr_performance_paramset.py` 纳入默认 `PERFORMANCE_CASES`，后续执行 `run_chunk_gdr_performance_msprof.py` 会和原有 shape 一起跑。

## profiling compare CSV 改为每 case 三行

**用户要求**:
- 不要把同一 case 的 normal/neumann 两个算子放在一行。
- 每个 case 输出三行：normal 一行、neumann 一行、第三行给出 `neumann/normal` 比值。
- 所有 shape 放在一起输出总表。

**实现动作**:
- 修改 `tests/pytest/summarize_chunk_gdr_performance_msprof.py`：
  - `performance_compare.csv` 改为 long format。
  - 新增 `row_type` 和 `ops` 字段。
  - `row_type=measurement, ops=normal`：普通算子实测值。
  - `row_type=measurement, ops=neumann`：Neumann 算子实测值。
  - `row_type=ratio_neumann_over_normal, ops=neumann/normal`：逐指标比值。
  - 支持 `--extra-root` 和 `--output-root`，用于把多个 profiling 根目录合并成一个总表。
- 用现有 profiling 重新汇总：
  - `tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv`：24 case × 3 行 = 72 行。
  - `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/performance_compare.csv`：8 case × 3 行 = 24 行。
  - `tests/pytest/prof_out/chunk_gdr_prefill_smoke/performance_compare.csv`：1 case × 3 行 = 3 行。
  - `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes/performance_compare.csv`：32 case × 3 行 = 96 行，包含原 full shapes 与 `S=16/S=64`。

**验证**:
- `python3 -m py_compile summarize_chunk_gdr_performance_msprof.py` 通过。
- 检查确认每个 case 都正好三行。
- 检查确认 compare CSV 不含 `prof_dir` 字段。
- 检查确认 compare CSV 不再有 `normal_*`/`neumann_*` 同行展开字段。
- 检查确认 ratio 字段仍只保留 `aic_mac_ratio` 与 `aiv_vec_ratio`。

**注意**:
- 按用户最新协作规则，本次只做本地修改，不执行 `git push`。

## 清理 prof_out 并准备讨论算子修改

**用户要求**:
- 清空 `tests/pytest/prof_out` 目录。
- 接下来讨论如何修改算子。

**执行动作**:
- 已执行 `rm -rf tests/pytest/prof_out/*`，保留 `tests/pytest/prof_out/` 空目录。
- 清理前目录约 `73M`，清理后为空目录约 `4.0K`。
- 按用户协作规则，本次未执行 `git push`。

**当前代码讨论重点**:
- `S=16` 反常慢点已经定位：Neumann MAC 降低，但 6 次 tiny matmul + 3 轮 AIC/AIV `INV += tmp` 往返导致 scalar/sync/GM 开销压过收益。
- 优先讨论修改方向：
  1. 将 `INV += tmp` 从 AIV 移到 AIC/FixPipe atomic add。
  2. 评估 `S<=16` 走 AIV 直接 inverse 特化，避免 tiny matmul 调度。
  3. 评估回到 C64 的分块 Neumann 方案，降低 chunk 数放大。

## 方案A：C16 FP32 Neumann 累加移入 AIC

**用户目标**:
- 先保存 `chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h` 当前版本。
- 再在该文件内修改：除初始矩阵生成继续用 AIV 外，后续 Neumann 求逆累加完全在 AIC 内完成。
- 同时分析该方案的性能收益点。

**执行动作**:
- 新增基线副本：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse_AIVAddBaseline_20260429.h`。
- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h`：
  - AIV 仍负责 `InverseCompute()`，写出 `L_powWsGm_ = L` 和 `INVWsGm_ = I-L`。
  - AIC 三轮改为 `NeumannMmulAddCompute()`，通过 `mmFp32.IterateAll(INVWsGm_, 1)` 将 `INV @ L_pow` 直接 atomic add 到 `INVWsGm_`。
  - 移除主流程中三轮 AIC↔AIV add 同步：`0x1/0x2`、`0x3/0x4`、`0xa/0xb`。
- 新增方案文档：`docs/2026-04-29/C16_AIC_ONLY_NEUMANN_ACCUMULATION_PLAN.md`。

**验证**:
- 临时切 selector 到 `C16_FP32_CVInverse` 后构建双算子包通过：`bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 安装后执行三方 smoke：`python3 compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64`，结果 PASS。
- smoke 中 CPU vs Neumann out/state 均 PASS；ChunkGDR vs Neumann out/state `max_abs=2.441406e-04`。

**性能收益判断**:
- 直接收益是去掉三轮 AIC→GM→AIV→GM→AIC 的累加往返，减少 flag 同步、AIV vec、AIV MTE2 和 GM 读写。
- AIV 不再等待三轮 tmp，可在 `InverseCompute()` 后立即计算 `GBKCompute()`/`VBetaCompute()`，与 AIC Neumann matmul 重叠。
- 方案不会减少 Neumann matmul 调用次数，收益重点应体现在 scalar/sync/MTE 和端到端流水重叠，而不是 AIC MAC time 大幅下降。

**风险**:
- 当前使用 `INVWsGm_` 同时作为 matmul 输入和 atomic add 输出；C16 smoke 已通过，但仍需全量精度和 msprof 验证所有 shape 稳定性。
- selector 已恢复到当前默认 `C16_FP16_CVInverse`，本次只修改实验文件；正式验证需显式切 selector。


## 方案A全量精度验证与基线清理

**执行动作**:
- 正式将 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` selector 切到 `chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h`。
- 重新构建并安装双算子包：`bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 执行全量三方精度：`cd tests/pytest && bash test_run.sh compare`。
- 全量精度通过后，按用户要求删除基线副本 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse_AIVAddBaseline_20260429.h`。

**验证结果**:
- `compare_chunk_gdr_three_way.py` 共 20 个 pytest case 全部 PASS。
- pytest 汇总：`20 passed in 5.96s`。
- 覆盖了 `S=5/16/31/48/64/80/96/128/129/160/257`、变量长度、GQA、`dk/dv=32/64/128` 和多 batch/head 组合。

**当前状态**:
- 方案A已成为当前默认 stage1 实现。
- 下一个必要验证是 32 shape msprof 性能批测，重点观察 AIV vec/MTE2、AIC scalar/FixPipe 和端到端耗时。

## 方案A 32-shape msprof 性能结果

**执行动作**:
- 执行包含 `S=16/S=64` 的 32 个 prefill shape 批测：`python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/chunk_gdr_prefill_all_shapes_aic_accum`。
- 结果保存到 `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes_aic_accum/performance_compare.csv` 和 `performance_raw.csv`。
- 输出格式为每 case 三行：`normal`、`neumann`、`ratio_neumann_over_normal`。

**总体现象**:
- 32 个 shape 中，Neumann 仅在 3 个 `S=16` case 更快；其余 29 个 case 普通 `ChunkGDR` 更快。
- 全量平均 `task_duration_us`：normal `11996.635us`，neumann `44766.741us`，Neumann/normal 平均约 `3.015x`。
- `S=16` 平均：normal `2275.980us`，neumann `2270.590us`，Neumann/normal 平均 `0.994x`，基本持平且略优。
- `S=64` 平均：normal `2363.630us`，neumann `5044.865us`，Neumann/normal 平均 `2.130x`，仍明显更慢。

**关键指标**:
- `S=16`：AIV vec 平均 ratio `0.179x`，说明 AIC atomic add 方案确实消掉了大部分 AIV add vec 开销；但 AIC scalar 平均 ratio `1.207x`，收益基本被 tiny matmul 固定开销抵消。
- `S=64`：AIV vec 平均 ratio `0.646x`，仍有改善；但 AIC scalar 平均 ratio `3.888x`、AIC fixpipe 平均 ratio `2.684x`，端到端被 4 个 C16 chunk 的 matmul 调度/FixPipe/MTE 放大主导。
- `S>=128`：Neumann/normal task ratio 约 `2.4x~4.0x`，随序列增长接近 4 倍 chunk 数放大上限。

**结论**:
- 方案A正确降低了 AIV 累加开销，并让 `S=16` 从之前平均 `1.119x` 慢变为平均 `0.994x` 基本持平/略优。
- 方案A没有减少 Neumann tiny matmul 调用次数；当 `S>=64` 时，C16 chunk 数放大带来的 AIC scalar、FixPipe、MTE 开销仍是主因。
- 下一步优化不应继续只搬 AIV add；应优先减少 matmul 调用/固定调度次数，或切到 C64 block inverse 方案。

## C16 当前方案剩余性能收益点分析

**用户问题**:
- 当前方案还能否通过减少重复搬运等方式获得性能收益。

**分析结论**:
- 当前 AIC atomic add 已经有效降低 AIV vec 开销，但 `S>=64` 的主要慢点仍是 C16 chunk 数放大后的 AIC scalar、AIC MTE2、FixPipe 和 tiny matmul 调度。
- 低风险清理项：方案A后 `tmpWsGm_` 不再被主流程使用，但 kernel 和 host tiling 仍按 `L_pow + tmp + INV` 分配三块 Neumann workspace；可以删除 `tmpWsGm_`、未调用的 `NeumannAddCompute()`，并把 workspace 从 `3*C*C` 改为 `2*C*C`。
- 最大 stage1 内部重复搬运：AIV 将 `query/key` 从 BF16 cast 为 FP32 后写 `queryContinousGm_`/`keyContinousGm_`，AIC 再读 FP32 GM 做 `K@K^T` 和 `Q@K^T`。若能改成 AIC 直接吃 BF16/half 输入、FP32 输出，可删除两块连续化 GM 和对应 AIV cast/write。
- 仍然保留的硬依赖：`gBKWsGm_`、`vBetaWsGm_` 是 AIV 生成、AIC matmul 消费，短期难删；`L_powWsGm_`/`INVWsGm_` 是 Neumann matmul 的 GM 输入/输出，也不能直接删除。
- 跨 stage 大搬运链路：Stage1 写 `outQkGm_`，Stage3 AIV 读后乘 scale/gamma/mask，再写 `tmpGM_`，Stage3 AIC 再读。可考虑前移 scale 或融合 masked qkt，但风险高于 stage1 内部清理。

**新增文档**:
- `docs/2026-04-29/C16_REMAINING_PERFORMANCE_OPPORTUNITIES.md`

**推荐顺序**:
1. P0：删除 `tmpWsGm_` / `NeumannAddCompute()` / workspace 第三块，低风险清理。
2. P1：优先推进 C64 分块求逆或 BF16/half 输入 AIC，真正减少 chunk 数、tiny matmul 调度或重复 GM 搬运。
3. P2：再考虑 Stage3 qkt mask 链路融合或 scale 前移。

## S=64 MAC 计算量分析

**用户问题**:
- `S=64` 时为什么新算子的 Matmul/MAC 看起来更多，需要把变量、求逆轮次全部乘上解释。

**结论**:
- 求逆子项确实更多：normal 为 `2*32^3=65,536`，Neumann 为 `4 chunks * 3 rounds * 2 * 16^3=98,304`，即 `1.5x`。
- 但全 Stage1 理论 MAC 并没有更多：normal `1,114,112`，Neumann `360,448`，ratio `0.324x`。
- 全算子理论 MAC 也没有更多：normal `2,162,688`，Neumann `1,212,416`，ratio `0.561x`，这里是每 `B,Nv` 的计算量，乘 `B*Nv` 后比例不变。
- profiling 中 `S=64` 的 `aic_mac_time_us` 稳定为约 `1.724x`，原因不是理论 MAC 总量增加，而是 C16 将 matmul 调用次数从每 `B,Nv` 约 `10` 次放大到约 `56` 次，且新增 `24` 次 `16×16×16` tiny matmul，Cube 利用率和调度摊销很差。

**新增文档**:
- `docs/2026-04-29/S64_MAC_ANALYSIS.md`

---

## 只读分析：C64_3232 / C64_1616 推导 32x32 Neumann 方案

**用户请求**: C64_1616 全量 msprof 并行期间，只读分析 `stage1_C64_3232_FP32_CVInverse.h` 与当前 C64_1616，实现 32x32 Neumann 方案的最小改造点、数学公式、matmul 次数、tail/validLen 建议，以及 C/AIC 全程完成的风险点。

**静态结论**:
- `C64_3232` 文件当前仍是旧版整 `64x64` Neumann + AIV `NeumannAddCompute()` 三轮加法，并不是完整 32x32 分块实现。
- 当前 `C64_1616` 已具备更接近目标的结构：AIV 仅生成初始 `L_pow=L` 和 `INV=I-L`，AIC 完成 inverse 数学，并增加 `SyncFixToMte2()` 与 chunk workspace 读完同步 `0xc`。
- 32x32 Neumann 最小方案应以当前 `C64_1616` 为基底：新增 `INVERSE_SHAPE=32` 或独立 `NEUMANN_BLOCK=32`，实现两个 32x32 对角块 Neumann，再用一次 64-from-32 block compose。
- 理论调用次数：每个 32x32 对角块 5 轮倍增，每轮 `Lpow` + `INV@Lpow` 共 10 次；两个对角块共 20 次；64-from-32 compose 2 次；总 inverse 22 次，比 16x16 方案的 30 次少 8 次，但单次 32x32 MM 更大。
- tail/validLen 应保留当前分段保护：`validLen<=32` 只算上半 32x32；`validLen>32` 再算下半并 compose；尾块无效行依赖已有 AIV zero padding 和 mask，AIC 不应读取未初始化有效范围之外的数学贡献。

**关键证据**:
- `C64_3232` 旧三轮 AIC/AIV 协同：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h:315`
- `C64_3232` 旧 `NeumannMmulCompute()` 写 `tmpWsGm_`：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h:809`
- 当前 `C64_1616` 全 AIC inverse 入口：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h:317`
- 当前 `C64_1616` 16x16 block inverse/compose：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h:789`
- 当前 tail gating：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h:847`
