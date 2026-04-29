# Neumann 算子进度与问题记录 - 2026-04-29

**最后更新**: 2026-04-29  
**主题**: 文档整理、当前算子状态确认

---

## 当前算子状态

| 项目 | 当前状态 |
|---|---|
| 算子 | `chunk_gated_delta_rule_neumann` |
| 数据类型 | 输入/输出 BF16，`g` 为 FP32，内部主要 workspace 为 FP32 |
| chunk | `chunkSize = 16` |
| `INVERSE_SHAPE` | `16` |
| `TASK_RATIO` | 保持 `2` |
| 求逆方式 | Neumann 三步倍增：`INV0 = I - L`，依次累加 `INV@L²`、`INV@L⁴`、`INV@L⁸` |
| 初始矩阵生成 | 当前由 AIV 两个 subBlock 拼完整 `16×16` 的 `L_powWsGm_` 和 `INVWsGm_` |
| AIC 职责 | 执行 `L_pow @ L_pow` 和 `INV @ L_pow` 等矩阵乘 |
| AIV 职责 | 生成 `L/I-L`、执行三轮 `INV += tmp`、处理 `GBK/VBeta/QPrime` |
| 当前长期方案 | 纯 AIC `InitInverseMatrix()` 仍是待评估方案，不是当前已落地实现 |

---

## 前序 direct-call 精度状态

验证方式：aclnn 单算子直调 `tests/test_neumann_direct`，不走 `torch_npu` pytest。该链路已被后续 torch_npu 三方精度脚本替代，临时直调文件已删除。

| 用例 | 参数 | 结果 |
|---|---|---|
| default | `B=2,T=128,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000001` |
| var_seqlens | `seqlens=17,65,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000000` |
| short_tail | `B=1,T=5,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000000`，FinalState max_abs_err `0.000031` |
| GQA spot check | `B=2,T=128,nk=2,nv=4,dk=64,dv=64` | PASS |
| dims_32 spot check | `B=2,T=128,nk=4,nv=4,dk=32,dv=32` | PASS |

结论：当前 direct-call 精度通过阈值 `0.01`，默认和尾块场景均已覆盖。

---

## 已完成整理

| 事项 | 状态 | 说明 |
|---|---|---|
| 错误年份修正 | 完成 | `docs/2025-*` 已移动为 `docs/2026-*`，文内日期同步修正 |
| 重复文档归档 | 完成 | 旧执行计划和初始进度放入 `docs/2026-04-28/archive/` |
| 文档索引 | 完成 | 新增 `docs/README.md`，标明当前有效文档和归档文档 |
| 今日进度 | 完成 | 新增本文件作为 2026-04-29 最新进度 |

---

## 当前待办

1. 若只要求当前可运行版本：以 `compare_chunk_gdr_three_way.py` 作为默认三方精度入口。
2. 若继续推进原始长期方案：设计并验证 AIC 端 `InitInverseMatrix()`，明确 AIC 是否能安全完成逐元素初始化。
3. 明确是否必须从当前 FP32 matmul / FP32 add 改为方案文档中记录的 FP16 MMA + FP16 Add。
4. 提交前拆分工作区中历史调试文件，避免把无关二进制和临时文件纳入变更。

---

## 本次只读分析记录

- 已完成 AscendC FP16/half A/B 输入、FP32 C 输出 matmul 写法检索。
- 推荐参考 `posembedding/rotary_position_embedding` 的 kernel `MatmulType<half, half, float>` 和 host `DT_FLOAT16/DT_FLOAT16/DT_FLOAT` 配置。
- 当前算子仍是 FP32 A/B/C matmul；若切换到 half A/B，需要新增 half GM workspace 或在进入 cube 前完成 GM dtype 转换。
- 暂未发现仓库内完整 float GM cast 到 half GM 再 cube MM 的端到端示例，仅发现通用 GM↔GM `CopyCast` 和 half GM 直接 matmul 示例。

## 风险和说明

- 当前代码精度通过，但实现与“纯 AIC 生成初始矩阵”方案不完全一致。
- 当前 Neumann matmul 类型仍是 FP32，不是文档早期目标中的 FP16 MMA。
- `docs/2026-04-28/archive/` 内文档仅作历史参考，不代表当前最新状态。

---

## AIC-only FP16 Neumann 设计判断

- 未发现 CANN matmul 公开接口支持 `float` GM 输入在 AIC 随路 cast 成 `half` 后参与 FP16 MMA。
- 可用能力是 matmul C 搬出 atomic add：`IterateAll/GetTensorC(..., 1)`，可把 `INV @ L_pow` 的 FP32 结果直接累加进 `INVWsGm_`。
- 推荐实验路径：AIC 侧完成 `float -> half` 显式 cast，再用 `half x half -> float` matmul；三轮 `INV += tmp` 改为 AIC FixPipe atomic add，移除 AIV add 路径。
- 风险：下一轮 `INV` 如需作为 half 输入，需要维护 half 镜像或每轮重新 cast，不能只维护 FP32 `INVWsGm_`。

---

## torch_npu 接口与三方精度对比进展

| 事项 | 状态 | 说明 |
|---|---|---|
| `chunk_gated_delta_rule` torch_npu 接口 | 已实跑通过 | 新增 `npu_ops_transformer` builder/C++ aclnn wrapper，并挂载 `torch_npu.npu_chunk_gated_delta_rule` |
| `chunk_gated_delta_rule_neumann` torch_npu 接口 | 已实跑通过 | 新增 `npu_ops_transformer` builder/C++ aclnn wrapper，并挂载 `torch_npu.npu_chunk_gated_delta_rule_neumann` |
| 三方精度脚本 | 已实跑通过 | `tests/pytest/compare_chunk_gdr_three_way.py` 比较 CPU golden、普通 NPU、Neumann NPU |
| CPU golden 逻辑 | 保持原样 | 三方脚本直接调用原 `cgdr_golden()`，未复制或改写 golden 算法 |
| Neumann pytest 启动脚本 | 已修复 | `test_run.sh` 指向 `test_chunk_gated_delta_rule_neumann_single.py` |
| 静态验证 | 已通过 | Python `py_compile` 通过 |
| JIT/NPU 实跑 | 已通过 | 已补 `ninja`，并重装包含两个算子的 custom OPP 包 |
| pytest 10 case | 已通过 | 独立 `compare_chunk_gdr_three_way_paramset.py` 维护批测输入，`bash test_run.sh compare` 收集 10 个三方精度 case，全部 PASS |

### 三方精度结果

| 用例 | 参数 | 结果 |
|---|---|---|
| default | `B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128` | PASS |
| var_seqlens | `B=2,seqlens=17,65,nk=4,nv=4,dk=64,dv=64` | PASS |
| GQA | `B=2,seqlen=64,nk=2,nv=4,dk=64,dv=64` | PASS |
| dims_32 | `B=2,seqlen=64,nk=4,nv=4,dk=32,dv=32` | PASS |
| short_tail | `B=1,seqlen=5,nk=4,nv=4,dk=64,dv=64` | PASS |
| exact_chunk | `B=1,seqlen=16,nk=4,nv=4,dk=64,dv=64` | PASS |
| tail31 | `B=1,seqlen=31,nk=4,nv=4,dk=64,dv=64` | PASS |
| seq128 | `B=1,seqlen=128,nk=4,nv=4,dk=64,dv=64` | PASS |
| mixed_dims | `B=2,seqlen=32,nk=4,nv=4,dk=128,dv=64` | PASS |
| gqa_nk1 | `B=1,seqlen=64,nk=1,nv=4,dk=64,dv=32` | PASS |

默认用例关键误差：
- CPU vs `ChunkGDR` out/state max_abs：`9.736717e-04` / `4.882067e-04`。
- CPU vs `ChunkGDRNeumann` out/state max_abs：`9.736717e-04` / `4.882067e-04`。
- `ChunkGDR` vs `ChunkGDRNeumann` out/state max_abs：`1.220703e-04` / `1.220703e-04`。

精度高低判断：默认 case 中两者相对 CPU golden 的误差几乎完全一致，差异只在 BF16 量化级别；当前结果不能说明普通 `ChunkGDR` 或 `ChunkGDRNeumann` 哪个实质精度更高。

### 已解决问题

- `npu_ops_transformer` 首次导入原本会触发无关 MoE JIT；已将 `ops/__init__.py` 改成懒加载，三方脚本只显式加载两个目标算子。
- 本机缺少 `ninja` 导致 PyTorch C++ extension JIT 失败；已安装 `ninja`。
- 已安装包最初只包含 Neumann `aclnn` 符号，普通 GDR 调用失败；已使用 `--ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann` 重包并安装到 `${ASCEND_HOME_PATH}/opp`。

---

## 输入范围和 direct-call 清理策略

| 项目 | 状态 | 结论 |
|---|---|---|
| 三方脚本随机范围 | 已核对 | 与原 pytest/golden 一致：`rand [0,1)`、`g=-rand`、`q/k normalize`、BF16 主输入、FP32 `g`、INT32 seqlens |
| CPU golden 调用 | 已核对 | 直接复用原 `cgdr_golden()`，默认 `use_float64=False`，不复制/改写逻辑 |
| 默认维度 | 已对齐短用例 | 默认改为 `B=1,seqlen=64,nk=4,nv=4,dk=128,dv=128` |
| aclnn direct-call 删除 | 已执行 | torch_npu 三方精度脚本已通过默认、变量长度、GQA、`dk=dv=32` 覆盖 |

### 已删除对象

- `tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp`
- `tests/generate_aclnn_test_data_cpu.py`
- `tests/test_neumann_direct`、`tests/test_neumann`、`tests/test_original` 等本地直调二进制
- `tests/test_aclnn_chunk_gated_delta_rule_original.cpp` 等旧临时对比文件
- `attention/chunk_gated_delta_rule/` 下旧临时直调文件和临时测试数据

### 不删除对象

- `op_host/op_api/aclnn_chunk_gated_delta_rule_neumann.*`：这是算子 aclnn API 实现，不是测试脚本。
- `tests/ut/op_host/op_api/test_aclnn_chunk_gated_delta_rule_neumann.cpp`：这是 op_api 单元测试，不是精度直调脚本。

---

## 精度 Case 扩展与 msprof 性能对比进展

| 事项 | 状态 | 说明 |
|---|---|---|
| 三方精度 case 扩展 | 已完成 | `compare_chunk_gdr_three_way_paramset.py` 已扩展到 20 个 pytest case |
| 三方精度批测 | 已通过 | `bash test_run.sh compare` 结果 `20 passed in 6.03s` |
| prefill 性能 shape | 已完成 | `compare_chunk_gdr_performance_paramset.py` 收录用户给定 24 个 `(B,S,D,Hq,Hkv)` shape |
| msprof payload | 已完成 | `compare_chunk_gdr_performance.py` 改为 warmup + cache flush + measurement 的 application 载体 |
| msprof 批测 launcher | 已完成 | `run_chunk_gdr_performance_msprof.py` 批量执行 normal/neumann 两个算子 |
| test_run 入口 | 已完成 | `test_run.sh performance` 调用 msprof 批测入口 |
| 性能脚本冒烟 | 已通过 | 小 flush 下 normal/neumann 均完成输出，未做 Python event benchmark |

### 当前性能测试方法

- 单 shape 单算子：
```bash
msprof --application="python3 compare_chunk_gdr_performance.py --ops neumann --B 8 --S 128 --D 64 --Hq 16 --Hkv 4" \
       --output=./prof_out/neumann_b8_s128_d64_h16_hkv4 --task-time=on
```

- 批量跑用户给定 prefill shapes：
```bash
cd tests/pytest
python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/chunk_gdr_prefill
```

- 只生成命令不执行：
```bash
python3 run_chunk_gdr_performance_msprof.py --dry-run --limit 1
```

### 当前风险和说明

- 真实性能结果需要从 `msprof` 输出解析，不再使用 Python event 或 wall time。
- 默认 flush 张量较大，完整 24 shape × 2 算子 profiling 会产生大量输出；建议先单 shape 确认 msprof 环境和输出解析链路。
- 当前性能脚本沿用三方精度脚本的数据范围：BF16 主输入、FP32 `g`、`q/k` normalize、`scale=1/sqrt(D)`、prefill 场景 `actual_seq_lengths=[S]*B`。

---

## 性能批测状态澄清

| 项目 | 状态 | 说明 |
|---|---|---|
| 完整性能批测 | 未执行 | 尚未跑完 24 个 prefill shape × 2 个算子的真实 `msprof` |
| 性能脚本验证 | 已完成 | 已通过 `py_compile`、`dry-run` 和小 flush payload 冒烟 |
| 可比性能结果 | 暂无 | 目前没有完整 `msprof` 结果，不能判断两算子性能高低 |

建议先执行单 shape：
```bash
cd tests/pytest
python3 run_chunk_gdr_performance_msprof.py --case prefill_b8_s128_d64_h16_hkv4 --output ./prof_out/chunk_gdr_prefill_smoke
```

---

## 完整性能批测结果与 C64 方案计划

| 项目 | 状态 | 说明 |
|---|---|---|
| 单 case msprof | 已通过 | `prefill_b8_s128_d64_h16_hkv4` 的 normal/neumann 均通过 |
| 完整性能批测 | 已完成 | 24 个 prefill shape × 2 个算子均完成真实 `msprof` |
| 性能结果落盘 | 已完成 | `prof_out/chunk_gdr_prefill_full/performance_compare.csv` 与 `performance_raw.csv` |
| 自动汇总脚本 | 已完成 | `summarize_chunk_gdr_performance_msprof.py` 抽取 Phase4 measurement |
| C64 两方案计划 | 已完成 | `C64_NEUMANN_INVERSE_IMPLEMENTATION_PLAN.md` |

### 当前性能结论

- 当前普通 `ChunkGDR` 在全部 24 个 prefill shape 上都快于当前 `ChunkGDRNeumann`。
- `normal_us / neumann_us`：最小 `0.2095`，平均 `0.2564`，最大 `0.3652`。
- 换算看，当前 Neumann 平均耗时约为普通 GDR 的 `3.90x`。
- 越大的 `S/D/Hq` 场景，当前 Neumann 相对普通 GDR 的劣势更明显。

### 关键文件

- 性能对比表：`tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv`
- 性能原始表：`tests/pytest/prof_out/chunk_gdr_prefill_full/performance_raw.csv`
- C64 方案计划：`docs/2026-04-29/C64_NEUMANN_INVERSE_IMPLEMENTATION_PLAN.md`

### 下一步

1. 修改 tiling 与 stage selector，切到 chunk=64。
2. 优先实现 `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`，因为它复用 16×16 Neumann 基础，风险低。
3. 方案1通过编译、小 case 精度、局部性能后，再实现 `C64_3232_FP32_CVInverse.h`。

---

## profiler 指标修正与上传状态

| 项目 | 状态 | 说明 |
|---|---|---|
| cube 指标取列 | 已修正 | `summarize_chunk_gdr_performance_msprof.py` 改为读取 Y 列 `aic_mac_ratio` |
| CSV 重生成 | 已完成 | `performance_compare.csv` 和 `performance_raw.csv` 已基于已有 profiling 输出重生成 |
| 最后一列处理 | 已保留审计 | `cube_utilization(%)` 仅保留为 `msprof_last_cube_utilization_pct`，不作为本轮 cube 判断指标 |
| fork 上传 | 准备中 | 将创建独立分支提交代码；完整 `PROF_*` 目录不纳入提交 |

### 修正后的性能字段

- `cube_utilization_y_col_ratio`: 直接来自 `op_summary_*.csv` 的 Y 列 `aic_mac_ratio`。
- `cube_utilization_y_col_pct`: `aic_mac_ratio * 100`，用于 CSV 中按百分比阅读。
- `msprof_last_cube_utilization_pct`: 原最后一列，仅用于排查工具字段差异。

**上传结果**:
- fork 分支已推送：`chunk-gdr-neumann-20260429-upload`。
- 远端提交：`61d5a61c57502fdc80b1899f68e579bacb364213`。
- MR 创建入口：`https://gitcode.com/chuqihang/ops-transformer/merge_requests/new?source_branch=chunk-gdr-neumann-20260429-upload`。

---

## profiling 上传完成

| 项目 | 状态 | 说明 |
|---|---|---|
| 完整 profiling 原始目录 | 已上传 | `tests/pytest/prof_out/` 已提交并推送到 fork 分支 |
| full 批测输出 | 已上传 | `chunk_gdr_prefill_full/` 包含 24 个 prefill shape × 2 算子的 `PROF_*` 输出 |
| smoke 输出 | 已上传 | `chunk_gdr_prefill_smoke/` 一并保留 |
| 最新远端提交 | 已推送 | `c69375e7 Upload chunk GDR profiling artifacts` |

---

## fork master 状态

| 项目 | 状态 | 说明 |
|---|---|---|
| fork master 推送 | 已完成 | `fork/master` 已更新到 `44eff3448a9a206e3ae1eff22c292eae078aa2d0` |
| 后续默认上传分支 | 已明确 | 用户要求以后都直接推 `fork/master` |

---

## C16 profiling 开销分析进展

| 项目 | 状态 | 说明 |
|---|---|---|
| profiling 分项分析 | 已完成 | AIC MAC 平均仅 `1.52x`，AIC/AIV 总时间约 `4x` 主要来自 scalar/MTE/FixPipe/同步 |
| Stage2 复杂度核对 | 已完成 | Stage2 总 MAC 约 `3*S*Dv*Dk`，不随 chunk=16 下降到 1/4 |
| Stage3 复杂度核对 | 已完成 | Stage3 MAC 约降到 1/4，但调用/搬运/同步次数变 4 倍 |
| 重复计算排查 | 发现小问题 | `InverseCompute()` 对 `attnWsGm_` 写 `I-L` 后立即被 `L@L` 覆盖，属于死写 |
| 死写修复 | 已完成 | 四个 stage1 CVInverse 变体均删除无效 `attnWsGm_` 初始化 |
| 构建验证 | 已通过 | Neumann 单算子包构建通过，双算子包重装后精度 smoke 通过 |
| 关键风险 | 未解决 | `C16_FP16_CVInverse` 实际仍是 FP32 MatmulType/host tiling，名称与实现不一致 |

### 当前结论

C16 当前性能差的主因不是同一 chunk 被重复计算，而是 chunk 数变 4 倍后，小 FP32 matmul 调用、Stage2 state 搬运、AIC/AIV 同步和 MTE/FixPipe 次数被放大。已修复的 `attnWsGm_` 死写只影响少量搬运，不足以解释 4x。

## 性能批测 case 补充

| 项目 | 状态 | 说明 |
|---|---|---|
| `S=16` 性能 case | 已完成 | 为 `B=8/16/32/64` 补充 `D=64,Hq=16,Hkv=4` |
| `S=64` 性能 case | 已完成 | 为 `B=8/16/32/64` 补充 `D=64,Hq=16,Hkv=4` |
| 性能 case 总数 | 已更新 | `PERFORMANCE_CASES` 从 24 扩展为 32 |
| 静态校验 | 已通过 | `py_compile` 通过；`B=8,S=16/64` dry-run 命令生成通过 |

### 说明

三方精度 case 已有 `seqlen=16` 与 `seqlen=64` 覆盖；本次新增的是 msprof prefill 性能批测 case。完整性能数据需要重新执行：

```bash
cd tests/pytest
python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/chunk_gdr_prefill_s16_s64
```

## S=16/S=64 性能结果

| 项目 | 状态 | 说明 |
|---|---|---|
| `S=16/64` msprof | 已完成 | 8 个 prefill shape × 2 算子均完成 |
| 结果目录 | 已保存 | `tests/pytest/prof_out/chunk_gdr_prefill_s16_s64/` |
| `S=16` 结论 | 已确认 | Neumann 平均为 normal 的 `1.119x` 耗时 |
| `S=64` 结论 | 已确认 | Neumann 平均为 normal 的 `2.441x` 耗时 |
| 更快算子 | 已确认 | 8/8 case 都是普通 `ChunkGDR` 更快 |

### 关键判断

`S=16` 只有一个 C16 chunk，Neumann 相对 normal 只慢约 `8%~14%`；`S=64` 时 Neumann 变成 4 个 C16 chunk，对比 normal 的 1 个 C64 chunk，per-chunk 调度、同步、搬运和小 matmul 开销开始主导，因此退化到约 `2.4x`。
