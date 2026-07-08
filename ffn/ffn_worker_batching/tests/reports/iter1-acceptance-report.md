# 迭代一 测试工程师验收报告（FfnWorkerBatching）

> 生成时间：2026-07-08　算子：FfnWorkerBatching（A5 / Ascend950 / arch35 / DAV_3510 新增，范围对齐 A2）
> 验收口径：迭代一 = 骨架（skeleton），验收范围 = **L0 标准用例（NORM / TilingKey=100 / fp16 / 基础 shape / 单 dtype）覆盖完整 + ST 通过率 100%**。
> 测试方式：**C++ 原生 ST（Mock 模式，CPU golden 自测）** 快速复核 L0 用例。骨架 NPU 精度已由汇合验证 probe5 真机 8 路二进制一致确认（见 iter1-integration-report.md），本报告为 ST 层快速复核。
> 精度标准来源：`docs/spec.yaml numerical_tolerance.per_dtype`（全 dtype `bitwise_equal`, rtol=atol=0）；非计算类算子（sort + index-gather + scan）。

## 状态: ✅通过

## 验收摘要

| 验收项 | 结果 | 详情 |
|--------|------|------|
| 用例覆盖 | 通过 | 覆盖率: 100%（迭代一范围 L0 NORM/fp16 基础 shape 全覆盖：3 内置 fp16-NORM + 3 CSV fp16-NORM + 1 CSV int8-NORM + 2 CPU golden 自测；RECV 单条按范围外 SKIP） |
| ST通过率 | 通过 | 通过率: 100% (10/10)，PASS 9 + SKIP 1（RECV，范围外，非阻塞） + FAIL 0 |
| 回归测试 | 不适用 | C++ ST 层为新增骨架特性验证，无存量 ST 可回归；A2(ascend910b)/A3(ascend910_93) 无回归已由汇合验证 op_host UT（`tiling_ascend910b_norm_no_regression` / `tiling_ascend910_93_recv_no_regression`）确认 PASS |
| 黑盒CSV覆盖 | 通过 | 期望: L0=5（`blackbox_case_targets.L0`）；执行: 5 全加载（4 golden 对拍 + 1 RECV 文档化 SKIP）；路由: NORM/need_schedule=0 → golden 对拍，RECV/need_schedule=1 → SKIP（范围外）；缺口: L1(515)/L2(13) 及 RECV/多核/int8/bf16 全整合属后续迭代，非迭代一验收范围 |

## 关键指标

- **总用例数**: 10（2 CPU golden 自测 + 3 内置 L0 fp16-NORM + 5 CSV L0）
- **通过**: 10（PASS 9 + SKIP 1；SKIP 为 need_schedule=1 RECV 路径，迭代一范围外，harness 记为非阻塞通过）
- **失败**: 0
- **通过率**: **100%**
- **确定性（ST 层）**: 连跑 2 次（130300 / 130433），全部 10 条结果行与关键值（actual_token_num=103/128/64/11664/66477/2458/24320）逐行一致，仅 CMake 缓存/编译探测行不同 → ST 层可复现
- **运行环境**: CANN 9.1.0（`/home/zl/Ascend`，`ASCEND_HOME_PATH=/home/zl/Ascend/cann-9.1.0`），cmake 3.28.3 / g++ 13.3.0；Mock 模式无需 NPU

---

## 一、验收范围（诚实标注）

迭代一是**骨架**迭代。本次 C++ ST 验收严格限定在 **L0 标准用例（NORM/TilingKey=100/fp16/基础 shape/单 dtype）** 的覆盖完整性与 100% 通过率上，与任务书验收标准一致。

**明确不在迭代一验收范围**（属后续迭代整合，不判失败）：
- RECV 路径（need_schedule=1 / TilingKey=101）—— CSV L0_002 一条按此范围外 SKIP。
- 多核 sort / int8(P2) / bf16(P1) 全整合 —— 其可行性已由 probe1~4 分别在真机 NPU 上 8/8 二进制一致验证（见 PROBE_SUMMARY.md）。
- L1（515）/ L2（13）黑盒用例 —— 迭代一不要求执行。

**NPU 精度已由汇合验证收口**：本算子输入 `schedule_context`（1024B）内嵌二级 GM 指针，TTK kernel 模式无法直驱（已证据化）；骨架 NPU 精度由 probe5 真机（Ascend950DT_9584 / DAV_3510）8 路输出 `np.array_equal` 二进制一致、连跑 3 次 bitwise 确定确认（汇合验证 ✅通过）。本报告以 C++ ST Mock 模式（CPU golden 自测）在 ST 层快速复核 L0 用例，非以 Mock 冒充 NPU 精度。

---

## 二、执行证据（可追溯）

### 2.1 编译 + 运行日志

| 项 | 路径 |
|----|------|
| 本次 run1 日志 | `tests/reports/st_cpp_mock_20260708_130300.log`（首次 clean build + 运行） |
| 本次 run2 日志 | `tests/reports/st_cpp_mock_20260708_130433.log`（复现确认） |
| B 任务历史日志 | `tests/reports/st_cpp_mock_20260708_105113.log`（10/10 PASS，与本次一致） |
| 复现命令 | `source /home/zl/Ascend/ascend-toolkit/set_env.sh && cd tests/st && bash run.sh --mock` |

编译：clean build（`rm -rf build-mock`）→ CMake 配置 `-DUSE_MOCK=ON` → `make -j88` → `[100%] Built target` → 实际运行被测程序，退出码 0。**非仅编译，已实际运行。**

### 2.2 逐用例结果（run1 = run2，二进制一致，除时间戳/编译探测行）

| # | 用例 | 路径 | dtype | 关键值 | 结果 |
|---|------|------|-------|--------|------|
| 1 | CPU golden 自测-1（手工可核对 A=2,M=1,BS=2,K=2,H=4,ep=4） | NORM | fp16 | 手工核对一致 | PASS |
| 2 | CPU golden 自测-2（k=0 全 mask） | NORM | fp16 | actual_token_num=0, group_list 首行[0,0] | PASS |
| 3 | L0_std_fp16_basic（A=8,BS=4,K=4,H=16,ep=8,Y=128） | NORM | fp16 | actual_token_num=103/128, 有效行 8 | PASS |
| 4 | L0_std_fp16_full_valid（Y=128） | NORM | fp16 | actual_token_num=128/128, 有效行 8 | PASS |
| 5 | L0_std_fp16_same_bin（Y=64,ep=16） | NORM | fp16 | actual_token_num=64/64, 有效行 16 | PASS |
| 6 | aclnnFfnWorkerBatching_L0_001（Y=14580,ep=1） | NORM | fp16 | actual_token_num=11664/14580, 有效行 1 | PASS |
| 7 | aclnnFfnWorkerBatching_L0_002（Y=88360,ep=8） | **RECV** | bf16 | need_schedule=1 | **SKIP（范围外）** |
| 8 | aclnnFfnWorkerBatching_L0_003（Y=83096,ep=256） | NORM | int8 | actual_token_num=66477/83096, 有效行 256 | PASS |
| 9 | aclnnFfnWorkerBatching_L0_004（Y=3072,ep=5120） | NORM | fp16 | actual_token_num=2458/3072, 有效行 2459 | PASS |
| 10 | aclnnFfnWorkerBatching_L0_005（Y=30400,ep=8192） | NORM | fp16 | actual_token_num=24320/30400, 有效行 8192 | PASS |

**总计: 10  通过: 10  失败: 0**（harness tally；其中 #7 为文档化 SKIP，返回 non-blocking pass）。

### 2.3 SKIP 诚实性说明

SKIP 唯一一条（#7，need_schedule=1 RECV）由被测程序 `RunCaseMock` 显式判定并日志打印「[SKIP] need_schedule=1 (RECV/扫描路径) 非迭代一范围，参考模型仅覆盖 NORM(need_schedule=0) 路径」后返回 true。这是**范围外主动跳过**，非静默假通过——CPU 参考模型只实现 NORM 路径 golden，RECV 归后续迭代。

---

## 三、精度口径核对（spec.yaml 一致）

- `spec.yaml numerical_tolerance.per_dtype`：float16/bfloat16/int8/int32/int64/float32 全部 `{rtol:0, atol:0, metric:bitwise_equal}`。
- Mock 模式断言 = CPU golden 不变式核对（permutation / group_list 计数守恒 / actual_token_num / 静态 shape 契约），口径与 spec `math_semantics.reference_oracle.absent=true` 的替代 golden 来源（TEST.md §6）一致。
- 二进制一致（rtol=atol=0）由汇合验证 probe5 真机 8 路 `np.array_equal` 收口，SHA256 恒定，本 ST 层复核与其一致。

---

## 四、验收结论

迭代一 C++ ST 快速复核 **✅通过**：

1. ✅ L0 标准用例（NORM/100/fp16 基础 shape 单 dtype）覆盖完整——3 内置 fp16-NORM + 3 CSV fp16-NORM 全覆盖，另附 int8-NORM 与 2 CPU golden 自测扩充。
2. ✅ ST 通过率 100%（10/10，0 失败）；唯一 SKIP 为范围外 RECV 路径，文档化非阻塞。
3. ✅ 实际编译 + 运行，非仅编译；退出码 0；连跑 2 次结果行二进制一致（ST 层确定性）。
4. ✅ 保留可追溯执行证据（编译 + 运行日志 2 份）。
5. ➖ 回归测试 C++ ST 层不适用；A2/A3 无回归已由汇合验证 UT 确认。

**遗留观察项（非阻塞，迭代二整合前收敛）**：Real 模式需安装自定义算子包（当前 `opp/vendors/` 未安装 ffn_worker_batching_custom），迭代二整合 RECV/多核/int8/bf16 时补齐 Real 模式端到端 NPU 对拍；穿刺3 host/kernel UB 容量口径差异（248KB vs 216KB）大 shape 下需复核（见汇合验证报告 §六）。
