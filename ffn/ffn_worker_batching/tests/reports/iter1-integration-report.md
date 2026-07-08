# 迭代一 联调验证报告（汇合验证）

> 生成时间：2026-07-08　算子：FfnWorkerBatching（A5 / Ascend950 / arch35 / DAV_3510 新增）
> 联调聚焦：**A5 NORM/100/fp16 单核骨架路径在真实 NPU 上 ST 精度验证通过（二进制一致）**
> 判定口径：非计算类算子，精度 = 8 路输出与 CPU golden `np.array_equal`（rtol=atol=0）逐 bit 相同。

## 状态: ✅通过

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| UT验证 | 通过 | 通过率: 100%（op_host UT 10/10，本次独立复跑确认） |
| ST验证 | 通过 | 通过率: 100%（NORM/100/fp16 骨架路径 NPU 8 路二进制一致，连跑 3 次 bitwise 确定） |
| 前序回归 | 通过 | A2(ascend910b) NORM + A3(ascend910_93) RECV no-regression UT 均 PASS |

## 关键指标

- **UT 总用例数**: 10, 通过: 10, 失败: 0（op_host：7 Tiling + 3 InferShape），通过率 **100%**
- **ST 总用例数（骨架路径 NPU 精度）**: 8 路输出全部比对通过（1 个 NORM/100/fp16 单核 case 的 8 路输出），失败: 0，通过率 **100%**
- **ST 骨架路径确定性**: 连跑 3 次，8 路输出合并 SHA256 恒为 `b3c6c20c…4ab4111f`（bitwise reproducible）
- **运行环境**: 真实 NPU **Ascend950DT_9584**（arch35 / DAV_3510），CANN 9.1.0（`/home/zl/Ascend`），驱动 7.0.t9.0.B849；**非 Mock / 非 CPU 仿真**

---

## 一、验证范围与口径（诚实标注）

迭代一是**骨架**（skeleton）迭代，本次联调只对「A5 arch35 NORM(TilingKey=100) / P0 fp16 / 单核 sort」这一条骨架路径做 NPU 精度收口。
- RECV(101) / 多核 sort / int8(P2) / bf16(P1) 属后续迭代整合范围，**本次不要求全整合**（其可行性已由 probe1~4 分别在 NPU 上验证，见 PROBE_SUMMARY.md）。
- 本算子输入 `schedule_context`（1024B）内嵌 GM 设备地址（二级指针），**TTK kernel 模式无地址注入能力、无法直驱**（已证据化，见 PROBE_SUMMARY / LOG）。NPU 精度验证以已验证可行的 **Kernel 直调**通路为准：host 侧组装 `ScheduleContextInfo`（标量 + GM 指针），`<<<>>>` 直调，复用**主线 arch35 骨架头**。

---

## 二、ST 骨架路径 NPU 精度验证（核心验收项）

### 2.1 验证工程

- **工程**: `probe/probe5_norm_fp16_skeleton/`（本次联调新建，从 probe4_norm_bf16 派生，dtype 翻转为 fp16）
- **复用主线骨架头**（与 `op_kernel/ffn_worker_batching_apt.cpp` 的 `FfnWorkerBatchingProcess`（TilingKey 100）逐步对应）:
  - `arch35/ffn_wb_sort_one_core.h`（单核 sort，`KernelSortMaskOneCore`）— **与主线 identical**
  - `arch35/ffn_wb_gather_out_all.h`（字节 gather，`KernelFfnWBGatherOutAll<false>`）— 仅删去 schedule_context 解析 include，gather 逻辑与主线 identical
  - `arch35/ffn_wb_group_listing.h`（`KernelFfnWBGroupListing`）— **与主线 identical**
  - `arch35/ffn_wb_sort_base.h` — **与主线 identical**；`ffn_wb_common.h` — 仅差 tiling_def include
- **三步链**（忠实镜像主线 NORM 路径）: 单核 sort → `ValidGatherIdxLengthCompute`（写 actual_token_num）→ 核间分工 gather + group_listing → `SyncAll`。

### 2.2 fp16 骨架分支的真实性

- gather 数据类型分支位于 `ffn_wb_gather_out_all.h:47`：`tokenDtypeSize_ = (tokenDtype==2) ? sizeof(int8_t) : sizeof(half)`。
- 即 **fp16(tokenDtype=0) 与 bf16(tokenDtype=1) 走完全相同的 2 字节 byte-gather 代码路径**。本工程设 `tokenDtype=0`、token_data 用真实 `np.float16` 值，命中的正是 fp16 骨架分支。

### 2.3 穿刺形状

`A=4, M=2, BS=2, K=3, H=16, expert_num=8`，`Y=24`；expert_ids 含 6 个 mask（无效 token）+ 重复键（测稳定排序）；`totalLength(24) ≤ sortLoopMaxElement` → 锁定**单核 sort**；`coreNum=2`（core0 gather / core1 group_listing）。valid_len(actual_token_num)=18。
- `ubSize` / `aivNum` / `sortLoopMaxElement` 全部由运行时平台接口（`PlatformAscendCManager` / `GetCoreMemSize`）+ 属性派生，**无 arch UB/核数写死常量**（运行时实测 UB=248KB / 64 核，符合 MEMORY「UB size no hardcode」）。

### 2.4 结果

真实 Ascend950DT_9584 NPU 实跑，8 路输出与 CPU golden 逐 bit 比对：

| 输出 | dtype | 元素数 | 结果 |
|------|-------|-------|------|
| y | fp16(uint16 视图) | 384 | ✅ 二进制一致 |
| group_list | int64 | 16 | ✅ 二进制一致 |
| session_ids | int32 | 24 | ✅ 二进制一致 |
| micro_batch_ids | int32 | 24 | ✅ 二进制一致 |
| token_ids | int32 | 24 | ✅ 二进制一致 |
| expert_offsets | int32 | 24 | ✅ 二进制一致 |
| dynamic_scale | fp32 | 24 | ✅ 二进制一致（fp16 不产出，恒 0） |
| actual_token_num | int64 | 1 | ✅ 二进制一致（=18） |

**8/8 PASS**。连跑 3 次合并 SHA256 恒为 `b3c6c20c9fec990e4df0f69f693dbb69d343fd3c3c8ba57e95be386d4ab4111f`（bitwise 确定性）。

### 2.5 复现

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe5_norm_fp16_skeleton
bash run.sh   # 编译(dav-3510) + 生成数据 + NPU 实跑 + 8 路二进制一致校验
```

---

## 三、UT 验证（复用 + 独立复跑确认）

op_host UT 二进制 `build/tests/ut/framework_normal/op_host/transformer_op_host_ut`（ops-transformer 仓 UT 框架）本次独立复跑（`--gtest_filter=FfnWorkerBatching*`，`BUILD_PATH` 就绪），**10/10 PASS**：

- `FfnWorkerBatchingTilingTest`（7）：
  - `tiling_test01`
  - `tiling_arch35_ascend950_norm_key100`（**骨架路径 tiling**：tilingKey=100, blockDim=64, workspace=16819232, tilingData=`[Y=1152, H=4096, tokenDtype=0, expertNum=8, coreNum=64, ubSize=262144, sortLoopMaxElement=8192, sortNumWorkSpace=1152]`）
  - `tiling_arch35_ascend950_recv_key101`
  - `tiling_arch35_ascend950_bf16_norm`
  - `tiling_arch35_ascend950_int8_recv`
  - `tiling_ascend910b_norm_no_regression`（**A2 回归**）
  - `tiling_ascend910_93_recv_no_regression`（**A3 回归**）
- `FfnWorkerBatchingInfershapeTest`（3）：`infershape_test01` / `arch35_max_session` / `arch35_max_topk`

历史报告：`tests/reports/op_host_ut_arch35_run_20260708.log`（10/10 PASS，与本次复跑一致）。

---

## 四、前序回归

- **A2/A3 零回归**：由 op_host UT 中 `tiling_ascend910b_norm_no_regression`（A2）与 `tiling_ascend910_93_recv_no_regression`（A3）两条用例覆盖，本次复跑均 PASS。
- 结合 A1-Main 阶段的双路径二进制生成结论（ascend950 + ascend910b kernel 二进制均生成、opFile 隔离 + `IsRegbaseSocVersion` 守卫实测生效），A5 新增路径对 A2/A3 分支无侵入。

---

## 五、支撑性穿刺（NPU 已验证，供整合参考，非本次强制项）

以下穿刺均在真实 Ascend950DT_9584 NPU 上 8 路二进制一致（详见各 `probe/<task>/RESULT.md` 与 `PROBE_SUMMARY.md`）：

| 穿刺 | 分支 | 状态 | 本次动作 |
|------|------|------|---------|
| probe2 | NORM/100/fp16 多核 sort + MrgSort 4 路归并 | ✅ NPU 8/8 | 本次 `/home/zl` 环境复跑确认 8/8 二进制一致 |
| probe4 | NORM/100/bf16 单核（主线骨架头） | ✅ NPU 8/8 | 本次 `/home/zl` 环境复跑确认 8/8 二进制一致 |
| probe3 | NORM/100/int8 P2 单核 | ✅ NPU 8/8 | 沿用 PROBE_SUMMARY 结论 |
| probe1 | RECV/101/fp16 | ✅ NPU 8/8 | 沿用 PROBE_SUMMARY 结论 |

---

## 六、结论

迭代一骨架汇合验证 **通过**：
1. ✅ UT 验证通过（op_host 10/10，独立复跑确认）。
2. ✅ ST 骨架路径（NORM/100/fp16 单核）在真实 Ascend950 NPU 上精度验证通过——8 路输出 `np.array_equal` 二进制一致，通过率 100%，连跑 3 次 bitwise 确定；**非编译通过或 CPU Mock 冒充**。
3. ✅ A2/A3 无回归（UT 覆盖并复跑确认）。

**遗留观察项（非阻塞，迭代二整合前收敛）**：穿刺3 kernel 侧 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`=216KB 与 host 侧 `GetCoreMemSize(UB)`=248KB 不一致，大 shape 下有 UB 溢出风险；`ffn_wb_sort_mrgsort_out.h` 整合时注意 workspace 基址与 proposal 偏移不可重复叠加（probe2 排障记录）。
