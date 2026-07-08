# 迭代一 联调 probe5 (probe5_norm_fp16_skeleton) 验证结果

## 状态: ✅ 成功（真实 Ascend950 NPU，8 路二进制一致）

迭代一联调（汇合验证）专用工程：验证 **A5 arch35 NORM(TilingKey=100) / P0 fp16 / 单核骨架路径**在真实 NPU 上 8 路输出与 CPU golden `np.array_equal` **二进制一致**（rtol=atol=0），连跑 3 次 bitwise 确定。**非 Mock / 非 CPU 仿真**。

## 工程信息

| 项 | 值 |
|----|----|
| Task Name | probe5_norm_fp16_skeleton |
| 目标 | 迭代一骨架汇合验证（NORM/100/fp16 单核） |
| TilingKey | 100 (NORM) |
| Dtype | **fp16 (P0)** — token_dtype=0 |
| 排序模式 | 单核 sort（totalLength ≤ sortLoopMaxElement，锁定单核） |
| 运行环境 | **NPU（Ascend950DT_9584 / arch35 / DAV_3510）** |
| 编译 arch | `--npu-arch=dav-3510` |
| CANN 环境 | `/home/zl/Ascend`（cann-9.1.0） |
| 验证方式 | **Kernel 直调**（host 组装 ScheduleContextInfo 标量 + GM 指针；非 TTK kernel 模式，schedule_context 二级指针无法注入已证据化） |

## 与主线的对应关系

忠实镜像 `op_kernel/ffn_worker_batching_apt.cpp` 中 `FfnWorkerBatchingProcess`（TilingKey=100）三步链，**复用主线 arch35 骨架头**：

- `arch35/ffn_wb_sort_one_core.h`（`KernelSortMaskOneCore`）— 与主线 identical
- `arch35/ffn_wb_gather_out_all.h`（`KernelFfnWBGatherOutAll<false>`）— 仅删 schedule_context 解析 include，gather 逻辑与主线 identical
- `arch35/ffn_wb_group_listing.h`（`KernelFfnWBGroupListing`）— 与主线 identical
- `arch35/ffn_wb_sort_base.h` — identical；`ffn_wb_common.h` — 仅差 tiling_def include

**fp16 分支真实性**：`ffn_wb_gather_out_all.h:47` 判定 `tokenDtypeSize_ = (tokenDtype==2)?sizeof(int8_t):sizeof(half)`，fp16(0) 与 bf16(1) 走完全相同的 2 字节 byte-gather；本工程 tokenDtype=0 + 真实 np.float16 值命中 fp16 骨架分支。

## 穿刺形状

`A=4, M=2, BS=2, K=3, H=16, expert_num=8`，`Y=24`；6 个 mask + 重复键；单核 sort；coreNum=2（core0 gather / core1 group_listing）。valid_len=18。`ubSize/aivNum/sortLoopMaxElement` 运行时派生（实测 UB=248KB / 64 核），无 arch 常量写死。

## 验证结果

| 输出 | dtype | 元素数 | 结果 |
|------|-------|-------|------|
| y | fp16(uint16) | 384 | ✅ 二进制一致 |
| group_list | int64 | 16 | ✅ 二进制一致 |
| session_ids | int32 | 24 | ✅ 二进制一致 |
| micro_batch_ids | int32 | 24 | ✅ 二进制一致 |
| token_ids | int32 | 24 | ✅ 二进制一致 |
| expert_offsets | int32 | 24 | ✅ 二进制一致 |
| dynamic_scale | fp32 | 24 | ✅ 二进制一致（恒 0） |
| actual_token_num | int64 | 1 | ✅ 二进制一致（=18） |

**8/8 PASS**。连跑 3 次合并 SHA256 恒为 `b3c6c20c9fec990e4df0f69f693dbb69d343fd3c3c8ba57e95be386d4ab4111f`。

## 复现

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe5_norm_fp16_skeleton
bash run.sh
```
