# 迭代一 穿刺2 (probe2_norm_multicore) 验证结果

## 状态: ✅ 成功（新环境 /home/zl/Ascend 复核通过）

真实 Ascend950DT_9584（arch35 / DAV_3510，CANN 9.1.0）NPU 上实跑，8 路输出与 CPU golden **二进制一致**（`np.array_equal`，rtol=atol=0），多次运行 bitwise 可复现。**非 Mock/CPU 仿真**。

> **新环境复核（2026-07-08）**：CANN 环境已由 `/root/Ascend` 统一切换到 `/home/zl/Ascend`（cann-9.1.0，编译器 `/home/zl/Ascend/cann-9.1.0/bin/bisheng`）。在新环境下 **clean rebuild**（`rm -rf build` 后 `--npu-arch=dav-3510` 重编）+ NPU 实跑复核，结论与旧环境完全一致：8/8 输出 `np.array_equal` 二进制一致；连跑 3 次 8 路输出合并 SHA256 恒为 `fd43123e39c7074e77873eb291f3d6730289c9c333762d9a1908fb4914e39389`（bitwise 确定性）；`gather_idx` 与 numpy `argsort(kind="stable")` 逐 bit 相同（稳定排序契约成立）。运行路径 `aclInit → aclrtSetDevice → <<<>>> → aclrtSynchronizeStream`（host 侧组装 schedule_context 内嵌 GM 地址、二级指针注入），确认 NPU 直驱、无 Mock/CPU 回退。TTK kernel 模式因无地址注入能力仍不适用（沿用旧结论）。

## 穿刺信息

| 项 | 值 |
|----|----|
| Task Name | probe2_norm_multicore |
| TilingKey | 100 (NORM) |
| Dtype | fp16 (P0) |
| 排序模式 | 多核 sort + MrgSort 4 路归并 |
| Memory Strategy | 大 shape 触发多核切分 |
| 运行环境 | **NPU（Ascend950DT_9584 / arch35 / DAV_3510）** |
| 编译 arch | `--npu-arch=dav-3510` |
| CANN 环境 | `/home/zl/Ascend`（cann-9.1.0，新环境复核 2026-07-08） |
| 验证方式 | **Kernel 直调**（`.asc` + `<<<>>>`，host 组装 schedule_context 内嵌 GM 地址；非 TTK kernel 模式） |

## 验证摘要

穿刺 shape：`Y=4096`（= 4 sort 核 × 1024/核，触发多核 + 4 路归并），`H=64`，`expert_num=128`，全 token 有效。

数据流（3 次核函数启动，`aclrtSynchronizeStream` 做核间栅栏，规避直调工程里脆弱的核间 SyncAll）：

1. **ffn_wb_probe_sort**（多核 sort，blockNum=4）：每核对自己 1024 元素 tile 做 `Sort<float,true>`（负 expert → 降序 = expert 升序聚合），proposal 落 workspace。
2. **ffn_wb_probe_merge**（MrgSort 4 路归并，blockNum=1）：`MrgSort<float,true>` 4 路合并 → `Extract` → 还原 expert → `sorted_expert_ids` + `gather_idx`。
3. **ffn_wb_probe_divide**（核间分工，blockNum=8）：core 0-3 gather（`DataCopyPad` 逐行搬 y + 索引 gather session/micro/token），core 4-7 group_listing（本地直方图 + `SetAtomicAdd<int32_t>` 聚合逐专家计数）。

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅通过 | 新环境 `/home/zl/Ascend` clean rebuild，`--npu-arch=dav-3510`，无错误 |
| NPU 运行 | ✅通过 | 3 stage 全部在 Ascend950DT_9584 实跑完成 |
| 二进制一致 | ✅通过 | 8/8 输出 `np.array_equal` bitwise 相同 |
| 确定性 | ✅通过 | 连跑 3 次合并 SHA256 恒为 `fd43123e…e39389`，bitwise 完全一致 |
| 稳定排序 | ✅通过 | `gather_idx` == numpy `argsort(kind="stable")` 逐 bit 相同 |

8 路输出逐项二进制一致：`sorted_expert_ids`(int32) / `gather_idx`(int32) / `y`(fp16) / `session_ids`(int32) / `micro_batch_ids`(int32) / `token_ids`(int32) / `group_count`(int32) / `actual_token_num`(int32) 全部 PASS。

## 关键发现（arch35 移植结论）

1. **多核 Sort + MrgSort 4 路归并在 arch35/DAV_3510 完全可行**：`Sort<float,true>` / `MrgSort<float,true>` / `MrgSortSrcList` / `Extract` / `ArithProgression` / `GetSortLen<float>` / `GetSortOffset<float>` 与 arch22 **同签名同语义**，直接可用（与 moe_init_routing_v3/arch35 同族原语实证一致）。逐核 sort proposal 经 workspace dump 验证完全正确（每核 tile 升序聚合、稳定索引升序、全局行号正确）。
2. **Sort/MrgSort 元素上限**（本穿刺聚焦点）：单核 Sort 1024 元素/tile（32 repeat）、4 路 MrgSort 归并 4×1024=4096 元素，经 `oneLoopMaxElements=MGR_SORT_MAX_ELEMENT(1024)` 分 loop + `listSortedNums` 消耗跟踪，在 arch35 上正确工作，无元素上限溢出。**结论：MrgSort 分 loop 归并框架（对齐 A2 `ffn_wb_sort_mrgsort_out.h`）在 arch35 有效。**
3. **稳定排序确定性契约成立**：含大量重复专家键（128 专家、4096 token），NPU `Sort→MrgSort` 输出的 `gather_idx` 与 numpy `argsort(kind='stable')` **bitwise 一致**——等键按原始行号升序稳定（跨 tile 归并保序）。满足 spec `determinism.bitwise_reproducible`，为迭代二多核整合去风险。
4. **核间 gather/group_listing 分工可行**：同一次 launch 内按 `GetBlockIdx()` 分流（gather 核 / group_listing 核）成立；`SetAtomicAdd<int32_t>` 跨核聚合逐专家计数正确；`DataCopyPad` 逐行字节 gather（hidden states）+ UB 驻留索引 gather 正确。

## 排障记录（供整合参考）

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 核函数 kernel type 未标记告警 | 直调工程需显式标记 vector 核；`__aicore__` 未标记触发 auto-derivate 告警 | 改用 `__global__ __vector__`（对齐 add_custom 直调模板约定；op 框架内则由 def.cpp KERNEL_TASK_TYPE 指定） |
| 2 | 归并输出前 2048 全 0、`gather_idx` 错乱 | **MrgSortOut 输入列表双重偏移**：`gmInputs_[i]` 已按 `i*GetSortLen(perList)` 设基址，又叠加 `offsets_[i]=GetSortOffset(perList*i)`，导致 list1 读到 core2、list2/3 读到未初始化 GM | 对齐 `moe_v3_mrgsort_out.h`：**所有列表共用同一 workspace 基址**，仅用 `offsets_[i]` 区分。1 行修复后 8/8 PASS |

> 整合提示：迭代二移植 `ffn_wb_sort_mrgsort_out.h` / `ffn_wb_sort_multi_core.h` 时，注意 workspace 基址与 proposal 偏移不可重复叠加（A2 原实现 `gmInputs[i]=workspaceGms[srcWsIndex]` 共用基址，靠 `offsets[i]` 区分，本穿刺已复现验证该约定正确）。

## 范围与简化说明（诚实标注）

- **无 mask（全 token 有效）**：本穿刺聚焦「多核 sort→MrgSort 4 路归并 + 分工」核心风险，全 token 有效使 golden 无歧义、validCnt=Y。被 mask 无效 token「置极大值排末尾」的语义与本穿刺的排序机制正交，arch35 采 `Select`（`__VEC_SCOPE__`/MicroAPI，见 `moe_v3_sort_one_core.h`）实现，由主线/其它穿刺验证。
- **group_list 简化为 group_count**：本穿刺输出逐专家 token 计数（histogram），验证 group_listing 计数 + `SetAtomicAdd` 多核聚合这一硬核；算子实际 `group_list=[expert_id, token_num]` 的打包为格式细节，迭代整合时补齐。
- **`GetSortTmpSize` 主机侧平台 API**：该 API 为 op 框架 host tiling 用（`sort_tiling_intf.h`），在独立 Kernel 直调工程中未作为可链接导出符号提供。本穿刺 kernel 侧 tmp/proposal 缓冲用 `GetSortLen<float>()`（2 float/proposal，与 moe arch35 in-kernel 一致）派生，Sort/MrgSort 实跑正确即证明 arch35 上缓冲尺寸充足。**host 侧 `GetSortTmpSize(platform, elemCount, dataTypeSize)` 的 arch35 平台参数需在主线 `_tiling_arch35.cpp` tiling 中验证**（本穿刺不阻塞）。
- **核间栅栏**：用 3 次 launch + `aclrtSynchronizeStream` 替代单 kernel 内 `SyncAll()`（SyncAll 语义验证由主线单 kernel 全流程承担）。

## 复现方式

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh   # 统一 CANN 环境（新环境）
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe2_norm_multicore
bash run.sh                 # 编译 + 生成数据 + NPU 实跑 + 8 路二进制一致校验
# 调试: PROBE_DUMP_WS=1 ./build/ffn_wb_probe  # dump 排序后 workspace proposal
```
