# 迭代一 穿刺汇总（PROBE_SUMMARY）

> 判定环境：真实 Ascend950DT_9584（arch35/DAV_3510），CANN 9.1.0，**NPU 实跑**，8 路输出与 CPU golden `np.array_equal` 二进制一致为通过。
> 各行由对应穿刺 Agent 填写；权威结论见各 `probe/<task>/RESULT.md`。

| 任务 | 状态 | 重试次数 |
|------|------|---------|
| 主线 (100 NORM, fp16) | - | - |
| 穿刺1 (101 RECV, fp16) | ✅ 成功 | 0 |
| 穿刺2 (100 NORM, fp16, 多核 sort+MrgSort) | ✅ 成功 | 0 |
| 穿刺3 (100 NORM, int8 P2, 单核 sort, 32B scale 连续 gather) | ✅ 成功 | 0 |
| 穿刺4 (100 NORM, bf16 P1) | ✅ 成功 | 0 |

## 迭代二 穿刺汇总

> 验证迭代三全覆盖难点（P2 量化、边界 shape、DCCI 一致性），判定环境同上（NPU 实跑，8 路 `np.array_equal` 二进制一致）。

| 任务 | 状态 | 重试次数 |
|------|------|---------|
| 穿刺1 (101 RECV, int8 P2, 32B scale 连续 gather) | ✅ 成功 | 0 |

### 迭代二 穿刺1 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，CANN `/home/zl/Ascend` cann-9.1.0，`--npu-arch=dav-3510`；UB/核数运行时查 216KB/64 核，派生 coreNum=32/ubSize=204800/sortLoopMaxElement=6400，未写死）。
- shape A=8,M=2,BS=2,K=3,H=128,HS=160(int8 hidden+32B FP32 scale),expertNum=16,curMB=0 → Y=48，BSK=6→pad=2→BSK_PAD=8，~56% 有效 → validTokens=21（专家 e0×4/e3×9/e7×8 含重复测稳定序）。8 路输出（y int8 / group_list / session_ids / micro_batch_ids / token_ids / expert_offsets / **dynamic_scale fp32 有效** / actual_token_num=21）全部 `np.array_equal` 二进制一致；连跑 3 次 8 路合并 SHA256 恒为 `7a68a360…dab6d`（bitwise reproducible）。
- 结论：**RECV(`isScanFlag=true`) 与 int8+32B scale(`tokenDtype==2`) 两分支正交、可叠加**，在同一次 NPU 运行同时命中并二进制一致，无需 RegBase / 无需绕行；`ffn_wb_gather_out_all.h` 与 probe1/probe3 字节一致（0 kernel 逻辑改动，纯组合复用 probe1 RECV 链 + int8 profile）。RECV pad row_idx(BSK_PAD=8) 与 int8 gather 的 HS(160B) 字节步长两套索引互不干扰，K=3 非 2 幂对齐 stride 公式 `(M-1)*F+2==M*F-BSK` 成立。
- 降风险：迭代三 P2×RECV(6 组合中的一条)组合已实测通过，可直接纳入整合。
- 详见 `probe/probe_iter2_int8_recv/RESULT.md`。

## 穿刺1 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，`--npu-arch=dav-3510` 编译；**新 CANN 环境 `/home/zl/Ascend` 复核**，UB/核数运行时查 216KB/64 核，未写死）
- shape Y=16（RECV，curMicroBatchID=0，~50% mask → validTokens=7），8 路输出（y fp16 / group_list / session_ids / micro_batch_ids / token_ids / expert_offsets / dynamic_scale fp32 恒 0 / actual_token_num=7）全部 `np.array_equal` 二进制一致；连跑 3 次 8 路 SHA256 完全一致（bitwise reproducible）。
- 结论：RECV 路径「`scan_token_info` 前置扫描等待 micro batch ready → 单核稳定 sort → gather + group_listing」三步链在 arch35 端到端二进制一致可行，RECV 前置扫描不破坏 arch35 移植结论；含 e2 三并列/e5 两并列的稳定排序确定性成立。
- 机制：RECV 输入 1024B `schedule_context` 内嵌 device 二级指针（token_info_buf@384 / token_data_buf@400 / polling_index@416），host 侧组装地址后整体拷 device，`<<<>>>` 直调 + kernel 内逐字段解析——**TTK kernel 模式无地址注入能力、必须 Kernel 直调**（新环境无需改动即复现）；`SetSysWorkspaceForce` + `GetUserWorkspace`（偏移 16MB）在 3510 手工配置 workspace 正确。
- 新旧环境一致性：旧 `/root/Ascend` 与新 `/home/zl/Ascend` 的 tiling 参数（coreNum=32 / ubSize=204800 / sortLoopMaxElement=6400）与 8 路验证结论完全一致，复核通过。
- 详见 `probe/probe1_recv_fp16/RESULT.md`。

## 穿刺2 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，`--npu-arch=dav-3510` 编译；**新 CANN 环境 `/home/zl/Ascend` 复核**，clean rebuild + NPU 实跑）
- shape Y=4096（4 核×1024，触发多核 + 4 路归并），全 token 有效；8 路输出（sorted_expert_ids / gather_idx / y fp16 / session_ids / micro_batch_ids / token_ids / group_count / actual_token_num）全部 `np.array_equal` 二进制一致；连跑 3 次 8 路输出合并 SHA256 恒为 `fd43123e…e39389`（bitwise reproducible）。
- 结论：多核 `Sort<float,true>` + 4 路 `MrgSort<float,true>` 归并在 arch35 可行（与 A2 同签名同语义，proposal/tmp 缓冲用 `GetSortLen<float>` 派生正确）；稳定排序确定性契约成立（重复键 gather_idx == numpy stable argsort，逐 bit 相同）；核间 gather / group_listing(`SetAtomicAdd`) 分工可行。
- 机制：3 次核函数启动（sort blockNum=4 → merge blockNum=1 → divide blockNum=8），`aclrtSynchronizeStream` 做核间栅栏；host 侧组装 schedule_context 内嵌 GM 地址（二级指针注入）后 `<<<>>>` 直调——**TTK kernel 模式无地址注入能力、必须 Kernel 直调**（新环境无需改动即复现）。
- 新旧环境一致性：旧 `/root/Ascend` 与新 `/home/zl/Ascend`（cann-9.1.0，编译器 bisheng）的 8 路验证结论 + 稳定排序契约完全一致，复核通过（新环境首跑即过，0 重试）。
- 排障（旧环境定位，新环境已内含修复）：①核函数需 `__global__ __vector__` 标记（直调工程）；②MrgSortOut 输入列表**不可双重偏移**（共用 workspace 基址 + `offsets_` 区分，对齐 `moe_v3_mrgsort_out.h`）— 迭代二整合 `ffn_wb_sort_mrgsort_out.h` 时重点注意。
- 详见 `probe/probe2_norm_multicore/RESULT.md`。

## 穿刺3 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35）
- 8 路输出（y int8 / group_list / session_ids / micro_batch_ids / token_ids / expert_offsets / **dynamic_scale fp32** / actual_token_num）全部二进制一致，确定性（bitwise reproducible）通过。
- 结论：P2 dynamic_quant int8 + BLOCK_BYTES=32 尾部 FP32 scale 连续 gather 在 arch35 可行，无需 RegBase；gather 分支可纳入迭代二/三整合。
- 详见 `probe/probe3_norm_int8/RESULT.md`。

## 穿刺4 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，UB 运行时查 248KB / 64 核，未写死）
- 8 路输出（**y bf16** / group_list / session_ids / micro_batch_ids / token_ids / expert_offsets / dynamic_scale(bf16 恒 0) / actual_token_num=18）全部二进制一致，连跑 3 次 SHA256 一致（bitwise reproducible）。
- 结论：P1 bf16 字节 gather（`tokenDtypeSize_=sizeof(half)=2B`，与 fp16 同构）在 arch35 二进制一致可行；A2 经典 SIMD 全栈（含 `GatherMask`/`CompareScalar`/`Sort`/`DataCopyPad`/`SetAtomicAdd`）直编 dav-3510 并 NPU 实跑正确，主路线无需 RegBase。
- 详见 `probe/probe4_norm_bf16/RESULT.md`。

---

# 迭代二 穿刺汇总（PROBE_SUMMARY 迭代二区）

> 判定环境同迭代一：真实 Ascend950DT_9584（arch35/DAV_3510），CANN 9.1.0（`/home/zl/Ascend`），**NPU 实跑**，8 路输出与 CPU golden `np.array_equal` 二进制一致 + 多次运行 bitwise 可复现为通过。目标：验证迭代三全覆盖难点，为迭代三降风险。

| 任务 | 验证目标 | TilingKey×路径 | Profile | 状态 | 重试次数 |
|------|---------|--------|--------|------|---------|
| 迭代二穿刺1 | P2 int8 + 32B scale 在 RECV(101) gather | 101 RECV | int8 (P2) | - | - |
| 迭代二穿刺2 | 边界 shape：A=1024 + topK+1=64 上界 + UB 收敛不溢出 (issue #01) | 100 NORM | fp16 (P0) | ✅ 成功 | 0 |
| 迭代二穿刺3 | 边界：全 mask(k=0) + same_bin(expert_num=8192) | 100 NORM | fp16 (P0) | ⚠️ 部分成功 | 0 |
| 迭代二穿刺4 (101 RECV, fp16, 稀疏 ~50% mask) | 稀疏 mask 稳定性 + actual_token_num<Y | 101 RECV | fp16 (P0) | ✅ 成功 | 0 |

## 迭代二穿刺2 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，`--npu-arch=dav-3510`；`/home/zl/Ascend` cann-9.1.0；UB 运行时查两口径，未写死）
- shape `Y=27648 H=128 expert_num=256`（= SORT_CORE_NUM(4)×sortLoopMaxElement(6912)），NORM/100/fp16，全 token 有效(k=N)；每 sort 核 tile=6912 **打到物理 UB 216KB 上界**（比标称边界 shape 更严的 UB 极限）。8 路输出全部 `np.array_equal` 二进制一致，连跑 3 次合并 SHA256 恒为 `a121961e…3bd95b`（bitwise reproducible），`gather_idx`==numpy stable argsort 逐 bit 相同。
- **issue_20260708_ub-size-host-kernel-mismatch_01 收敛结论（本穿刺核心）**：同进程实测两口径 UB —— kernel 侧 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`=**221184 B(216KB)** vs host 侧 `GetCoreMemSize(UB)`=**253952 B(248KB)**，差 32KB，与 issue 记录一致坐实。主线公式 `sortLoopMaxElement=ubSize/32`，多核 sort 每核 4-buffer footprint=`4*SLM*8=ubSize`：**kernel 侧派生 SLM=6912→footprint 216KB 恰好装下(FITS，NPU 实跑不溢出)**；**host 侧派生 SLM=7936→footprint 248KB 超物理 216KB 达 32KB(OVERFLOW)**。故主线 tiling 应改按 kernel 侧真实可用 UB（或 `min(host,kernel)`）派生 sortLoopMaxElement，仍禁写死 arch 常量。
- 是否 UB 溢出：**保守口径(kernel 216KB)不溢出，NPU 实跑成功无 507035/踩内存**；对照口径(host 248KB)经 footprint 记账+216KB 恰好装满推得必溢出（未跑越界配置避免挂死）。标称边界 shape `[1024,4,64,128]`(Y=262144) 主线切 64 核后每核 4096<SLM，sort footprint 128KB 亚上界不溢出——溢出仅当单核每 loop tile 打到 SLM 时触发，本穿刺 4 核×tile=SLM 正是暴露该隐患的正确构造。
- 详见 `probe/probe_iter2_boundary_maxshape/RESULT.md`。

## 迭代二穿刺3 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，`--npu-arch=dav-3510`；`/home/zl/Ascend` cann-9.1.0；UB 运行时查 248KB / 64 核，未写死）
- shape `A=4 M=2 BS=2 K=3 H=16 Y=24`，单核 sort，NORM/100/fp16；两场景共用 shape，仅 expert_ids/expert_num 不同。复用 `probe5_norm_fp16_skeleton` arch35 骨架头 + `probe_ffn_wb_norm` kernel。
- **same_bin（expert_num=8192，全部 token 路由同一专家 e=5）**: ✅ 8/8 `np.array_equal` 二进制一致（group_list[0]=[5,24]、[1]=[0,0]、其余 0，符合 single-bin 契约），3x 合并 SHA256 恒 `4d87a9bd…90209113`（bitwise reproducible）。边界可行，无需改动。
- **k=0（全部 token 被 mask）**: 穿刺**暴露主线 `KernelFfnWBGroupListing` 单核路径缺陷**——`validGatherIdxLength==0` 缺守卫，`ComputeOneCore(0)` 读 0 字节搬运后未初始化 UB 并 flush `group_list[0]=[脏 expert_id(实测=expertNum=8), 0]`（应为 [0,0]；其余 7 路含 actual_token_num=0 / y / ids / scale 全 0 均正确）。缺陷确定性复现（脏值稳定）。单行守卫 `validGatherIdxLength > 0` 修复后 NPU 实测 kzero 8/8 二进制一致、3x SHA256 恒 `e253c6a8…bad9b11`，且不影响 samebin。
- 结论：same_bin(expert_num 上界) 边界 arch35 完全可行；k=0 边界暴露真实缺陷并给出可验证修复，迭代三须在主线 arch35 `ffn_wb_group_listing.h` 落地守卫。缺陷 + 根因 + 迭代三行动项归档 `issues/issue_20260708_group-listing-kzero-dirty-expertid_01.md`（含：核对 arch22/A2 基线口径、多核 group_listing k=0 另穿刺、UT 加 k=0 断言）。
- 详见 `probe/probe_iter2_boundary_kzero_samebin/RESULT.md`。

## 迭代二穿刺4 摘要
- 运行环境: **NPU**（Ascend950DT_9584 / arch35，`--npu-arch=dav-3510` 编译；`/home/zl/Ascend` cann-9.1.0；UB 运行时查 216KB / 64 核，未写死）
- shape `A=8 BS=2 K=2 Y=32`，**精确 50% mask（16 valid / 32，actual_token_num=16<Y）**，`totalLengthWithPad=64`（单核 scan-sort GatherMask 单 repeat 上限）；有效专家直方图 `{e0×3,e2×4,e5×6,e7×3}`，重键 e5×6 分散在非单调 session `[0,1,4,4,7,7]`。8 路输出全部 `np.array_equal` 二进制一致；连跑 3 次 8 路合并输出 SHA256 恒为 `be9b4dc5…85b1bb79`（bitwise reproducible）。
- 结论：**稀疏 ~50% mask 在 RECV 单核 scan-sort 路径 arch35 二进制一致、排序稳定、可复现**。`CompareScalar(GT -expertStart)` + `GatherMask` 丢弃无效/pad token（无效值以极大值排末尾后剔除），`Sort<float,true>` 以 expert_id 主键 + row_idx 次键稳定排序；NPU 实跑结果与 numpy `argsort(kind="stable")` 逐 bit 相同，e5 六并列严格按 row_idx 升序聚合。为迭代三 `normal_sparse`（spec.extreme_inputs）全覆盖降风险，无 arch35 风险点。
- 机制：复用迭代一穿刺1 RECV 直调模板（13 个 `ffn_wb_*.h` + kernel 入口**原样复用、逻辑未改**），仅改 shape(A=4→8)/稀疏配置/golden/确定性校验；`SetSysWorkspaceForce`+`GetUserWorkspace` 手工 workspace、host 侧 1024B schedule_context 二级指针注入 + `<<<>>>` 直调机制不变——印证「TTK kernel 模式无地址注入能力、必须 Kernel 直调」。
- 详见 `probe/probe_iter2_sparse_mask_recv/RESULT.md`。
