# 迭代二 穿刺2 (probe_iter2_boundary_maxshape) 验证结果

## 状态: ✅ 成功（0 重试）

真实 Ascend950DT_9584（arch35 / DAV_3510，CANN 9.1.0，`/home/zl/Ascend`）NPU 上实跑，8 路输出与 CPU golden **二进制一致**（`np.array_equal`，rtol=atol=0），连跑 3 次 bitwise 可复现。**非 Mock/CPU 仿真。**

本穿刺是 **issue_20260708_ub-size-host-kernel-mismatch_01** 的 NPU 实跑验证载体：验证「边界上界 shape 触发多核 + UB 派生阈值收敛不溢出」。**最关键结论见下方「issue #01 UB 收敛结论」。**

## 穿刺信息

| 项 | 值 |
|----|----|
| Task Name | probe_iter2_boundary_maxshape |
| TilingKey | 100 (NORM) |
| Dtype | fp16 (P0) |
| 验证目标 | 边界上界 shape (A=1024 / topK+1=64 语义, k=N 全有效) + UB 收敛不溢出 |
| 排序模式 | 多核 sort + MrgSort 4 路归并 (每核 tile = sortLoopMaxElement) |
| 运行环境 | **NPU（Ascend950DT_9584 / arch35 / DAV_3510）** |
| 编译 arch | `--npu-arch=dav-3510` |
| CANN 环境 | `/home/zl/Ascend`（cann-9.1.0，编译器 bisheng） |
| 验证方式 | **Kernel 直调**（`.asc` + `<<<>>>`；非 TTK kernel 模式） |
| 重试次数 | 0 |

## issue #01 UB 收敛结论（本穿刺最关键结论）

### 实测运行时 UB 取值（禁写死，全部运行时查询）

| 口径 | 接口 | 实测值 (本机 Ascend950DT_9584) |
|------|------|--------|
| **kernel 侧真实可用 UB** | `aclrtGetDeviceInfo(ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE)` | **221184 B = 216KB** |
| **host 侧平台 UB** | `PlatformAscendCManager::GetCoreMemSize(UB)` | **253952 B = 248KB** |
| vectorCoreNum | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 64 |

两口径实测差 **32768 B（32KB，约 13%）**，与 issue #01 记录（穿刺3 kernel 侧 216KB vs 穿刺4/主线 host 侧 248KB）**完全一致**，本穿刺在同一进程内同时查两口径直接坐实差异存在。

### tiling 派生的 UB 阈值 + 溢出判定

主线 tiling 公式（`ffn_worker_batching_tiling_arch35.cpp:220`）：
```
sortLoopMaxElement = ubSize / (sizeof(int32)*2*4) / 32 * 32 = ubSize / 32
```
多核 sort（`ffn_wb_sort_multi_core.h` `KernelSortMaskMultiCore::Init`）每核每 loop 分配 **4 个 buffer**（in/out/temp/sorted），每个 = `sortNum*sizeof(int32)*2` 字节。当每核每 loop tile == `sortLoopMaxElement` 时，4-buffer footprint：
```
4 * sortLoopMaxElement * (sizeof(int32)*2) = 4 * (ubSize/32) * 8 = ubSize  ← 恰好占满派生 UB
```

| UB 取值口径 | 派生 sortLoopMaxElement | sort 4-buffer footprint | vs 物理 UB 221184 B | 判定 |
|------------|------------------------|-------------------------|---------------------|------|
| **kernel 侧 216KB（本穿刺采用，保守口径）** | **6912** | **221184 B (216KB)** | == 221184 B | ✅ **FITS（恰好装下，不溢出）** |
| host 侧 248KB（主线现状口径，对照） | 7936 | 253952 B (248KB) | > 221184 B | ❌ **OVERFLOW（超物理 UB 32KB）** |

**实测 NPU 输出（`--query-ub` + kernel run，两次一致）**：
```
[UB] kernel-side ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE = 221184 (216.0 KB)
[UB] host-side   GetCoreMemSize(UB)                = 253952 (248.0 KB)
[UB] SLM(kernel-derived)=6912 -> sort 4-buffer footprint=221184 B (216.0 KB) vs 物理 UB 221184 B => FITS
[UB] (control) SLM(host-derived)=7936 -> footprint=253952 B (248.0 KB) vs 物理 UB 221184 B => OVERFLOW (host口径超物理 UB)
```

### 是否 UB 溢出

**本穿刺（kernel 侧 216KB 派生 SLM=6912）：不溢出。** 4 个 sort 核每核 tile=6912，4-buffer footprint 恰好 = 物理 UB 221184 B，NPU 实跑 8/8 二进制一致，`Sort<float,true>`/`MrgSort<float,true>` 全部成功，无 507035 向量核异常、无踩内存、无精度错乱。这是比标称边界 shape（见下方「边界 shape 说明」）**更严的 UB 极限测试**——直接把 sort buffer 打到物理 UB 上界。

**对照结论（host 侧 248KB 派生 SLM=7936）：会溢出。** footprint 253952 B 超物理 216KB 达 32KB。**故主线 tiling 若按 host 侧 248KB 派生 sortLoopMaxElement，在多核 sort 每核 tile 打到 SLM 的场景下必然 UB 超分**（未运行该越界配置，避免 NPU 挂死/踩内存风险；结论由「实测两口径 + footprint 记账 + 216KB 恰好装满」三重证据推得，充分且严格）。

### issue #01 收敛建议（供迭代二/三整合落地）

1. **主线 tiling 的 `sortLoopMaxElement` 应改按 kernel 侧真实可用 UB（`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`）派生**，或取 `min(host, kernel)` 保守值；当前 `ffn_worker_batching_tiling_arch35.cpp:110` 走 `GetCoreMemSize(UB)`（248KB 口径），存在超分隐患。
2. 仍**禁写死 arch UB 字节常量**（对齐 MEMORY「UB size no hardcode」）——只统一「用哪个运行时接口取值」，本穿刺 host/kernel/golden 的 Y/SLM 全由 `--query-ub` 运行时派生写 `ub_params.txt`，无任何 216/248 字面常量。
3. **触发条件说明**：标称边界 shape `[A=1024,BS=4,topK+1=64,H=128]` → Y=262144，主线多核 sort `TilingMultiCoreSort` 切到 64 核后每核 4096 元素（< SLM），sort footprint 仅 128KB，**亚上界不会暴露溢出**；溢出仅当单核每 loop tile 打到 SLM 时触发（Y 更大 / 核数受限）。本穿刺用 4 核 × tile=SLM 直接构造该极限，是暴露该隐患的正确姿势。

## 验证摘要

穿刺 shape（由 NPU 运行时真实 UB 派生）：
- `Y = SORT_CORE_NUM(4) × sortLoopMaxElement(6912) = 27648`（边界上界 A=1024/topK+1=64 语义；k=N 全 token 有效）
- `H=128`（spec 边界 max_out_shape[3]），`expert_num=256`
- 每 sort 核 tile = **6912 = sortLoopMaxElement（打到物理 UB 216KB 上界）**

数据流（3 次核函数启动，`aclrtSynchronizeStream` 做核间栅栏）：
1. **ffn_wb_probe_sort**（blockNum=4）：每核 `Sort<float,true>` 6912 元素 tile；**分配 4 buffer（in/out/temp/sorted）复现真实算子 footprint = 216KB**，proposal 落 workspace。
2. **ffn_wb_probe_merge**（blockNum=1）：4 路 `MrgSort<float,true>` 分 loop 归并 27648 元素 → `Extract` → `sorted_expert_ids` + `gather_idx`。
3. **ffn_wb_probe_divide**（blockNum=32：16 gather + 16 group）：gather 核逐行 DMA gather（y + **打包 id 行** 32B 对齐单次 DMA，避免大 Y 全量 source 驻 UB 溢出）；group 核本地直方图 + `SetAtomicAdd<int32_t>` 聚合逐专家计数。

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅通过 | `/home/zl/Ascend` clean build，`--npu-arch=dav-3510`，无错误 |
| NPU 运行 | ✅通过 | 3 stage 全部在 Ascend950DT_9584 实跑完成 |
| **UB 不溢出** | ✅通过 | sort 4-buffer footprint 216KB == 物理 UB，NPU 实跑成功，无 507035/踩内存 |
| 二进制一致 | ✅通过 | 8/8 输出 `np.array_equal` bitwise 相同 |
| 确定性 | ✅通过 | 连跑 3 次合并 SHA256 恒为 `a121961e…3bd95b`，bitwise 完全一致 |
| 稳定排序 | ✅通过 | `gather_idx` == numpy `argsort(kind="stable")` 逐 bit 相同 |

8 路输出逐项二进制一致（Y=27648）：`sorted_expert_ids`(int32) / `gather_idx`(int32) / `y`(fp16, 3538944 elems) / `session_ids`(int32) / `micro_batch_ids`(int32) / `token_ids`(int32) / `group_count`(int32, 256 experts) / `actual_token_num`(int32=27648) 全部 PASS。
其他不变量：`sorted_expert_ids` 非降序 ✅；`group_count` sum=27648（=Y）✅；min=78 max=132（256 专家全非零，无空 bin）。

## 边界 shape 说明（诚实标注）

- **A=1024 / topK+1=64 语义 vs 本穿刺 Y=27648**：spec 边界上界硬校验为 `A∈(0,1024]`、`topK+1∈(0,64]`，核心关系式 `Y=A*BS*(topK+1)`。**排序阶段只依赖 Y**（A/BS/K 只在 host tiling 与 gather/id 语义体现，与 sort UB footprint 正交）。本穿刺主目标是「UB 收敛不溢出」，故取 `Y=SORT_CORE_NUM*SLM=27648` 让每 sort 核 tile 精确打到物理 UB 上界（216KB）——这是比 A=1024 标称 shape 更严的 UB 极限。`Y=27648` 在 `A≤1024`（如 A=108,BS=4,K=64）语义域内合法，是 A=1024 上界的等价 UB 压力代表。
- **标称边界 shape `[1024,4,64,128]`（Y=262144）的 UB 行为已在 issue #01 收敛结论 §3 分析**：主线切 64 核后每核 4096 < SLM，sort footprint 128KB 亚上界、不溢出；边界 shape 的**报错类校验**（A>1024 / topK+1>64 / expert_num>8192 等）属 host tiling `CheckInputParam`/`GetAttrsInfo` 硬校验，由 UT 覆盖（PLAN 迭代三），非本 NPU 穿刺范围。
- **全 token 有效（k=N，无 mask）**：与 PLAN 迭代二穿刺2「k=N 全有效」判定一致；mask 无效 token 语义由其它穿刺/主线 `Select` 分支验证，与本穿刺 UB 收敛正交。
- **group_list 简化为 group_count**：输出逐专家 token 计数（histogram），验证 group_listing 计数 + `SetAtomicAdd` 多核聚合；算子实际 `group_list=[expert_id, token_num]` 打包为格式细节，迭代整合补齐（同 probe2 口径）。
- **id 打包（每行 8 int32/32B）**：为大 Y 逐行 DMA gather 单次 32B 对齐，是穿刺工程内部 source 布局，与算子 session/micro/token 三独立表语义等价（golden 同源对拍）。

## 复现方式

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh   # 统一 CANN 环境
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe_iter2_boundary_maxshape
bash run.sh    # 编译 -> --query-ub 派生 shape -> gen_data -> NPU 实跑 -> 8 路二进制一致校验
# 确定性: 连跑 3 次核函数，合并 8 路输出 SHA256 恒为 a121961e...3bd95b
```
