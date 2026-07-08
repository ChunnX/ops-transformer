# probe_iter2_int8_recv 验证结果

## 状态: ✅ 成功

## 任务信息
- Task Name: probe_iter2_int8_recv（迭代二 穿刺1）
- TilingKey×路径: **101 (RECV)**
- Dtype: **int8 (P2, dynamic_quant)**
- 排序模式: 单核 sort（scan_token_info 前置）
- Memory Strategy: 尾部 32B FP32 scale 连续 gather（HS = H + 32B）
- 穿刺验证目标: **P2 int8 + 32B FP32 scale 连续 gather 在 RECV(101) 路径**的搬运语义可行性，
  为迭代三 P2×RECV 全覆盖降风险。

## 运行环境
- **运行环境: NPU**（真实硬件，非 Mock/CPU 仿真）
- 芯片: **Ascend950DT_9584**（arch35 / DAV_3510）
- CANN 根: **/home/zl/Ascend**（`ascend-toolkit → cann-9.1.0`）；编译器 `bisheng`
- NPU 设备: `/dev/davinci0`；host x86_64
- 平台运行时查询值（禁写死 arch 常量）：`vectorCoreNum=64`、`ubPerCore=221184 字节 (216KB)`
  - 派生 tiling：`coreNum=32`（RECV 且 Y≤200000 限核 `min(64,32)`）、`ubSize=204800`
    （ubPerCore-16KB）、`sortLoopMaxElement=6400`、`sortNumWorkSpace=48`。
  - 全部经 `aclrtGetDeviceInfo` 运行时取值，无 arch UB/核数常量写死（对齐 MEMORY「UB size no hardcode」）。
- launch blockDim=32（末 1 核 group_listing，其余 gather）。

## 验证方法
- **Kernel 直调**（`.asc` + `<<<blockNum,nullptr,stream>>>`），`--npu-arch=dav-3510` 编译。
  RECV 输入 1024B `schedule_context` 内嵌 device 二级指针（token_info@384 / token_data@400 /
  polling_index@416），host 侧 `aclrtMalloc` 子 buffer 后把地址 `memcpy` 进 ctx 整体拷 device，
  kernel 内 `ScheduleContextInfoCompute<true>` 逐字段解析。**TTK kernel 模式无地址注入能力、必须 Kernel 直调**。
- **组合复用（0 kernel 逻辑改动）**：
  - RECV 链（`ffn_wb_scan_token_info.h` / `ffn_wb_scan_sort_one_core.h` /
    `ffn_wb_scan_group_listing_one_core.h` / `ffn_wb_recv_probe_kernel.asc`）← 复用 probe1_recv_fp16。
  - int8 + 32B scale 分支 ← `ffn_wb_gather_out_all.h` 本身即模板 `<isScanFlag>` + 运行时
    `tokenDtype==2` 双分支（与 probe1/probe3 **字节一致**，diff 已核对）。RECV(`isScanFlag=true`)
    与 int8(`tokenDtype==2`) 两路正交，本穿刺首次在 **同一次运行同时命中**并验证。
- CPU golden 按 spec.yaml RECV 三步语义（scan→稳定 sort(含 pad row_idx)→gather+group_listing）
  独立复算，精度口径 = 二进制一致（`np.array_equal`, rtol=atol=0）。

## 测试用例
| 项 | 值 |
|----|----|
| A (session_num) | 8 |
| M (micro_batch_num) | 2 |
| BS (micro_batch_size) | 2 |
| K (topK+1) | 3 |
| H (hidden, int8) | 128 |
| HS (attn_to_ffn_token_size) | 160 = H(128) int8 + 尾部 32B FP32 scale 块 |
| expert_num | 16 |
| token_dtype | 2 (P2 int8) |
| curMicroBatchID | 0 |
| BSK / pad / BSK_PAD | 6 / 2 / 8（RECV 每 session 按 int32=8 对齐补 pad） |
| Y = A·BS·K | 48 |
| 有效 token (valid_len) | 21（~56% 有效，其余 expert_id=2000000 mask 剔除） |
| 有效专家分布 | e0×4, e3×9, e7×8（含重复专家测稳定序） |

## 验证摘要
| 输出 | dtype | 元素数 | 结果 |
|------|-------|-------|------|
| y | int8 | 6144 (48×128) | ✅ bitwise equal |
| group_list | int64 | 32 (16×2) | ✅ bitwise equal（`[0,4][3,9][7,8][0,0]…`） |
| session_ids | int32 | 48 | ✅ bitwise equal |
| micro_batch_ids | int32 | 48 | ✅ bitwise equal（恒 curMB=0） |
| token_ids | int32 | 48 | ✅ bitwise equal |
| expert_offsets | int32 | 48 | ✅ bitwise equal |
| **dynamic_scale** | float32 | 48 | ✅ bitwise equal（**P2 关键路径**，有效 FP32 非恒 0） |
| actual_token_num | int64 | 1 | ✅ bitwise equal（=21） |

- **8 路输出全部与 CPU golden 二进制一致**（`np.array_equal`, rtol=atol=0）。
- **确定性**：相同 `schedule_context` 连跑 3 次，8 路输出合并 SHA256 恒为
  `7a68a360…dab6d`（spec `determinism.bitwise_reproducible` ✅）。

## 关键发现（arch35 int8 + 32B scale 在 RECV 路径的搬运语义）
1. **P2 int8 + 32B scale gather 与 RECV scan/sort 两路正交、可叠加**：`ffn_wb_gather_out_all.h`
   的 `isScanFlag=true`（RECV：`sessionIndices=aIndices`、`microbatchIndices=curMicroBatchID`、
   `bskProduct=BS*K+BsKPaddingCount`）与 `tokenDtype==2`（int8 字节 gather + 尾部 4B FP32 scale
   读取 `tokenDataBufGm_[base+H]` + scratch 32B 步长 `ArithProgression`/`Gather` → 连续
   `dynamic_scale`）在同一次 NPU 运行同时生效，`dynamic_scale` 二进制一致，**无需 RegBase 反量化管线**。
2. **RECV 的 pad row_idx 与 int8 gather 的 stride 计算兼容**：RECV `bskProduct=BS*K+pad=8`
   参与 `aIndices/bsIndices/kIndices` 拆解，而 int8 token_data 的 `globalXOffset` 用
   `HS(=160 字节)` 步长（`strideK=HS`），两套索引体系（sort 用 pad 后的 8 对齐、gather 用真实 HS 字节）
   互不干扰，K=3(BSK=6→pad=2) 非 2 幂对齐下 stride 公式 `(M-1)*F+2 == M*F-BSK` 仍成立。
3. **ScheduleContext 1024B 解析 + DCCI 语义在 int8 RECV 正确**：`ScheduleContextInfoCompute<true>`
   指针解析、`ValidGatherIdxLengthCompute` 跨核读 `validCnt` 的 `DataCacheCleanAndInvalid` 均正确
   （actual_token_num=21 命中）。
4. **UB 运行时取值链路正确**：`sortLoopMaxElement=6400`、gather `maxBlockSize_` 由运行时 ubSize 派生，
   int8 场景 `tokenDtypeSize_=1`、`blockBufferSize` 含 `BLOCK_BYTES=32` scale 预留，216KB 下自动适配。

## 结论
迭代二 穿刺1 目标达成：**P2 dynamic_quant int8（BLOCK_BYTES=32 scale 连续 gather + 有效 dynamic_scale）
在 RECV(TilingKey 101) 路径、arch35/DAV_3510 上功能与精度可行**，8 路输出对 CPU golden 二进制一致，
确定性满足。**RECV 与 int8 两分支正交可叠加，无需额外绕行**——为迭代三 P2×RECV 全覆盖显著降风险，
该组合可直接纳入迭代三整合。

## 复现
```bash
cd probe/probe_iter2_int8_recv
bash run.sh              # source /home/zl/Ascend/set_env.sh → 编译(dav-3510) → gen_data+golden → NPU 直调 → 8 路二进制对拍
bash run.sh --skip-build # 复用已编译产物
```

## 工程文件
```
probe/probe_iter2_int8_recv/
├── op_kernel/
│   ├── ffn_wb_probe_tiling.h              # 扁平 TilingData(8×int64，无 arch 常量)
│   ├── ffn_wb_recv_probe_kernel.asc       # RECV 入口(scan+sort+gather+group_listing) —— 复用 probe1
│   ├── ffn_wb_common.h ffn_wb_sort_base.h
│   ├── ffn_wb_get_schedule_context.h      # 1024B schedule_context 解析(isScanMode=true)
│   ├── ffn_wb_scan_token_info.h           # RECV 专属：扫描等待 micro batch ready
│   ├── ffn_wb_scan_sort_one_core.h        # 单核稳定 sort(含 pad)
│   ├── ffn_wb_gather_out_all.h            # gather 8 路(isScanFlag + int8/32B scale 双分支)
│   └── ffn_wb_scan_group_listing_one_core.h  # 末核 group_listing
├── op_host/
│   ├── ffn_wb_recv_probe.asc              # main + KernelCall(int8 token_data / HS=H+32 / tokenDtype=2)
│   └── data_utils.h
├── scripts/{gen_data,verify_result}.py    # int8 RECV 输入+CPU golden / 8 路 np.array_equal 对拍
├── CMakeLists.txt run.sh RESULT.md
```
