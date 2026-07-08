# probe_iter2_sparse_mask_recv —— 模板穿刺结果

**任务**：迭代二 穿刺4（`probe_iter2_sparse_mask_recv`）
**TilingKey**：101（RECV）　**Dtype**：fp16（token_dtype=0）　**排序模式**：单核 scan-sort（scan_token_info 前置）
**穿刺验证目标**（PLAN 迭代二穿刺4 / spec.extreme_inputs `normal_sparse`）：稀疏 ~50% mask 场景下——无效
token 初始化极大值稳定排到序列末尾、有效 token 按 expert_id 升序聚合且同专家保持 row_idx 升序（**排序稳定性**），
`actual_token_num < Y`，8 路输出与 CPU golden `np.array_equal` 二进制一致，且多次运行 bitwise 可复现（确定性）。

> 为迭代三 `normal_sparse` 全覆盖降风险。相较迭代一穿刺1（Y=16，7 valid），本穿刺放大排序稳定性压力：
> A=8 跨 8 session、**精确 50% mask（16 valid / 32）**、直方图含重键 e5×6 / e2×4 且重复分散在**非单调 row_idx**
> 的不同 session/slot 上，`totalLengthWithPad=64` 逼近单核 scan-sort 的 GatherMask 单 repeat 上限。

---

## 状态：✅ 成功

## 运行环境：NPU（**真实实跑，非 Mock**）

| 项 | 值 |
|----|----|
| CANN 根 | **/home/zl/Ascend**（`ascend-toolkit → cann-9.1.0`） |
| SoC | Ascend950DT_9584 |
| 架构 | arch35 / DAV_3510（`--npu-arch=dav-3510` 编译） |
| CANN | 9.1.0 |
| 编译器 | `bisheng` |
| host 架构 | x86_64 |
| NPU 设备 | `/dev/davinci0` |
| 运行时 vectorCoreNum | 64（`ACL_DEV_ATTR_VECTOR_CORE_NUM`） |
| 运行时 ubPerCore | 221184 B = 216 KB（`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`，**未写死**） |
| launch blockDim | 32（RECV 且 Y≤200000 限核 `min(coreNum,32)`；末 1 核 group_listing，其余 gather） |

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅ 通过 | `--npu-arch=dav-3510` 无 error（仅 12 条 `cce_global attribute ignored` host-cast 提示，与穿刺1 同源，非阻塞） |
| NPU 运行 | ✅ 通过 | 真实 Ascend950 实跑，`aclrtSynchronizeStream` 返回 0，`kernel finished on NPU`，8 路输出全部产出 |
| 精度（二进制一致） | ✅ 通过 | 8 路输出 `np.array_equal`（rtol=atol=0）全 PASS |
| 稀疏 / 排序稳定性 | ✅ 通过 | `actual_token_num=16 < Y=32`；无效 token 全部丢弃（排末尾），有效 token 按 expert 升序聚合，同专家 row_idx 升序 |
| 确定性 | ✅ 通过 | 同输入连跑 3 次，8 路合并输出 SHA256 恒为 `be9b4dc5…85b1bb79`（bitwise reproducible） |

**8 路输出对拍结果（全 PASS）**：`y`(fp16/256) · `group_list`(int64/16) · `session_ids`(int32/32) ·
`micro_batch_ids`(int32/32) · `token_ids`(int32/32) · `expert_offsets`(int32/32) · `dynamic_scale`(fp32/32，fp16 恒 0) ·
`actual_token_num`(int64/1=**16**)

**连跑 3 次 SHA256（设备输出，节选，3 次完全一致）**：
- `y.bin`：`9c620f8168196896…`
- `group_list.bin`：`dba6c314b7b351b2…`
- `actual_token_num.bin`：`5eb6da0e0e522104…`
- 8 路合并：`be9b4dc54d70298b…85b1bb79`（3 次一致）

## 穿刺形状与稀疏配置

`A=8, M=2, BS=2, K(topK+1)=2, H=8, expert_num=8, token_dtype=0(fp16), curMicroBatchID=0`
→ `Y = A*BS*K = 32`，`HS = H*2 = 16B/token`，`BSK=4 → 32B 对齐补 pad=4 → BSK_PAD=8`，
`totalLengthWithPad = A*BSK_PAD = 64`（单核 scan-sort 分支，GatherMask 单 repeat 上限）。

micro batch0 每 session 的 expert 分配：**精确 50% mask**（`expert_id=2000000 ≥ 1e6` 视无效丢弃，另每 session
BS*K 段补 pad=INT_MAX 亦无效）；32 slot 中 16 个有效 → `actual_token_num=16`。

有效专家直方图 `{e0×3, e2×4, e5×6, e7×3}`（重键分散在非单调 row_idx）：
- 排序后 `sorted_experts=[0,0,0, 2,2,2,2, 5,5,5,5,5,5, 7,7,7]`
- `session_ids=[0,2,5, 1,3,3,6, 0,1,4,4,7,7, 2,5,6]`（e5 六并列的 session 序列 `[0,1,4,4,7,7]` 严格按 row_idx
  升序聚合 → **稳定排序确定性成立**）
- `group_list=[0,3][2,4][5,6][7,3]` + `[0,0]` 终止符，其余 [0,0]

## 关键结论（对迭代三 normal_sparse 全覆盖的降风险）

1. **稀疏 ~50% mask 在 RECV 单核 scan-sort 路径 arch35 二进制一致、稳定、可复现**：`CompareScalar(GT -expertStart)`
   + `GatherMask` 丢弃无效/pad token（无效值以极大值形式天然排到序列末尾后被 GatherMask 剔除），`Sort<float,true>`
   以 expert_id 为主键、row_idx 为次键做稳定排序。本穿刺 16/32 有效、重键 e5×6 分散在非单调 session 上，NPU 实跑
   排序结果与 numpy `argsort(kind="stable")` 逐 bit 相同，`actual_token_num=16<32`，证实迭代三 `normal_sparse`
   分支在 arch35 无风险。
2. **单核 scan-sort GatherMask 单 repeat 承载 totalLengthWithPad=64 正确**：`gatherMaskParams.repeatTimes=1`
   （8 blocks = 64 fp32）在 `totalLengthWithPad=64` 边界处理 A=8 全 session 一次搬运 + mask + gather 正确无溢出，
   为迭代二 RECV 单核分支（`KernelScanSortMaskOneCore`）标定了单核处理上限的可行区间。
3. **确定性契约（spec.determinism.bitwise_reproducible）在稀疏场景成立**：相同 schedule_context 连跑 3 次，8 路合并
   输出 SHA256 完全一致；稳定排序 + 整数精确累加计数保证稀疏 mask 下亦 bitwise 可复现。
4. **UB/核数运行时取值、无 arch 常量写死**：`vectorCoreNum`/`ubPerCore` 经 `aclrtGetDeviceInfo` 运行时查询
   （64 核 / 216KB），`ubSize=ubPerCore-16KB`、`sortLoopMaxElement=6400` 由 ubSize 派生，对齐 MEMORY「UB size no hardcode」。

## 复用来源与改动

复用迭代一 `probe/probe1_recv_fp16/` RECV 直调模板（13 个 `ffn_wb_*.h` 头 + kernel 入口 `ffn_wb_recv_probe_kernel.asc`
原样复用，未改 kernel 逻辑）。仅改动：
- `scripts/gen_data.py`：shape A=4→8、稀疏配置改为精确 50% mask + 重键分散（含稳定性压力）、golden 重算。
- `op_host/ffn_wb_sparse_recv_probe.asc`：`A_=4→8`（其余 shape/tiling/workspace 公式不变，随 A 自动派生）。
- `scripts/verify_result.py`：A=4→8。
- `run.sh`：新增连跑 3 次 8 路 SHA256 确定性校验步骤。
- `CMakeLists.txt`：host 源文件名指向 `ffn_wb_sparse_recv_probe.asc`（target/kernel 入口名不变）。

## 复现

```bash
cd probe/probe_iter2_sparse_mask_recv
bash run.sh              # source /home/zl/Ascend/set_env.sh -> 编译(dav-3510) -> gen_data+golden -> NPU 直调 -> 8 路二进制对拍 -> 3 次确定性 SHA256
bash run.sh --skip-build # 复用已编译产物
```

## 工程文件

```
probe/probe_iter2_sparse_mask_recv/
├── op_kernel/                              # 与穿刺1 同源, kernel 逻辑未改
│   ├── ffn_wb_probe_tiling.h
│   ├── ffn_wb_recv_probe_kernel.asc        # RECV scan+sort+gather+group_listing 入口
│   ├── ffn_wb_common.h ffn_wb_sort_base.h
│   ├── ffn_wb_get_schedule_context.h
│   ├── ffn_wb_scan_token_info.h
│   ├── ffn_wb_scan_sort_one_core.h         # 单核稳定 sort（本穿刺核心验证对象）
│   ├── ffn_wb_gather_out_all.h
│   └── ffn_wb_scan_group_listing_one_core.h
├── op_host/
│   ├── ffn_wb_sparse_recv_probe.asc        # main + KernelCall（A=8, 运行时查 UB/核数）
│   └── data_utils.h
├── scripts/{gen_data,verify_result}.py     # 稀疏 50% mask 输入+CPU golden / 8 路对拍
├── CMakeLists.txt run.sh RESULT.md
```
