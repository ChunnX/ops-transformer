# MegaMoe A8W8 第二轮 PIPE_ALL 诊断设计

日期：2026-07-31

## 背景与目标

第一轮已启用 Dispatch D1～D5 和 Combine C1～C4 共 9 个 `PipeBarrier<PIPE_ALL>();`，原 A8W8
用例仍然精度失败。

第二轮保留第一轮全部诊断点，只在尚未覆盖且经二次确认仍有价值的三个边界增加
`PipeBarrier<PIPE_ALL>();`，验证 Hcomm 源数据发布、Dispatch sender acquire 和 GMM1 epilogue
状态发布是否影响复现结果。

本轮仍是临时诊断，不作为最终同步修复。

## 范围与约束

- 仅新增 3 个 `PipeBarrier<PIPE_ALL>();` 及稳定的 `SYNC_DIAG[R2-*]` 标签。
- 保留现有 D1～D5、C1～C4 和 resetTensor 诊断 barrier。
- 不新增或移动 `hcomm_.Drain()`，不修改 Hcomm、flag、counter 或控制流逻辑。
- 不添加 A5 不需要的纯 V→V 同步。
- 不修改 AIC 分支。
- 不在轮询循环体中增加 barrier。
- 涉及文件：
  - `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
  - `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h`

## 第二轮插点

| 标签 | 插入位置 | 诊断边界 | 审计结论 |
| --- | --- | --- | --- |
| `SYNC_DIAG[R2-H1]` | `CrossRankSyncInWorldSize` 写 `syncCount` 后、进入 rank 循环并执行 `WriteNbi` 前 | Scalar 写本地 `syncCount` → Hcomm 将该地址作为 DMA 源 | 缺失同步，高优先级 |
| `SYNC_DIAG[R2-H2]` | `DispatchToken` 完成 READY/tokenIdx 读取和地址计算后、`WriteWithNotifyNbi` 前 | READY acquire → Hcomm 读取 relay payload | 诊断可疑，高优先级 |
| `SYNC_DIAG[R2-G1]` | `BlockEpilogueSwigluMxQuant::operator()` 的可选 `AtomicAdd` 分支结束后、函数返回和 caller `NotifyCube` 前 | GMM1 progress AtomicAdd → NotifyCube/下一 tile | 诊断可疑，中高优先级 |

插入形式：

```cpp
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-H1]: syncCount publication -> Hcomm source read
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-H2]: READY acquire -> relay Hcomm send
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-G1]: GMM1 progress publication -> NotifyCube
```

每行仅出现在对应位置，不连续堆叠。

## 诊断判定

1. 一次性启用 R2-H1、R2-H2、R2-G1，运行与第一轮相同的 A8W8 复现用例。
2. 如果精度恢复：
   - 先移除 R2-G1，保留 H1/H2；
   - 再在 H1/H2 之间二分；
   - 对命中的最小点重新选择真实同步语义，删除全部无关诊断 barrier。
3. 如果仍然精度失败：
   - 记录第二轮三个点均未改变复现；
   - 不再继续用本地 `PIPE_ALL` 排查 Hcomm completion，因为 `PIPE_ALL` 不等价于 `Drain`；
   - 返回数据地址、workspace 布局、计数与通信完成协议排查。

## 验证

- 使用 `rg -n "SYNC_DIAG\\[R2-"` 确认第二轮标签恰好 3 个且各出现一次。
- 使用 `rg -n "SYNC_DIAG"` 确认第一轮 9 个标签仍然存在。
- 使用 `git diff --check` 检查格式。
- 按用户要求不执行本地编译；最终结果以原 A8W8 复现用例为准。
