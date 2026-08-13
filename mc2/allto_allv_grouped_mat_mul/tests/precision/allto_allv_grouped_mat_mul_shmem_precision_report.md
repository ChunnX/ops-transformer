# AllToAllV GroupedMatMul SHMEM 精度验证报告

**状态**：NOT_RUN（当前无可用 Ascend950 SHMEM 多卡编译/执行环境）
**算子名称**：`AlltoAllvGroupedMatMulShmem`
**计算公式**：SHMEM AllToAllV 路由后，按本地专家执行 BF16 GroupedMatMul
**参考基线**：PyTorch FP32 路由拼接 + `torch.matmul`，结果转 BF16
**支持 dtype**：BF16
**计划精度标准**：MERE < 2^-7 且 MARE < 10 × 2^-7

## 总览

| 指标 | 值 |
|---|---:|
| 计划合法用例数 | 36 |
| 已执行 | 0 |
| 通过 | 0 |
| 失败 | 0 |
| 未执行 | 36 |
| 通过率 | N/A（不得用未执行用例计算） |

## 覆盖

| 类别 | 计划数 | 已执行 | 状态 |
|---|---:|---:|---|
| 典型/源参考 shape | 9 | 0 | NOT_RUN |
| 泛化/边界合法 shape | 27 | 0 | NOT_RUN |
| EP=1..8 | 已覆盖元数据 | 0 | NOT_RUN |
| M0=128/256 两个 tiling 家族 | 已覆盖元数据 | 0 | NOT_RUN |
| 不均匀、稀疏、零 expert、零 peer 路由 | 已覆盖元数据 | 0 | NOT_RUN |

## 离线完成项

- 36 条用例由确定性脚本生成，所有源 rank 的 `sum(sendCounts) == M`。
- `recvCounts[srcRank, localExpert]` 由全 rank `sendCounts[dstRank, dstLocalExpert]` 转置生成并逐项校验。
- 元数据覆盖 BF16、EP 1..8、E 1..64、残块和 1004 MiB SHMEM heap 近边界。

## 待真实环境执行

- Ascend950 CCE 编译与算子包安装。
- EP=1/2/3/4/5/7/8 多进程 SHMEM/UDMA 初始化和拉起。
- 逐 rank 输出与 FP32 golden 的 MERE、MARE、MaxAbsErr、CosSim 比较。
- 源参考用例 `M=4084, K=2048, N=4096, EP=2, E=4` 的端到端验证。

未产生任何精度数值，也未将未执行用例标为通过。
