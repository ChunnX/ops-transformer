# AI Core Error：Cube 指令在 AIV 核执行（MIX 模式架构问题）

## 现象

```
ctypes aclnn function 'aclrtSynchronizeStream' failed! error code: 507015
EZ9999: ... there is an aicore error exception, core id is 0,
errorStr: timeout or trap error. subErrType: 0x4.
```

SoC 问题修复后跑 id=61，kernel 在 NPU 上崩溃（AI Core Error / timeout）。

## 二分法定位

1. Process 开头 early return → 不崩溃 → 崩溃在 Process 阶段
2. 只保留 Bmm1、注释 Vec1/Bmm2/Vec2 → 卡死（同步不配对）
3. 修同步后只保留 DataCopy Nd2Nz → AI Core Error → 崩溃在 DataCopy Nd2Nz 到 L1
4. `ExecuteSingleMatmul` 开头加 `if (g_coreType != AscendC::AIC) return;` → **不崩溃，success 1** → 根因确认

## 根因

**MatmulAdapterA5 的 Cube 指令在 Vector 核（AIV）上执行导致 trap。**

- A5 融合核架构下，`__CCE_AICORE__ == 310` 对 AIC 和 AIV **都为 true**。
- `s1s2_bn2gs1.h:161` 的 `#if !((__CCE_AICORE__ == 310) || ...)` 让 A5 的 AIV 核也走 MatmulAdapterA5。
- adapter 里的 DataCopy(Nd2Nz→L1)/LoadData(L1→L0)/Mmad/Fixpipe 都是 Cube 核专有指令。
- **AIV 核没有 L1/L0A/L0B/L0C 硬件**，执行这些指令即 trap。

## MIX 模式架构确认

kernel binary 含 `mix_aic` 和 `mix_aiv` 两版本：`coreType: MIX`，`taskRation: 1:2`（1 AIC : 2 AIV），`intercoreSync: 1`。

- AIV 核无 L1/L0 → matmul 指令 trap。
- **AIC 核跑完整流程也卡死**：AIV 直接 return 后，runtime 的 intercoreSync 等待 AIV 参与 CrossCore 同步，AIV 不响应 → 死锁。
- `SyncAll()` 解决不了：它可能只同步同类型核，不适用于 AIC↔AIV 跨核对齐。

## 正确方向（架构改造，非 guard/SyncAll 能解决）

A5 MIX 要求 AIC / AIV 协同：
- **AIC**：做 Bmm1/Bmm2（Cube matmul），`CrossCoreSetFlag` 通知 AIV。
- **AIV**：做 Vec1(softmax)/Vec2，`CrossCoreWaitFlag` 等 AIC 结果。
- 数据经 GM 传递（Bmm 写 GM，Vec 读 GM）。
- 同步：`CrossCoreSetFlag<FIA_SYNC_MODE, PIPE_FIX>` / `CrossCoreWaitFlag<FIA_SYNC_MODE, PIPE_MTE2>`。
- 参照 gqa_kernel.asc 的 `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分工，但不能全盘照搬。

## 已完成的配套修复（可保留）

| 修复项 | 文件 | 改动 |
|--------|------|------|
| PIPE_FIX→PIPE_ALL | matmul_adapter.h:119,123,217 | 3 处 `PipeBarrier<PIPE_FIX>` → `PipeBarrier<PIPE_ALL>` |
| 同步机制 | matmul_adapter.h:165-228 | SetFlag/WaitFlag 改 FetchEventID 动态分配 + M_FIX/FIX_M 同步 + FixpipeParamsV220 |
| L0C TPosition | matmul_adapter.h:248 | CO2 → CO1 |
