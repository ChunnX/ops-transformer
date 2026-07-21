# qCoreOffset 未初始化导致 Output2 几乎全零

## 现象

恢复完整分工（AIC Bmm1+Bmm2 / AIV Vec1+Vec2）后，kernel 跑通但精度错：

```
Output 2 (attention_out): Cosine: -0.001
  NPU range: [-1.98, 2.03], zero ratio: 99.94%   ⚠️ 几乎全 0
```

Output 2 非零元素 ≈ 8192 = 一个 slice [128,64] 的大小 → 14 个 AIV slice 都写到了**同一个/错误位置**，互相覆盖。

## 排查中被否定的猜测

**taskIdMod2 不同步**：case 61 中 splitFactor=1、s2LoopLimit=0，taskId 恒为 0，taskIdMod2 本来就是 0，不是问题。相关修改已撤销。

## 根因

`qCoreOffset`（和 `k1CoreOffset`）是在 `Bmm1SetTensorA`（line 991-1014）里计算的：

- **AIC 调用 Bmm1SetTensorA** → 有正确 qCoreOffset。
- **AIV 从不调用 Bmm1SetTensorA**，只调 `SetExtraInfo`，而 SetExtraInfo 不设 qCoreOffset。
- 所以 AIV 的 `extraInfoA5_.qCoreOffset` 是**残留/未初始化值**。
- ProcessVec2 用它作为 attentionOutGm 的写入偏移 → 所有 slice 写错位置。

## 修复

在 `SetExtraInfo`（AIC/AIV 都调用）的 A5 分支下，补充计算 `qCoreOffset` 和 `k1CoreOffset`。

> softmaxMaxOffset 在 ProcessVec1 内设置、AIV 会调用，无需额外处理。

## 验证结果（部分成功）

```
Output 2: Cosine=0.0005, NPU_zero=0.0000   ← zero 从 99.94% 降到 0%
```

- ✅ 位置修对了：448 个 block 全部写入数据，不再全零。
- ❌ 但数值仍错（cosine≈0）→ 从**位置错误**转为**数据内容错误**，见 [issue-09](issue-09-bmm2-v-layout-error.md) / [issue-10](issue-10-bmm1-systematic-error.md)。
