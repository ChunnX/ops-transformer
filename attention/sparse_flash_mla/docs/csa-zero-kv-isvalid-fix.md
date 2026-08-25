# CSA arch22：零长 KV / cmp CalcParams isValid 修复说明

## 问题

CSA + TND 多 batch 场景下，当某 batch 的 `actOriS2Size` 与 `actCmpS2Size` 均为 0（KV 双零，或 cmp 压缩后长度为 0），`ProcessBalance` preload 排空轮仍可能满足旧条件 `s2LoopIdx < s2LoopTimes`（含 stale loop），从而下发 Mm1/V0 并访问无效 `cmp_kv` GM。

## 修复

在 `CalcParams` 增加 KV 有效长度 OR 守卫（1 处）：

```cpp
info.isValid = (tempLoopInfo.actOriS2Size > 0 || tempLoopInfo.actCmpS2Size > 0) &&
               (s2LoopIdx < tempLoopInfo.s2LoopTimes);
```

当 ori/cmp 均无有效长度时不再发起新 S2 计算；preload 排空仍由滞后 snapshot 与 `extraLoop=2` 完成。正常 batch 行为不变。

## 与 MR 9448 区别

9448 针对 SWA **零长 Q**（`actS1Size > 0`）；本修复针对 CSA **零长 KV/cmp**，不修改 Q 侧判断。
