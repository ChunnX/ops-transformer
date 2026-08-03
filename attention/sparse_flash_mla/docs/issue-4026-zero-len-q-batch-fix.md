# Issue #4026：TND 尾部零长 Q batch 导致 arch22 SWA 前向 OOB

| 项 | 说明 |
| --- | --- |
| Issue | [cann/ops-transformer#4026](https://gitcode.com/cann/ops-transformer/issues/4026) |
| MR | [cann/ops-transformer!9448](https://gitcode.com/cann/ops-transformer/merge_requests/9448) |
| 修改文件 | `attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_swa_kernel.h` |
| 合入改动 | **仅** `CalcParams` 中 `info.isValid` 增加 `actS1Size > 0`（约 603 行） |

---

## 1. 问题是什么

**根因场景：`cu_seqlens_q` 尾部出现零长 batch**（相邻前缀和相等，例如 `cu_q[b]==cu_q[b+1]`），**不是** KV 长度算错。

典型复现：

- TND / TND，连续 SWA（`oriWinLeft=127`, `oriWinRight=0`）
- `cu_q = [0, 480, 1024, 1024]` → **batch 2 的 Q 长为 0**（前缀和 `tBase=1024`，无 token 1024）
- `cu_seqlens_ori_kv` 可与 `cu_q` 相同或不同；**`GetActualSeqLenKV` 仍能得到正确的 `actOriS2Size`**（TND 下 `cu_kv[b+1]-cu_kv[b]`）。Issue 中两者均为 0 时，KV 侧长度为 0 是 **正确读数**，OOB 来自 **Q 为 0 仍下发 Mm1** + **TND Q 前缀 1024 的 GM offset**。

现象：cube/vector UB→GM OOB；`cu_seqlens_ori_kv` 被写坏；反向 507057 等。

---

## 2. 因果链（精简）

1. 本核 **最后一个** `bN2` 且 **Q 或 KV 序列长为 0** 时，`isS1S2ZeroAndLastBatch` → **`gS1LoopEnd + 1`**，零长 Q batch **多进 1 次 gS1**（preload 收尾设计）。
2. 该轮仍走 840–868、895–904；旧 **`isValid = s2LoopIdx < s2LoopTimes`**，**未判断 `actS1Size > 0`**。
3. 若 **`s2LoopIdx < s2LoopTimes`**（含 **上一 batch 残留的 `s2LoopTimes`**，或见下文 mask 推导在 **非 127/0** 下的 `N>0`），则 **`isValid == true`** → **`ComputeMm1`**，**`tensorAOffset` 按 batch2 的 Q 前缀 1024** → 越界。

**修复：** 仅增加

```cpp
info.isValid = (tempLoopInfo.actS1Size > 0) && (s2LoopIdx < tempLoopInfo.s2LoopTimes);
```

零长 Q batch 上 **`extraInfo0` 永不发起新 Mm1**；preload 排空仍靠 **`extraInfo1/2` 滞后 snapshot** 与 **`isEnd` 时 `extraLoop=2`**。

---

## 3. `s2LoopTimes` 在零长 Q batch 上怎么算（连续 SWA 127/0）

840–868 在 **`actS1Size == 0` 且 `actOriS2Size == 0`**（与 #4026 相同 `cu`）时：

842–846：

- `s1EndIdx = Min(..., actS1Size - 1)` → **`actS1Size - 1` 无符号下溢 → `0xFFFFFFFF`**
- 作 **`(int32_t)s1EndIdx` → -1**

848–851（`oriWinRight=0`, `oriWinLeft=127`）：

- `oriMaskRight = actOriS2Size - actS1Size + (int32_t)s1EndIdx + oriWinRight = 0 - 0 + (-1) + 0 = **-1**`
- `oriMaskLeft = max(actOriS2Size - actS1Size + (int32_t)s1StartIdx - oriWinLeft, 0) = **0**`（`oriWinLeft≥0`）
- **窗口长度** = `oriMaskRight - oriMaskLeft + 1 = **0**`

863–868：

- **`N = s2LoopTimes = CeilDiv(0, s2BaseSize) = 0`**
- `isEnd` 时 s2 循环 **`0 + 2`** 步；**旧 `isValid`** 在 **`s2LoopIdx >= 0`** 且 **`N=0`** 时为 false，**但若未重算、残留上一 batch 的 `s2LoopTimes > 0`，仍会 `isValid==true`** → 这是 #4026 更直接的触发条件之一。

**若 `oriWinRight > 0`（非连续 SWA）**，零长 Q/KV 时可能得到：

- `oriMaskRight = -1 + oriWinRight`，`oriMaskLeft = 0`
- **窗口长度 = `oriWinRight`**
- **`N = s2LoopTimes = CeilDiv(oriWinRight, s2BaseSize)`**（可能 **> 0**）

此时 **仅 603** 仍保证 **`actS1Size==0` 时不 Mm1**；与 127/0 下 **`N=0`** 并列，**603 是统一兜底**。

---

## 4. 正式代码修改（当前 MR）

| 位置 | 改动 |
| --- | --- |
| `CalcParams` ~603 | `info.isValid = (tempLoopInfo.actS1Size > 0) && (s2LoopIdx < tempLoopInfo.s2LoopTimes);` |

**不改** cube/vector；**不**改 Host tiling。

---

## 5. 废弃方案（未合入）：`ProcessBalance` 826–839 与 `s1EndIdx` cast

曾考虑在 **`actS1Size <= 0`** 时 **提前 `continue`**，并在 **`isEnd`** 时 **手动 2 次 `PreloadPipeline` drain**（826–839），以及 **`s1EndIdx` 的 `static_cast<uint32_t>(actS1Size-1)`**（845–846）。

**废弃原因（与 #4026 + 603 结论一致）：**

- **OOB 主因**是旧 **`isValid`**，**603 已足够**。
- 连续 SWA **127/0** 下零长 batch **`N=0`**，**`isEnd` 时本就 0+2 步**；826 与正常路径 **重复**。
- 826 **不解决**「仅 KV=0、Q>0」等其它零长形态（本来就不看 `actOriS2Size`）；KV 长度由 **`GetActualSeqLenKV`** 正确得到 **0** 时，**467/861/`N=0`** 等原有逻辑处理。

**保留作参考的废弃代码（勿合入）：**

```cpp
            // TND packed: zero-length Q batch (e.g. cu[bIdx]==cu[bIdx+1]) must not touch q/kv GM or outputs.
            if (tempLoopInfo.actS1Size <= 0) {
                bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == gS1LoopEnd);
                if (isEnd) {
                    tempLoopInfo.oriLoopTimes = 0;
                    tempLoopInfo.s2LoopTimes = 0;
                    for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < PRELOAD_NUM; s2LoopIdx++) {
                        PreloadPipeline(gloop, cmpLoop, constInfo.s2Start, s2LoopIdx, extraInfo);
                        ++gloop;
                    }
                    constInfo.s2Start = 0;
                }
                continue;
            }
```

```cpp
            tempLoopInfo.s1EndIdx =
                Min((tempLoopInfo.s1StartIdx + constInfo.mBaseSize / constInfo.gSize - 1),
                    static_cast<uint32_t>(tempLoopInfo.actS1Size - 1));
```

---

## 6. 其它零长 batch（简要）

| 场景 | 603 | 说明 |
| --- | --- | --- |
| **中间 batch Q=0**（TND） | 若进 gS1/`PreloadPipeline` 则挡 Mm1 | 中间 batch 常 **0 次 gS1**，不靠 826 |
| **KV=0，Q>0** | Q>0 时 `isValid` 仍看 `s2LoopTimes` | 127/0 下 **`N=0`**；`InitAllZeroOutput` 等原逻辑 |
| **末尾 Q=0**（#4026） | **必须** | 多 1 次 gS1 + 可能 **残留 `s2LoopTimes`** |

---

## 7. 建议验证

1. `cu_q = [0, 480, 1024, 1024]`，`cu_ori_kv` 同或按用例；前向无 OOB；`cu_seqlens_ori_kv` 不被写坏。
2. 仅 `cu_q` 尾部零长、`cu_ori_kv` 正常：仍靠 **603**。
3. 回归：非零 TND/BSND SWA。

---

## 8. 补丁

`attention/sparse_flash_mla/patches/issue-4026-zero-len-q-batch-master.patch` — 应与 **仅 603 行** 一致（合入前请同步更新 patch 内容）。
