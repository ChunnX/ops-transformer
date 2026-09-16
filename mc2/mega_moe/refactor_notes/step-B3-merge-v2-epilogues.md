# B3 — merge the per-token V2 epilogue family, and unify the epilogue numbering

Commit: `refactor(mega_moe): merge V2 epilogues and unify epilogue numbering`

## Scope

Four V2 headers collapsed to one with two specialisations, plus the epilogue-numbering
unification that collapse forces.

| Before | After |
|---|---|
| `utils/block_epilogue_pertoken_v2.hpp` — `EpilogueAtlasA2PerTokenDequantV2` (A3) | kept, becomes the shared platform-aware implementation |
| `utilsA2/block_epilogue_pertoken_v2_int8_a2.hpp` — `...V2Int8` (A2) | **deleted** — served by the shared `...V2` via `PerTokenV2Tag<IS_A2, isInt8>` |
| `utilsA2/block_epilogue_pertoken_v2_bf16_a2.hpp` — `...V2BF16` (A2) | **deleted** — served by the shared `...V2` |
| `utils/block_epilogue_w4a8post_pertoken_v2.hpp` — `...W4A8PostPerTokenDequantV2` | folded into the same header, unchanged |

The shared header carries the machinery the pre-rebase refactor introduced:

- `detail::PerTokenV2Tag<IS_A2_, NEED_DEQUANT_>` — the 7th template argument that selects the
  platform and the dequant/non-dequant overload set. The W4A8 path keeps the pre-existing
  `IS_A2_T<IS_A2_>` tag (no per-token dequant).
- `detail::PerTokenV2ServerIdStorage<HAS_SERVER_ID>` + `detail::PerTokenV2Params<LayoutC, IS_A2,
  HAS_SERVER_ID>` — the `Params` struct shared by both specialisations. Standard V2 uses
  `HAS_SERVER_ID = true` (it needs `serverId` for the A2 cross-server write), W4A8 uses `false`.
- `detail::ApplyPerTokenScales(...)` — the per-row `Muls` loop both specialisations were
  duplicating, extracted once.
- `CopyOutput(...)` on the standard specialisation — the output write loop, which carries the
  A2 cross-server branch (`(dstEpIdx / SERVER_RANK_SIZE_A2) != params.serverId` writes to
  `windowsOutAddr() + offsetWinOutD`, else to the peer). The non-dequant `operator()` overload
  that replaces the deleted `...V2BF16` path also routes through it.

## The numbering unification (the reason this step also touches two class templates)

Master's `mega_moe_arch22.h` numbers the epilogues the opposite way round from `mega_moe_a2.h`:

| | `mega_moe_a2.h` (A2) | master `mega_moe_arch22.h` (A3) | after this step (both) |
|---|---|---|---|
| `EpilogueDispatchPolicy2` | V2 family | `EpilogueAtlasA2PerTokenDequant` (V1 row) | **V2 family** |
| `EpilogueDispatchPolicy3` | — | V2 family | `EpilogueAtlasA2PerTokenDequant` (V1 row) |
| `BlockEpilogue2` | V2 | V1 | **V2** |
| `BlockEpilogue3` | — | V2 | V1 |

Master's freshly-landed route-cache wiring in `mega_moe_kernel_a3.hpp` is written against its own
numbering:

```cpp
epilogueParams3.useUbRouteCache = true;
epilogueParams3.ubTokenPerExpert = ubTokenPerExpertCache;
epilogueParams3.ubPreSumBeforeRank = ubPreSumBeforeRankCache;
```

`epilogueParams3` is master's **V2** epilogue — the one whose `Params` has those three fields,
because the route cache is consumed by the V2 read loops. Under the unified numbering V2 is
`BlockEpilogue2`, so all three assignments moved to `epilogueParams2`, and the two
`typename BlockEpilogueN::Params` initialisers swapped (V2's ten-argument form now initialises
`epilogueParams2`, V1's four-argument form `epilogueParams3`). Consistent with that, the
`CombineV1` / `CombineV2` signatures traded epilogue types, and the dispatch became
`blockEpilogue2.InitFlag(); CombineV2(params, blockEpilogue2);` for the int4 path and
`blockEpilogue3.SetFlag(); CombineV1(params, blockEpilogue3);` for the CombineV1 path — matching
the new meaning of each name.

This is a silent-misrouting-shaped change (a wrongly-numbered epilogue would associate the route
cache with the wrong read loops), so it is worth noting that the two specialisations have
different method sets — V2 has `InitFlag`, V1 has `SetFlag`, and only V2 has the route-cache
`Params` — so the compiler rejects most mismatches outright. The A2 and A3 builds were both
re-run after the swap for exactly this reason.

## Upstream behaviour that had to survive

Master added the route-table UB cache to this file after the refactor was first drafted. Taking
the pre-rebase merged header verbatim would have dropped it. It is restored in the shared form
master uses:

```cpp
// 路由表标量读取：优先走 UB 缓存（紧凑索引），否则回落 GM（保持既有直接读路径）
CATLASS_DEVICE int32_t GetLenRankInExpert(int32_t dstEpIdx, int32_t groupIdx) const
{
    if (params.useUbRouteCache) {
        return params.ubTokenPerExpert.GetValue(dstEpIdx * params.expertPerRank + groupIdx);
    }
    return tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
}
```

plus `GetDstExpertOffset`, the three `Params` fields (`useUbRouteCache`, `ubTokenPerExpert`,
`ubPreSumBeforeRank`) added to `detail::PerTokenV2Params`, and both V2 read sites converted from
the direct GM reads to the getters. The getters are defined once per specialisation, matching
upstream's shape.

## Runtime cost

**None.** `PerTokenV2Tag` is a type-level selector; the dequant/non-dequant choice is overload
selection; `ApplyPerTokenScales` and `CopyOutput` are refactors of existing loops with identical
bodies. `useUbRouteCache` is read from `Params` exactly as upstream wrote it — this step
introduces no new runtime branch, argument or indirection.

## Verification

- `bash build.sh --pkg --soc=ascend910b --ops=mega_moe` — green, 76 targets, 0 errors.
- `bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe` — green, 76 targets, 0 errors.

## Follow-up flagged, not fixed here

`mega_moe_arch22.h` still defines `isW4A8 = is_same_v<BType_, int4b_t> || is_same_v<BType_,
int32_t>`, while `mega_moe_a2.h` was narrowed to `int4b_t` only during the master sync. Since the
arch22 op-def no longer advertises `DT_INT32` for weights, the `int32_t` arm is unreachable
plumbing rather than a behaviour difference — but the two platforms should agree. Left for the
class-template merge (B6), where the two predicates become one.
