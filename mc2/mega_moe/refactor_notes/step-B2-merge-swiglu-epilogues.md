# B2 — merge the SwiGLU epilogue family

Commit: `refactor(mega_moe): merge SwiGLU epilogues`

## Scope

Four headers, four `BlockEpilogue` specialisations, collapsed to one header with two.

| Before | After |
|---|---|
| `utils/block_epilogue_pertoken_swiglu.hpp` — `EpilogueAtlasA2PerTokenDequantSwigluQuant` (A3) | kept, becomes the shared implementation |
| `utilsA2/block_epilogue_pertoken_swiglu_a2_int8.hpp` — `...QuantInt8` (A2) | **deleted** — served by the shared `...Quant` quantised overload |
| `utilsA2/block_epilogue_pertoken_swiglu_a2_bf16.hpp` — `...QuantBF16` (A2) | **deleted** — served by the shared `...Quant` non-quantised overload |
| `utils/block_epilogue_w4a8post_pertoken_swiglu.hpp` — `...W4A8PostPerTokenDequantSwigluQuant` | folded into the same header, unchanged |

Consumers updated: `mega_moe_a2.h` (dispatch-policy selection + includes), `mega_moe_kernel_a2_int8.hpp`,
`mega_moe_kernel_a2_bf16.hpp`, `mega_moe_kernel_a2_bf16_w4a8.hpp`, `mega_moe_kernel_a3.hpp` (includes),
and the four A2 SwiGLU call sites (two per kernel — both the `dequantSum1` and the `dequantSum2` path).

## Why the A2 variants are genuinely redundant

This is the load-bearing claim of this step, so it is worth stating with evidence rather than
assertion. Diffing the A3 generic against `...QuantInt8` outside comments and includes leaves 148
lines, but every one of them falls into one of four categories:

1. **Task partitioning — equivalent.** The A2 variant inlines
   `subblockIdx / moveDataCoreNum / perCoreData / remainderData` to pick this core's row range.
   The A3 generic calls `GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)`, whose
   body is *the same computation, line for line* (`block_epilogue_pertoken_swiglu.hpp`, `GetTaskRange`).
   A3 was already using A2's subblock partition; it was just factored out.
2. **Overload set — additive.** The generic carries a second `operator()` for the non-quantised
   path, which the A2-int8 variant does not need and the A2-bf16 variant replaces with its own.
3. **`ComputeLoadedTile` — generalised.** The generic added a `bool applyPerTokenScale` argument.
   The quantised path passes `true`, which is exactly what the A2 variants always did; the
   non-quantised path passes `false`.
4. Include guards, include paths, blank lines, brace/style churn.

So the shared `...Quant` specialisation covers A2-int8 (quantised overload), A2-bf16 (non-quantised
overload) and A3 quantised, and the A2-only policies `...QuantInt8` / `...QuantBF16` are left declared
and unused in `utils/dispatch_policy_custom.hpp`.

## Signature change at the A2 call sites

The shared overloads take `Arch::Resource<ArchTag> const &` where the A2-only variants took
`uint32_t epilogueCoreNum` directly, so each A2 SwiGLU call gains a `resource` argument in front of
its core-count argument. `resource` is already in scope at all four sites (`BlockEpilogue1
blockEpilogue1(resource, n);` is constructed a few lines above each). Nothing else about the calls
changes: in particular the second call site in each kernel passes `coreNum` where the first passes
`params.epilogueCoreNum`, and that asymmetry is pre-existing and preserved verbatim.

## Upstream behaviour that had to survive

Master added a per-token-scale batch prefetch to all four SwiGLU headers after this refactor was
first written, so taking the pre-rebase merged file verbatim would have silently dropped it. It is
ported into **both** surviving specialisations:

```cpp
// 输入侧 per-token scale 批量预读：批量加载到 UB，避免逐个标量读 GM（读旧值 + 性能差）
static constexpr uint32_t SCALE_INPUT_COUNT = 256;
static_assert(SCALE_INPUT_COUNT % (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) == 0,
              "The scale input batch must be block aligned");
static constexpr uint32_t SCALE_INPUT_BYTES = SCALE_INPUT_COUNT * sizeof(float);
```

plus `SCALE_INPUT_BYTES` added to `BUFFER_SIZE`, a `ubPerTokenScaleInput` UB slot allocated after
`ubRowAbsMax` (so the trailing `sharedTmpBuffer` shrinks by 1024 B, matching master), the member
declaration, and the batched read replacing the per-row scalar GM read:

```cpp
uint32_t scaleOffsetInBatch = (loopIdx - loopStartIdx) % SCALE_INPUT_COUNT;
if (scaleOffsetInBatch == 0) {
    uint32_t remain = tasksForIdx - (loopIdx - loopStartIdx);
    uint32_t copyCount = remain >= SCALE_INPUT_COUNT ? SCALE_INPUT_COUNT : <remain rounded up to a block>;
    AscendC::DataCopy(ubPerTokenScaleInput, gmPerTokenScale1[loopIdx], copyCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
}
ElementPerTokenScale perTokenScale = ubPerTokenScaleInput.GetValue(scaleOffsetInBatch);
```

The W4A8 specialisation also loses its `dcci(gmPerTokenScale1.GetPhyAddr(loopIdx), 0);`, which was
the stale-D-cache workaround the MTE2 path makes unnecessary — matching master.

Note this is a *perf* change with a *bugfix* character (the comment's "读旧值" — the scalar reads
could observe stale values), and it is upstream's, not this refactor's.

## Runtime cost

**None.** The quantised/non-quantised split is overload selection resolved at compile time; the
platform differences were already `if constexpr`/type-level. No runtime branch, argument or
indirection was added.

## Verification

- `bash build.sh --pkg --soc=ascend910b --ops=mega_moe` — green, 76 targets, 0 errors.
- `bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe` — green, 76 targets, 0 errors.

## Carried-forward risk (upstream, not introduced here)

`copyCount` rounds *up* to a multiple of `BYTE_PER_BLK / sizeof(ElementPerTokenScale)` = 8 floats, so
the last batch can `DataCopy` up to 7 floats past this core's `tasksForIdx` — and past the end of
`gmPerTokenScale` on the final core. Whether that is benign depends on GM allocation slack, which was
not verified. Same shape in every file that got this optimisation upstream. Worth confirming with the
owner before the series is accepted.
