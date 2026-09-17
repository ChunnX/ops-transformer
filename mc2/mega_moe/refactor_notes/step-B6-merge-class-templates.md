# B6 — merge `MegaMoeA2` into `MegaMoe`

Goal: one class template `MegaMoe<..., IS_A2_>` driving both platforms, and one
branch in `mega_moe.cpp`, replacing the two parallel class templates that exist
today (`MegaMoeA2Impl::MegaMoeA2` in `mega_moe_a2.h` and `MegaMoeImpl::MegaMoe` in
`mega_moe_arch22.h`).

## Measured duplication

Counting non-comment, non-blank lines:

| | Lines |
|---|---|
| `MegaMoeA2` (`mega_moe_a2.h`) | 371 |
| `MegaMoe` (`mega_moe_arch22.h`) | 280 |
| Identical lines between them | **211 (75.4% of the smaller)** |
| Union | 326 |

So 208 lines are literally duplicated and a merge can plausibly remove ~150–200
of them. The overlap is not incidental — it is the `Init` GM_ADDR assignments,
the member declarations, and the whole block of `Element`/`Layout`/`Tile`/
`BlockEpilogue` type aliases in `Process()`.

## The six real differences

Everything that is *not* shared falls into six places. Three are deliberate
A2/A3 behavioural differences that must survive verbatim; three are platform
plumbing.

| # | Difference | Kind | Handling |
|---|---|---|---|
| D1 | `Init` takes an extra `maskBufferGM` (A3 only), and A3's `Params` carries `ptrMaskBuffer` | plumbing | unify the signature; A2 passes `nullptr`, and `Params` stays per-kernel |
| D2 | `kRoutingIsQuant` counts `int32_t` on A3, not on A2 | **behavioural** | unify — see step 1; A2 cannot reach the int32 arm |
| D3 | `isW4A8` counts `int32_t` on A3, not on A2 | plumbing | unify — see step 1 |
| D4 | int8 `epilogueCoreNum`/`epilogueGranularity` differ: A3 `aivNum_` with `(epr>4)? epr-3 : (epr>1)? epr-1 : 1`, A2 `aivNum_/2` with `(epr>1)? epr-1 : 1` | **behavioural** | gate on `IS_A2_`; README lists this as a hard constraint |
| D5 | A2 pads `layoutA1`'s row stride for W4A8 (`k_ > k2 ? k_ : k2`); A3 does not | plumbing | gate on `IS_A2_` |
| D6 | different device kernels (`MegaMoeKernel` vs `MegaMoeKernelA2`/`MegaMoeKernelA2Bf16W4A8`) with different `Params` | plumbing | gate on `IS_A2_`; the kernels stay separate |

D4 is the one that can silently corrupt results: getting `epilogueCoreNum` wrong
mis-partitions the epilogue across AIVs. It is gated, never merged.

D6 matters for scoping. **Merging the class templates does not require merging
the device kernels.** The two `Params` structs are near-identical (the A3 one has
three extra fields: `recvRoundsMax`, `gmm1InputExpertSlotWidth`, `ptrMaskBuffer`),
so the merged class can pick the kernel and build its `Params` under
`if constexpr (IS_A2_)`, leaving A2 with exactly the struct it has today and zero
extra members. That is the plan; anything else would add cost to A2 to buy tidiness.

## Steps

Each step is its own commit and its own build.

| Step | Commit | Content | Codegen |
|---|---|---|---|
| 1 | *unify the dtype predicates* | D2 + D3 | **0/15 — byte-identical** |
| 2 | *share Init and the member set* | D1, extract `LoadCommonTiling` | TBD |
| 3 | *share the Process() type aliases* | the 208 duplicated lines | TBD |
| 4 | *one class, one entry point* | D4, D5, D6; delete `mega_moe_a2.h`; collapse `mega_moe.cpp` | TBD |

## Step 1 — unify the dtype predicates

Both predicates in `mega_moe_a2.h` gained the `int32_t` arm that
`mega_moe_arch22.h` already had:

```cpp
// mega_moe_a2.h, Init()
    constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t> || std::is_same_v<BType_, int8_t>;
// mega_moe_a2.h, Process()
    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t>;
```

### Why this is a no-op on A2

A3 packs W4A8 weights as `int32`, which is why its predicates count `int32_t`.
A2 does not, and more importantly **A2 can never be instantiated with an int32
weight**: `mega_moe.cpp` wraps the entire A2 branch in

```cpp
if constexpr (isA2) {
#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)
    ...
#endif
}
```

so no `MegaMoeA2<..., int32_t, ...>` translation unit is ever generated. The
`int32_t` arm is therefore unreachable on A2, and adding it cannot change what A2
compiles to.

### Codegen prediction, and why this step is worth a commit on its own

Because the added arm is unreachable, the expected codegen delta is **zero across
all 15 kernels**. That prediction is the point of landing this as a separate
commit: it is a direct test of the claim that A2 never sees an int32 weight. If
the fingerprint moves, the claim is wrong and the whole D2/D3 handling in the
later steps needs rethinking before any merge happens. (Measured result: see
below.)

### Codegen result

| | |
|---|---|
| B5 HEAD → B6 step 1 | **0 / 15 kernels changed — byte-identical** |

Prediction confirmed. All 15 `ascend910b` kernel objects are bit-identical after
the predicate unification, which proves the load-bearing claim directly: **A2
never sees an int32 weight**, so the `int32_t` arm is unreachable there and the
`#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)` guard in `mega_moe.cpp` does what it looks
like it does.

That matters beyond this step. D2 and D3 were the only *predicate-level*
divergences between the platforms, and the merged class in step 4 will carry one
`kRoutingIsQuant` and one `isW4A8` for both — this is the evidence that doing so
cannot change A2.
