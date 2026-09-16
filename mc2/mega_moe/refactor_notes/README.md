# arch22 A2/A3 de-duplication — step notes

This directory documents the arch22 de-duplication refactor of `mega_moe`, one note per
commit, so that a regression can be attributed to a single step and bisected.

## Ground rules

- **One step = one commit = one note.** Each commit must build on its own for the SoCs it
  touches (`ascend910b` for A2, `ascend910_93` for A3, `ascend950` for arch35 which must stay
  untouched).
- **No runtime cost.** All dtype/platform selection is `if constexpr` on template parameters
  or type aliases. Do not introduce runtime branching, extra arguments threaded through hot
  loops, or virtual dispatch to express a difference that was previously compile-time. If a
  step genuinely cannot avoid a runtime cost, it must be called out in that step's note under
  a `## Runtime cost` heading with the exact code and the reason.
- **Upstream first.** Where a step merges two implementations, any behaviour that upstream
  added to either side must survive. When the two sides disagree semantically, stop and ask
  rather than picking one.

## Step index

| Step | Commit subject (prefix) | Scope | Status |
|------|-------------------------|-------|--------|
| —    | `docs(mega_moe): correct CLAUDE.md` | CLAUDE.md correction + this notes scaffold | done |
| B1   | `refactor(mega_moe): merge A2/A3 BlockMmad` | `utilsA2/block_mmad_preload_async_fixpipe_quant_a2.hpp` → `utils/block_mmad_preload_async_fixpipe_quant.hpp` | done — A2+A3 green |
| B2   | `refactor(mega_moe): merge SwiGLU epilogues` | `utilsA2/block_epilogue_pertoken_swiglu_a2_{bf16,int8}.hpp` + `utils/block_epilogue_w4a8post_pertoken_swiglu.hpp` → `utils/block_epilogue_pertoken_swiglu.hpp` (4 specialisations → 2) | done — A2+A3 green |
| B3   | `refactor(mega_moe): merge per-token V2 epilogues` | `utilsA2/block_epilogue_pertoken_v2_{bf16,int8}_a2.hpp` + `utils/block_epilogue_w4a8post_pertoken_v2.hpp` → `utils/block_epilogue_pertoken_v2.hpp` | planned |
| B4   | `chore(mega_moe): drop dead A2 epilogue headers` | remove the unreferenced `utilsA2/block_epilogue_pertoken_row_a2_*.hpp` | planned |
| B5   | `refactor(mega_moe): unify the three A2 device kernels` | `mega_moe_kernel_a2_{int8,bf16,bf16_w4a8}.hpp` → `mega_moe_kernel_a2.hpp`, `template<..., bool IsInt8, bool IsW4A8>` | planned |
| B6   | `refactor(mega_moe): merge MegaMoeA2 into MegaMoe` | `mega_moe_a2.h` folded into `mega_moe_arch22.h` as `MegaMoe<..., IS_A2_>`; single entry in `mega_moe.cpp` | planned |
| B7   | `docs(mega_moe): no-loss audit and review` | audit table + final review doc | planned |

## Baseline

The refactor sits on top of the master-sync commit
`chore(mega_moe): sync arch22 onto master, keep host-side validation and csrc sizing fix`,
which took master's rewritten chunked A2/A3 path verbatim and re-applied only the host-side
validation, the arch22 `DT_INT32` narrowing and the csrc CCL-sizing fix. Verified at that
point: `ascend910b` package green, `ascend910_93` package green, host UT 8/8.

## Why the A2 kernels are worth merging

Measured on the master-sync baseline: the three A2 kernels share 878 unique source lines;
per-file divergence is 266 (int8), 245 (bf16) and 348 (w4a8) — roughly 78% shared. The
divergences are dtype plumbing (`ElementABefore`, presence of the per-token scale buffers,
`copyInNum` stride, flag payload shape, `RuntimeRank` caching), all of which are expressible
as compile-time conditions.
