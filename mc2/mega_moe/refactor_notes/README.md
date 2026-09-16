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
- **"No runtime cost" is not "no codegen change".** Merging two template bodies re-emits
  them, and the compiler's register allocation and scheduling move as a result — see the
  table below. Every step must therefore state its *measured* codegen impact, and no step
  may claim byte-equivalence it has not actually measured. Byte-equivalence, where it is
  achieved (B3→B4 and B5 steps 1–2b), is a bonus that makes a step provable; its absence is
  not by itself a defect, but it does move the burden onto golden comparison.

## Step index

| Step | Commit subject (prefix) | Scope | Status |
|------|-------------------------|-------|--------|
| —    | `docs(mega_moe): correct CLAUDE.md` | CLAUDE.md correction + this notes scaffold | done |
| B1   | `refactor(mega_moe): merge A2/A3 BlockMmad` | `utilsA2/block_mmad_preload_async_fixpipe_quant_a2.hpp` → `utils/block_mmad_preload_async_fixpipe_quant.hpp` | done — A2+A3 green |
| B2   | `refactor(mega_moe): merge SwiGLU epilogues` | `utilsA2/block_epilogue_pertoken_swiglu_a2_{bf16,int8}.hpp` + `utils/block_epilogue_w4a8post_pertoken_swiglu.hpp` → `utils/block_epilogue_pertoken_swiglu.hpp` (4 specialisations → 2) | done — A2+A3 green |
| B3   | `refactor(mega_moe): merge V2 epilogues and unify epilogue numbering` | `utilsA2/block_epilogue_pertoken_v2_{bf16,int8}_a2.hpp` + `utils/block_epilogue_w4a8post_pertoken_v2.hpp` → `utils/block_epilogue_pertoken_v2.hpp` (4 specialisations → 2); also swaps the A3 epilogue numbering so BlockEpilogue2 = V2 on both platforms | done — A2+A3 green |
| B4   | `chore(mega_moe): drop dead A2-only epilogue variants` | remove the unreferenced `utilsA2/block_epilogue_pertoken_row_a2_*.hpp` and the six A2-only policy declarations orphaned by B1–B3 | done — A2+A3 green |
| B5   | `refactor(mega_moe): unify the three A2 device kernels` | `mega_moe_kernel_a2_{int8,bf16,bf16_w4a8}.hpp` → `mega_moe_kernel_a2.hpp`, `template<..., bool IsInt8, bool IsW4A8>`; five commits, see `step-B5-unify-a2-kernels.md` | done — A2 green, codegen measured |
| B6   | `refactor(mega_moe): merge MegaMoeA2 into MegaMoe` | `mega_moe_a2.h` folded into `mega_moe_arch22.h` as `MegaMoe<..., IS_A2_>`; single entry in `mega_moe.cpp` | planned |
| B7   | `docs(mega_moe): no-loss audit and review` | audit table + final review doc | planned |

## Codegen impact of the whole series

Every step was rebuilt from a clean `build/` for `ascend910b` and its 15 kernel
objects fingerprinted (`.text` + `.rodata` bytes hashed; method and caveats in
`step-B5-unify-a2-kernels.md`). Per step, relative to the step before it:

| Step | Kernels whose codegen changed |
|------|-------------------------------|
| baseline → B1 (BlockMmad) | 12 / 15 |
| B1 → B2 (SwiGLU epilogues) | 15 / 15 |
| B2 → B3 (V2 epilogues + numbering) | 15 / 15 |
| B3 → B4 (drop dead epilogues) | **0 / 15** |
| B4 → B5 step 1 / 2a / 2b | **0 / 15** each |
| B5 step 3 (rank cache, SendTokensV3 path A) | 6 / 15 |
| B5 step 4 (bf16 routed to unified kernel) | 12 / 15 |
| **baseline → HEAD, cumulative** | **15 / 15** |

**No kernel is byte-identical after the full series.** The refactor is not
codegen-neutral and must not be described as such. What the ELF layout does
support (baseline → HEAD, aggregated over all 15 objects):

- `.text` 13 309 920 → 13 245 852 B: **−64 068 B (−0.48%)**, and **no kernel grew**
- `.rela.text` unchanged to the byte — the code's reference structure is identical
- `.ascend.stack.size.record` unchanged — no instantiation needs more stack
- `.data` and `.rodata` unchanged — no constant or table was rewritten

That is the signature of removed instructions, not added work. It is
**corroborating evidence, not proof**: correctness has to come from golden
comparison of `yOut` + `expertTokenNumsOut` per dtype/arch, which has **not** been
run. Green compilation and passing UT prove nothing here — the UT suites do not
execute kernels.

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
