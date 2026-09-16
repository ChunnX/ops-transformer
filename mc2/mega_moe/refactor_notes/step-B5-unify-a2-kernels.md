# B5 — unify the three A2 device kernels

Five commits, one logical step (kept separate so the fold can be bisected):

| # | Commit | Subject |
|---|--------|---------|
| 1  | `455abb610` | introduce unified A2 kernel from int8 |
| 2a | `94571c04a` | fold first bf16 divergences into the unified A2 kernel |
| 2b | `280798cdd` | split workspace/peermem layouts per dtype |
| 3  | `ec96ca014` | unify rank caching and SendTokensV3 path A |
| 4  | `94baae0da` | route the bf16 A2 path through the unified kernel |

## Scope

| Before (B4 `7854a4088`) | Lines | After (HEAD `94baae0da`) | Lines |
|---|---|---|---|
| `mega_moe_kernel_a2_int8.hpp` | 1682 | `mega_moe_kernel_a2.hpp` | 1866 |
| `mega_moe_kernel_a2_bf16.hpp` | 1656 | ↑ same file | |
| `mega_moe_kernel_a2_bf16_w4a8.hpp` | 1786 | `mega_moe_kernel_a2_bf16_w4a8.hpp` | 1786 |

The int8+bf16 pair collapses from 3338 to 1866 lines (−44%). The w4a8 kernel is
**not** merged — see "Why w4a8 stays separate" below.

## Shape of the unified kernel

```cpp
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, bool IsInt8_, bool IsW4A8_ = false>
class MegaMoeKernelA2 {
public:
    static constexpr bool IS_INT8 = IsInt8_;
    static constexpr bool IS_W4A8 = IsW4A8_;
    ...
    using ElementABefore = std::conditional_t<IsInt8_, int8_t, ElementA>;
    using WorkspaceInfo  = std::conditional_t<IS_INT8, WorkspaceInfoInt8,  WorkspaceInfoBf16>;
    using PeermemInfo    = std::conditional_t<IS_INT8, PeermemInfoInt8,    PeermemInfoBf16>;
```

Every dtype difference is a template parameter, a type alias, or an
`if constexpr` — **nothing is a runtime branch**. The kernel carries 8 `IS_INT8`
sites (below) plus 3 pre-existing `DispatchPolicy::ASYNC` sites that predate this
step.

## The 8 divergence sites

| Line (HEAD) | What differs |
|---|---|
| 258 | `gmPerTokenScale1/2` GM views — the per-token-scale regions exist only in the int8 peermem/workspace layouts |
| 398 | `SendTokensV3` path A (self-rank): int8 separates the row-tail scale into its own `gmScale` |
| 412 | same loop: int8 issues one extra `DataCopyPad` per row for that scale |
| 537 | `RecvTokensV3` tail: int8 copies received tokens back out of peer mem and lifts the row-tail scale; bf16 returns right after the flag |
| 1054 | `CombineSetFlag`: int8 arms 8 events (`V_MTE2`×4, `S_MTE2`×2, `MTE3_V`×2), bf16 arms 2 `MTE3_MTE2` |
| 1357 | `blockEpilogue1` (SwiGLU, dequantSum1 path) — int8 uses the dequant overload carrying both per-token scales |
| 1385 | `blockEpilogue1` (SwiGLU, dequantSum2 path) — same reason |
| 1501 | `blockEpilogue` (GMM2) — int8 uses the overload that takes `gmPerTokenScale2` |

`kRoutingIsQuant` is `IS_INT8 || IS_W4A8`; it was `true` in the int8 kernel,
`false` in the bf16 kernel, and `true` in w4a8, so the expression reproduces all
three.

## Where the divergent regions were kept verbatim, and why

The fold does not rewrite both sides into one shared body. Each divergent region
keeps the int8 text and the bf16 text **verbatim** on its two branches. That was
a deliberate methodological choice, forced by the evidence: rewriting the two
sides into a single body with gates — hoisting locals, inlining constructors,
adjusting scope — changed 6 of 15 kernels' codegen during B2 even though the
rewrites were semantically equivalent, because register allocation and
instruction scheduling shifted.

Byte-identical codegen is a strictly stronger property than semantic
equivalence, and it is the property that makes a fold *provable* rather than
*plausible*. Keeping both sides verbatim is what buys it (see the step-1..2b
results below).

Steps 3 and 4 are the points where that rule was relaxed, under the agreed
approach ("merge where the two dtypes are semantically equivalent, gate only
where they genuinely differ"). Their codegen deltas are real and are recorded in
the codegen section rather than waved away.

## Why w4a8 stays separate

`mega_moe_kernel_a2_bf16_w4a8.hpp` is untouched. Folding it was attempted and
reverted: the two files do not share the shape the int8/bf16 pair does. In
particular `SendTokensV3` has a different signature (w4a8 takes an extra
`uint64_t gmOffsetA`), so a function-level fold cannot work without changing the
callers, and the fold was converging on two complete pipelines living side by
side inside one file — 3390 lines versus 3652 apart versus 5308 unmerged — for
about 7% fewer lines. That is not worth it.

Consequently `IsW4A8_` is currently **never instantiated as `true`**: both call
sites in `mega_moe_a2.h` (`:364` for int8, `:411` for bf16) leave it at its
default. It is kept because it is the exact hook a future w4a8 fold needs and it
costs nothing at runtime, but it should be treated as reserved-not-used.

## Runtime cost

**None.** Every dtype difference resolves at template instantiation:
`std::conditional_t` for the type aliases, `if constexpr` for the divergence
sites, template parameters for `IS_INT8`/`IS_W4A8`. No runtime branch, no extra
function argument, no indirection, no virtual dispatch was introduced. The
per-token-scale work that the int8 path does and the bf16 path does not was
already there before the fold; the fold only decides **at compile time** which
side a given instantiation compiles.

## Codegen evidence

Method (reproducible, script at `refactor_notes/` — see the note below): the NPU
ELF machine type (0x1029) is unknown to `objdump`/`objcopy`, so equivalence is
established by reading the `.text` and `.rodata` section offsets and sizes with
`readelf -S -W`, extracting those bytes with `dd`, and hashing them. All 15
kernel objects in the `ascend910b` package are compared, keyed by object name.

### Per-step, B4 → HEAD

| Step | Kernels whose codegen changed |
|---|---|
| B4 → **step 1** (455abb610) | **0 / 15** |
| **step 2a** (94571c04a) | **0 / 15** |
| **step 2b** (280798cdd) | **0 / 15** |
| **step 3** (ec96ca014) | **6 / 15** |
| **step 4** (94baae0da) | **12 / 15** |

Steps 1–2b are **byte-identical** across all 15 kernels. That is the strongest
result in this step and the justification for the verbatim-branch methodology:
those three commits fold the entire bf16 divergence set into the unified file,
and because `mega_moe_a2.h` still routes bf16 to its own header, the folded bf16
code is dead for every instantiation — so identity is a hard invariant, not a
coincidence. It proves the folds are neutral *as written*.

Step 3's 6 changes are exactly the two int8 object groups (6 objects), and it
leaves the bf16/fp16 groups untouched — which is what it must do, since step 3
only rewrites int8-live code (the `RuntimeRank` cache, the `SendTokensV3` path A
merge, the `RecvTokensV3` early-return and path B/C flag-slot indices).

Step 4's 12 changes cover all four non-w4a8 groups. Six of them are the bf16/fp16
objects switching to the unified kernel. The other six are the int8 objects,
which step 4 re-touches, and those are the ones that need calling out: step 4
rewrites int8-live code in ways that are semantically identical but
lexically different — `LayoutA layoutA = IS_INT8 ? A : B` where int8 previously
constructed `A` directly, `gmOffsetS`/`gmGroupOffsetA` likewise, and two
`blockEpilogue1` call sites wrapped in `if constexpr (IS_INT8)`. Nothing about
int8's values, ordering or synchronisation changes, but the compiler no longer
emits the same bytes, so this is the first commit in the series whose behavioural
equivalence is **argued, not proven**.

### Whole refactor, master-sync baseline → HEAD

| | |
|---|---|
| Kernels whose codegen changed | **15 / 15** |
| `.rodata` sections changed | **0 / 15** |
| `.text` total | 13 309 920 → 13 245 852 B (**−64 068 B, −0.48%**) |
| Kernels whose `.text` grew | **0 / 15** |

Every kernel shrank (between −0.09% and −0.78%) and no `.rodata` section moved by
a byte, which is consistent with a pure reduction in emitted instructions rather
than a change in what the code computes or in its constant tables. That is
corroborating evidence, **not** proof of equivalence.

The rest of the ELF layout corroborates it much more sharply. Across all 15
objects, aggregating each section's size, the only sections that move at all are
`.text` and `.strtab`:

| Section | Baseline | HEAD | Δ |
|---|---|---|---|
| `.text` | 13 309 920 | 13 245 852 | **−64 068** |
| `.rodata` | — | — | 0 |
| `.data` | 120 | 120 | 0 |
| `.rela.text` | 90 000 | 90 000 | **0** |
| `.ascend.stack.size.record` | 2400 | 2400 | **0** |
| `.ascend.meta.*` | 174 000 | 174 000 | 0 |
| `__CCE_KernelArgSize` | 1200 | 1200 | 0 |
| `.comment` | 1605 | 1605 | 0 |
| `.bl_uninit` | 498 | 498 | 0 |
| `.symtab` | 26 640 | 26 640 | 0 |
| `.shstrtab` | 21 450 | 21 450 | 0 |
| `.strtab` | 49 820 | 50 288 | +468 |

Three of those zeroes carry real weight:

- **`.ascend.stack.size.record` is unchanged**, so no instantiation needs more
  stack. If the fold had introduced a runtime branch or a new local, this is
  where it would show.
- **`.rela.text` is unchanged to the byte**, so the number and kind of symbol
  references the code makes is identical. The instruction *stream* shrinks;
  its *references* do not multiply.
- **`.data` and `.rodata` are unchanged**, so no constant or table was rewritten.

`.strtab` growing by 468 bytes is expected: the unified kernel's mangled symbol
names are longer.

Read together: the refactor removed instructions without adding locals, changing
constants, or adding references. That is the signature of dead-code elimination
and instruction folding, not of added work. It still is not a proof — a wrong
fold could in principle be exactly this shape — which is why golden comparison
remains the gate.

**This is the honest bottom line for the whole B-series, not just B5: the
refactor is not codegen-neutral.** B1 (BlockMmad) and B2 (SwiGLU epilogue) alone
already moved all 15 kernels. Correctness therefore rests on golden comparison of
`yOut` + `expertTokenNumsOut` per dtype/arch, which has not been run. Compilation
being green and the UT suites passing prove nothing here — the UT suites do not
execute kernels.

## Corrections to the commit messages

Two of the five commit messages carry statements about codegen that were written
before the numbers in this note existed. This section supersedes them; the
messages themselves were left alone because rewriting published history is not
worth it for a comment.

- **`94baae0da`** ends with the literal placeholder *"N of 15 kernels changed"*,
  never filled in. The measured value is **12 of 15** relative to `ec96ca014`.
- **`ec96ca014`** claims *"6 of 15 kernels changed"*. That number is **correct**
  — 6 of 15 relative to `280798cdd` — though it was arrived at by a route that
  cannot be reconstructed now, and the surrounding prose ("Correctness therefore
  rests on golden comparison, not on this commit") is if anything more true than
  it reads, given that B1–B4 had already moved all 15 kernels.
- Both messages describe their step's delta as an unexplained side effect. With
  the per-step chain now measured, each delta has a named cause: step 3's 6 are
  the int8 object groups and step 4's 12 are the bf16/fp16 groups switching
  kernel plus the int8 groups being re-lexified (see "Per-step" above).

## Notes for whoever re-runs this

- The first fingerprint script emitted only `sort`ed signatures, discarding the
  object name. Comparing two such files *positionally* is meaningless — the sort
  interleaves changed and unchanged entries, so a 12-of-15 delta looks like
  15-of-15. Only multiset comparison is valid on that format. The current script
  keys each row by object name; use that one.
- `fp_a2_pre.txt` and friends are **not** a pre-refactor baseline. They were
  taken after B1–B4 had landed and their signatures match B4 exactly. The real
  baseline is `2cd217bc7` (`chore(mega_moe): sync arch22 onto master`).
- Kills the build: if the `ascend910b` package build ever hangs, check for orphan
  `asc_opc` / `opc.py` processes. When kernel compilation fails, CANN's `asc_opc`
  does not exit, and the waiting `make`/`cmake --build` hang indefinitely. This
  happened during B5 (a mid-edit compile error in step 4) and left 46 processes
  wedged for 2h40m.
