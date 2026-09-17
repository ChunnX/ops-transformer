# B7 — no-loss audit

Audit of B1–B7: what changed, what provably did **not** change, and what remains
unverified. Written to be checked, not to be believed — every claim below names
the command or the file that establishes it.

## Headline numbers

| | Baseline `2cd217bc7` | HEAD | Δ |
|---|---|---|---|
| arch22 code (`.h`/`.hpp`/`.cpp` lines) | 16 039 | 11 782 | **−4 257 (−26.5%)** |
| Files deleted outright | — | 12 | |
| `refactor_notes/` + `CLAUDE.md` | 0 | 1 143 | +1 143 (documentation, not code) |

Of the 4 257 removed lines, 3 951 are the contents of the deleted files and the
rest is code that was folded. Twelve files are gone: `mega_moe_a2.h`, the orphaned
`mega_moe_kernel_a2_bf16.hpp`, `mega_moe_kernel_a2_int8.hpp` (renamed, not lost),
the two `block_epilogue_w4a8post_*` headers, two dead `row_a2_*` epilogues, four
`utilsA2/` epilogue variants and the `utilsA2/` `BlockMmad`. `utilsA2/` no longer
exists.

## 1. Frozen interfaces — verified byte-identical

Each was diffed against the baseline and is byte-for-byte identical:

| Item | Why it matters |
|---|---|
| `__global__ mega_moe` signature and its 23 `GM_ADDR` parameters | the op binary's entry ABI |
| `MegaMoeTilingDataQuant` / `MegaMoeTilingDataNonQuant` | the tiling struct layout the host writes and the kernel reads |
| `mega_moe_tiling_key.h` (`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL`) | the tiling key space; changing it re-numbers every kernel |
| `op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp` | host tiling, including the arch22 validation added during the master sync |
| `mega_moe_def.cpp`'s `aicore_config` (Ascend950) | arch35's dtype/format declaration — untouched by this refactor |

The arch22 `aicore_config` **was** changed earlier (the `DT_INT32` group was
removed from the dtype lists), but that happened in the master-sync commit, not
in B1–B7, and this audit confirms the 950 config is byte-identical.

## 2. Behavioural differences that must survive — verified

These are the places where A2 and A3 genuinely disagree. They are the ones that
would silently corrupt results if a merge flattened them, so each was checked
expression by expression rather than assumed.

### `epilogueCoreNum` / `epilogueGranularity`

| Case | A2 (baseline) | A3 (baseline) | HEAD |
|---|---|---|---|
| w4a8 | `aivNum_`, `0U` | `aivNum_`, `0U` | same, shared |
| int8 | `aivNum_ / 2`, `(epr>1) ? epr-1 : 1` | `aivNum_`, `(epr>4) ? epr-3 : (epr>1) ? epr-1 : 1` | both bodies kept verbatim under `if constexpr (IS_A2_)` |
| bf16/fp16 | `aivNum_`, `(epr>2) ? epr-2 : 1` | `aivNum_`, `(epr>2) ? epr-2 : 1` | same, shared |

The int8 row is the dangerous one — A2 gives the epilogue half the AIVs and a
shorter granularity curve. The two bodies are not merged into one expression;
they sit side by side and are chosen at compile time.

### `layoutA1` row stride

A2 pads it for W4A8 (`k_ > k2 ? k_ : k2`), and uses dense `k_` otherwise. A3 uses
`k_` in both cases. At HEAD:

```cpp
const uint32_t layoutA1RowStride =
    static_cast<uint32_t>(IS_A2_ ? (isW4A8 ? (k_ > k2 ? k_ : k2) : k_) : k_);
LayoutA layoutA1{static_cast<uint32_t>(m_), static_cast<uint32_t>(k_), layoutA1RowStride};
```

A3 previously wrote `LayoutA layoutA1{m_, k_}`. Catlass defines
`RowMajor(rows, cols)` as `RowMajor(rows, cols, cols)`, so passing `k_` explicitly
is the same layout — and the fingerprint confirms the compiler emits identical
code (B6 step 3, 0/15).

### dtype predicates

`kRoutingIsQuant` and `isW4A8` differed between the platforms; they are now shared
and both count `int32_t`. This is safe **only** because A2 cannot be instantiated
with an `int32_t` weight. That is enforced at `mega_moe.cpp`:

```cpp
if constexpr (isA2) {
#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)
    ... A2 instantiation ...
#endif
}
```

Verified, not inferred: B6 step 1 added the `int32_t` arms and all 15 kernel
objects were byte-identical afterwards, which is only possible if the arm is
unreachable. **This guard must stay a preprocessor `#if`.** Converting it to
`if constexpr (isA2) { return; }` would still instantiate `MegaMoe<..., int32_t,
...>` and the property would be lost.

## 3. Upstream machinery — verified present

Master's newer arch22 machinery had to survive the merge. Each is still present
and still wired:

| Mechanism | Where it lives at HEAD |
|---|---|
| chunked `MoePermutePrologue` (`PERMUTE_CHUNK` = 1024) | `moe_permute_prologue/`, referenced from 8 files |
| `recvRoundsMax` / `recvRoundBudget` receive rounds | 4 files (A3 kernel + host tiling) — same count as baseline |
| `mask_buffer` rank masking | 4 files — same count as baseline; still gated to A3 by `mega_moe_tiling_arch22.cpp:591` |
| route-table UB cache (`useUbRouteCache`, `ubTokenPerExpert`, `ubPreSumBeforeRank`) | `utils/block_epilogue_pertoken_v2.hpp` + both A2/A3 kernels |
| per-token-scale UB batch prefetch | the SwiGLU and V2 epilogues |

The `useUbRouteCache` file count drops from 8 to 5, which looks like a loss and is
not: the three that disappeared are `utilsA2/block_epilogue_pertoken_v2_{bf16,int8}_a2.hpp`
and `utils/block_epilogue_w4a8post_pertoken_v2.hpp`, all of which were folded into
`utils/block_epilogue_pertoken_v2.hpp`, which is still present and still the one
the route cache is consumed from.

## 4. What was removed, and what took over

| Removed | Lines | Taken over by |
|---|---|---|
| `mega_moe_a2.h` (`MegaMoeA2`) | 456 | `mega_moe_arch22.h` as `MegaMoe<..., IS_A2_>` |
| `mega_moe_kernel_a2_int8.hpp` | — | renamed to `mega_moe_kernel_a2.hpp` |
| `mega_moe_kernel_a2_bf16.hpp` | 1 656 | `MegaMoeKernelA2<..., false>` in the same unified kernel |
| `utilsA2/block_mmad_preload_async_fixpipe_quant_a2.hpp` | 493 | `utils/block_mmad_preload_async_fixpipe_quant.hpp` |
| `utilsA2/block_epilogue_pertoken_swiglu_a2_{bf16,int8}.hpp` | 920 | `utils/block_epilogue_pertoken_swiglu.hpp` (two overload sets) |
| `utils/block_epilogue_w4a8post_pertoken_swiglu.hpp` | 555 | ↑ same header |
| `utilsA2/block_epilogue_pertoken_v2_{bf16,int8}_a2.hpp` | 500 | `utils/block_epilogue_pertoken_v2.hpp` via `PerTokenV2Tag<IS_A2, isInt8>` |
| `utils/block_epilogue_w4a8post_pertoken_v2.hpp` | 366 | ↑ same header |
| `utilsA2/block_epilogue_pertoken_row_a2_{bf16,int8}.hpp` | 418 | **nothing — dead before this refactor** |
| `utils/dispatch_policy_custom.hpp` (8 policy declarations) | 43 | **nothing — orphaned by the merges above** |

The `row_a2_*` pair is worth a note: it was already unreferenced at the merge
base, and one of the two contained a hard syntax error
(`AscendC::RoundMode::,`) that never fired only because nothing ever included it.
It is gone now; if that variant is ever wanted it must be recovered from history
*and* fixed.

## 5. The one thing this audit had to fix

`mega_moe_kernel_a2_bf16.hpp` survived B5 step 4. That step's message said the
file was "no longer included", which was true — the A16W16 path had moved to the
unified kernel — but the file itself was left on disk, 1 656 lines of unreachable
code. Found by walking every arch22 header and asking which file includes it; it
was the only one with no answer. Removed here. The deletion is byte-identical on
all 15 kernel objects, as it must be for code nothing compiles.

## 6. Codegen

The fingerprint of every step is in `README.md`'s summary table and in each step's
note. The two facts that matter for this audit:

- **The refactor is not codegen-neutral.** Baseline → HEAD changes all 15 kernel
  objects. It is a pure *reduction* — `.text` 13 309 920 → 13 245 852 B (−0.48%),
  no kernel growing; `.rela.text`, `.ascend.stack.size.record`, `.data` and
  `.rodata` unchanged to the byte — but that is corroboration, not equivalence.
- **Each kernel object carries both the A2 and the A3 instantiation.**
  `ASCENDC_TPL_SEL` does not vary with the build's SoC, so `--soc=ascend910b` and
  `--soc=ascend910_93` emit the same 15 objects (verified by md5), and `readelf`
  shows 20 `.ascend.meta.*` sections per object with 5 of them carrying the arch
  bit (`_1092_` = `_68_ | 1024`). One fingerprint run therefore covers both
  platforms — worth knowing, because it is the opposite of what the build
  commands suggest.

## 7. What is NOT proven

- **No golden comparison has been run.** Nothing here shows that `yOut` or
  `expertTokenNumsOut` is numerically unchanged for any dtype or arch. That is the
  only thing that can establish correctness for the steps whose codegen moved,
  and it needs hardware.
- **The UT suites prove nothing about kernels.** They cover tiling, infershape and
  the aclnn surface; they never execute a kernel.
- **Compilation being green proves only that it compiles.** Every step here was
  built, but a clean build is not evidence of equivalence.
- **The steps that moved codegen are the ones to scrutinise**: B1 (12/15),
  B2 (15/15), B3 (15/15), B5 step 3 (6/15), B5 step 4 (12/15). B4, all of B5's
  steps 1–2b, and all of B6 are byte-identical and carry correspondingly less
  risk. A golden matrix should weight its effort accordingly.
- **W4A8 on A2 remains unimplemented**, as it was before. `MegaMoeKernelA2Bf16W4A8`
  and the `IsW4A8_` template parameter are reserved and never instantiated. This
  refactor neither implements nor removes that path.

## Reproduction

```bash
source <cann>/bin/setenv.bash
cd /root/workspace/Q2/trans_tyf
bash build.sh --pkg --soc=ascend910b --ops=mega_moe -j16
bash mc2/mega_moe/refactor_notes/codegen_fingerprint.sh \
     build/cann-ops-transformer-custom_linux-aarch64.run /tmp/fp.txt
```

One caveat learned the hard way: if the build appears to hang, check for orphan
`asc_opc` / `opc.py` processes. When kernel compilation fails, CANN's `asc_opc`
does not exit, and the waiting `make` / `cmake --build` hang indefinitely — this
happened once during B5 and left 46 processes wedged for 2h40m.
