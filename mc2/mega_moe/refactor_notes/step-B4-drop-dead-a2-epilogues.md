# B4 — drop the dead A2-only epilogue variants

Commit: `chore(mega_moe): drop dead A2-only epilogue variants`

## Scope

| Removed | Why it is dead |
|---|---|
| `utilsA2/block_epilogue_pertoken_row_a2_bf16.hpp` | no include, no instantiation |
| `utilsA2/block_epilogue_pertoken_row_a2_int8.hpp` | no include, no instantiation |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantInt8` | declared, never instantiated |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantBF16` | declared, never instantiated |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantSwigluQuantInt8` | orphaned by B2 |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantSwigluQuantBF16` | orphaned by B2 |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantV2Int8` | orphaned by B3 |
| `dispatch_policy_custom.hpp`: `EpilogueAtlasA2PerTokenDequantV2BF16` | orphaned by B3 |

## Evidence

Nothing in the repository references them. For each of the six policies, a repo-wide grep
excluding `build/`, their own declaration site and the two dead headers returns **zero** uses;
the two headers themselves are included by no file. Their policies are only ever defined and
declared, never instantiated, so the `BlockEpilogue` specialisations they provide are unreachable
code.

The `row_a2_*` pair was already dead *before* this refactor — it was equally unreferenced at the
merge base — so this is pre-existing dead code, not something the refactor orphaned. (The A3 row
epilogue, `utils/block_epilogue_pertoken_row.hpp`, specialises a different policy and *is* live; it
is untouched.)

## The syntax error this removes

`block_epilogue_pertoken_row_a2_bf16.hpp` line 174 reads:

```cpp
AscendC::Cast(ubD, ubCFp32, AscendC::RoundMode::, blockN);
```

The `CAST_RINT` token is missing. This is a hard compile error, introduced upstream, and it
survived only because the file is never included — no translation unit ever parses it. Deleting
the file removes the landmine; if this variant is ever wanted again it must be recovered from
history *and* fixed.

## Runtime cost

**None.** Nothing that was reachable changes. The only generated-code difference is that the
unused policy types no longer exist.

## Verification

- `bash build.sh --pkg --soc=ascend910b --ops=mega_moe` — green, 76 targets, 0 errors.
- `bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe` — green, 76 targets, 0 errors.
- `grep -rn "RoundMode::,"` across the arch22 tree — no hits.
- `utilsA2/` no longer exists; every A2-only epilogue has now been folded into `utils/` by B1–B3.
