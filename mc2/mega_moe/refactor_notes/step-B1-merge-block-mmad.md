# B1 — merge the A2 and A3 `BlockMmad` specialisations

Commit: `refactor(mega_moe): merge A2/A3 BlockMmad`

## Scope

| | |
|---|---|
| Folded away | `op_kernel/arch22/utilsA2/block_mmad_preload_async_fixpipe_quant_a2.hpp` (deleted) |
| Kept and extended | `op_kernel/arch22/utils/block_mmad_preload_async_fixpipe_quant.hpp` |
| Consumers updated | `mega_moe_kernel_a2_int8.hpp`, `mega_moe_kernel_a2_bf16.hpp` (include path only) |

Before: two headers, one `BlockMmad` partial specialisation each — the A3 one on
`MmadAtlasA3PreloadAsyncFixpipeQuant`, the A2 one on `MmadAtlasA2PreloadAsyncFixpipe`. Both
policies were already declared side by side in `utils/dispatch_policy_custom.hpp`, so the
duplication was in the *implementation* of the specialisation, not in the policy.

After: one header defining both specialisations on top of a shared implementation.

## Design

The two implementations differ in exactly one structural way: A3 must first copy the L1
scale tensor into the FixPipe buffer (`copyL1ToFP`) before the FixPipe can consume it, and
must fence that copy; A2 dequantises per-channel straight from the L1 scale tensor and has
no FixPipe copy at all.

That difference is expressed as one boolean template parameter:

- `detail::FixpipeScaleHandler<bool NEED_L1_TO_FP_, ...>` — `false` is an empty base whose
  `Init`/`Copy` are no-ops; `true` holds `fixpipeBuf` + `copyL1ToFP` and performs the copy,
  the `FIX_MTE2` re-arm and the `PIPE_FIX` barrier. Making `false` an *empty base* rather
  than a member with a dead flag means A2 objects carry no FixPipe state and the compiler
  emits nothing for the calls.
- `detail::BlockMmadPreloadAsyncFixpipeImpl<DispatchPolicy_, NEED_L1_TO_FP_, ...>` — the
  former A3 body, with the FixPipe step delegated to the handler and `CopyL0CToGm` selected
  by `std::conditional_t` on `ElementA == int8_t` and `NEED_L1_TO_FP_` (the A2/int8 path uses
  `CopyL0CToGmPerChannel` from `copy_l0c_to_gm_custom.hpp`).
- Two thin `BlockMmad` specialisations re-open the class with `using Base::Base;`, pinning
  `NEED_L1_TO_FP_` to `false` (A2) and `true` (A3).

## Equivalence check

The one thing that could silently break here is the `FIX_MTE2` event-0 protocol, which
compiles fine and deadlocks on hardware if a `SetFlag`/`WaitFlag` pairing is lost.
Operation-by-operation, for the `int8` case (the only case that touches `FIX_MTE2`):

| Step in `L1TileMmad` | A2 before | A2 after | A3 before | A3 after |
|---|---|---|---|---|
| `WaitFlag<FIX_MTE2>(0)` | yes | yes | yes | yes |
| `copyGmToL1S` + `MTE2_FIX` set/wait pair | yes | yes | yes | yes |
| `copyL1ToFP` into FixPipe | **absent** | **absent** (`Copy` is a no-op) | yes | yes (inside `Handler<true>::Copy`) |
| `SetFlag<FIX_MTE2>(0)` re-arm after the copy | absent | absent | yes | yes (inside `Handler<true>::Copy`) |
| `PipeBarrier<PIPE_FIX>` | absent | absent | yes | yes (inside `Handler<true>::Copy`) |
| end-of-loop `SetFlag<FIX_MTE2>(0)` re-arm | yes, guarded by `is_same_v<ElementA,int8_t>` | yes — guard became `!NEED_L1_TO_FP_ && int8_t`, which is `true` for A2 | **absent** | **absent** — the same guard is `!true && int8_t` = `false` |
| final `Finalize(...)` | yes | yes | yes | yes |

So the single merged guard `!NEED_L1_TO_FP_ && is_same_v<ElementA, int8_t>` reproduces both
original guards exactly, and the handler reproduces A3's copy/fence sequence exactly. The
protocol stays closed: `InitL1` arms, each `L1TileMmad` consumes and re-arms on whichever
path owns the arming, and the destructor drains the last one.

## Runtime cost

**None.** Every branch is `if constexpr` on template parameters, or a `std::conditional_t`
type selection. The `NEED_L1_TO_FP_ = false` instantiation has an empty base, so A2 carries
no `fixpipeBuf`/`copyL1ToFP` members and the no-op calls compile away.

## Dead code removed

The end-of-loop block

```cpp
#ifdef __TILE_SYNC__
    if (params.flag > 0) { ... AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId); }
#else
    Finalize(params.syncLoopIdx, params.flag);
#endif
```

became an unconditional `Finalize(params.syncLoopIdx, params.flag)`. `__TILE_SYNC__` is not
defined anywhere in the repository or the build (`grep -rn "__TILE_SYNC__"` across the tree,
`cmake/` and `build.sh` finds only these two `#ifdef` sites), so the `#else` branch was the
only one ever compiled on both files. This removes a cross-core set-flag path that has never
executed; if `__TILE_SYNC__` is ever reintroduced, this step must be revisited.

## Include migration

The body was carried over from the pre-rebase branch, which predates master's CATLASS
switch. Its includes were rewritten to master's scheme:

```
../template_linear_algebra_v2/catlass.hpp          -> catlass/catlass.hpp
../template_linear_algebra_v2/arch/resource.hpp    -> catlass/arch/resource.hpp
../template_linear_algebra_v2/tla_v2_coord.hpp     -> catlass/coord.hpp
../template_linear_algebra_v2/gemm_coord.hpp       -> catlass/gemm_coord.hpp
../template_linear_algebra_v2/gemm/dispatch_policy.hpp -> catlass/gemm/dispatch_policy.hpp
../template_linear_algebra_v2/gemm/helper.hpp      -> catlass/gemm/helper.hpp
```

`copy_l0c_to_gm_custom.hpp` and `dispatch_policy_custom.hpp` are unchanged by master, and
their contents (`CopyL0CToGmPerChannel`, both `MmadAtlas*` policies) are what the merged
implementation needs — no upstream behaviour was dropped, because master's only change to
this header was the include rewrite.

## Codegen impact

| | |
|---|---|
| B1 relative to the master-sync baseline | **12 / 15 kernels changed** |
| Unchanged | the three w4a8 objects (`8642b7…`, `b0a23f9…`, `fd9010…`) |

Measured with the fingerprint method described in `step-B5-unify-a2-kernels.md`
(`.text`/`.rodata` bytes hashed per kernel object, all 15 objects of the
`ascend910b` package, keyed by object name).

The BlockMmad merge does not reach the w4a8 path — that kernel keeps its own
file and its own `BlockMmad` instantiation — so those three objects staying
byte-identical is the expected result. It doubles as a control: it shows the
method does not manufacture differences.

For the other twelve kernels this step is **not** codegen-neutral. A merged
`BlockMmad` is a rewrite of a template shared by every path, and re-emitting it
shifts register allocation and scheduling. Its equivalence is argued from the
source merge, not proven from bytes.

## Verification

- `bash build.sh --pkg --soc=ascend910b --ops=mega_moe` — green, 76 targets, 0 errors.
- `bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe` — green, 76 targets, 0 errors.

Kernel execution is not covered (see the note in `README.md`); the `FIX_MTE2` equivalence
argument above is the substitute for a hardware run at this step, and should still be
confirmed by golden comparison before the series is accepted.
