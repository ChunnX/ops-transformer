# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mega_moe` is a single fused MoE (Mixture-of-Experts) operator for Huawei Ascend NPUs. It fuses the full expert-FFN pipeline — **Dispatch + Linear1 + SwiGLU + Linear2 + Combine** — and the surrounding inter-rank communication into one operator, overlapping communication with computation.

It is **one operator inside the larger `ops-transformer` (CANN) monorepo**. The repo root is two levels up at `../../` (the `trans_tyf/` checkout, a working copy of `ops-transformer`). The CMake build system, `build.sh`, CI scripts, `.clang-format`, and `pre-commit` config all live at that repo root — **not** inside this directory. Run all build/test commands from the repo root.

## Build, test, lint (run from repo root `../../`)

Environment prerequisite: source the CANN toolkit env (`source ~/Ascend/ascend-toolkit/latest/bin/setenv.bash` or equivalent) so `cmake`, the BiSheng compiler, and `ASCEND_3RD_LIB_PATH` are available.

```bash
# Build a run package (kernel bin) for one SoC, one op:
bash build.sh --pkg --soc=ascend910b --ops=mega_moe -j16

# Build host + api libraries only (no kernel bin):
bash build.sh --ophost --opapi --ops=mega_moe -j16

# Unit tests (build + run). Pick the test category and SoC:
bash build.sh --ophost_test --ops=mega_moe --soc=ascend910b       # tiling + infershape UT
bash build.sh --opapi_test  --ops=mega_moe --soc=ascend910b       # aclnn UT
bash build.sh -u --ops=mega_moe --soc=ascend910b                  # all UT
bash build.sh --ophost_test --ops=mega_moe --noexec               # compile UT, don't execute
bash build.sh --ophost_test --ops=mega_moe --cov                  # with coverage

# System/e2e tests (needs real NPU or simulator):
bash scripts/ci/ops_st_test.sh --soc_version=ascend910b --ops=mega_moe --test_type=kernel,aclnn,e2e

# Lint / format (matches CI gate):
pre-commit run --all-files                  # clang-format (style=file), ruff, codespell
```

UT binaries are GTest executables emitted under `build/`. To run a single case, invoke the compiled binary with `--gtest_filter=MegaMoeArch22TilingTest.Test0` (test names live in `tests/ut/`).

SoC versions matter — they select the kernel architecture (see below). Supported for this op: `ascend910b`, `ascend910_93`, `ascend950`.

**The UT suites do not execute kernels.** They cover tiling, infershape and the aclnn API surface only. Kernel correctness has to be shown separately by golden-comparing `yOut` + `expertTokenNumsOut` on hardware (see the matrix below).

### Third-party dependencies are fetched and checked out at configure time

`CANN_3RD_LIB_PATH` (normally `<repo>/third_party`) holds `catlass`, `opbase` and `ops-tensor`. `cmake/third_party/*.cmake` pins each to a commit and checks it out during configure, so **a stale or absent local clone breaks the build in non-obvious ways**:

- `catlass.cmake` — master switched CATLASS from a git submodule to a CMake fetch. The in-tree `op_kernel/arch22/template_linear_algebra_v2/` no longer exists; arch22 now includes headers as `catlass/...` resolved from `third_party/catlass`. Do not reintroduce `template_linear_algebra_v2/` includes.
- `opbase.cmake` — in the `CANN_3RD_LIB_PATH` branch it runs `git checkout ${OPBASE_TAG_ID}` **without a preceding `git fetch`** (unlike `ops-tensor.cmake`). If the pinned commit is not already in the local clone the checkout fails silently, the tree stays on the old commit, and you get `fatal error: op_host/util/op_const_def.h: No such file or directory`. Fix: `git -C third_party/opbase fetch origin <TAG> --depth=1 && git -C third_party/opbase checkout <TAG>`.
- `ops-tensor.cmake` — does fetch before checkout, so it self-heals when the network is reachable.

## Architecture: two hardware paths, one op

The op has two completely separate device-kernel implementations, selected by SoC:

| SoC | Product | Kernel dir | Scenarios supported |
|-----|---------|------------|---------------------|
| `ascend910b`, `ascend910_93` | Atlas A2 / A3 | `op_kernel/arch22/` | A16W16 (BF16/FP16), A8W8-INT, A8W4-INT |
| `ascend950` | Ascend 950PR/950DT | `op_kernel/arch35/` | A8W8-FP (MXFP, FP8/FP4 weights) |

The two paths share **none** of their compute kernels: arch22 uses the externally-fetched `catlass` GEMM template library; arch35 uses the in-tree `blaze/` GEMM library. They share the op def, infershape, and aclnn API.

### Standard CANN op layout (this dir)

- `op_host/mega_moe_def.cpp` — op definition: inputs/outputs/attrs **and a separate `OpAICoreConfig` per SoC** (`ascend950` vs `ascend910b`/`ascend910_93`). The two configs declare different dtype/format lists — editing one does not affect the other. The arch22 lists do **not** include `DT_INT32` weights (removed deliberately; W4A8-on-A2 weight unpacking is not implemented).
- `op_host/mega_moe_infershape.cpp` — `InferShape`/`InferDataType` (output `y` mirrors `x`; `expert_token_nums` has dim `moe_expert_num / ep_world_size`).
- `op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp`, `op_host/op_tiling/arch35/mega_moe_tiling.cpp` — host-side tiling. Note the build globs these differently: arch35 tiling is matched explicitly (`op_tiling/arch35/*.cpp`), everything else under `op_tiling/` recursively. `op_host/op_tiling/mega_moe_tiling_host.h` holds what the two arch tilings share.
- `op_api/aclnn_mega_moe.cpp` — `aclnnMegaMoe` C-API wrapper (also handles INT4-packed-as-INT32 view conversion).
- `op_graph/` — graph plugin registration.
- `torch_extension/` — Python/pybind extension, relocated here from the repo-root `torch_extension/cann_ops_transformer/ops/csrc/`.
- `tests/ut/` — `op_host/test_mega_moe_infershape.cpp`, `op_host/arch{22,35}/test_mega_moe_tiling.cpp`, `op_api/test_aclnn_mega_moe.cpp`.

### arch22 kernel internals (`op_kernel/arch22/`)

- `mega_moe.cpp` — the single `__global__ mega_moe` entry. Selects A2 vs A3 via `TPL_ARCH` and the tiling struct via `TPL_QUANT_MODE`. **W4A8 on A2 is reserved but not yet implemented** (A3 supports it).
- `mega_moe_arch22.h` — the **single** class template `MegaMoe<..., IS_A2_>` for both platforms; owns the GEMM/epilogue wiring and `Init`/`Process`. The A2/A3 differences are exactly three `if constexpr (IS_A2_)` sites: `epilogueCoreNum`/`epilogueGranularity` in the int8 arm, `layoutA1RowStride`, and the device-kernel/`Params` tail.
- Device kernels (the class template is merged; the kernels are deliberately still two):
  - `mega_moe_kernel_a2.hpp` — `MegaMoeKernelA2<..., IsInt8, IsW4A8>`, shared by the A16W16 (bf16/fp16) and A8W8-INT paths on A2
  - `mega_moe_kernel_a2_bf16_w4a8.hpp` — `MegaMoeKernelA2Bf16W4A8`; **reserved, never instantiated** (W4A8 on A2 is not implemented, and the tiling-key selection has no INT4 entry)
  - `mega_moe_kernel_a3.hpp` — `MegaMoeKernel` (adds a third epilogue for the A3 combine path)
- `mega_moe_tiling_key.h` — declares the tiling-key template axes (`TPL_IS_TRANSPOSE_W1/W2`, `TPL_QUANT_MODE`, `TPL_QUANT_OUT_TYPE`, `TPL_ARCH`) via `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL`, binding each axis combination to a tiling struct.
- `mega_moe_tiling_a2a3.h` — `MegaMoeA2A3TilingData` (common) + `MegaMoeTilingDataQuant`/`NonQuant` wrappers. Defines the `MEGA_MOE_QUANT_MODE_*` / `MEGA_MOE_QUANT_OUT_TYPE_*` / `SOC_ASCEND*` constants.
- Sub-kernels:
  - `moe_permute_prologue/` — `MoePermutePrologue`, the dispatch-side permute/sort/gather prologue. Replaced the older `moe_init_routing_v2/` + `moe_init_routing_quant_v2/` pair.
  - `unpermute/` — token combine.
  - `utils/` — shared infrastructure: per-token epilogues (SwiGLU, V2 dequant, W4A8 post-processing, quant fixpipe), `const_args.hpp` (named constants such as `PERMUTE_CHUNK`, `AIV_PER_AIC`, `GATE_UP_ROWS_PER_TOKEN`, `ICACHE_PRELOAD_LINES`), `dispatch_policy_custom.hpp`, `copy_l0c_to_gm_custom.hpp`, `hccl_shmem.hpp`. These are now shared by A2 and A3; the former `utilsA2/` duplicates were removed by this refactor.

Both A2 and A3 now run a **per-chunk pipeline**: tokens are processed in `PERMUTE_CHUNK` (1024) blocks with per-chunk cumsum/route-table buffers, and A3 additionally splits each chunk into receive rounds bounded by `recvRoundBudget` (`recvRoundsMax` rounds). `maxOutputSize_` is the per-round receive budget, not a token total. Route tables are served from per-core UB caches (`useUbRouteCache`) and per-token scales are batch-prefetched into UB.

### arch35 kernel internals (`op_kernel/arch35/`)

- `mega_moe_apt.cpp` — entry; selects `MegaMoe` vs `MegaMoeLayered` by comm mode (MTE vs URMA), MXFP dispatch quant only, BF16 activation + FP8/FP4 weights.
- `mega_moe_arch35.h` / `mega_moe_layered.h` — the arch35 class hierarchy; `common/` holds shared types/workspace/peermem utilities; `stage/` holds the pipeline stages (dispatch, quant, GMM1/gmm1-activation, GMM2/combine, unpermute, send-mask, workspace reset).
- `mega_moe_wave_a8w8.h` / `mega_moe_wave_a8w4.h` / `mega_moe_wave_a4w4.h` — per-scenario wavefront loops.
- `blaze/` — arch35's GEMM/prologue/epilogue library (`blaze/epilogue/activation/` holds the SwiGLU/Situglu/SwiGLU-OAI variants; `blaze/epilogue/block_epilogue_activation_mx_quant.h` the MXFP SwiGLU+quant epilogue).

## Quantization attribute cheat sheet

Attrs in `mega_moe_def.cpp` drive scenario selection; the host tiling translates them into the kernel's `TPL_QUANT_MODE`/`TPL_QUANT_OUT_TYPE` template axes:

- `dispatch_quant_mode`: `0`=non-quant (A16W16), `2`=INT8 quant (A8W8-INT & A8W4-INT), `4`=MXFP (A8W8-FP, arch35 only). arch22 tiling accepts only `0` and `2` and rejects anything else — so do not derive arch22 behaviour from `4`.
- `dispatch_quant_out_dtype`: `1`=INT8, `23`=FLOAT8_E5M2, `24`=FLOAT8_E4M3FN, `296`=FLOAT4_E2M1.
- A8W4-INT additionally requires `bias1`/`bias2` (offline-computed MSD precision-compensation biases) — see README for the math.

On A2/A3 the CCL buffer size helper in `torch_extension/csrc/mega_moe.cpp` must derive `isQuantRouting` from `dispatch_quant_mode == 2`; it previously used `== 4`, which is never true there and made quant-routing calls allocate an undersized buffer.

Full input/output/attr semantics and per-scenario constraints are in `README.md` (this dir) and the user-facing interface doc at `../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md`.

## Refactor status (arch22 de-duplication)

**Done.** The three near-identical A2 kernels are one, the two parallel class templates are one, and the A2-only `utilsA2/` copies of the epilogue/`BlockMmad` headers are gone. Work was staged as independently buildable commits so a regression can be bisected; each step left a note under `refactor_notes/`. Read `refactor_notes/README.md` for the step index and the **measured** codegen impact of every step.

### Codegen is not neutral — do not assume a merge is free

Every step was rebuilt from a clean `build/` and its 15 kernel objects fingerprinted. **Baseline → HEAD changes all 15 objects.** It is a pure reduction (`.text` −0.48%, no kernel growing; `.rela.text`, `.ascend.stack.size.record`, `.data` and `.rodata` unchanged to the byte), but it is not byte-equivalence. Individual steps differ sharply — B4 and all of B6 are 0/15 byte-identical, B1 is 12/15, B2/B3/B5-step-4 are 15/15. **Correctness rests on golden comparison, which has not been run.**

One caveat before re-measuring: `ASCENDC_TPL_SEL` does not vary with the build's SoC, so **each kernel object contains both the A2 and the A3 instantiation** (20 `.ascend.meta.*` sections per object, 5 with the arch bit set). The same 15 objects come out of `--soc=ascend910b` and `--soc=ascend910_93`, so fingerprinting either package covers both platforms.

**Hard constraints to preserve** (violating these breaks the op binary/tiling ABI or silently changes results):
- The `__global__ mega_moe` signature and `GM_ADDR` argument list stay byte-identical.
- `MegaMoeTilingDataQuant`/`NonQuant` struct memory layout stays byte-identical.
- `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` output (the tiling key) is unchanged.
- The three `if constexpr (IS_A2_)` sites in `mega_moe_arch22.h` keep their two bodies **verbatim** — `epilogueCoreNum`/`epilogueGranularity` in the int8 arm (A2 uses `aivNum_/2` and a shorter curve; A3 uses `aivNum_` and a wider one), `layoutA1RowStride`, and the device-kernel/`Params` tail. A wrong `epilogueCoreNum` mis-partitions the epilogue silently.
- The `kRoutingIsQuant` and `isW4A8` predicates are **shared** by both platforms and count `int32_t`. That is safe only because A2 can never be instantiated with an `int32_t` weight, which `#if (ORIG_DTYPE_WEIGHT1 != DT_INT32)` in `mega_moe.cpp` enforces. **Do not convert that guard to `if constexpr`** — `if constexpr (isA2) { return; }` would still instantiate `MegaMoe<..., int32_t, ...>`.
- Master's newer arch22 machinery must survive intact: the chunked `MoePermutePrologue`, `recvRoundsMax`/`recvRoundBudget` receive rounds, `mask_buffer` rank masking, the route-table UB cache (`useUbRouteCache`) and the per-token-scale UB batch prefetch.

When editing arch22, verify against `{W4A8, INT8, BF16, FP16} × {A2, A3} × {Quant, NonQuant} × {transpose_w1=0/1} × {transpose_w2=0/1}` and golden-compare `yOut` + `expertTokenNumsOut` per dtype/arch. Compilation and the UT suites are necessary but nowhere near sufficient.

## Conventions

- Every source file carries the Huawei CANN copyright/license header — new files must include it (copy from any existing file). CI runs an OAT license check (`OAT.xml` at repo root).
- C++ formatting is `clang-format --style=file` (`.clang-format` at repo root); enforced by pre-commit and CI. Match surrounding style.
- External deps (from `op_host/CMakeLists.txt`): `mc2/common`, `mc2/3rd`, `mc2/moe_distribute_dispatch_v2`, `mc2/moe_distribute_dispatch_v3`. The op also relies on `mc2/common/op_kernel/moe_distribute_base.h` for the communication primitives.
