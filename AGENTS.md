# AGENTS.md

## Commit Message Convention

**MANDATORY**: Before every `git commit`, read `COMMIT_CONVENTION.md` and follow the format exactly. The commit-msg hook will reject non-conforming messages. Key points:
- Format: `<type>[optional scope]: <subject>`
- Types: feat | fix | test | revert | refactor | style | docs | perf | chore
- Chinese subject, no trailing period
- Test results go in footer as `Test:` section
- Add `Co-Authored-By:` footer when AI-generated

## Project Overview

CANN ops-transformer: Ascend NPU transformer operator library (C++17 kernels + Python tests). Operators run on Huawei Ascend AI chips via the CANN toolchain. Requires CANN >= 8.5 installed with `ASCEND_HOME_PATH` set.

## Build Commands

All builds go through `build.sh` (not raw `cmake`). Build artifacts land in `build/`, packages in `build_out/`.

```bash
# Build custom operator package (most common)
bash build.sh --pkg --soc=ascend910b --ops=<op_name> -j16

# Build all operators for a module
bash build.sh --pkg --soc=ascend910b --module=attention -j16

# Build specific library only
bash build.sh --ophost -j16        # ophost_transformer.so
bash build.sh --opapi -j16         # opapi_transformer.so
bash build.sh --opgraph -j16       # graph plugin
bash build.sh --onnxplugin -j16    # ONNX plugin

# Build binary kernel only
bash build.sh --opkernel --soc=ascend910b --ops=<op_name>

# Build JIT package (without kernel binary, for tiling-only changes)
bash build.sh --jit --soc=ascend910b --ops=<op_name> -j16

# Clean
bash build.sh --make_clean
```

**`--pkg` behavior**: Without `--ops`, `--pkg` builds a built-in package. With `--ops` or `--module`, it builds a custom vendor package.

### SOC Versions (chip targets)

| Flag | Hardware |
|------|----------|
| `ascend910b` | Atlas A2 (default) |
| `ascend910_93` | Atlas A3 |
| `ascend950` | Ascend 950PR/DT |
| `ascend310p` | Atlas Inference |
| `kirinx90` | KirinX90 |
| `kirin9030` | Kirin 9030 |
| `mc62` | MC62 |

## Testing

```bash
# Run all UTs
bash build.sh -u --soc=ascend910b

# Run UTs for specific operators
bash build.sh -u --ops=<op_name> --soc=ascend910b

# Compile UTs without executing
bash build.sh -u --noexec

# Specific UT categories
bash build.sh --ophost_test          # ophost UTs
bash build.sh --opapi_test           # opapi UTs
bash build.sh --opgraph_test         # opgraph UTs
bash build.sh --opkernel_test        # kernel UTs (requires --ops)

# Run example (eager mode, custom package)
bash build.sh --run_example <op_name> eager cust --vendor_name=custom

# Run example (graph mode)
bash build.sh --run_example <op_name> graph

# Run example with CANN Simulator (no real hardware needed, currently Ascend950 only)
bash build.sh --run_example <op_name> eager --simulator=camodel --soc=ascend950
```

Python tests under `<op>/tests/` use pytest. Test configuration for CI is in `tests/test_config.yaml`.

```bash
# Run Python tests for a specific operator directly
pytest <op>/tests/pytest/ -k "test_name"
```

## Operator Directory Structure

Standard operator layout (e.g. `attention/flash_attention_score/`):

```
<op_name>/
  op_host/           # Op definition, tiling, infershape (C++)
    <op>_def.cpp     # Op registration
    <op>_tiling.cpp  # Tiling strategy
  op_kernel/         # Kernel implementation (C++, AscendC)
    <op>.cpp         # Kernel entry
    <op>.h           # Kernel implementation
    arch22/          # ascend910b/910_93 specific
    arch35/          # ascend950 specific
    arch38/          # mc62 specific
  op_api/            # ACLNN API layer
  op_graph/          # Graph plugin layer
  tests/             # UT (C++) and pytest (Python)
  examples/          # test_aclnn_*.cpp / test_geir_*.cpp
  CMakeLists.txt     # Required, do not modify structure
  README.md          # Required for all operators
```

## Module Directories

Top-level modules: `attention/`, `moe/`, `mc2/`, `ffn/`, `mhc/`, `posembedding/`, `gmm/`, `mamba/`

- `common/` - Shared utilities, framework plugins (ONNX), common kernel headers
- `experimental/` - Community-contributed operators (mirrors module structure); use `--experimental` flag to build
- `examples/` - Reference operators (`add_example`) and `fast_kernel_launch_example` template
- `3rdparty/catlass` - Git submodule (run `git submodule update --init`)
- `torch_extension/` - PyTorch extension wrapper (built automatically with `--pkg`)

## Creating a New Operator

```bash
# Scaffold a new AscendC operator
bash build.sh --genop=<module>/<op_name>
# e.g. bash build.sh --genop=attention/my_new_attention

# Scaffold a new AI CPU operator
bash build.sh --genop_aicpu=<module>/<op_name>
```

Contributed operators go under `experimental/<module>/`. Standard operators require richer deliverables (op_host + op_kernel + UT).

## Code Style and Pre-commit

Pre-commit hooks run: trailing-whitespace, end-of-file-fixer, check-yaml, clang-format (C++), ruff-check + ruff-format (Python), codespell, OAT compliance.

```bash
pre-commit run --all-files
```

- C++: `.clang-format` at root (ColumnLimit: 120, IndentWidth: 4, PointerAlignment: Right, BreakBeforeBraces: Custom with AfterFunction: true)
- Python: ruff for linting and formatting
- Codespell skips `.py`, `.cpp`, `.hpp`, `.c`, `.h`, `.md` files; domain terms (CANN, ASCEND, ND, etc.) are allowlisted

## Key Conventions

- `build.sh` is the single entry point; do not invoke `cmake` directly from the repo root
- The `--ops` flag uses snake_case operator names, comma-separated for multiple
- The `--module` flag auto-includes `gmm` when `mc2` is specified
- `ENABLE_EXPERIMENTAL` flag (`--experimental`) is required to compile operators under `experimental/`
- Build output `.run` packages install to `${ASCEND_HOME_PATH}/opp/vendors/<vendor>_transformer/`
- Default vendor name is `custom`; override with `--vendor_name=<name>`
- CMake option `ENABLE_CCACHE` is ON by default; ccache is auto-detected
- Kernel files may have arch-specific subdirectories (`arch22/`, `arch35/`, `arch38/`) for chip-specific implementations

## Multi-Environment Development

**Build environment**: A2 (Atlas A2, Ascend910B) — used for code editing and compilation.
**Test environment**: A5 (Ascend 950) — used for running tests on actual hardware.

This repo involves cross-compilation: building on A2 and testing on A5. Key gotchas:

- **Always verify the target environment** before debugging. Check `npu-smi info` to confirm chip type (Ascend950PR vs Ascend950DT) and CANN version (`cat $ASCEND_HOME_PATH/version.info`).
- **Custom operator packages must be installed** on the test machine. If plog shows kernel loading from `built-in` instead of `vendors/custom_transformer`, the custom package isn't installed or isn't taking precedence.
- **CANN built-in kernels override vendor kernels with the same hash**. When CANN has both `built-in` and `vendors/custom_transformer` kernels with identical filenames, it loads the built-in version. Fix: delete the conflicting built-in `.o` and `.json` files from `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/<soc>/<op_dir>/`. Back them up first. This is a one-time fix per CANN install.
- **Kernel cache can cause stale binaries**. After rebuilding, if behavior doesn't change, clear: `rm -rf build/` and check for `$HOME/atc_data/kernel_cache/` or similar cache directories.
- **NPU device state persists across crashes**. A kernel crash (error 507015 = aicore timeout) can leave the device in a bad state, causing subsequent tests to fail. Reset with `npu-smi set -t reset -i <device_id>` or wait for device recovery.

## Debugging Kernel Issues

### DumpTensor and PRINTF

Both are available for kernel-side debugging but have significant limitations:

- **DumpTensor**: Dumps tensor data from UB/L0C/GM (not L1 on Ascend 950). Outputs to plog or requires `show_kernel_debug_data` tool to parse binary dumps.
- **PRINTF/printf**: Prints scalar values. Output goes to plog or stdout depending on execution mode.
- **Critical limitation**: Adding debug instrumentation to kernels with tight AIC/AIV cross-core synchronization can cause kernel crashes (507015 timeout). Even lightweight PRINTF has been observed to break timing-sensitive kernels.
- **Enable with compile flag**: `OPS_COMPILE_OPTIONS="-DDEBUG_DUMP" bash build.sh --pkg ...`
- **Disable at runtime**: `export ASCENDC_DUMP=0`

### Analyzing plog

When kernels crash, check plog for the actual kernel binary being loaded:

```bash
grep "binPath.*<OpName>" $HOME/ascend/log/debug/plog/plog-*.log | head -5
```

If path shows `built-in` instead of `vendors/custom_transformer`, the custom operator package isn't installed or isn't being used.

### Common Error Codes

- **507015**: aicore timeout - kernel hung or crashed, often due to synchronization issues or invalid memory access
- **507014**: aicore error - typically memory access violation or illegal instruction

## Testing Quirks

### Device Selection

Python tests default to device 0. To use a different device:

```python
# In test paramset or conftest.py
params["device_id"] = 3  # Use NPU device 3
```

Or set environment variable before running tests:

```bash
ASCEND_DEVICE_ID=3 pytest ...
```

### Test Isolation

- Tests that crash the kernel can poison subsequent tests in the same session. Run failing tests in isolation: `pytest -k "specific_test_name"`
- After a crash, reset the NPU device or wait for recovery before running more tests.
- Some tests use `ThreadPoolExecutor` with `max_workers=1`, which can mask the actual exception. Check the full traceback carefully.

### Pytest Configuration

Each operator has its own `pytest.ini` under `<op>/tests/pytest/`. The `conftest.py` files provide fixtures like `golden_mode` and `cache_dir`. Test parameters are defined in `*_paramset_*.py` files.

## Active Debugging: D256 MxFP8 Prefill

**Baseline**: master `6ec9e1bd6` achieves **25% pass rate**. Extended branch `21a3ac09a` (unmerged, 53 commits since feature, 12 reverts) is the latest work.

**Test case**: `TND_B1_QS128_KVS128_Nq8_Nkv8_D256_SP0` in `attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/`

**Current status**: V1 bug fixed (x_exp_1=32), pass rate 25.18%, T=0..31 correct. Remaining T=32 boundary bug in C2 (FixpipeMm2).

**Error pattern** (after V1 fix):
- T=0..31: All correct (65536 elements, first_fail_idx=65536)
- T=32..127: All wrong (axis0 range=[32,127], 96/128 errors)
- All 8 heads affected, all 256 D dimensions affected

### Bisection Results (DumpTensor + sentinel experiment)

**Method**: Used DumpTensor (desc=10001/20001/30001) at stage boundaries + sentinel fill (Duplicate<half>(1000.0)) to detect unwritten UB positions.

**Stage boundaries**:
```
C1(AIC) → [BMM1结果 UB] → V1(AIV) → [P数据 UB] → C2(AIC) → [BMM2结果 UB] → V2(AIV) → [输出]
           desc=20001❌        desc=10001✅       desc=30001✅
           (float没dump出)    (fp8,8192B)       (float,256B)
```

**V1 bug (FIXED)**: VF softmax StoreAlign wrote P data to only half the NZ positions (alternating 32-element blocks: 32 non-zero + 32 zero per 128-element row). Root cause: `x_exp_1 = x_exp + (ubN * 4)` should be `x_exp + 32` for MxFP8 fp8 path. The second StoreAlign wrote to position 512 instead of position 32, leaving odd 32-blocks as zeros. BMM2 read these zeros, corrupting output.

**Fix**: `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` lines 880, 1285 — MxFP8 NoUpdate/Update functions, fp8 branch: `x_exp_1 = x_exp + 32` (was `x_exp + (ubN * 4)`). Commit `6dd2100ad`.

**Note**: The extended branch `21a3ac09a` had a "revert" commit (`ef4424a03`) intended to change x_exp_1 back to 32, but the diff was in the wrong direction (re-applied ubN*4 instead of reverting to 32). The fix was never actually applied until now.

**C2 bug (REMAINING)**: T=32 boundary error. P data confirmed correct (5792/8192 values match golden P after fp8 roundtrip, remaining 2400 zeros are NZ padding not read by DataCopy). Bug is in FixpipeMm2 with `dualDstCtl=1` (M-split) where cube-side `mSize=mBaseSize` (T-independent) vs VEC-side `actVecMSize` (T-dependent) mismatch at T=32.

### Key D=128 vs D=256 tiling differences

| Parameter | D=128 (works) | D=256 (broken) |
|-----------|--------------|----------------|
| sInnerFactor_ | SINNER_512 | SINNER_256 |
| s2BaseSize | 512 | 256 |
| s2BaseSizeCur (ubN) | 256 | 128 |
| s2SplitSize | 256 | 128 |
| blockStride (StoreAlign) | 65 | 33 |
| nSize (FixpipeMm2) | 128 | 256 |
| mBaseSize | 128 | 128 |
| srcStride (FixpipeMm2) | 128 | 128 |

D=128 works because `nSize == mBaseSize == 128` (coincidence). D=256 has `nSize=256 ≠ mBaseSize=128`.

### All failed fix attempts (all reverted unless noted)

| # | Attempt | Result | Commit |
|---|---------|--------|--------|
| 1 | splitD mode | L0C large enough, not needed | `cf9f28019`→`27f18af48` |
| 2 | BMM2 DN N-split | Back to K-split | `d17b29f50`→`27f18af48` |
| 3 | PScale dual copy stride | Partially kept | `2f94fb881` |
| 4 | P data UB→L1 srcOffset hardcoded 65 | No effect (65 = mBaseSize/2+1, M-derived, correct) | `0a1a5046c`→`f5135c65a` |
| 5 | P store second group offset x_exp+(ubN/8) | Crash | `77c9e0a28`→`93de962ae` |
| 6 | FixpipeMm2 srcStride mBaseSize→nSize | Worse (25%→1.66%) | `30ac6eee8`→`fed381273` |
| 7 | BMM2 M-half split (tried twice!) | Worse (25%→1%) | `86f1754a0`→`5380aca4e`, `0a6def89e`→reverted |
| 8 | BMM2 ndNum=2 N-split | Worse (25%→1%) | `41a5fb07d`→`eec05cc07` |
| 9 | D-gate 止血 (disable DN for D256) | Host/kernel mismatch, D256 all 0 | `8f9f9d83c`→`e8309ca24` |
| 10 | T=32 dump instrumentation | PRINTF/DumpTensor caused AICore error | `f5999f1b4`→`c1ecf82c7` |
| 11 | x_exp_1 ubN*4→32 (MxFP8 path) | **FIXED V1, 25%→25.18%** (kept) | `6dd2100ad` |
| 12 | srcOffset 65→vec1BlockStride | Worse (25%→23.27%, 65 is M-derived not N-derived) | `0522bccac`→`5f212e89e` |

### Debug instrumentation (currently in code)

DumpTensor probes in `fia_block_vec_fullquant_mx.h`:
- `desc=10001`: P data (stage1CastTensor, fp8, 8192B) — after V1 softmax, before UB→L1 copy
- `desc=20001`: BMM1 result (mmRes, float) — does not work (float type not dumped)
- `desc=30001`: BMM2 result (mmRes in ProcessVec2OnUb, float, 256B) — after C2 fixpipe

Enable: `export ASCENDC_DUMP=1` before pytest. Probes are unconditional (no compile flag needed).

### Key files

- `attention/common/op_kernel/arch35/fia_block_cube_fullquant_mx.h` — IterateBmm2Dn, FixpipeMm2 (C2 bug location)
- `attention/common/op_kernel/arch35/fia_block_vec_fullquant_mx.h` — ProcessVec1Dn (V1 fixed), ProcessVec2OnUb (V2), P copy UB→L1, DumpTensor probes
- `attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h` — VF softmax, x_exp_1 offset (V1 fix), StoreAlign write pattern
- `attention/common/op_kernel/arch35/fia_template_dispatcher.h` — Template dispatch, useDn, bmm2Write2Ub, D-gate location
- `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp:1246` — `sInnerFactor_ = (vHeadDim==DSIZE_256) ? SINNER_256 : SINNER_512`

### Next steps to try

1. **D-gate 止血** (retry #9 with host+kernel fix) — Disable DN for D=256 in both `CheckEnableDN` (host) and `UpdateTilingKeyQuantMode` (kernel tiling key). Non-DN path is verified correct.
2. **CANN Simulator** — Simulate L0C/UB data flow without code instrumentation
3. **GQA path comparison** — GQA path doesn't force `bmm2Write2Ub=true` for D=256 (`fia_template_dispatcher.h:155-161`), bypassing the dualDstCtl issue. Study why MX path forces it (`fia_template_dispatcher.h:88-94`).
4. **Consult Huawei support** for FixpipeMm2 `dualDstCtl=1` behavior with nSize≠mBaseSize

## CANNBot Skills

Specialized debugging skills are installed in `.agents/skills/`. Load them with the `skill` tool when relevant:

- `ascendc-precision-debug` - Precision issue diagnosis (pipeline sync, DataCopy alignment, FP16/BF16)
- `ascendc-crash-debug` - Kernel hang/crash debugging (plog parsing, mssanitizer)
- `ascendc-regbase-best-practice` - DAV_3510 RegBase operator API constraints
- `npu-arch` - Chip model to architecture mapping, feature support matrix
