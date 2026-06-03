# AGENTS.md - ops-transformer Repository Guide

CANN operator library for transformer-class LLM computations on Huawei Ascend NPU. Contains attention, MoE, mc2, FFN, and grouped matmul operators.

## Repository Structure

```
ops-transformer/
├── attention/          # 50+ attention variants (FA, MLA, lightning indexer)
├── moe/                # MoE routing/finalization
├── mc2/                # Communication-computation fusion (requires HCCL)
├── experimental/       # User contributions (REQUIRED for new operators)
├── examples/           # Reference: add_example, fast_kernel_launch_example
├── torch_extension/    # PyTorch NPU extension (npu_ops_transformer)
├── tests/              # pytest + UT framework
├── docs/               # Chinese docs (QUICKSTART.md is entrypoint)
├── scripts/opgen/      # Operator skeleton generator
└── build.sh            # Main build entrypoint
```

## Critical Commands

### Build Operators

```bash
# Build specific operator package (most common)
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule -j16

# Build experimental operators
bash build.sh --pkg --experimental --soc=ascend910b --ops=<op_name> -j16

# Build all built-in operators
bash build.sh --pkg --soc=ascend910b -j16

# Build JIT package (without kernel binaries, for graph mode)
bash build.sh --jit --pkg --soc=ascend910b

# Build static library
bash build.sh --pkg --static --soc=ascend910b -j16
```

### SoC Version Mapping (CRITICAL - agents will guess wrong)

| Hardware | `--soc=` value | Architecture |
|----------|---------------|--------------|
| Atlas A2 Training/Inference | `ascend910b` | arch32 (dav-2201) |
| Atlas A3 Training/Inference | `ascend910_93` | arch32 |
| Ascend 950PR/950DT/KirinX90 | `ascend950` | arch35 |
| Atlas Inference (310P) | `ascend310p` | arch20 |

### Run Operator Examples

```bash
# Eager mode (aclnn direct call)
bash build.sh --run_example chunk_gated_delta_rule eager cust --vendor_name=custom

# Graph mode (GE IR)
bash build.sh --run_example chunk_gated_delta_rule graph

# With simulator (ascend950 only)
bash build.sh --run_example <op_name> eager --soc=ascend950 --simulator=camodel
```

### Unit Tests

```bash
# Build and run all UT
bash build.sh -u

# Build specific UT targets
bash build.sh --ophost_test --noexec    # Build ophost UT only
bash build.sh --opapi_test --noexec     # Build opapi UT only
bash build.sh --opkernel_test --noexec  # Build kernel UT only

# With coverage
bash build.sh -u --cov --noexec
```

### Generate Operator Skeleton

```bash
# Create standard operator structure under experimental/
bash build.sh --genop=attention/my_new_attention

# Create AICPU operator structure
bash build.sh --genop_aicpu=experimental/my_aicpu_op
```

## Operator Directory Layout

### Standard Operator (full delivery)

```
${op_class}/${op_name}/
├── op_host/                        # Tiling + OpDef + InferShape
│   ├── ${op_name}_def.cpp          # Operator prototype definition
│   ├── ${op_name}_tiling.cpp       # Tiling strategy implementation
│   ├── ${op_name}_infershape.cpp   # Shape inference (optional)
│   └── op_api/
│       ├── aclnn_${op_name}.cpp    # aclnn C API implementation
│       └── ${op_name}.cpp          # OpAPI wrapper
├── op_kernel/                      # Ascend C Kernel
│   ├── ${op_name}.cpp              # Kernel entrypoint
│   ├── ${op_name}.h                # Kernel implementation
│   ├── ${op_name}_tiling_data.h    # TilingData struct
│   └── ${op_name}_tiling_key.h     # TilingKey enum
├── op_graph/                       # Graph mode proto (optional)
├── tests/
│   ├── ut/op_host/                 # Host-side UT
│   ├── ut/op_kernel/               # Kernel UT
│   └── pytest/                     # PyTorch API tests
├── examples/                       # Usage examples (test_aclnn_*.cpp)
├── CMakeLists.txt
└── README.md
```

### Simple Operator (fast_kernel_launch)

Single-file approach via PyTorch Extension. See `examples/fast_kernel_launch_example/README.md`.

## Testing Conventions

### pytest Tests (PyTorch API)

Located in `${op}/tests/pytest/`. Structure:

```
tests/pytest/
├── test_run.sh                     # Execution script
├── ${op}_golden.py                  # CPU golden implementation
├── test_${op}_single.py             # Single case test main
├── ${op}_operator_single.py         # NPU invocation wrapper
└── test_${op}_paramset.py           # Parameter configuration
```

Run: `bash test_run.sh single` (after configuring paramset)

### UT Tests (C++)

Located in `${op}/tests/ut/`. Compile with `bash build.sh --ophost_test` or `--opkernel_test`.

## Pre-commit Hooks

- **clang-format**: C++/C code formatting (`.clang-format` config present)
- **OAT check**: Open source compliance audit

Before commit, ensure code passes pre-commit checks.

## Dependencies

Required environment variables (set by CANN installation):

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/latest  # or ~/Ascend/latest
ASCEND_OPP_PATH=${ASCEND_HOME_PATH}/opp
ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
```

Install deps: `bash install_deps.sh` (Python 3.7+, GCC 7.3+, CMake 3.16+)

## Key Reference Documents

- **Quick Start**: `docs/QUICKSTART.md` - End-to-end walkthrough with AddExample
- **Build Options**: `docs/zh/install/build.md` - Full build.sh parameter reference
- **Operator Development**: `docs/zh/develop/aicore_develop_guide.md` - Standard operator workflow
- **Fast Kernel Launch**: `examples/fast_kernel_launch_example/README.md` - PyTorch Extension approach
- **Contribution Guide**: `CONTRIBUTING.md` - PR requirements, CI gates, deliverables

## Common Gotchas

1. **Source/CANN version mismatch**: Use matching branch tags from [release-management](https://gitcode.com/cann/release-management). Master branch may be incompatible with installed CANN.

2. **Custom package installation**: After building, run `.run` file installs to `${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/`. Must set:
   ```bash
   export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
   ```

3. **pytest requires torch_npu**: Install compatible torch_npu from [pytorch releases](https://gitcode.com/Ascend/pytorch/releases).

4. **Experimental operators**: User contributions must go to `experimental/${op_class}/`, not main directories.

5. **mc2 operators require HCCL**: Linked with `-lhccl -lhccl_fwk` for distributed communication.