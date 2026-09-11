# AGENTS.md

## Build

**Prerequisites**: CANN SDK installed (provides `bisheng` compiler and ASCEND toolchain). Python 3.7+, GCC 7.3+, CMake 3.16+.

Before building, source the CANN environment:
```bash
source $ASCEND_HOME_PATH/bin/setenv.bash   # or $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
```

Clone submodules:
```bash
git submodule update --init   # 3rdparty/catlass
```

**Key build commands** (use `bash build.sh` for everything):

| Task | Command |
|------|---------|
| Build custom op package | `bash build.sh --pkg --soc=ascend910b --ops=op_name -j16` |
| Build experimental op | `bash build.sh --pkg --soc=ascend910b --ops=op_name -j16 --experimental` |
| Build all UT | `bash build.sh -u --soc=ascend910b` |
| Build single op UT | `bash build.sh --ophost_test --soc=ascend910b --ops=op_name` |
| Build opkernel UT | `bash build.sh --opkernel_test --soc=ascend910b --ops=op_name` |
| Compile UT without running | Add `--noexec` |
| Run single example | `bash build.sh --run_example op_name eager` |
| Generate new op scaffold | `bash build.sh --genop=class/op_name` |
| Clean all | `bash build.sh --make_clean` |
| Build specific lib | `bash build.sh --opapi` / `--ophost` / `--opgraph` / `--onnxplugin` |
| Install system deps | `bash install_deps.sh` |

SOC values: `ascend910b` (Atlas A2), `ascend910_93` (Atlas A3), `ascend950`, `ascend310p`, `kirinx90`, `kirin9030`, `mc62cm12a`.

**Multiple SOCs**: comma-separated: `--soc=ascend910b,ascend950`.

## Architecture

This is a **Huawei Ascend NPU operator library** for transformer-class models, part of the CANN stack.

- **`attention/`**, **`moe/`**, **`mc2/`**, **`gmm/`**, **`ffn/`**, **`mhc/`**, **`posembedding/`** — standard operator directories. Each op has its own subdirectory.
- **`experimental/`** — community-contributed operators. Requires `--experimental` flag to build.
- **`common/`** — shared host/kernel utilities (tiling helpers, buffer management, etc.).
- **`3rdparty/catlass`** — git submodule (customized ATLAS-like library for Ascend).
- **`torch_extension/`** — PyTorch bindings (build with `python3 -m build --wheel -n`).
- **`examples/`** — example ops and fast-kernel-launch demos.
- **`tests/`** — CI test config (`test_config.yaml` maps changed files to test scopes) and shared UT utilities.
- **`scripts/`** — CI scripts, op generator, utility tools.
- **`cmake/`** — all build infrastructure, third-party deps, custom build targets.

### Standard operator directory layout
```
op_name/
├── op_host/         # Host-side: op definition, tiling, infershape
│   ├── op_name_def.cpp
│   ├── op_name_tiling.cpp
│   └── CMakeLists.txt
├── op_kernel/       # Device-side: Ascend C/C kernel implementation
│   ├── op_name.cpp
│   ├── op_name.h
│   ├── arch22/arch32/arch35/arch38/  # SOC-specific kernel variants
│   └── op_name_tiling_data.h / op_name_tiling_key.h
├── op_api/          # Optional: API layer
├── op_graph/        # Optional: graph fusion support
├── tests/           # Unit tests
├── examples/        # Usage examples
├── CMakeLists.txt
└── README.md
```

### Minimal community operator (experimental/) layout
```
op_name/
├── op_name.cpp       # Kernel implementation (Ascend C)
├── tests/
│   └── test_op_name.py
├── CMakeLists.txt
└── README.md
```

## C++ conventions

- **Standard**: C++17 (`-std=c++17` for host, `-std=gnu++1z` for kernels)
- **Format**: `.clang-format` at repo root: 120 columns, 4-space indent, no tabs, pointer alignment right (`int *p`), braces on new line after functions, always break template declarations
- **Pre-commit**: `clang-format -i` + OAT compliance check (`scripts/oat_check.sh`)

## Testing

- GTest-based C++ unit tests. Test binaries are built via CMake and run through `build.sh -u`.
- Python-based tests exist for community/experimental operators.
- CI uses `tests/test_config.yaml` to determine which tests to run based on changed files.
- UT targets: `transformer_op_host_ut`, `transformer_op_api_ut`, `transformer_op_graph_ut`, `transformer_op_kernel_ut`.

## Versioning

Current version: `9.0.0` (in `version.cmake`). Depends on CANN packages >= 8.5.

**Use a tagged branch**, not `master`. Tags match CANN release versions (e.g. `9.0.0`). Using `master` with a mismatched CANN version will break at compile time.

## Gotchas

- Certain `--ops` selections auto-add dependent operators. For example: `moe_distribute_dispatch_v2` automatically includes `moe_distribute_dispatch_v3`; `distribute_barrier` includes `distribute_barrier_extend`.
- Tiling sink is disabled by default when using `--ops`, except for `fused_infer_attention_score` and `incre_flash_attention`.
- Build output goes to `build/` (intermediates) and `output/` (packages). Both are cleaned by `--make_clean`.
- The `3rdparty/catlass` submodule must be initialized separately; cloning alone is not enough.
