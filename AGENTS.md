# ops-transformer - CANN Ascend C Operator Library

Huawei Ascend AI processor transformer operators (attention, moe, mc2, gmm, ffn, posembedding).

## Key Commands

```bash
# Build (default: ascend910b)
bash build.sh --pkg --soc=ascend910b -j8
bash build.sh --pkg --ops=op1,op2 --soc=ascend910b    # Specific operators
bash build.sh --pkg --experimental --soc=ascend910b  # Experimental operators
bash build.sh --make_clean                           # Clean

# Tests
bash build.sh -u                                     # All UT
bash build.sh --ophost_test --noexec --cov           # ophost UT with coverage
bash build.sh --run_example flash_attention_score eager  # Run example

# Generate operator template
bash build.sh --genop=attention/my_new_op            # Standard AiCore
bash build.sh --genop_aicpu=attention/my_aicpu_op    # AI CPU
```

## SoC → Arch Mapping

| SoC | Hardware | Arch |
|-----|----------|------|
| ascend910b | Atlas A2 | arch32 |
| ascend910_93 | Atlas A3 | arch32 |
| ascend950 | Ascend 950PR/DT | arch35 |
| ascend310p | Atlas Inference | arch20 |
| mc62cm12a | MC2 | arch38 |

## Directory Structure

- `attention/`, `moe/`, `mc2/`, `gmm/`, `ffn/`, `posembedding/`, `mhc/` - Production operators
- `experimental/` - New operator contributions (submit PRs here)
- `examples/fast_kernel_launch_example/` - Minimal operator template (`<<<>>>` syntax)

Standard operator:
```
${op_class}/${op_name}/
├── op_host/              # Tiling + definition
├── op_kernel/            # Kernel implementation
│   └── arch32/arch35/    # SoC-specific kernels
├── op_api/               # aclnn API
├── examples/             # Test examples
├── tests/ut/             # UT tests
└── README.md
```

## Pre-commit Hooks

- `clang-format -i` on C/C++ files
- OAT compliance check via `scripts/oat_check.sh`

## CI Triggers

`tests/test_config.yaml` defines:
- `src`: paths to watch for changes
- `options`: operators to test when src changes
- `test.ut/example`: enable/disable tests

Trigger CI by commenting `compile` on PR.

## Environment

- `ASCEND_HOME_PATH` must be set (CANN Toolkit)
- `bisheng` compiler must be available

## Ascend C API Rules

- **Allowed**: Basic vector APIs (Add/Mul/Sub/Div/Exp/Log/ReduceSum/ReduceMax/Cast)
- **Forbidden**: High-level APIs (Softmax/LayerNorm/BatchNorm) - implement manually

## Contribution Flow

1. Issue → 2. Sig assigns `experimental/${op_class}` path → 3. PR → 4. Comment `compile` → 5. CI pass → 6. Review

Docs: [QuickStart](docs/QUICKSTART.md), [Dev Guide](docs/zh/develop/aicore_develop_guide.md), [CONTRIBUTING.md](CONTRIBUTING.md)