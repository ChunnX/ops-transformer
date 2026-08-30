# MoeTutelDispatch ST design

This directory contains generated and contract-specific test assets for the
Ascend950 GEIR-only implementation. The operator has no ACLNN/E2E API.

## Assets

- `design/03_参数定义.yaml` through `design/07_因子值.csv`: parameter,
  factor, constraint, solver and frozen factor-pool assets.
- `design/MoeTutelDispatch_L0_design_source.csv`: 15-row API-shaped design
  source retained for traceability; do not pass it to TTK.
- `design/materialize_ttk_cases.py`: deterministic conversion to standard
  `op_name/input_shapes/output_shapes` Kernel/GEIR CSVs and the harness manifest.
- `testcases/MoeTutelDispatch_L0_test_cases.csv`: 15 distinct executable cases,
  97/97 factor values covered.
- `testcases/MoeTutelDispatch_L1_test_cases.csv`: 500 distinct,
  constraint-valid executable cases.
- `testcases/MoeTutelDispatch_required_{kernel,geir}.csv`: 20 Kernel and 15
  GEIR mode-specific materialized cases.
- `testcases/MoeTutelDispatch_L2_test_cases.csv`: 25 Host/GEIR negative cases
  using only the four spec error categories.
- `testcases/MoeTutelDispatch_required_test_cases.csv`: 38 mandatory P0,
  bitwise, guard, long-H, concurrency, performance and test-infra scenarios.
- `testcases/MoeTutelDispatch_contract_harness_manifest.csv` plus
  `run_contract_harness.py`: deterministic target for every required/L2 row;
  `MTD_REQ_033` uses GEIR `--compile-only` because its caller E is intentionally
  unknown and cannot be inferred from the operator inputs.
- `design/prepare_manual_data.py`: writes exact raw-bit route/gate/payload inputs
  and positive-zero goldens outside the repository for the 35 directed TTK cases.

## Reproduce

Run these commands from this directory:

```bash
PYTHONHASHSEED=0 python3 "${ASCENDC_ST_DESIGN_DIR:?set ASCENDC_ST_DESIGN_DIR}/scripts/generate_factor_values.py" \
  design/06_求解配置.yaml design/05_约束定义.yaml design/04_测试因子.yaml \
  design/07_因子值.csv --max-cases 10000
python3 design/postprocess_l1_cases.py design/07_因子值.csv \
  testcases/MoeTutelDispatch_L1_test_cases.csv \
  testcases/MoeTutelDispatch_L1_coverage_report.yaml --target 500 --seed 42
python3 design/materialize_ttk_cases.py --check
python3 run_contract_harness.py
```

`ASCENDC_ST_DESIGN_DIR` must point to the installed `ascendc-st-design`
tool directory.

The factor pool has 10,000 rows / 6,804 distinct full rows. Its frozen SHA256
is `b4d12638b707d6fdb1c46229b860773e456ddcd6e2132681a3622e7f86812c71`.
`PYTHONHASHSEED=0` is required because the shared generator derives local seeds
from Python hashes.

## TTK validation and execution

`fixtures/validation_opp` is a signature-only fixture for pre-package
`--validate`; it is not deployment or execution evidence. With that fixture,
L0 is 15/15, L1 is 500/500, required Kernel is 20/20 and required GEIR is
12/12 valid in both applicable parser paths.

Real CP4/CP6 execution must use the built/deployed package and explicitly lock
bitwise comparison. Prepare the exact data once, then let the harness honor each
case's repeat, dynamic and parallelism fields:

```bash
python3 design/prepare_manual_data.py --ttk-root <ops-test-kit> --output-dir <artifact-dir>
python3 run_contract_harness.py --execute --ttk-root <ops-test-kit> \
  --manual-data-dir <artifact-dir> --result-dir <result-dir> \
  --host-ut-binary <host-ut-binary> \
  --build-scan <build-scan> --static-contract <static-contract>
```

The harness writes one result CSV per TTK invocation outside the repository and
checks `precision_status`; the dynamic GE-build entry additionally requires
`precision=BUILD_ONLY`, while Kernel entries require
`bin_compile_s=BINARY_MATCH` and `bin_precision=100%`. This is required because
TTK can return process status zero even when a GEIR case records an execution
error in its result CSV.

The complete rationale and gates are in
`deliverables/MoeTutelDispatch/TEST.md` at the workspace root.
