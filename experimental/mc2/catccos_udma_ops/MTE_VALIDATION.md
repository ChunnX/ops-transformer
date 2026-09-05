# MTE validation record

Local verification on 2026-09-05 used Windows and Python 3.13.2. This environment
has no CANN compiler, torch/torch_npu or NPU. The results below are CPU and source
checks; synthetic profiler fixtures are never device performance measurements.

## Verified locally

- **163 tests and 63 subtests passed** in the selected MTE and existing regression
  suite. This covers operator signatures, backend packaging and loading,
  lifecycle/launch lock ordering, dependency checks, worker partitioning,
  padded-row addressing, peer completion, persistent barrier generations,
  case validation and profiler parsing.
- Tuning integration tests run real mock subprocesses that produce accuracy
  artifacts and profiler CSVs. They cover ranking, rejected/extra launches,
  UDMA/probe exclusion, resume, retries, replay, independent top-K/stress
  processes, interrupted-attempt isolation and timeout cleanup.
- The five a2avgmm and three gmma2av fixed TilingProfile records match
  commit 17adcfad8949b1aa9f29ced0e685f3dc25831f28 field for field. No legacy
  NPU/Data split dimension is searched.
- a2avgmm retains the baseline GMM loop and export remapping. Its generic
  non-aligned K path now uses the communication writer's aligned GM stride
  on AIC and in the host workspace bound; fixed K=2048 cases keep their
  original layout. gmma2av retains the baseline target-order AIC producer.
- All directly included CatCCOS/CATLASS headers resolve against the pinned
  dependency commits. SHMEM APIs and bootstrap declarations were checked
  against the pinned source.
- Both Bash build scripts pass syntax checks. The selected CMake dependency
  and dynamic-symbol-check tests pass. Git whitespace checks pass.
- Planning the six modes in mte_frozen_profiles.yaml yields **996 legal
  candidates** with the declared 32 AIC / 256 KiB UB platform and the default
  search space. This is enumeration, not a measured best configuration.

From experimental/mc2/catccos_udma_ops, the CPU regression command is:

~~~bash
python -m pytest \
  tests/test_mte*.py \
  tests/test_production_build.py tests/test_production_sync.py \
  tests/test_workspace_layout.py tests/test_runtime.py \
  tests/test_direct_input_path.py tests/test_dependency_versions.py \
  tests/test_case_schema.py tests/test_msprof_parser.py \
  tests/test_all_to_allv_grouped_mat_mul_tiling.py \
  tests/test_validation_export_default.py -q -p no:cacheprovider
~~~

## Target-host work still required

1. Build and install the MTE wheel with the pinned CANN-compatible dependencies.
   Configuration verifies the actual SHMEM bootstrap symbol; the local test of
   this gate used controlled symbol-table fixtures.
2. Check the generated device binaries and UB allocation, then execute the
   initialization probe and both operators on all participating peers.
3. Run BF16 accuracy for smoke cases, the six fixed business modes, optional
   export and output reuse. Device-only routes, uneven/empty routes, repeated
   launches and stream changes also need device regression coverage.
4. Generate a plan for the actual queried platform, run search and independent
   retests, and compare with UDMA in separate processes using equal cases and
   profiler settings. Native compile success, device correctness, memory
   safety and a performance optimum have **not** been established locally.

Build/API instructions are in [MTE.md](MTE.md). Search commands and artifact
definitions are in [the tuning guide](ops_transformer_catccos_mte/tuning/README.md).
