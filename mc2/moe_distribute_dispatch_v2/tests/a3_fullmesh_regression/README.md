# A3 dispatch/combine fullmesh regression

This suite uses the Stage v2 native plugins in `ep_test_mega_moe_T`. It does not
replace its NPU preflight, HCCL lifecycle, CPU Golden, or precision thresholds.

## Reproduce

1. Build the original and modified revisions separately for `ascend910_93`, using
   different vendor names and installation directories:

   ```bash
   bash build.sh --pkg --soc=ascend910_93 \
     --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2 \
     --vendor_name=<vendor> -j16
   ```

2. Copy the test framework to an isolated directory. Apply
   `ep_test_current_abi.patch` there. The patch adapts the old fullmesh expert
   offsets to the current `(source_rank, token, topk)` triples, checks the entire
   valid triple range and dispatched weights, preserves requested communication
   quantization, and supplies a CPU reference for A3 block INT8 communication.
   No precision tolerance is increased.

   The patch synchronizes ranks between repeated accuracy calls. This bounds
   rank skew while exercising both communication windows. Performance calls
   use the separate profiler path and do not include these accuracy barriers.
   Unsynchronized sparse repetition is a separate stress condition.

3. Generate the case tables with `python generate_cases.py --output <cases>`.
   Use one local node with 16 ranks, PTA V2, a 2048 MiB HCCL buffer, and an unused
   communication port. When the default device listener is still occupied,
   use a private `HCCL_NPU_SOCKET_PORT_RANGE` in the local configuration; see
   [CANN's port configuration reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md).
   Fullmesh tests must not inherit hierarchy-selection
   environment variables. Keep the environment configuration outside Git.

4. For each variant, select only its installed vendor in
   `ASCEND_CUSTOM_OPP_PATH`, source that vendor's `bin/set_env.bash`, and give it
   a separate `ASCEND_CACHE_PATH`. Use an isolated OPP overlay: symlink the
   toolkit's non-vendor resources, create a private `vendors` directory, and
   set `ASCEND_OPP_PATH` to that overlay. Do not modify system installations.
   Verify that the package contains device `.o` files, the selected kernel path
   belongs to the expected vendor, and its hash matches the built artifact.
   A `--jit` package alone can fall back to a system binary and is insufficient
   for this ACLNN regression. Set `EP_TEST_PROVENANCE_PATH` to retain loaded
   library records. Use the framework's normal node entry:

   ```bash
   python <framework>/core -o dispatch -c <config> -m <local-ip> \
     -f <cases>/dispatch_accuracy.xlsx
   python <framework>/core -o combine -c <config> -m <local-ip> \
     -f <cases>/combine_accuracy.xlsx
   python <framework>/core -o both -c <config> -m <local-ip> \
     -f <cases>/both_accuracy.xlsx
   ```

5. Run `dispatchperformance.xlsx` and `combineperformance.xlsx` with the same
   operation names. Performance cases use the framework profiler and 100 loops;
   compilation and setup must not be reported as operator latency. Compare
   identical case names, inputs/seeds and collection settings. Retain per-rank
   raw results, structured run summaries, source hashes and vendor provenance.
   Recheck the two larger non-quantized combine shapes using
   `combine_nonquant_recheck.xlsx` on both variants. Keep the same repeat count
   per shape and variant. The comparison averages all completed, accuracy-passed
   runs in each input directory; keep superseded binaries' runs elsewhere.

   ```bash
   python compare_performance.py --baseline <baseline-results> \
     --optimized <optimized-results> --cases <cases>/cases.json \
     --output <comparison-results> --keep 50
   ```

## Coverage

- 1/8/16/32/64/128/256 input tokens per rank; H=3584/7168; K=1/8/16.
- BF16/FP16; dispatch per-token INT8 including smooth scales; combine INT8.
- Uniform expert routing for the principal performance shapes, random and
  concentrated routing for correctness; 16/256 experts.
- Partial/empty token masks, expert masks, both count formats, performance info.
- Ten consecutive executions per accuracy case exercise window reuse.
- Dispatch/combine round trips exercise the public metadata handoff.

The smallest `force` cases need not occupy every expert; do not describe them
as perfectly balanced across all 256 experts. Single-node results do not imply
EP64/EP128 or A5 performance.
