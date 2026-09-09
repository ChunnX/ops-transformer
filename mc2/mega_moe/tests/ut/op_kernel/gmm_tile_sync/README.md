# GMM hardware notification protocol test

Run from the repository root with a C++17 host compiler and CMake:

```bash
cmake -S mc2/mega_moe/tests/ut/op_kernel/gmm_tile_sync -B /tmp/mega-moe-gmm-sync-test
cmake --build /tmp/mega-moe-gmm-sync-test -j2
ctest --test-dir /tmp/mega-moe-gmm-sync-test --output-on-failure
```

The test exercises both `Gmm1SwigluSync` and `Gmm2CombineSync` from the production `mega_moe_gmm_epilogue_sync.h` with a minimal `kernel_operator.h` stub. Stubs validate exact mode, pipeline and event ID; condition variables model independent Ready/ACK counters with bounded waits. Assertions remain enabled in release builds.

All 98 schedules must complete with matched notifications, counters at zero and no counter exceeding 15. One test per AIV0/AIV1 mapping accumulates 15 notifications on each channel before any consumer runs, then consumes GMM2 first. This checks isolation and independent limits; it does not suggest changing the kernel's stage order.

Real Ascend950 tests are still required for pipeline ordering, memory visibility, scheduler agreement and numerical correctness.
