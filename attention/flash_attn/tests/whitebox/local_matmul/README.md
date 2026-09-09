# FlashAttn local matmul whitebox checks

Run from the repository root with a C++17 host compiler and Python 3:

```bash
python attention/flash_attn/tests/whitebox/local_matmul/run.py --mutations
```

The test compiles the actual `op_kernel/arch35/flash_attn_matmul.h` against a
host-only AscendC instruction recorder. It does not implement floating-point
Mmad or emulate device pipeline execution. The mock include directory must never
be added to device builds. No CANN installation or NPU is needed for this check.

## Contracts checked

- FP16 and BF16; Full, K-split and N-split; the three transpose combinations
  used by ND/DN BMM1/BMM2.
- M = 1, 2, 15, 16, 17, 31, 32, 63, 64, 127, 128;
  S2 = 1, 15, 16, 17, 127, 128, 129, 255, 256;
  D = 64, 72, 128, 192, 256.
- L1 NZ source byte offsets, LoadData fractal counts, source/destination strides,
  transpose flags, L0 slot addresses and L0C result offsets.
- M=1 workaround; actual M versus stored M; actual K/N versus fractal-aligned
  loads; first K block clears C and subsequent blocks accumulate.
- All initial A/B index parities followed by persistent state across alternating
  BMM calls; double buffering rotates, single L0B always uses slot 0/mutex 8,
  including an intentionally stale initial B index.
- Exact mutex ordering: Full uses both forward barriers; split paths use B's
  forward barrier for both loads; N splitting holds A until all N blocks finish.
  Slot ownership is released at the end of each call.

The matrix includes primitive-level cases beyond the host's currently selected
combinations. It does not establish coverage of the host tiling dispatcher or
of all device kernel template branches.

Expected result: **20,160 instruction-contract cases PASS**. With `--mutations`,
five temporary faulty header copies must each fail the contract check: A stops
rotating, single B uses the stale index, K continuation clears C, N results
overlap, and the forward B barrier disappears. Production files are never
modified by mutation checks.

## Complementary device regression

Select an upstream operator package, then the candidate package, with identical
host/metadata and compiler options:

```bash
python attention/flash_attn/tests/pytests/check_local_matmul.py --whitebox --save /tmp/fa-base.pt
python attention/flash_attn/tests/pytests/check_local_matmul.py --whitebox --reference /tmp/fa-base.pt
```

This runs 130 deterministic FP16/BF16 BSND, continuous-KV, unmasked cases:
all supported D/DV pairs, M/S2 boundaries, multiple query tiles, B=2 and KV heads=2.
It checks finite output/LSE, output against FP32 SDPA, and bitwise equality of
both output and LSE to upstream. A host recorder cannot prove asynchronous
pipeline safety, so this hardware comparison complements the instruction checks.

Recorded on Ascend950PR / CANN 9.1.0: 130/130 exact comparisons passed.
PA, TND, mask and sink hardware coverage is not claimed. See
`docs/matmul_scalar_optimization.md` for compile coverage and performance results.
