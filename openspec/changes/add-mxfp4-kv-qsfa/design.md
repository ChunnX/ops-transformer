# Design: B-I combined MXFP4 NoPE cache record

## Frozen B-I record contract

For every logical KV token, `key` stores one 400-byte raw record. Its public
storage dtype is `DT_INT8`; it is intentionally **not** advertised as a pure
`DT_FLOAT4_E2M1` tensor because the record includes raw RoPE and E8M0 bytes.

```text
byte [0, 256)   : 512 NoPE values as 256 packed FP4 E2M1 bytes
byte [256, 384) : 64 raw RoPE values in query dtype Q_T (128 bytes)
byte [384, 400) : 16 E8M0 scales; scale g covers NoPE D[32g, 32g+32)
```

Packed-FP4 order is low nibble then high nibble: record byte `i` contains
NoPE D[2i] in bits [3:0] and NoPE D[2i+1] in bits [7:4]. E8M0 byte `0xff` is
invalid for this supported cache format and MUST be rejected by the reference
oracle. Raw-cache production is outside this operator change; the kernel
defensively maps any `0xff` scale that reaches it to bf16 NaN rather than a
wrong finite value.

The DMA input row is aligned from 400 to 416 bytes. The V0 output remains the
existing `Q_T[576]` representation: D[0:512] is dequantized NoPE and
D[512:576] is byte-preserved RoPE, placed in the existing D-block × 17-row-slot
UB layout before the existing L1/GM copy-out.

## Kernel route

The kernel remains on the A5 SIMD-VF route:

```text
GM raw key record -> V0 input UB -> Reg VF decode -> V0 output UB -> L1/GM
```

`sparse_indices` still names the KV tokens to gather. A pair gather brings two
complete 400-byte records; eight pairs fill the fixed 16-row UB batch. There
is no additional scale GM gather.

The VF logical scale group is D=32. For aligned vector processing, one D=128
NoPE VF tile consumes 64 packed FP4 bytes and four E8M0 scales. The intended
implementation reuses existing A5 SIMD VF FP4 unpack/cast patterns and MUST
validate the exact E8M0-to-BF16 conversion and store API on the target CANN
toolchain before implementation. SIMT is out of scope.

## Compatibility and validation

- A5 B path supports only `sparse_block_size=1`.
- Existing BSND, TND, and PA row routing semantics remain unchanged; only raw
  record bytes/row change from 656 to 400 and DMA stride from 672 to 416.
- `key_dequant_scale` retains existing schema/host-presence compatibility in
  combined scale mode but is not a numerical source for this B-I V0 path.
- The clean-room oracle defines record decoding and does not copy any operator
  implementation code.

## Contract Architect Confirmation

This section is the contract architect's independent confirmation of the frozen
B-I ABI. It records the exact record layout, the decoder assumptions, and the
explicit limitations. It is authored against repository revision
`f258f93332429a1948bf908c66b75d1048aa4508`.

### Exact ABI

Per logical KV token, `key` is an opaque `DT_INT8` record of exactly 400 bytes.
The public storage dtype is `DT_INT8`; the record is a byte container, never a
pure FP4 tensor.

| Byte range | Size | Content |
|------------|------|---------|
| `[0, 256)` | 256 B | 512 NoPE values packed as FP4 E2M1, two values per byte |
| `[256, 384)` | 128 B | 64 RoPE values stored raw in query dtype Q_T (BF16 or FP16) |
| `[384, 400)` | 16 B | 16 E8M0 scales; scale `g` covers NoPE `D[32g, 32g+32)` |

Decoder semantics (frozen):

- Packed-FP4 order is low nibble then high nibble: record byte `i` holds
  NoPE `D[2i]` in bits `[3:0]` and NoPE `D[2i+1]` in bits `[7:4]`.
- FP4 E2M1 bit layout is `S-E-E-M` (bit 3 sign, bits 2:1 exponent, bit 0
  mantissa); normal value `(-1)^S * 2^(E-1) * (1 + M/2)`, subnormal (`E=0`)
  value `(-1)^S * (M/2)`. There is no inf/NaN code point; `E=3` encodes `±4.0`
  and `±6.0`.
- E8M0 is exponent-only with bias 127: `scale = 2^(byte - 127)`. Byte `0xff`
  is the E8M0 NaN encoding and is rejected for this supported cache format.
- V0 output is `Q_T[576]`: `D[0:512]` is dequantized NoPE (`fp4 * scale`),
  `D[512:576]` is byte-preserved RoPE. The output is placed in the existing
  D-block × 17-row-slot UB layout (17 = 16 rows + 1 anti-bank-conflict slot).
- DMA input row is aligned from 400 to 416 bytes (32-byte align); the current
  656-byte record uses 672-byte stride.

### Decoder assumption provenance

Each assumption is confirmed against in-repository evidence; no fallback
behavior was invented.

| Assumption | Evidence |
|------------|----------|
| FP4 E2M1 nibble order: low = `D[2i]`, high = `D[2i+1]` | `experimental/attention/quant_flash_attn/tests/ttk/qfa_mxfp4_test/common/mx_quant_fp4_tool.py` `pack_fp4_to_uint8`/`unpack_uint8_to_fp4` ("低 4 位存前者, 高 4 位存后者"; "OCP MX nibble 序: low = elem[2k], high = elem[2k+1]") |
| FP4 E2M1 value set `[0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]`, no inf/NaN | `mx_quant_fp4_tool.py` `FP4_E2M1_VALUES` / `generate_fp4_e2m1_encoding_table` |
| E8M0 bias = 127, `0xff` reserved (NaN) | `mx_quant_fp4_tool.py` `E8M0_BIAS = 127`, `_to_e8m0_uint8`/`pack_mxfp4_full` clamp to `[0, 254]`; `gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/weight_quant_basic_block/gmmsq_quant_mx_vf.h` `fp8NanRegTensor = 0x00ff`, `scaleOffset = 0x7f00` |
| Current A5 record 656 B (NoPE 512 + RoPE 128 + scale 16) | `op_host/kv_quant_sparse_flash_attention_tiling.cpp:1244` (`kHeadDim_ != 656`) |
| V0 16-row UB batch, 17-row-slot layout, RoPE reinterpreted as Q_T | `op_kernel/arch35/kv_quant_sparse_flash_attention_service_vector_mla_arch35.h` (`blockStride = 17`, `DequantKv`, `kRopeUb = srcTensor[dSizeNope].ReinterpretCast<Q_T>()`) |
| Current DMA row stride 672 (32-byte align of 656) | `..._service_vector_mla_arch35.h` `dVTemplateTypeInput = 672`, `combineDim = 672` |

Verification conclusion: the oracle decoder (`oracle/mxfp4_b_oracle.py`) and the
spec scenarios agree with the repository evidence for both the FP4 nibble order
and the E8M0 special-value behavior. No evidence conflict was found; no
spec/oracle revision was required for these two points.

### Explicit limitations

- A5 only: this contract is scoped to the `__CCE_AICORE__ == 310` (A5 SIMD-VF)
  route; SIMT and the arch22 route are out of scope.
- `sparse_block_size=1` only; any other value is rejected for B-I.
- No full-D=576 MXFP4 and no separate FP4-data/E8M0-scale gather path.
- `key_dequant_scale` is a compatibility input only; it is not a numerical
  source for B-I V0 dequantization.
- The E8M0-to-BF16 conversion and the exact CANN store API used by the A5
  implementation are not validated against the target CANN toolchain at
  contract-authoring time; T003 must confirm them before implementation freeze.
- This confirmation authorizes the contract, not implementation. Implementation
  authorization and completion remain with the Forge run gates.
