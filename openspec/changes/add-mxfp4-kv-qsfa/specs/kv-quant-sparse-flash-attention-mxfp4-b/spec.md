## ADDED Requirements

### Requirement: Combined MXFP4 NoPE record layout

The A5 B-I implementation MUST interpret every supported `key` KV-token record
as exactly 400 `DT_INT8` storage bytes in the order `[256B packed FP4 E2M1
NoPE][128B raw Q_T RoPE][16B E8M0 scale]`; it MUST NOT treat this mixed record
as a pure FP4 tensor.

#### Scenario: A packed record decodes in D order

- **GIVEN** one 400-byte key record with distinct packed NoPE nibble values, distinct BF16 RoPE values, and 16 valid E8M0 scales
- **WHEN** the B-I reference decoder processes the record for BF16 query dtype
- **THEN** NoPE D[0:512] uses low-nibble then high-nibble order and RoPE D[512:576] equals the source RoPE values

### Requirement: E8M0 per-32 NoPE dequantization

The B-I implementation MUST decode each FP4 E2M1 NoPE value and multiply it
by the E8M0 scale assigned to its 32-element D group; it MUST use exactly 16
scales for NoPE D=512.

#### Scenario: Scale boundaries are observable

- **GIVEN** all 512 packed NoPE values encode the same nonzero FP4 number and the 16 E8M0 scales have distinct exponents
- **WHEN** the B-I reference decoder dequantizes the record
- **THEN** all 32 values in group g equal that FP4 value multiplied by scale g and adjacent groups may differ

### Requirement: RoPE preservation and V0 output compatibility

The B-I implementation MUST dequantize only NoPE D[0:512], MUST copy raw
RoPE D[512:576] to the final V0 `Q_T[576]` output, and MUST preserve the
existing sparse routing, 16-row V0 UB batch, output layout, and post-V0
BMM/softmax behavior.

#### Scenario: One sparse V0 batch preserves the RoPE suffix

- **GIVEN** a sparse gather selects up to 16 valid B-I records for one V0 batch
- **WHEN** V0 prepares K for BMM1
- **THEN** each output row has dequantized NoPE in D[0:512] and its own unchanged RoPE suffix in D[512:576]

### Requirement: A5 B-I input validity

The A5 B-I implementation MUST accept only `DT_INT8` raw key records with
storage width 400 and `sparse_block_size=1`; host validation MUST reject an
invalid record width or an unsupported sparse block size rather than silently
decoding a different layout. The clean-room oracle MUST reject an E8M0 scale
byte `0xff` (the E8M0 NaN encoding). Raw-cache production is outside this
operator change and is an input precondition, not an in-kernel validation
mechanism: if a malformed `0xff` byte nevertheless reaches V0, the kernel MUST
map it to bf16 NaN rather than silently decoding a finite value.

#### Scenario: Invalid layout is rejected

- **GIVEN** a B-I request with key storage width 656, sparse_block_size 2, or an E8M0 scale byte 0xff
- **WHEN** host validation validates the record width and sparse block size, and the clean-room oracle validates the E8M0 scale bytes
- **THEN** an invalid width or sparse block size fails with an explicit unsupported-format error, and an E8M0 0xff fails oracle decoding rather than producing a finite value
