## ADDED Requirements

### Requirement: PA_NZ MXFP4 mode 6 eligibility

`FusedInferAttentionScore` SHALL accept the new route only when key and value use `FLOAT4_E2M1`, both antiquant modes are mode 6/per-token-group, KV storage is PageAttention, and KV layout is NZ. The implementation SHALL require `headDim % 32 == 0` and define `G = headDim / 32`.

#### Scenario: FIA accepts a supported PA_NZ invocation

- **GIVEN** `FLOAT4_E2M1` key/value, mode 6 for key/value, PageAttention storage, NZ KV layout, and a head dimension divisible by 32
- **WHEN** FIA validates and tiles the invocation
- **THEN** FIA SHALL select the PA_NZ per-token-group antiquant route

#### Scenario: A non-divisible head dimension is rejected

- **GIVEN** an otherwise supported MXFP4 mode 6 PA_NZ invocation with `headDim % 32 != 0`
- **WHEN** the host validates the invocation
- **THEN** the invocation SHALL fail validation before kernel dispatch

### Requirement: PA_NZ scale and offset storage contract

For PA_NZ mode 6, scale and non-null offset SHALL be physical page metadata. The producer SHALL transform logical parameter values `[B,N,S,G]` through PageAttention `[P,N,Bs,G]` and then store the final parameter tensor as `[P,N,G,Bs]`, where `P=totalBlockNum` and `Bs=blockSize`. The value identity SHALL be `final[p,n,g,u] = logical[b,n,s,g]` for `p=blockTable[b,s/Bs]` and `u=s%Bs`.

Scale SHALL be E8M0 and offset SHALL either be null or have dtype equal to Q (`FP16` or `BF16`). Key and value scale/offset tensors SHALL each satisfy the same storage contract for their respective head dimensions.

#### Scenario: A non-identity page map preserves scale coordinates

- **GIVEN** two logical sequences whose `blockTable` entries map to non-monotonic physical pages and scale values unique by `(b,n,s,g)`
- **WHEN** the producer creates PA_NZ scale storage
- **THEN** each value SHALL appear at `[blockTable[b,s/Bs],n,g,s%Bs]`

#### Scenario: Offset follows the same storage transform

- **GIVEN** a non-null FP16 or BF16 offset with values unique by `(b,n,s,g)`
- **WHEN** the producer creates PA_NZ offset storage
- **THEN** its value SHALL appear at the same `[p,n,g,u]` coordinate as the matching scale parameter

#### Scenario: A logical sequence-shaped PA parameter is rejected

- **GIVEN** a PA_NZ mode 6 invocation whose scale or non-null offset is supplied as logical `[B,N,S,G]` storage instead of final `[P,N,G,Bs]` storage
- **WHEN** the host validates the invocation
- **THEN** the invocation SHALL fail validation

### Requirement: Page-aware NZ parameter loading

For every PA_NZ antiquant tile, FIA SHALL translate each logical token range into physical page segments through `blockTable`. For each segment, scale and non-null offset SHALL be read using source coordinates `[physicalPage,kvHead,group,tokenInPage]` and placed in a UB group-plane view compatible with the NZ VF path. The loader SHALL preserve all groups, KV heads, page boundaries and tile tails.

Scale and offset SHALL use separate source element types and address units: scale as E8M0 raw bytes and offset as Q_T elements. The implementation SHALL not derive either source address by casting the other type or by using a shared half-element adjustment.

#### Scenario: A tile crosses physical page boundaries

- **GIVEN** a PA_NZ tile whose logical token range spans at least two `blockTable` pages
- **WHEN** the kernel loads scale and non-null offset
- **THEN** it SHALL load every token from its mapped physical page without treating separate pages as contiguous GM memory

#### Scenario: Group-plane data remains distinguishable

- **GIVEN** scale and non-null offset values distinct for every physical page, KV head, group and page token
- **WHEN** the kernel prepares the NZ VF inputs
- **THEN** each VF group/token lane SHALL receive the value from its matching `[p,n,g,u]` coordinate

### Requirement: Symmetric and asymmetric MXFP4 dequantization

For every decoded MXFP4 value at logical head coordinate `d`, the implementation SHALL select `g=floor(d/32)`. With null offset, it SHALL compute `decode_fp4(x[d]) * scale[g]`. With non-null offset, it SHALL compute `(decode_fp4(x[d]) + offset[g]) * scale[g]`.

The implementation SHALL convert decoded FP4 data, E8M0 scale and non-null Q_T offset to BF16 before the Add/Mul operations, then store to the established Q_T output format and rounding path. The symmetric path SHALL not load or consume offset.

#### Scenario: Symmetric PA_NZ dequantization

- **GIVEN** supported PA_NZ scale values and a null offset
- **WHEN** FIA dequantizes key/value data
- **THEN** each group of 32 logical D elements SHALL be multiplied by its matching scale without an offset addition

#### Scenario: Asymmetric PA_NZ dequantization with a negative offset

- **GIVEN** supported PA_NZ scale values and FP16 or BF16 offsets including negative values
- **WHEN** FIA dequantizes key/value data
- **THEN** each group of 32 logical D elements SHALL use `(decode + offset) * scale` with its matching signed offset

### Requirement: Isolation from unsupported and existing paths

This change SHALL not add mode 6 PageAttention support for BNBD or BBH. A mode 6 PageAttention invocation using either layout SHALL fail validation rather than dispatching the PA_NZ parameter loader. Non-PageAttention mode 6 SHALL retain its existing symmetric input contract, address calculation and numerical behavior.

#### Scenario: BNBD or BBH is not misrouted as PA_NZ

- **GIVEN** a mode 6 PageAttention invocation with BNBD or BBH KV layout
- **WHEN** the host validates the invocation
- **THEN** it SHALL reject the invocation before dispatch

#### Scenario: Existing non-PA mode 6 remains unchanged

- **GIVEN** a pre-existing valid non-PageAttention mode 6 symmetric invocation
- **WHEN** the updated operator executes it
- **THEN** it SHALL use the pre-existing non-PA route and preserve its prior input contract and numerical behavior

### Requirement: Required build and numerical evidence

The implementation SHALL not be accepted without target-environment compilation and numerical evidence for FIA. The evidence SHALL include both symmetric and asymmetric PA_NZ, non-identity `blockTable`, a cross-page tile, multiple KV heads, at least two groups, and FP16/BF16 offset coverage.

#### Scenario: Required PA_NZ test matrix passes

- **GIVEN** a target CANN environment and the completed FIA implementation
- **WHEN** the PA_NZ build and correctness suite runs
- **THEN** all required symmetric/asymmetric, page-mapping, group, dtype and regression cases SHALL pass against the approved reference oracle
