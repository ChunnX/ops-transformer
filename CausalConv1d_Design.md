# CausalConv1d — Detailed Design Document

## 1. Overview

CausalConv1d performs a **causal depthwise 1D convolution** along the sequence dimension,
optionally followed by a SiLU activation. The operator supports two execution modes:

| Mode | `run_mode` | Meaning | Typical Use |
|------|-----------|---------|-------------|
| **FN (Prefill)** | 0 | Full-sequence forward pass | Prompt prefill |
| **Update (Decode)** | 1 | Incremental single-token update | Autoregressive decode |

Each mode supports two input formats:

| Format | x shape | InputModeKey | Description |
|--------|---------|-------------|-------------|
| **BATCH** | `[batch, seqLen, dim]` | FN_BATCH / UPDATE_BATCH | Fixed-length sequences |
| **VARLEN** | `[cuSeqlen, dim]` | FN_VARLEN / UPDATE_VARLEN | Ragged/variable-length sequences |

The kernel width `W` is limited to {2, 3, 4}.

Target hardware: **Ascend 950 (arch35)**, 64 AI cores, 253952 bytes UB per core.

---

## 2. Mathematical Definition

```
y[t, d] = SiLU( sum_{k=0}^{W-1} weight[k, d] × ring[t-k, d] + bias[d] )
```

where SiLU(x) = x / (1 + e^(-x)), and `ring` contains the W most recent tokens:
- t=0: `ring = [x[-W+1], ..., x[-1], x[0]]`, with `x[-i]` from `conv_states` (or zero for cold start)
- t=1: `ring = [x[-W+2], ..., x[0], x[1]]`, with `x[1]` from the input

**Cold-start semantics** (`hasInit=false`): The first `W-1` output tokens are zeroed.
The ring's pre-sequence history slots are filled with zeros instead of loading from `conv_states`.

---

## 3. Operator Interface

### Inputs

| # | Name | Shape | Dtype | Required | Description |
|---|------|-------|-------|----------|-------------|
| 0 | x | `[batch, seqLen, dim]` or `[cuSeqlen, dim]` | fp16/bf16 | Yes | Input sequence |
| 1 | weight | `[W, dim]` | same as x | Yes | Conv kernel, `W ∈ {2,3,4}` |
| 2 | bias | `[dim]` | same as x | Optional | Per-channel bias |
| 3 | conv_states | `[numCacheLines, stateLen, dim]` | same as x | Yes | State cache, `stateLen ≥ W-1` |
| 4 | query_start_loc | `[batch+1]` | int64 | Optional | Varlen cumulative offsets |
| 5 | cache_indices | `[batch]` | int64 | Optional | Seq→cache slot mapping |
| 6 | initial_state_mode | `[batch]` | int64 | Optional | Per-seq: 1=cached, 0=cold (FN only) |
| 7 | num_accepted_tokens | `[batch]` | int64 | Optional | Speculative decode offset (Update only) |

### Outputs

| # | Name | Shape | Dtype | Description |
|---|------|-------|-------|-------------|
| 0 | conv_states | `[numCacheLines, stateLen, dim]` | same as x | Updated state cache |
| 1 | y | same as x | same as x | Convolution output |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| activation_mode | int64 | 0 | 0=Identity, 1=SiLU |
| pad_slot_id | int64 | -1 | Cache slot for padding (skipped) |
| run_mode | int64 | 0 | 0=Prefill(FN), 1=Decode(Update) |

---

## 4. Tiling Key Dispatch

The kernel is compiled as a **static template** dispatched via a 4-field tiling key:

| Field | Bits | Values | Meaning |
|-------|------|--------|---------|
| `inputModeKey` | 2 | 0–3 | FN_VARLEN / FN_BATCH / UPDATE_VARLEN / UPDATE_BATCH |
| `widthKey` | 2 | 2–4 | Kernel width |
| `hasBiasKey` | 1 | 0/1 | Whether bias is present |
| `activationKey` | 1 | 0/1 | Activation: NONE / SILU |

Tiling key encoding: `(inputModeKey) | (widthKey << 2) | (hasBiasKey << 4) | (activationKey << 5)`.

Total combinations: 4 × 3 × 2 × 2 = **48**. Each combination produces a separate compiled kernel instance.

Dispatch flow (from `causal_conv1d_apt.cpp`):
```
GetTilingKey() → switch on (inputModeKey, widthKey, hasBiasKey, activationKey)
  → RunCausalConv1dFn<T, keys...>() or RunCausalConv1dUpdate<T, keys...>()
```

---

## 5. Tiling Data Structure

```cpp
struct CausalConv1dTilingData {
    // Shape params (host → kernel)
    int64_t dim;            // Channel dimension
    int64_t cuSeqlen;       // Total token count (batch × seqLen for BATCH mode)
    int64_t seqLen;         // Per-sequence length (0 for varlen)
    int64_t batch;          // Number of sequences
    int64_t kernelWidth;    // W, convolution kernel width
    int64_t stateLen;       // Length of state buffer per cache line
    int64_t numCacheLines;  // Number of cache slots

    // Attribute params
    int64_t activationMode; // 0=none, 1=SiLU
    int64_t padSlotId;      // Cache slot ID for padding

    // Partitioning params
    int32_t baseDim;        // Dimension per UB slice (aligned to cache line)
    int64_t tokensPerBlock; // Tokens per core block
    int64_t tokenBlockCnt;  // Number of token blocks

    // Runtime flags
    bool hasNumAcceptedTokens;  // Speculative decode
    bool hasCacheIndices;       // Per-seq cache slot mapping
    bool hasInitialStateMode;   // Per-seq init state flag (FN only)
    bool hasInitStateWorkspace; // Workspace prefetch enabled
};
```

`baseDim` is `int32_t` to match index arithmetic on the device. All other shape fields are `int64_t`.

---

## 6. UB Memory Layout

Each core has **253952 bytes** of Unified Buffer (UB). The UB is partitioned into three regions:

### 6.1 Buffer Allocation

```
inBuf:  (W + 1) × baseDim × sizeof(T)      // Ring buffer
outBuf: 2 × baseDim × sizeof(T)              // Double-buffered output
calcBuf: (W + hasBias) × baseDim × sizeof(float)  // Float32 compute buffer
```

For fp16, `W=4`, `hasBias=true`:
```
inBuf:  5 × baseDim × 2  = 10 × baseDim bytes
outBuf: 2 × baseDim × 2  =  4 × baseDim bytes
calcBuf: 5 × baseDim × 4 = 20 × baseDim bytes
Total:                     = 34 × baseDim bytes
```

### 6.2 CalcBufPool Layout

```
calcBuf.Get<float>() →
  [0 ............. W×baseDim)          → weight (float32, W rows)
  [W×baseDim ..... (W+1)×baseDim)      → bias   (float32, 1 row, only if hasBias)
```

```cpp
CalcBufPool::Init(calcBuf, W, baseDim):
  pool.weight = calc[0]
  pool.bias   = calc[W × baseDim]
```

### 6.3 Ring Buffer Layout

The ring buffer occupies `inBuf` with `W+1` rows. Each row has `baseDim` elements.

Slot functions map logical time `t` to physical row index:

```
SlotCurr(t)    = (t + W - 1) % (W + 1)   // Current token slot
SlotPrefetch(t) = (t + W) % (W + 1)       // Next token prefetch slot
```

Example for `W=4` (5-slot ring):

```
t=0: Curr=3, Prefetch=4, History slots=[2,1,0]   (W-1=3 older tokens)
t=1: Curr=4, Prefetch=0, History slots=[1,0,3]
t=2: Curr=0, Prefetch=1, History slots=[0,3,4]
t=3: Curr=1, Prefetch=2, History slots=[3,4,0]
```

Rows wrap around modulo `W+1`, creating a circular window into the sequence.

---

## 7. Host-Side Tiling

### 7.1 UB-Limited Base Dimension

The host first computes the maximum `baseDim` that fits in UB:

```
alignElems = cacheLineSize / elemBytes              // e.g. 64/2 = 32
ringBufCnt = W + 1
calcBufCnt = W + (hasBias ? 1 : 0)
outBufCnt  = 2

bytesPerElem = ringBufCnt × elemBytes
             + outBufCnt × elemBytes
             + calcBufCnt × sizeof(float)

rawBaseDim = ubSize / bytesPerElem
ubLimitedBaseDim = FloorPow2Multiple(rawBaseDim, alignElems)
```

For fp16, W=4, hasBias=true: `bytesPerElem = 5×2 + 2×2 + 5×4 = 34`, `rawBaseDim = 253952/34 ≈ 7469`.

`FloorPow2Multiple(7469, 32)` = `128 × 32 = 4096`. So `ubLimitedBaseDim = 4096`.

### 7.2 Block Decomposition (`ChooseBaseDim`)

```
if dim ≤ ubLimitedBaseDim:
    baseDim = dim                    // Whole dim fits in UB
else:
    // Binary search for best baseDim
    for candidate in powers-of-2 × alignElems (descending):
        baseDimCnt = ceil(dim / candidate)
        group = coreNum / baseDimCnt
        if group ≤ 0: continue
        tokensPerBlock = ceil(cuSeqlen / group)
        actualCores = tokensPerBlock × baseDimCnt
        if actualCores ≥ 0.8 × coreNum:
            baseDim = candidate; break
```

After `baseDim` is chosen:

```
baseDimCnt = ceil(dim / baseDim)
group = coreNum / baseDimCnt
tokensPerBlock = ceil(cuSeqlen / group)
tokenBlockCnt = ceil(cuSeqlen / tokensPerBlock)
blockDim = tokenBlockCnt × baseDimCnt
```

### 7.3 FN Mode Tiling Example

`fn_b8_d2048_w4`: batch=8, seqLen=16, dim=2048, W=4.

```
cuSeqlen = 8 × 16 = 128
dim (2048) ≤ ubLimitedBaseDim (4096) → baseDim = 2048

baseDimCnt = ceil(2048 / 2048) = 1
group = 64 / 1 = 64
tokensPerBlock = ceil(128 / 64) = 2
tokenBlockCnt = ceil(128 / 2) = 64
blockDim = 64 × 1 = 64
```

Each of the 64 cores processes **2 consecutive tokens** along the full dim=2048.

### 7.4 Update Mode Tiling Example

`upd_b128_d8192_w4`: batch=128, seqLen=1, dim=8192, W=4.

```
cuSeqlen = 128 × 1 = 128
dim (8192) > ubLimitedBaseDim (4096) → split

baseDim = 4096
baseDimCnt = ceil(8192 / 4096) = 2
group = 64 / 2 = 32
tokensPerBlock = ceil(128 / 32) = 4
tokenBlockCnt = ceil(128 / 4) = 32
blockDim = 32 × 2 = 64
```

Each core processes **4 sequences**, handling half the dimension (4096 of 8192).
The other half is processed by a different core for the same set of sequences.

---

## 8. FN Mode — Prefill (run_mode=0)

### 8.1 ProcessBlock Flow

```
ProcessBlock():
    1. Compute blockIdx, coreCnt
       → tokenTileId = blockIdx / baseDimCnt
       → baseDimIdx  = blockIdx % baseDimCnt
       → channelStart = baseDimIdx × baseDim
       → curBaseDim   = min(baseDim, dim - channelStart)

    2. tokenStart = tokenTileId × tokensPerBlock
       tokenEnd   = min(tokenStart + tokensPerBlock, cuSeqlen)

    3. valid = (tokenStart < cuSeqlen) && (curBaseDim > 0) && (tokenEnd > tokenStart)

    4. If hasInitStateWorkspace && tokenTileId==0:
         PrefetchInitStatesToWorkspace()  // Owner core copies conv_states → workspace
         SyncAll()                         // All cores wait

    5. If !valid: return

    6. InitCalcBuf(curBaseDim)
       LoadWeightAndBias(channelStart, curBaseDim)  // DMA weight+bias, cast to float32

    7. Find initial sequence:
         VARLEN: seq = FindVarlenSeqByToken(tokenStart)  // Binary search on queryStartLoc
         BATCH:  seq = tokenStart / seqLen

    8. while cursor < tokenEnd and seq < batch:
         GetSeqWindow(seq) → seqStart, curSeqLen
         Skip if cursor before/after current sequence
         tileEnd = min(tokenEnd, seqStart + curSeqLen)
         tileLen = tileEnd - cursor

         ResolveCacheIndex(seq) → cacheIdx (skip if padSlotId)
         ResolveSeqHasInit(seq) → hasInit

         InitRing(cacheIdx, hasInit, seqStart, cursor, tileLen, channelStart, curBaseDim, dim)
         RunSeq(cursor, tileLen, channelStart, curBaseDim, hasInit, seqStart)
         WriteBackStateIfLastChunk(cursor, tileLen, seqStart, curSeqLen, cacheIdx, ...)

         SetEvent(MTE3_MTE2)
         cursor = tileEnd; seq++
```

### 8.2 Sequence Navigation

**BATCH mode**: Fixed `seqLen` per sequence.
```
seq    = tokenStart / seqLen
start  = seq × seqLen
len    = seqLen
```

**VARLEN mode**: Variable lengths, binary search on `queryStartLoc`.
```cpp
FindVarlenSeqByToken(tokenIdx):
    left=0, right=batch
    while left < right:
        mid = (left + right) / 2
        if tokenIdx < queryStartLoc[mid+1]: right = mid
        else: left = mid + 1
    return left  // Sequence index
```

### 8.3 InitRing — History Initialization

The ring buffer must contain `W-1` history tokens before the first output token.
History can come from:
- **Within the current sequence** (`xGm`): tokens at/after `seqStart`
- **Before the current sequence** (`conv_states` or zeros): tokens before `seqStart`

```
historyCount   = W - 1                               // e.g. 3 for W=4
historyStartTok = tileStart - historyCount            // may be negative

preSeqCount = # of history rows before seqStart:
  if seqStart ≤ historyStartTok:  preSeqCount = 0               // All within sequence
  elif seqStart > tileStart - 1:  preSeqCount = historyCount    // All before sequence
  else:                           preSeqCount = seqStart - historyStartTok  // Split

xGmCount = historyCount - preSeqCount                 // Within-sequence history

// 1. Load pre-sequence history:
if preSeqCount > 0:
    if hasInit:
        statePos = historyStartTok - seqStart + historyCount
        stateOffset = stateBase + statePos × dim + channelStart
        DataCopyPad(ring[0], convStatesGm[stateOffset], preSeqCount rows)
        hasGmHistoryCopy = true
    else:
        Duplicate(ring[0], 0, preSeqCount × baseDim)    // Zero-fill (cold start)

// 2. Load within-sequence history:
if xGmCount > 0:
    xGmOffset = (historyStartTok + preSeqCount) × dim + channelStart
    DataCopyPad(ring[preSeqCount×baseDim], xGm[xGmOffset], xGmCount rows)
    hasGmHistoryCopy = true

// 3. Initialize events:
if hasGmHistoryCopy: SetEvent(MTE2_V)   // Set both EVENT_ID0 and EVENT_ID1

// 4. Load first token:
if tileLen > 0:
    slot0 = SlotCurr(0, W)
    DataCopy(ring[slot0×baseDim], xGm[tileStart×dim + channelStart], baseDim)
    SetFlag(MTE2_V, EVENT_ID0)
```

**InitRing Example** (fn_b8_d2048_w4, cold start, core 0):

```
tileStart=0, tileLen=2, seqStart=0, hasInit=false, W=4

historyCount=3, historyStartTok=-3
seqStart (0) > tileStart-1 (-1) → preSeqCount = 3, xGmCount = 0

preSeqCount > 0, hasInit=false:
  → Duplicate(ring[0], 0, 3×2048)  // Zero rows 0,1,2

hasGmHistoryCopy = false → no SetEvent

xGmCount = 0 → no within-sequence history load

tileLen > 0:
  slot0 = SlotCurr(0,4) = 3
  DataCopy(ring[3×2048], xGm[0×2048], 2048)  // x[0] → slot 3
  SetFlag(MTE2_V, EVENT_ID0)

Ring state: [zero, zero, zero, x[0], _]
```

### 8.4 RunSeq — Sliding Window Conv1d

```
RunSeq(start, len, channelStart, baseDim, hasInit, seqStart):
    for t = 0 to len-1:
        // 1. Wait for previous DMA to complete
        WaitFlag(MTE2_V, t&1 ? EVENT_ID1 : EVENT_ID0)

        // 2. Prefetch next token (if any)
        if t+1 < len:
            if t > 0: WaitFlag(MTE3_MTE2, t&1 ? EVENT_ID1 : EVENT_ID0)
            slotNext = SlotPrefetch(t, W)
            xOffset = (start+t+1) × dim + channelStart
            DataCopyPad(ring[slotNext×baseDim], xGm[xOffset], 1 row)
            SetFlag(MTE2_V, t&1 ? EVENT_ID0 : EVENT_ID1)  // See note below

        // 3. Handle output double-buffer
        outSlot = t & 1
        if t ≥ 2: WaitFlag(MTE3_V, t&1 ? EVENT_ID1 : EVENT_ID0)

        // 4. Compute or zero output
        if !hasInit && (start+t-seqStart) < W-1:
            Duplicate(outT[outSlot×baseDim], 0, baseDim)  // Cold-start zeroing
        else:
            ComputeConv1d(ring, weight, bias, outT[outSlot×baseDim], baseDim, t)

        // 5. Sync VEC → MTE3 then output
        SetFlag(V_MTE3, t&1 ? EVENT_ID0 : EVENT_ID1)
        WaitFlag(V_MTE3, t&1 ? EVENT_ID0 : EVENT_ID1)

        outOffset = (start+t) × dim + channelStart
        DataCopyPad(yGm[outOffset], outT[outSlot×baseDim], 1 row)

        // 6. Chain events for next iteration
        if t+2 < len: SetFlag(MTE3_V, t&1 ? EVENT_ID1 : EVENT_ID0)
        if t+2 < len: SetFlag(MTE3_MTE2, t&1 ? EVENT_ID0 : EVENT_ID1)
```

**Important event ping-pong fix**: The prefetch `SetFlag` uses `t&1 ? EVENT_ID0 : EVENT_ID1`.
This ensures that `t=0`'s prefetch signals `EVENT_ID1`, which `t=1`'s `WaitFlag` consumes.
`EVENT_ID1` is thus initialized by the MTE2 prefetch DMA (not by VEC-based `SetEvent`),
which is critical for correct synchronization on Ascend 950.

Event flow for `len=2`:

```
t=0: WaitFlag(EVENT_ID0) ← InitRing's SetFlag
     Prefetch x[1], SetFlag(EVENT_ID1)    // EVENT_ID1 now set
     Compute, SetFlag+WaitFlag(V_MTE3, EVENT_ID1)
     Output y[0]

t=1: WaitFlag(EVENT_ID1) ← t=0's prefetch
     (no prefetch: t+1=2 ≥ len=2)
     Compute, SetFlag+WaitFlag(V_MTE3, EVENT_ID0)
     Output y[1]
```

### 8.5 State Write-Back

After processing a chunk of tokens within a sequence, the **last chunk** must write
back the final `W-1` ring rows to `conv_states` for future use.

```cpp
WriteBackStateIfLastChunk(chunkStart, chunkLen, seqStart, seqLen, cacheIdx, ...):
    if chunkStart + chunkLen == seqStart + seqLen:  // This chunk covers the seq end
        WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim, destStart=0)
```

**WriteBackState** extracts the last `W-1` alive ring rows:

```
firstAliveSlot = len % ringSize
lastAliveSlot  = (len + W - 2) % ringSize

if firstAliveSlot ≤ lastAliveSlot:
    // Contiguous segment
    DataCopyPad(convStatesOutGm[stateBase], ring[firstAliveSlot×baseDim], W-1 rows)
else:
    // Wraps around: two segments
    seg1 = ringSize - firstAliveSlot rows from ring[firstAliveSlot×baseDim]
    seg2 = lastAliveSlot + 1 rows from ring[0]
```

Example (`W=4`, `len=2`): `ringSize=5`, `firstAliveSlot=2%5=2`, `lastAliveSlot=(2+2)%5=4`.
Writes rows 2, 3, 4 → contiguous segment case.

---

## 9. Update Mode — Decode (run_mode=1)

### 9.1 ProcessBlock Flow

```
ProcessBlock():
    1. Same blockIdx → tokenTileId, baseDimIdx decomp as FN
    2. seqStart = tokenTileId × tokensPerBlock
       seqEnd   = min(seqStart + tokensPerBlock, batch)
    3. InitCalcBuf() + LoadWeightAndBias() (same as FN)
    4. for seq in [seqStart, seqEnd):
         GetSeqWindow(seq) → start, len  (len=1 for BATCH, variable for VARLEN)
         ResolveCacheIndex(seq) → cacheIdx
         stateTokenOffset = GetStateTokenOffset(seq)  // For speculative decode
         InitRing(cacheIdx, stateTokenOffset, start, len, channelStart, curBaseDim, dim)
         RunSeq(start, len, channelStart, curBaseDim, cacheIdx, doVarLenStateWrite)
         WriteBackState(cacheIdx, len, ..., destStart)
         SetEvent(MTE3_MTE2)
```

Key differences from FN:
- Sequences (not tokens) are the tiling unit
- `len` is typically 1 (one token per sequence for BATCH mode)
- History always comes from `conv_states` (no cold-start zeroing)
- `WriteBackState` called unconditionally (not conditionally)

### 9.2 InitRing — Simpler History

Update's InitRing always loads `W-1` history rows from `conv_states`:

```
stateBaseOffset = cacheIdx × stateLen × dim + stateTokenOffset × dim + channelStart
DataCopyPad(ring[0], convStatesGm[stateBaseOffset], W-1 rows)

slot0 = SlotCurr(0, W)
DataCopyPad(ring[slot0×baseDim], xGm[start×dim + channelStart], 1 row)
SetFlag(MTE2_V, EVENT_ID0)
```

No preSeqCount analysis needed — all history is in the cache.

### 9.3 Speculative Decode (`num_accepted_tokens`)

When `hasNumAcceptedTokens=true`, the history offset within `conv_states` depends on
how many speculative tokens were actually accepted:

```
GetStateTokenOffset(seq):
    offset = numAcceptedTokens[seq] - 1
    return max(offset, 0)
```

### 9.4 In-Loop State Write

During RunSeq, if speculative decode is active (`doVarLenStateWrite=true`),
the ring's "dead" slots (tokens no longer needed for future computation) are written
back to `convStatesOutGm` within the loop:

```
if doVarLenStateWrite && t ≥ 1:
    deadSlot = t % (W + 1)
    stateOffset = stateBase + (t - 1) × dim
    DataCopyPad(convStatesOutGm[stateOffset], ring[deadSlot×baseDim], 1 row)
```

---

## 10. Address Offset Calculation

### 10.1 Global Memory (GM) Addressing

All GM addresses are byte offsets. Tensor data is row-major (C-order).

**x tensor** — shape `[batch, seqLen, dim]` or `[cuSeqlen, dim]`:

```
xGm[token_idx, channel] = x_base + (token_idx × dim + channel) × sizeof(T)

// Token-offset access:
xGm[token × dim + channelStart]    // row token, starting at channelStart, baseDim elements
```

**weight tensor** — shape `[W, dim]`:

```
weightGm[kernel_idx, channel] = weight_base + (kernel_idx × dim + channel) × sizeof(T)

// Channel-sliced access:
weightGm[channelStart]             // row 0, starting at channelStart, baseDim elements
weightGm[kernel × dim + channelStart]  // row kernel, ...
```

**bias tensor** — shape `[dim]`:

```
biasGm[channel] = bias_base + channel × sizeof(T)

// Channel-sliced:
biasGm[channelStart]               // starting at channelStart, baseDim elements
```

**conv_states tensor** — shape `[numCacheLines, stateLen, dim]`:

```
convStatesGm[cache, state_token, channel] =
    conv_states_base + (cache × stateLen × dim + state_token × dim + channel) × sizeof(T)

// Access with stateTokenOffset (speculative decode):
stateBase = cacheIdx × stateLen × dim + stateTokenOffset × dim + channelStart
```

**y (output) tensor** — same layout as x:

```
yGm[token, channel] = y_base + (token × dim + channel) × sizeof(T)

// Write:
yGm[token × dim + channelStart]    // row token, baseDim elements
```

**conv_states output** — same layout as conv_states input:

```
convStatesOutGm[cache, state_token, channel] = ... (same as convStatesGm)

// WriteBackState writes to:
stateBase = cacheIdx × stateLen × dim + destStart × dim + channelStart
```

### 10.2 Unified Buffer (UB) Addressing

UB addresses are element offsets (not byte offsets) for local tensors.

**Ring buffer** (`inBuf`):

```
ring[row × baseDim]  // Start of row `row` in the ring
ring[row × baseDim + col × V_LENGTH]  // Column chunk for VEC access
```

Row index computed by slot functions:

```
SlotCurr(t)    → ring[SlotCurr(t) × baseDim]      // Current input token
SlotPrefetch(t) → ring[SlotPrefetch(t) × baseDim]  // Prefetched next token
SlotHist(t,i)  → ring[SlotHist(t,i) × baseDim]    // i-th history token
```

**Output buffer** (`outBuf`):

```
outT[outSlot × baseDim]          // Double-buffer, outSlot = t & 1
outT[outSlot × baseDim + col × V_LENGTH]  // VEC access
```

**Calc buffer** (`calcBuf`):

```
calcBuf.Get<float>()[weight_row × baseDim + col × V_LENGTH]  // Weight (float32)
calcBuf.Get<float>()[W × baseDim + col × V_LENGTH]           // Bias (float32)
```

### 10.3 Address Calculation Summary Table

| Access Pattern | GM Expression | UB Expression |
|---------------|---------------|---------------|
| x[token] | `(token × dim + channelStart) × sizeof(T)` | `ring[slot × baseDim]` |
| weight[row] | `(row × dim + channelStart) × sizeof(T)` | `calcBuf[weight_row × baseDim]` (float32) |
| bias | `channelStart × sizeof(T)` | `calcBuf[W × baseDim]` (float32) |
| conv_states[cache, tok] | `(cache × stateLen × dim + tok × dim + channelStart) × sizeof(T)` | — (loaded to ring) |
| y[token] | `(token × dim + channelStart) × sizeof(T)` | `outT[outSlot × baseDim]` |

### 10.4 DMA DataCopyPad Parameters

Multi-row DMA via `DataCopyPad` uses a descriptor `{count, blockBytes, srcGap, dstGap, srcOffset}`:

```
count       → Number of rows to copy
blockBytes  → Bytes per row = baseDim × sizeof(T)
srcGap      → Source gap between rows = (dim - baseDim) × sizeof(T)
dstGap      → Destination gap between rows (0 for dense UB layout, gap for GM)
srcOffset   → Offset within each source row (0 for aligned)
```

Example: loading `W` rows of weight from GM (dim=2048, baseDim=1024):
```
DataCopyPad(weightDma[baseDim], weightGm[channelStart],
            {W, baseDim×sizeof(T), (dim-baseDim)×sizeof(T), dstGap, 0})
```
Copies W rows, each 1024 elements, skipping 1024 elements between rows in GM (the
portion of dim handled by another core).

---

## 11. ComputeConv1d — Kernel Computation

### 11.1 Data Flow

```
fp16 input (ring)  →  Cast to float32
float32 weight     →  (already float32 in calcBuf)
float32 bias       →  (already float32 in calcBuf)
                     ↓
               Mul + Add loop (W iterations)
                     ↓
          [SiLU activation] (optional)
                     ↓
              Cast to fp16  →  output
```

### 11.2 Unrolled Implementation (W ∈ {2,3,4})

For W=2,3,4, the convolution is fully unrolled at compile time:

```cpp
ComputeConv1dUnroll(ring, weight, bias, out, baseDim, t, colLoopTimes, dataCount):
    for col = 0 to colLoopTimes-1:
        mask = UpdateMask<float>(dataCount)

        // Load bias or init to zero
        bias_reg = hasBias ? Load(bias[col×V_LENGTH]) : Duplicate(0.0f)

        // W=4: unrolled MAC
        Load ringF from ring[(t+0)%ringSize × baseDim + col×V_LENGTH]  // cast fp16→fp32
        Load weight from weight[0×baseDim + col×V_LENGTH]
        Mul(tmp, ringF, weight); Add(bias_reg, bias_reg, tmp)

        Load ringF from ring[(t+1)%ringSize × baseDim + col×V_LENGTH]
        Load weight from weight[1×baseDim + col×V_LENGTH]
        Mul(tmp, ringF, weight); Add(bias_reg, bias_reg, tmp)

        // W≥3: extra MAC
        if constexpr (W >= 3):
            Load ringF from ring[(t+2)%ringSize × baseDim + col×V_LENGTH]
            Load weight from weight[2×baseDim + col×V_LENGTH]
            Mul(tmp, ringF, weight); Add(bias_reg, bias_reg, tmp)

        // W≥4: extra MAC
        if constexpr (W >= 4):
            Load ringF from ring[(t+3)%ringSize × baseDim + col×V_LENGTH]
            Load weight from weight[3×baseDim + col×V_LENGTH]
            Mul(tmp, ringF, weight); Add(bias_reg, bias_reg, tmp)

        // Activation: SiLU
        if constexpr (hasActivation):
            tmp = -bias_reg; Exp(tmp, tmp); tmp += 1.0f; bias_reg /= tmp

        // Output: cast fp32 → fp16
        Cast<float, fp16>(out_reg, bias_reg)
        Store(out[col×V_LENGTH], out_reg)
```

### 11.3 Column Loop

VEC instructions operate on `V_LENGTH = 32` floats (128 bytes) at a time.

```
colLoopTimes = ceil(baseDim / V_LENGTH)
dataCount = baseDim   // Last iteration uses partial mask
```

For `baseDim=2048`: `colLoopTimes = 64`, uniform mask throughout.

---

## 12. Event Synchronization

### 12.1 Pipeline Architecture

Four hardware pipes used for synchronization:

| Pipe | Direction | Purpose |
|------|-----------|---------|
| `MTE2_V` | GM→UB DMA → VEC | x prefetch completion |
| `V_MTE3` | VEC → UB→GM DMA | Compute output ready |
| `MTE3_V` | UB→GM DMA → VEC | Output DMA completion |
| `MTE3_MTE2` | UB→GM DMA → GM→UB DMA | Output done → safe to reuse ring slot |

Two events per pipe: `EVENT_ID0` and `EVENT_ID1`, used in ping-pong pattern.

### 12.2 Ping-Pong Pattern (FN RunSeq for len=3)

```
t=0:
  WaitFlag(MTE2_V, EVENT_ID0)            ← InitRing's SetFlag
  WaitFlag(MTE3_MTE2, EVENT_ID0) [skip: t=0]
  Prefetch x[1], SetFlag(MTE2_V, EVENT_ID1)
  Compute, SetFlag+WaitFlag(V_MTE3, EVENT_ID1)
  Output y[0]
  SetFlag(MTE3_V, EVENT_ID1)             [t+2=2 < 3]
  SetFlag(MTE3_MTE2, EVENT_ID0)          [t+2=2 < 3]

t=1:
  WaitFlag(MTE2_V, EVENT_ID1)            ← t=0's prefetch
  WaitFlag(MTE3_MTE2, EVENT_ID1)         ← t=0's SetFlag MTE3_MTE2... wait no
                                         ← Check: t=0 SetFlag MTE3_MTE2 EVENT_ID0
                                         ← t=1 WaitFlag MTE3_MTE2 EVENT_ID1
                                         ← MISMATCH! Actually...

Let me re-derive from the code:

t=0: SetFlag(MTE3_MTE2, t&1?E0:E1) = E1  // (0&1=0, so E1)
t=0: SetFlag(MTE3_V,    t&1?E1:E0) = E0  // (0&1=0, so E0)

t=1: WaitFlag(MTE3_MTE2, t&1?E1:E0) = E1 // (1&1=1, so E1 matches t=0's SetFlag E1 ✓)
     WaitFlag(MTE3_V,    t&1?E1:E0) = E1 // But t=0 set MTE3_V E0...

Hmm, this is getting complex. Let me verify against the actual code patterns.

Actually, looking at the code:
- SetFlag MTE3_V:     `(t & 1) ? EVENT_ID1 : EVENT_ID0`
- WaitFlag MTE3_V:    `(t & 1) ? EVENT_ID1 : EVENT_ID0`  (same as SetFlag)
- SetFlag MTE3_MTE2:  `(t & 1) ? EVENT_ID0 : EVENT_ID1`
- WaitFlag MTE3_MTE2: `(t & 1) ? EVENT_ID1 : EVENT_ID0`

For t=0: SetFlag MTE3_V E0, WaitFlag MTE3_V E0 (self-contained)
For t=0: SetFlag MTE3_MTE2 E1

For t=1: WaitFlag MTE3_MTE2 E1 (matches t=0's SetFlag E1 ✓)
For t=1: SetFlag MTE3_V E1, WaitFlag MTE3_V E1 (self-contained)
For t=1: SetFlag MTE3_MTE2 E0

For t=2: WaitFlag MTE3_V E0 (matches... wait, MTE3_V for t=2 uses `(2&1=0) ? E1 : E0 = E0`. But t=1's SetFlag MTE3_V was E1!

Actually, looking at the code again:
```
if (t >= 2) {
    WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
}
```
t=2: (2&1=0) → WaitFlag MTE3_V EVENT_ID0

And:
```
if (t + 2 < len) {
    SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
}
```
t=0: (0&1=0) → SetFlag MTE3_V EVENT_ID0

So t=2 waits for MTE3_V EVENT_ID0, which was set by t=0's SetFlag MTE3_V EVENT_ID0. ✓

OK, the events are correctly ping-ponged. Let me write this correctly in the document.
```

### 12.2 Event Ping-Pong Table (FN)

| Iteration | WaitFlag MTE2_V | SetFlag MTE2_V (prefetch) | WaitFlag MTE3_MTE2 | SetFlag V_MTE3 / WaitFlag V_MTE3 | SetFlag MTE3_V | SetFlag MTE3_MTE2 |
|-----------|--------|--------|---------|------|------|------|
| t=0 | `E0` (InitRing) | `E1` | — | `E1` ✓ | `E0` | `E1` |
| t=1 | `E1` (t0 prefetch) | `E0` | `E1` (t0) | `E0` ✓ | `E1` | `E0` |
| t=2 | `E0` (t1 prefetch) | `E1` | `E0` (t1) | `E1` ✓ | — | — |

Events used in a given iteration:
- `SetFlag(V_MTE3, E_X)` / `WaitFlag(V_MTE3, E_X)` — self-contained per iteration
- `WaitFlag(MTE2_V, E_prev)` — waits for previous iteration's prefetch or InitRing
- `SetFlag(MTE2_V, E_cur)` — signals current prefetch for next iteration
- `WaitFlag(MTE3_MTE2, E_prev)` — waits for previous iteration's output DMA release
- `SetFlag(MTE3_V, E_prev)` — signals output DMA completion (consumed 2 iterations later)
- `SetFlag(MTE3_MTE2, E_cur)` — signals ring slot release (consumed 1 iteration later)

### 12.3 InitRing Event Initialization

InitRing initializes `EVENT_ID0` on `MTE2_V` via `SetFlag` after the first token DMA.
When `hasGmHistoryCopy=true`, `SetEvent(MTE2_V)` initializes both `EVENT_ID0` and `EVENT_ID1`.

`EVENT_ID1` on `MTE2_V` is initialized by `t=0`'s prefetch `SetFlag` (which targets `EVENT_ID1`
due to the swapped event assignment), ensuring MTE2-based signaling.

---

## 13. Workspace Prefetch (FN hasInitialStateMode)

When `hasInitStateWorkspace=true`, the first `tokenTileId` core copies all `conv_states`
from GM to a workspace buffer in GM, then all cores read history from the workspace
(via `SyncAll` barrier). This reduces redundant GM reads.

```
PrefetchInitStatesToWorkspace(channelStart, baseDimSize):
    totalRows = numCacheLines × stateLen
    for row = 0 to totalRows step (W + 1):
        rowsThisBatch = min(W+1, totalRows - row)
        // GM conv_states → UB
        DataCopyPad(ubBuf, convStatesGm[row×dim + channelStart], rowsThisBatch)
        SetEvent(MTE2_MTE3)
        // UB → workspace GM
        DataCopyPad(workspaceGm[row×dim + channelStart], ubBuf, rowsThisBatch)
        SetEvent(MTE3_MTE2)
```

Then in InitRing, when `hasInitStateWorkspace=true`, history is loaded from
`initStateWorkspaceGm_` instead of `convStatesGm`.

---

## 14. State Write-Back Detail

### 14.1 Ring → convStatesOutGm Mapping

After processing `len` tokens, the ring buffer contains the most recent `W+len` tokens
(history + current tokens). The last `W-1` tokens are the new state.

The ring wraps every `W+1` slots. `WriteBackState` handles both contiguous and wrapped cases:

```
ringSize = W + 1
firstAlive = len % ringSize
lastAlive  = (len + W - 2) % ringSize

stateBase = cacheIdx × stateLen × dim + destStart × dim + channelStart
```

Contiguous case (`firstAlive ≤ lastAlive`):
```
ring[firstAlive×baseDim], ring[(firstAlive+1)×baseDim], ..., ring[lastAlive×baseDim]
→ convStatesOutGm[stateBase], convStatesOutGm[stateBase + dim], ...
```

Wrapped case (`firstAlive > lastAlive`):
```
Segment 1: ring[firstAlive×baseDim] → end of ring   (ringSize - firstAlive rows)
Segment 2: ring[0]                → ring[lastAlive×baseDim]  (lastAlive + 1 rows)
```

### 14.2 FN Write-Back: Conditional on Last Chunk

FN mode only writes state when the current chunk covers the **end of the sequence**:

```cpp
WriteBackStateIfLastChunk(chunkStart, chunkLen, seqStart, seqLen, ...):
    if chunkStart + chunkLen == seqStart + seqLen:
        WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim, destStart=0)
```

For example, seq 0 has 16 tokens, core 7 processes tokens 14–15:
`14 + 2 == 0 + 16` → true → write state back.

Core 0 (tokens 0–1): `0 + 2 != 16` → skip.

### 14.3 Update Write-Back: Unconditional

Update mode always writes state back (one token per sequence).
Optional `destStart=len-1` for speculative decode.

---

## 15. Memory Budget Analysis

**UB budget** for typical configuration (fp16, W=4, hasBias=true, baseDim=2048):

| Buffer | Size (bytes) | Formula |
|--------|-------------|---------|
| inBuf (ring) | 20,480 | `5 × 2048 × 2` |
| outBuf | 8,192 | `2 × 2048 × 2` |
| calcBuf | 40,960 | `5 × 2048 × 4` |
| **Total UB** | **69,632** | (27.4% of 253,952) |

Largest possible baseDim (fp16, W=4): `253952 / 34 ≈ 7469 → FloorPow2Multiple → 4096`.

| Buffer | Size (bytes) for baseDim=4096 |
|--------|------|
| inBuf | 40,960 |
| outBuf | 16,384 |
| calcBuf | 81,920 |
| **Total** | **139,264** (54.9%) |

**GM workspace** (FN hasInitStateWorkspace):

```
workspace = ASCENDC_RESERVED_WORKSPACE_SIZE (16 MB)
          + numCacheLines × stateLen × dim × elemBytes
```

---

## 16. Appendix: Tensor Shapes Summary

### FN Mode

| Tensor | Shape | Notes |
|--------|-------|-------|
| x | `[batch, seqLen, dim]` or `[cuSeqlen, dim]` | 3D batch or 2D varlen |
| weight | `[W, dim]` | W ∈ {2, 3, 4} |
| bias | `[dim]` | Optional |
| conv_states (in) | `[numCacheLines, stateLen, dim]` | stateLen ≥ W-1 |
| conv_states (out) | same as in | Written by last chunk of each seq |
| y | same as x | First W-1 tokens=0 when cold start |

### Update Mode

| Tensor | Shape | Notes |
|--------|-------|-------|
| x | `[batch, 1, dim]` or `[cuSeqlen, dim]` | 1 token per seq |
| weight | `[W, dim]` | Same as FN |
| bias | `[dim]` | Optional |
| conv_states (in) | `[numCacheLines, stateLen, dim]` | stateLen ≥ W (for spec decode) |
| conv_states (out) | same as in | Always written |
| y | same as x | |
