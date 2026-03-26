# MLA Prolog Kernel UB Usage Report & Max-Subrow Optimizer Design

## Context

The MLA Prolog split-N kernel (`kernel_mla_prolog_split_n.h`) processes vector operations on Ascend NPU with a 192KB UB (Unified Buffer). The UB is partitioned into **fixed buffers** (persistent constants like gamma, scales) and a **shareBuffer_** (reused across mutually-exclusive vector modules). This report analyzes the exact UB footprint of every vector module under each of the 10 quant modes, enabling an optimizer to calculate the maximum `subrow` (vectorRow_ or stepNumHeadDequant) that can be processed per iteration.

## Source Files

| File | Role |
|------|------|
| `op_kernel/kernel_mla_prolog_split_n.h` | Main kernel: VectorBufferInit (L507-578), all module entry points |
| `op_kernel/mla_prolog_comm.h` | MLAPType template (type derivations), constants (MAX_UB_SIZE=192KB) |
| `op_kernel/mla_prolog_tiling_data.h` | Tiling parameters: stepNumHeadDequant, dimension constants |
| `op_kernel/service_rms_norm.h` | RmsNormNormal / RmsNormDynamicQuant UB layouts |
| `op_kernel/service_rotary_position_embedding.h` | RotaryPosEmbPerTensor / PerHead UB layouts |
| `op_kernel/service_dequant.h` | DequantPerTokenQc / CastPerTokenQc / DequantSplitNQc UB layouts |
| `op_kernel/service_dynamic_quant_qn_mul_qr.h` | DynamicQuantQnWithMulQr UB layout |
| `op_kernel/service_rope.h` | RotaryPosEmb internal UB: 2 * cnt * sizeof(float) |
| `op_kernel/service_gather_sin_cos.h` | GatherSinCos UB: 2 * curVecToken * Dr * sizeof(bf16) |
| `op_kernel/mla_prolog_vector_comm.h` | DynamicQuant / QuantPerTile / QuantPerChannel / QuantPerTensor UB |

## Notation & Default DeepSeekV3 Dimensions

| Symbol | Default | Description |
|--------|---------|-------------|
| He | 7168 | headSizeX (input hidden) |
| Hcq | 1536 | headSizeCq (latent query dim) |
| Hckv | 512 | headSizeCkv (latent KV dim) |
| D | 128 | dimHeadSizeQc (per-head query dim) |
| Dr | 64 | dimHeadRope (per-head RoPE dim) |
| N | 32 | numHeadSize (number of attention heads) |
| Hqc | 4096 | headSizeQc = N * D |
| Hqr | 2048 | headSizeQr = N * Dr |
| stepBS | 32 | stepBatchSize (tokens per step) |
| tileSize | 128 | per-tile quant tile size |
| cvRatio | 2 | CV1:2 ratio (2 vec cores per cube core) |

---

## 1. Quant Mode Type Matrix

| Mode | Name | mmInputType | mmQcQrInputType | kvCacheType | mmCqOutputType | mmCkvKrOutputType | mmQcQrOutputType | rmsNormCqOutputType | rmsNormCkvOutputType | krCacheType | isPertile |
|------|------|------------|-----------------|-------------|----------------|-------------------|------------------|---------------------|----------------------|-------------|-----------|
| 0 | NO_QUANT | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | false |
| 1 | PARTIAL_KV_NOQUANT | bf16 | int8 | bf16 | bf16 | bf16 | int32 | int8 | bf16 | bf16 | false |
| 2 | PARTIAL_KV_PERCHANNEL | bf16 | int8 | int8 | bf16 | bf16 | int32 | int8 | int8 | int8 | false |
| 3 | FULL_KV_NOQUANT | int8 | int8 | bf16 | int32 | int32 | int32 | int8 | bf16 | bf16 | false |
| 4 | FULL_KV_PERTENSOR | int8 | int8 | int8 | int32 | int32 | int32 | int8 | int8 | bf16* | false |
| 5 | PARTIAL_KV_PERTILE | bf16 | int8 | int8 | bf16 | bf16 | int32 | int8 | int8 | bf16** | true |
| 6 | FULL_KV_PERTILE | int8 | int8 | int8 | int32 | int32 | int32 | int8 | int8 | bf16** | true |
| 7 | MXFP8_KV_NOQUANT | fp8 | fp8 | bf16 | float | float | float | fp8 | bf16 | bf16*** | false |
| 8 | MXFP8_KV_PERTENSOR | fp8 | fp8 | fp8 | float | float | float | fp8 | fp8 | bf16*** | false |
| 9 | MXFP8_KV_PERTILE | fp8 | fp8 | fp8 | float | float | float | fp8 | fp8 | bf16*** | true |

`*` krCacheType=bf16 because X=int8 && C=int8 triggers the conditional (mla_prolog_comm.h L232)
`**` krCacheType=bf16 because isPertile=true (mla_prolog_comm.h L232)
`***` krCacheType=bf16 because fp8 template specialization always sets it to bf16 (mla_prolog_comm.h L267)

---

## 2. Fixed UB Allocation per Quant Mode

### 2.1 Individual Fixed Buffers

From `VectorBufferInit()` (L507-578):

| Buffer | Condition | Size Formula | Default Bytes |
|--------|-----------|-------------|---------------|
| rmsnormGammaCqBuffer_ | always | Hcq * 2 | 3,072 |
| rmsnormGammaCkvBuffer_ | always | Hckv * 2 | 1,024 |
| deQuantScaleCqBuffer_ | always | (stepBS + 7) * 32 | 1,248 |
| dequantScaleWDqBuffer_ | mmInputType={int8,fp8} | Hcq * 4 | 6,144 |
| dequantScaleWDkvKrBuffer_ | mmInputType={int8,fp8} | (Hckv + Dr) * 4 | 2,304 |
| smoothScaleCqBuffer_ | mmQcQrInputType=int8 && smoothScale != null | Hcq * 4 | 6,144 |
| quantScaleCkrBuffer_ | krCacheType=int8 | Dr * 4 | 256 |
| quantScaleCkvBuffer_ | (rmsNormCkvOutputType={int8,fp8}) && !isPertile | Hckv*4 if mmCkvKrOutputType!=int32; else 32 | 2,048 or 32 |
| sincosBuffer_ (dequantOpt) | enableDequantOpt | 2 * Dr * 4 * ceil(stepBS/cvRatio) | 8,192 |
| sincosBuffer_ (normal) | !enableDequantOpt | 2 * Dr * 4 * curVecTokenMax | 512 |
| stepActualSeqBuffer_ | actualSeqMode=EN_Q_LEN | stepBS * 8 | 256 |

### 2.2 Per-Mode Fixed Totals

`base` = rmsnormGammaCq + rmsnormGammaCkv + deQuantScaleCq = 3072 + 1024 + 1248 = **5,344 B**

| Mode | Additional Fixed Buffers | Fixed (dequantOpt) | shareBuffer_ (dequantOpt) | Fixed (normal) | shareBuffer_ (normal) |
|------|-------------------------|---------------------|--------------------------|----------------|----------------------|
| 0 | (none) | N/A | N/A | 5,344 + 512 = **5,856** | **190,752** |
| 1 | smoothScaleCq(6144) | 5,344+6,144+8,192 = **19,680** | **176,928** | 5,344+6,144+512 = **12,000** | **184,608** |
| 2 | smoothScaleCq(6144) + quantScaleCkr(256) + quantScaleCkv_perchan(2048) | **22,240** | **174,368** | **14,304** | **182,304** |
| 3 | dequantWDq(6144) + dequantWDkvKr(2304) + smoothScaleCq(6144) | **28,128** | **168,480** | **20,448** | **176,160** |
| 4 | dequantWDq(6144) + dequantWDkvKr(2304) + smoothScaleCq(6144) + quantScaleCkv_pertensor(32) | **28,160** | **168,448** | **20,480** | **176,128** |
| 5 | smoothScaleCq(6144) | **19,680** | **176,928** | **12,000** | **184,608** |
| 6 | dequantWDq(6144) + dequantWDkvKr(2304) + smoothScaleCq(6144) | **28,128** | **168,480** | **20,448** | **176,160** |
| 7 | dequantWDq(6144) + dequantWDkvKr(2304) | **21,984** | **174,624** | **14,304** | **182,304** |
| 8 | dequantWDq(6144) + dequantWDkvKr(2304) + quantScaleCkv_perchan(2048) | **24,032** | **172,576** | **16,352** | **180,256** |
| 9 | dequantWDq(6144) + dequantWDkvKr(2304) | **21,984** | **174,624** | **14,304** | **182,304** |

---

## 3. Per-Module UB Footprint from shareBuffer_

### 3.1 GatherSinCos

**Source**: `service_gather_sin_cos.h`
**Active**: All modes
**Subrow param**: curVecToken (number of tokens to load sin/cos for)

```
UB = 2 * curVecToken * Dr * sizeof(bf16) = 4 * curVecToken * Dr
```

Default (curVecToken=1, Dr=64): **256 bytes** -- negligible.

---

### 3.2 RmsNormCq

**Source**: L1282-1364
**Active**: All modes
**Subrow param**: `vectorRow_` (currently 1)

**shareBuffer_ layout**:
```
[outputLocal]          rmsNormCqOutputType:  vectorRow_ * Hcq elements
[dequantScaleQcQr]     dequantScaleType:     curVecToken * dequantScaleCqElemNum elements
[dequantScaleXLocal]   float:               dequantScaleXElemNum elements
[shareTmpUb]           ...internal RmsNorm/DynQuant workspace...
```

**Header size** (before shareTmpUb):

| Component | Mode 0 | Modes 1-6 (int8 out) | Modes 7-9 (fp8 out) |
|-----------|--------|---------------------|---------------------|
| outputLocal sizeof | 2 (bf16) | 1 (int8) | 1 (fp8) |
| dequantScaleCqElemNum | 8 (=32/4) | 8 (=32/4) | 64 (=64/1) |
| dequantScaleType sizeof | 4 (float) | 4 (float) | 1 (fp8_e8m0) |
| dequantScaleXElemNum | 8 | 8 | 56 |
| **header** (vR=1, cvt=1) | 1536*2+8*4+8*4=**3,136** | 1536*1+8*4+8*4=**1,600** | 1536*1+64*1+56*4=**1,824** |

**shareTmpUb internal usage** (per vectorRow_):

| Variant | Condition | shareTmpUb bytes |
|---------|-----------|------------------|
| RmsNormNormal (bf16 in, bf16 out) | Mode 0 | 2 * Hcq * 4 + 32 = **12,320** |
| RmsNormDynQuant (bf16 in, int8 out, arch32) | Modes 1-2 | (4 * Hcq + 8) * 4 = **24,608** |
| RmsNormDynQuant (int32 in, int8 out, arch32) | Modes 3-6 | (4 * Hcq + 8) * 4 = **24,608** |
| RmsNormDynQuant (float in, fp8 out, arch35 VF) | Modes 7-9 | ~3 * Hcq * 4 = **18,432** |

**RmsNormDynQuant breakdown** (arch32, service_rms_norm.h L124-167):
```
xFp32Local [float]:    Hcq elements at shareTmpUb start
  -> RmsNormNormal called with shareTmpUb[Hcq*4], needs 2*Hcq*4+32 internally
  -> After RmsNorm: smooth multiply in-place
maxInt8Tensor [float]: Align(row, 8) at shareTmpUb[Hcq*4]
DynamicQuant:          shareTmpUb[(Hcq+Align(row,8))*4], needs computeSize*4 + Align(row,32)*4
CastFP32ToINT8:        reuses shareTmpUb, needs Hcq*4
Total = (4 * Hcq + 8) * 4 bytes
```

**Total UB from shareBuffer_ (vectorRow_=1, curVecToken=1)**:

| Mode | header + shareTmpUb | Total |
|------|---------------------|-------|
| 0 | 3,136 + 12,320 | **15,456** |
| 1-2 | 1,600 + 24,608 | **26,208** |
| 3-6 | 1,600 + 24,608 | **26,208** |
| 7-9 | 1,824 + 18,432 | **20,256** |

**Max vectorRow_ formula**:
```
maxVectorRow = floor((shareBuffer_ - scaleHeader) / (Hcq * sizeof(outType) + shareTmpUb_perRow))
```

---

### 3.3 RmsNormRopeScatterCkvKr (parent: L1660)

This function allocates a **parent header** from shareBuffer_ then calls RmsNormAndScatterCkv and RopeAndScatterKr **sequentially** (they share the same `shareTmpUb`, so we take the max of the two).

**Parent header** (L1668-1671):
```
dequantScaleXLocal [float]: 8 elements = 32 bytes
cosLocalCkvKr [float]:      Dr * curVecToken * 4 bytes  (only when enableDequantOpt)
sinLocalCkvKr [float]:      Dr * curVecToken * 4 bytes  (only when enableDequantOpt)
```

Parent header = 32 + (enableDequantOpt ? 2 * Dr * curVecToken * 4 : 0) bytes

Default (enableDequantOpt, curVecToken=1): 32 + 512 = **544 bytes**
Default (!enableDequantOpt): **32 bytes**

---

#### 3.3a RmsNormAndScatterCkv (L1428)

**Subrow param**: `vectorRow_` (currently 1)

**Layout from shareTmpUb**:
```
[outputLocal]          kvCacheType: vectorRow_ * Hckv
   (pertile: vectorRow_ * (Hckv * sizeof(kvCacheType) + tileNum * sizeof(float)), aligned 32B)
[rmsNormShareTmpUb]    ...
```

**rmsNormShareTmpUb usage by variant** (all with vectorRow_=1):

**When rmsNormCkvOutputType = bf16 (modes 0, 1, 3, 7)**:
- RmsNormNormal outputs directly to outputLocal
- Internal: 2 * vectorRow_ * Hckv * 4 + 32 bytes
- Total rmsNormShareTmpUb = **4,128 bytes**

**When rmsNormCkvOutputType = int8 (modes 2, 4)**:
- `inputLocal [float]`: vectorRow_ * Hckv * 4 inside rmsNormShareTmpUb
- RmsNormNormal outputs to inputLocal, needs 2 * Hckv * 4 + 32 internally
- Then quantization (QuantPerTensor or QuantPerChannel) needs Hckv * 4 (reuses rmsnorm internal space)
- Total rmsNormShareTmpUb = 3 * vectorRow_ * Hckv * 4 + 32 = **6,176 bytes**

**When rmsNormCkvOutputType = int8, isPertile (modes 5, 6)**:
- Same as int8 + QuantPerTile overhead:
  - PerTileClipWithAlpha: aMax[Align(row*tileNum, 8)*4] + aMaxBrcb[row*tileNum*8*4] + additional
  - DynamicQuant: computeSize*4 + Align(row*tileNum, 8)*32
  - CastFP32ToINT8: cnt*4
- Approximate total rmsNormShareTmpUb ~ **8,000-10,000 bytes**

**When rmsNormCkvOutputType = fp8 (modes 8, 9)**:
- Same structure as int8 variant
- QuantPerTensorToFP8e4m3 (mode 8) or QuantPerTileToFp8e4m3 (mode 9)
- Total rmsNormShareTmpUb ~ **6,176 bytes** (per-tensor) or **~8,000 bytes** (per-tile)

**Total for RmsNormAndScatterCkv** (vectorRow_=1):

| Mode | outputLocal bytes | rmsNormShareTmpUb | Total |
|------|-------------------|-------------------|-------|
| 0 | 512*2 = 1,024 | 4,128 | **5,152** |
| 1 | 512*2 = 1,024 | 4,128 | **5,152** |
| 2 | 512*1 = 512 | 6,176 | **6,688** |
| 3 | 512*2 = 1,024 | 4,128 | **5,152** |
| 4 | 512*1 = 512 | 6,176 | **6,688** |
| 5 | Align(512+tileNum*4, 32) ~ 528 | ~8,224 | **~8,752** |
| 6 | Align(512+tileNum*4, 32) ~ 528 | ~8,224 | **~8,752** |
| 7 | 512*2 = 1,024 | 4,128 | **5,152** |
| 8 | 512*1 = 512 | 6,176 | **6,688** |
| 9 | Align(512+tileNum*4, 32) ~ 528 | ~8,224 | **~8,752** |

---

#### 3.3b RopeAndScatterKr (L1577)

**Subrow param**: `vectorRow_` (currently 1)

**Layout from shareTmpUb**:
```
[outputKrLocal]    krCacheType: vectorRow_ * Dr
[ropeShareTmpUb]   ...RotaryPosEmbPerTensor internal...
```

RotaryPosEmbPerTensor internal (service_rotary_position_embedding.h L40-96, cnt = vectorRow_ * Dr):
```
kLocal [T]:          cnt elements at shareTmpUb start
kFp32Local [float]:  cnt elements at shareTmpUb[baseOffset]
  (baseOffset = cnt when T=int32; cnt/2 otherwise)
ropeShareUB [float]: 2 * cnt elements   (for RotaryPosEmb internal rearrangement)
kFp32OutputLocal:    cnt elements        (when O != C, i.e., output bf16 != compute float)
```

| Variant | T type | baseOffset | Total internal bytes |
|---------|--------|-----------|---------------------|
| bf16 CkvKr (modes 0,1,5) | bf16 | cnt/2 | cnt*2 + 3*cnt*4 = 14*cnt |
| int32 CkvKr (modes 3,4,6) | int32 | cnt | 5*cnt*4 = 20*cnt |
| float CkvKr (modes 7,8,9) | float | cnt/2 | 4.5*cnt*4 = 18*cnt |

Special: mode 2 (krCacheType=int8) adds RopePostQuantPerChannel requiring extra cnt*4 bytes.

**Total for RopeAndScatterKr** (vectorRow_=1, cnt=Dr=64):

| Mode | outputKrLocal | rope internal | Extra | Total |
|------|--------------|--------------|-------|-------|
| 0 | 64*2=128 | 14*64=896 | -- | **1,024** |
| 1 | 64*2=128 | 14*64=896 | -- | **1,024** |
| 2 | 64*1=64 | 14*64=896 | 256 (QuantPerChannel) | **1,216** |
| 3 | 64*2=128 | 20*64=1,280 | -- | **1,408** |
| 4 | 64*2=128 | 20*64=1,280 | -- | **1,408** |
| 5 | 64*2=128 | 14*64=896 | -- | **1,024** |
| 6 | 64*2=128 | 20*64=1,280 | -- | **1,408** |
| 7 | 64*2=128 | 18*64=1,152 | -- | **1,280** |
| 8 | 64*2=128 | 18*64=1,152 | -- | **1,280** |
| 9 | 64*2=128 | 18*64=1,152 | -- | **1,280** |

---

#### 3.3c Combined RmsNormRopeScatterCkvKr

```
Total = parentHeader + max(RmsNormAndScatterCkv, RopeAndScatterKr)
```

| Mode | parentHeader | max(CKV, KR) | **Grand Total** |
|------|-------------|-------------|-----------------|
| 0 | 32 | max(5,152, 1,024) = 5,152 | **5,184** |
| 1 | 544 | max(5,152, 1,024) = 5,152 | **5,696** |
| 2 | 544 | max(6,688, 1,216) = 6,688 | **7,232** |
| 3 | 544 | max(5,152, 1,408) = 5,152 | **5,696** |
| 4 | 544 | max(6,688, 1,408) = 6,688 | **7,232** |
| 5 | 544 | max(8,752, 1,024) = 8,752 | **9,296** |
| 6 | 544 | max(8,752, 1,408) = 8,752 | **9,296** |
| 7 | 544 | max(5,152, 1,280) = 5,152 | **5,696** |
| 8 | 544 | max(6,688, 1,280) = 6,688 | **7,232** |
| 9 | 544 | max(8,752, 1,280) = 8,752 | **9,296** |

---

### 3.4 RopeQr (L1714, non-enableDequantOpt path)

**Active**: All modes when `!enableDequantOpt`
**Subrow param**: None adjustable -- processes N heads fixed (N=numHeadSize). The per-token loop is external.

**Layout** (cnt = N * Dr):

**When mmQcQrInputType=int8 (modes 1-6)**:
```
channelDeqScaleLocal [float]: N * Dr elements = N * Dr * 4 bytes
outputLocal [bf16]:           N * Dr elements = N * Dr * 2 bytes
ropeShareTmpUb:               ...RotaryPosEmbPerTensor(int32 input)...
```
RotaryPosEmbPerTensor with T=int32: 5 * cnt * 4 = 20 * N * Dr bytes

**When mmQcQrInputType=bf16 (mode 0)**:
```
outputLocal [bf16]: N * Dr * 2 bytes
ropeShareTmpUb:     ...RotaryPosEmbPerTensor(bf16 input)...
```
RotaryPosEmbPerTensor with T=bf16: cnt*2 + 3*cnt*4 = 14 * N * Dr bytes

**When mmQcQrInputType=fp8 (modes 7-9)**:
```
outputLocal [bf16]: N * Dr * 2 bytes
ropeShareTmpUb:     ...RotaryPosEmbPerTensor(float input)...
```
RotaryPosEmbPerTensor with T=float, O=bf16: 4.5*cnt*4 = 18 * N * Dr bytes

**Total UB** (N=32, Dr=64):

| Variant | Formula | Default bytes |
|---------|---------|---------------|
| int8 (modes 1-6) | N * Dr * (4 + 2 + 20) = N * Dr * 26 | **53,248** |
| bf16 (mode 0) | N * Dr * (2 + 14) = N * Dr * 16 | **32,768** |
| fp8 (modes 7-9) | N * Dr * (2 + 18) = N * Dr * 20 | **40,960** |

---

### 3.5 DequantQc (L2084, non-enableDequantOpt path)

**Active**: Modes 1-6 when `!enableDequantOpt`
**Subrow param**: `stepNumHeadDequant` (number of heads per chunk, from tiling data)

Inside DequantPerTokenQc (service_dequant.h L37-80):
```
count = stepNumHeadDequant * D
inputLocal [int32]:    count * 4 bytes
scaleLocal [float]:    (count + 16) * 4 bytes
computeLocal [float]:  (count + 16) * 4 bytes
outputLocal [bf16]:    (count + 16) * 2 bytes
```

**UB formula**:
```
UB = count * 4 + (count + 16) * 4 + (count + 16) * 4 + (count + 16) * 2
   = count * 14 + 160
   = stepNumHeadDequant * D * 14 + 160
```

**Default** (stepNumHeadDequant=32, D=128): 32 * 128 * 14 + 160 = **57,504 bytes**

**Max stepNumHeadDequant**:
```
maxStepNumHeadDequant = floor((shareBuffer_ - 160) / (D * 14))
```

---

### 3.6 CastQc (L2112, non-enableDequantOpt path)

**Active**: Modes 7-9 when `!enableDequantOpt`
**Subrow param**: `stepNumHeadDequant`

Inside CastPerTokenQc (service_dequant.h L93-126):
```
count = stepNumHeadDequant * D
inputLocal [float]:   count * 4 bytes
outputLocal [bf16]:   (count + 16) * 2 bytes
```

**UB formula**:
```
UB = count * 4 + (count + 16) * 2 = count * 6 + 32
   = stepNumHeadDequant * D * 6 + 32
```

**Default** (stepNumHeadDequant=32, D=128): 32 * 128 * 6 + 32 = **24,608 bytes**

**Max stepNumHeadDequant**:
```
maxStepNumHeadDequant = floor((shareBuffer_ - 32) / (D * 6))
```

---

### 3.7 DequantQcQrSplitN (L1921, enableDequantOpt path)

**Active**: Modes 1-6 when `enableDequantOpt`
**Subrow param**: row = mmQcQrParam_.m = stepBS (fixed at tiling time)

**Layout from shareBuffer_** (L1946-1953):
```
inputLocal / computeLocal [reuse, int32/float]:  count * 4  (count = row * D/cvRatio)
scaleLocal / outputLocal [reuse]:                (count + 8) * 4
```

**UB formula**:
```
UB = count * 8 + 32 = stepBS * (D / cvRatio) * 8 + 32
```

**Default** (stepBS=32, D=128, cvRatio=2): 32 * 64 * 8 + 32 = **16,416 bytes**

---

### 3.8 CastQcQrSplitN (L1880, enableDequantOpt path)

**Active**: Modes 7-9 when `enableDequantOpt`

```
UB = count * sizeof(float) = stepBS * (D / cvRatio) * 4
```

**Default**: 32 * 64 * 4 = **8,192 bytes**

---

### 3.9 RopeQrSplitN (L1836, enableDequantOpt path)

**Active**: All modes (1-9) when `enableDequantOpt`
**Subrow param**: ropeCnt = ceil(stepBS / cvRatio)

**Layout from shareBuffer_**:
```
outputLocalRope [bf16]: ropeCnt * Dr elements = ropeCnt * Dr * 2 bytes
ropeShareTmpUb:         ...RotaryPosEmbPerHead internal...
```

RotaryPosEmbPerHead internal (cnt = ropeCnt * Dr):
- When T=int32: baseOffset=cnt, total = 5 * cnt * 4 = 20 * cnt bytes
- When T=float: baseOffset=cnt/2, total = 4.5 * cnt * 4 = 18 * cnt bytes

**UB formula**:

| Variant | Formula | Default (ropeCnt=16) |
|---------|---------|---------------------|
| int32 (modes 1-6) | ropeCnt * Dr * (2 + 20) = ropeCnt * Dr * 22 | 16 * 64 * 22 = **22,528** |
| float (modes 7-9) | ropeCnt * Dr * (2 + 18) = ropeCnt * Dr * 20 | 16 * 64 * 20 = **20,480** |

**Max ropeCnt**:
```
maxRopeCnt_int32 = floor(shareBuffer_ / (Dr * 22))
maxRopeCnt_float = floor(shareBuffer_ / (Dr * 20))
```

---

### 3.10 DynamicQuantQnAndMulQr (L2140)

**Active**: Mode 4 and Mode 8 only (full_quant + kv_quant + !isPertile)
**Subrow param**: `row` = curStepBatchSizeVec ~ stepBS / cvRatio

**Internal layout** (service_dynamic_quant_qn_mul_qr.h L218-231):

Phase 1 (DynamicQuant) and Phase 2 (MulQr) share UB space:
```
subRow = 16 (cv1:2) or 8 (cv1:1)
computeSize = subRow * Hckv
loadRopeGmSize = row * Dr
computeSizeRope = min(row, 64) * Dr
brcbCnt = Align(subRow, 8)

Layout:
  inputHalf [bf16]:       computeSize * 2
  qrInputLocal [bf16]:    loadRopeGmSize * 2
  outputLocalRope [bf16]: (computeSizeRope + 16) * 2
  inputLocal [float]:     computeSize * 4
  outputLocal [float]:    computeSize * 4
  maxInt8Tensor [float]:  brcbCnt * 4
  dynamicQuantUb [float]: (computeSize + brcbCnt * 8) * 4
  scaleOutputLocal [float]: Align(row, 8) * 4
  scaleBrcb [float]:      Align(row, 8) * 4
```

**Approximate UB formula** (cv1:2, subRow=16):
```
UB ~ subRow * Hckv * 2                       # inputHalf
   + row * Dr * 2                             # qrInputLocal
   + (min(row, 64) * Dr + 16) * 2             # outputLocalRope
   + subRow * Hckv * 4 * 2                    # inputLocal + outputLocal
   + Align(subRow, 8) * 4                     # maxInt8Tensor
   + (subRow * Hckv + Align(subRow, 8) * 8) * 4  # dynamicQuantUb
   + Align(row, 8) * 4 * 2                    # scaleOutputLocal + scaleBrcb
```

**Default** (row=16, subRow=16, Hckv=512, Dr=64):
```
= 16*512*2 + 16*64*2 + (16*64+16)*2 + 16*512*4*2 + 16*4 + (16*512+16*8)*4 + 16*4*2
= 16,384 + 2,048 + 2,080 + 65,536 + 64 + 33,280 + 128
= ~119,520 bytes
```

This is the **largest UB consumer** across all modules.

---

## 4. Module Activation Matrix

| Module | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|--------|---|---|---|---|---|---|---|---|---|---|
| GatherSinCos | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| RmsNormCq (Normal) | Y | | | | | | | | | |
| RmsNormCq (DynQuant) | | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| RmsNormCkvKr (combined) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| RopeQr (normal) | N* | N* | N* | N* | N* | N* | N* | N* | N* | N* |
| DequantQc (normal) | | N* | N* | N* | N* | N* | N* | | | |
| CastQc (normal) | | | | | | | | N* | N* | N* |
| DequantQcQrSplitN (dequantOpt) | | D* | D* | D* | D* | D* | D* | | | |
| CastQcQrSplitN (dequantOpt) | | | | | | | | D* | D* | D* |
| RopeQrSplitN (dequantOpt) | | D* | D* | D* | D* | D* | D* | D* | D* | D* |
| DynQuantQnAndMulQr | | | | | Y | | | | Y | |

`N*` = when NOT enableDequantOpt
`D*` = when enableDequantOpt

---

## 5. Bottleneck Summary per Mode

### 5.1 Normal path (!enableDequantOpt)

| Mode | Largest Module | UB (bytes) | Available shareBuffer_ | Headroom |
|------|---------------|------------|----------------------|----------|
| 0 | RopeQr (bf16) | 32,768 | 190,752 | 157,984 |
| 1 | DequantQc | 57,504 | 184,608 | 127,104 |
| 2 | DequantQc | 57,504 | 182,304 | 124,800 |
| 3 | DequantQc | 57,504 | 176,160 | 118,656 |
| **4** | **DynQuantQnAndMulQr** | **~119,520** | **176,128** | **56,608** |
| 5 | DequantQc | 57,504 | 184,608 | 127,104 |
| 6 | DequantQc | 57,504 | 176,160 | 118,656 |
| 7 | RopeQr (fp8) | 40,960 | 182,304 | 141,344 |
| **8** | **DynQuantQnAndMulQr** | **~119,520** | **180,256** | **60,736** |
| 9 | RopeQr (fp8) | 40,960 | 182,304 | 141,344 |

### 5.2 enableDequantOpt path

| Mode | Largest Module | UB (bytes) | Available shareBuffer_ | Headroom |
|------|---------------|------------|----------------------|----------|
| 1 | RmsNormCq (DynQuant) | 26,208 | 176,928 | 150,720 |
| 2 | RmsNormCq (DynQuant) | 26,208 | 174,368 | 148,160 |
| 3 | RmsNormCq (DynQuant) | 26,208 | 168,480 | 142,272 |
| 4 | RmsNormCq (DynQuant) | 26,208 | 168,448 | 142,240 |
| 5 | RmsNormCq (DynQuant) | 26,208 | 176,928 | 150,720 |
| 6 | RmsNormCq (DynQuant) | 26,208 | 168,480 | 142,272 |
| 7 | RmsNormCq (fp8) | 20,256 | 174,624 | 154,368 |
| 8 | RopeQrSplitN (fp8) | 20,480 | 172,576 | 152,096 |
| 9 | RmsNormCq (fp8) | 20,256 | 174,624 | 154,368 |

---

## 6. Optimizer Design

### 6.1 Algorithm

```python
def compute_max_subrow(quant_mode, enable_dequant_opt, dims):
    """
    Returns: dict mapping module_name -> (subrow_param_name, max_value)
    """
    # Step 1: compute fixed buffer total
    fixed = compute_fixed_buffers(quant_mode, enable_dequant_opt, dims)
    share_buf = MAX_UB_SIZE - fixed

    # Step 2: for each active module, compute max subrow
    results = {}
    for module in active_modules(quant_mode, enable_dequant_opt):
        header, per_row_cost, param_name = module.ub_formula(quant_mode, dims)
        if per_row_cost > 0:
            max_val = (share_buf - header) // per_row_cost
        else:
            max_val = float('inf')  # fixed-size module
        results[module.name] = (param_name, max_val)

    # Step 3: identify bottleneck (minimum max_subrow per param group)
    bottleneck = min(results.items(), key=lambda x: x[1][1])
    return results, bottleneck
```

### 6.2 Subrow Parameter Groups

Different modules use different subrow parameters. The optimizer should track them separately:

| Param Group | Modules | Typical Range |
|-------------|---------|---------------|
| `vectorRow_` | RmsNormCq, RmsNormAndScatterCkv, RopeAndScatterKr | 1 - 8 |
| `stepNumHeadDequant` | DequantQc, CastQc | 1 - N (32) |
| `ropeCnt` | RopeQrSplitN | 1 - stepBS/cvRatio (16) |
| N/A (fixed dims) | RopeQr, DequantQcQrSplitN, CastQcQrSplitN, GatherSinCos | fixed by config |
| `row` (curStepBatchSizeVec) | DynQuantQnAndMulQr | 1 - stepBS/cvRatio (16) |

### 6.3 Per-Param Max Subrow Formulas

**vectorRow_ max** (bottleneck is usually RmsNormCq):
```
vectorRow_max = min(
    (shareBuffer - rmsCq_scaleHeader) / (Hcq * sizeof(outType) + rmsCq_perRow),
    (shareBuffer - ckvkr_parentHeader) / max(ckv_outputSize + ckv_perRow, kr_outputSize + kr_perRow)
)
```

**stepNumHeadDequant max**:
```
# int8 DequantQc (modes 1-6)
stepNumHeadDequant_max = (shareBuffer - 160) / (D * 14)

# fp8 CastQc (modes 7-9)
stepNumHeadDequant_max = (shareBuffer - 32) / (D * 6)
```

**ropeCnt max** (RopeQrSplitN):
```
# int8 path (modes 1-6)
ropeCnt_max = shareBuffer / (Dr * 22)

# fp8 path (modes 7-9)
ropeCnt_max = shareBuffer / (Dr * 20)
```

### 6.4 Key Considerations

1. **vectorRow_ is currently hardcoded to 1** (L185). Increasing it reduces loop iterations but increases per-iteration UB. The optimizer should compute the maximum safe value.

2. **enableDequantOpt changes the bottleneck profile significantly** -- it trades larger sincosBuffer_ for simpler per-module UB (split-N modules vs full DequantQc/RopeQr).

3. **DynamicQuantQnAndMulQr is the clear bottleneck** for modes 4 and 8 (full quant + KV quant). Its subRow is already tuned (16 for cv1:2, 8 for cv1:1) -- the optimizer should verify this fits.

4. **shareBuffer_ is always the last allocation** (L576-578). Its size = MAX_UB_SIZE - all prior allocations. The optimizer must accurately replicate the fixed buffer calculation.

5. **Per-tile quant modes (5, 6, 9)** have larger RmsNormAndScatterCkv footprints due to QuantPerTile needing aMax, aMaxBrcb, DynamicQuant, and CastFP32ToINT8 buffers. These modes need more careful UB budgeting.

6. **Arch35 VF vs Arch32**: The RmsNormDynamicQuant path has different UB requirements on Arch35 (VectorFlow IP uses dedicated hardware, often lower UB) vs Arch32 (software implementation, higher UB). The optimizer should detect the target architecture.
