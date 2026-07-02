# stem_oam_prep_paged_kv

Stem OAM Prep Paged KV operator for block-sparse attention KV preparation.

## Overview

`stem_oam_prep_paged_kv` is the KV-side preprocessing module for Stem OAM (Output-Aware Metric) block-sparse attention. It extracts K/V data from Paged KV Cache, computes K Processing (per-token K-scale x group sum + anti-diagonal flip) and V Processing (per-head vScale x L2 Norm -> Global Normalize -> ReLU -> Block Average), producing `kFlat` and `vBias` for Stem OAM score computation.

## Supported Platforms

- Ascend950DT / Ascend950PR (arch35, DAV_3510)

## Supported Data Types

| Input | Dtype | Output | Dtype |
|-------|-------|--------|-------|
| kCache, vCache | FP8_E4M3FN | kFlat | BF16 |
| kvIndices, kvSeqLens | INT32 | vBias | FP32 |
| kScaleCache, vScale | FP32 | | |

## API

```cpp
aclnnStatus aclnnStemOamPrepPagedKVGetWorkspaceSize(
    const aclTensor *kCache,
    const aclTensor *vCache,
    const aclTensor *kvIndices,
    const aclTensor *kvSeqLens,
    const aclTensor *kScaleCache,
    const aclTensor *vScale,
    double lambdaMag,
    int64_t cacheLayout,
    int64_t kvBlockSize,
    int64_t stemBlockSize,
    int64_t stemStride,
    aclTensor *kFlat,
    aclTensor *vBias,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnStemOamPrepPagedKV(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## Constraints

- kCache/vCache dtype must be FP8_E4M3FN
- kScaleCache/vScale must not be nullptr
- kvBlockSize: {64, 128}
- stemBlockSize: divisible by 32, <= 256
- stemStride: divisible by 16, <= 64, stemStride <= stemBlockSize