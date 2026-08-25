# GatherPaKvCacheWithDequant - Fused Operator

## Overview

This is a fused operator that combines **GatherPaKvCache** and **Dequantization** operations for PagedAttention in KV Cache scenarios. The fusion eliminates intermediate tensor allocations and reduces memory bandwidth by performing gather and dequantization in a single kernel pass.

## Motivation

In PagedAttention with quantized KV Cache, the original workflow requires:
1. **GatherPaKvCache**: Gather INT8/FP8 data from KV Cache based on block tables
2. **Dequantization**: Convert INT8/FP8 to BF16 using scale factors

This creates intermediate tensors and requires multiple kernel launches. The fused operator:
- Reduces memory bandwidth by 50% (no intermediate tensor writes/reads)
- Eliminates kernel launch overhead
- Improves cache locality
- Expected 17% speedup in 64K-bs16 scenario

## Architecture

### Data Flow

```
Input:
  - key_cache: INT8 [num_blocks, block_size, num_heads_k, head_dim_k]
  - key_scale_cache: FLOAT [num_blocks, block_size, num_heads_k]
  - value_cache: FP8_E4M3FN [num_blocks, block_size, num_heads_v, head_dim_v]
  - value_scale: FLOAT [num_heads_v, head_dim_v]
  - block_tables: INT32/INT64 [batch, max_blocks_per_seq]
  - seq_lens: INT32/INT64 [batch]

Output:
  - key: BF16 [num_tokens, num_heads_k, head_dim_k]
  - value: BF16 [num_tokens, num_heads_v, head_dim_v]
```

### Kernel Implementation

The operator supports two data formats:

1. **ND Format** (gather_pa_kv_cache_with_dequant_nd.h)
   - Standard N-dimensional tensor layout
   - Optimized for contiguous memory access
   - Supports both INT8 key and FP8 value dequantization

2. **NZ Format** (gather_pa_kv_cache_with_dequant_nz.h)
   - Fractal NZ format for better memory efficiency
   - Requires 32-byte alignment
   - Optimized for Ascend 950 architecture

### Key Components

```
ops-transformer-gather-pa-kv-cache-with-dequant/
├── op_graph/
│   └── gather_pa_kv_cache_with_dequant_proto.h      # Operator prototype
├── op_host/
│   ├── gather_pa_kv_cache_with_dequant_def.cpp      # Operator definition
│   ├── gather_pa_kv_cache_with_dequant_infershape.cpp # Shape inference
│   ├── gather_pa_kv_cache_with_dequant_tiling.h     # Tiling data structures
│   ├── gather_pa_kv_cache_with_dequant_tiling.cpp   # Tiling implementation
│   ├── config/ascend950/
│   │   └── gather_pa_kv_cache_with_dequant_binary.json # Binary config
│   └── op_api/
│       ├── aclnn_gather_pa_kv_cache_with_dequant.h  # ACLNN API header
│       ├── aclnn_gather_pa_kv_cache_with_dequant.cpp # ACLNN API impl
│       ├── gather_pa_kv_cache_with_dequant.h        # Host API header
│       └── gather_pa_kv_cache_with_dequant.cpp      # Host API impl
├── op_kernel/
│   ├── gather_pa_kv_cache_with_dequant.cpp          # Kernel entry
│   └── arch35/
│       ├── gather_pa_kv_cache_with_dequant_nd.h     # ND format kernel
│       └── gather_pa_kv_cache_with_dequant_nz.h     # NZ format kernel
└── examples/
    └── test_aclnn_gather_pa_kv_cache_with_dequant.cpp # Example test
```

## API Usage

### ACLNN API

```cpp
// Phase 1: Get workspace size
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
aclnnStatus ret = aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(
    keyCache,           // INT8 tensor
    keyScaleCache,      // FLOAT tensor
    valueCache,         // FP8_E4M3FN tensor
    valueScale,         // FLOAT tensor
    blockTables,        // INT32/INT64 tensor
    seqLens,            // INT32/INT64 tensor
    key,                // BF16 output tensor
    value,              // BF16 output tensor
    seqOffsetOptional,  // Optional, can be nullptr
    cacheMode,          // "Norm" for ND, "PA_NZ" for NZ
    isSeqLensCumsum,    // Whether seq_lens is cumulative
    &workspaceSize,
    &executor);

// Allocate workspace
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// Phase 2: Execute operator
ret = aclnnGatherPaKvCacheWithDequant(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);

// Synchronize
aclrtSynchronizeStream(stream);
```

## Build Instructions

### Prerequisites

- CANN (Compute Architecture for Neural Networks) toolkit
- Ascend 950 hardware or simulator
- CMake 3.14+

### Build Steps

```bash
# Set CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DBUILD_OPEN_PROJECT=ON

# Build
make -j$(nproc)
```

## Testing

Run the example test:

```bash
./test_aclnn_gather_pa_kv_cache_with_dequant
```

Expected output:
```
Key output (first 10 elements):
  key[0] = 0x3f80
  key[1] = 0x3f80
  ...
Value output (first 10 elements):
  value[0] = 0x3f00
  value[1] = 0x3f00
  ...
Test completed successfully!
```

## Performance Characteristics

### Memory Bandwidth

- **Original**: Read INT8/FP8 + Write INT8/FP8 + Read BF16 + Write BF16
- **Fused**: Read INT8/FP8 + Write BF16
- **Savings**: ~50% reduction in memory transactions

### Compute Efficiency

- Single kernel launch instead of multiple
- Better instruction-level parallelism
- Reduced synchronization overhead

### Expected Speedup

Based on profiling analysis:
- **64K-bs16 scenario**: ~17% speedup in FA operator
- **10K-bs1 scenario**: ~0.3% speedup
- **64K-bs2 scenario**: ~0.14% speedup

## Implementation Notes

### Dequantization Logic

**Key (INT8 → BF16)**:
```
float_val = int8_val * scale
bf16_val = cast_to_bf16(float_val)
```

**Value (FP8_E4M3FN → BF16)**:
```
float_val = fp8_val * scale
bf16_val = cast_to_bf16(float_val)
```

### Tiling Strategy

- **Batch dimension**: Distributed across AI cores
- **Block dimension**: Processed sequentially per batch
- **Head dimension**: Parallelized within each block
- **UB utilization**: Maximized through careful buffer management

### Constraints

- Key cache must be INT8 format
- Value cache must be FP8_E4M3FN format
- Output is always BF16
- NZ format requires 32-byte alignment
- Block size must be positive integer

## Future Enhancements

1. **FP8 Key Support**: Add support for FP8 key cache (E4M3FN/E5M2)
2. **Dynamic Quantization**: Support per-token dynamic scale factors
3. **Multi-stream Optimization**: Better overlap for large batch sizes
4. **Ascend 910B Support**: Extend to 910B architecture

## References

- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [CANN Operator Development Guide](https://www.hiascend.com/document)
- [Ascend C Programming Guide](https://www.hiascend.com/document)

## License

Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under CANN Open Software License Agreement Version 2.0
