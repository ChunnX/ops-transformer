# moe_init_routing_v2 SIMT API Conversion Report

## 1. Public Header File Detection
| Item | Result |
|------|--------|
| Multiple Simt files detected | Yes (3 files) |
| Simt files list | moe_v2_expert_token_out_simt.h, moe_v2_src_to_dst_op_simt.h, moe_v2_src_to_dst_with_capacity_simt.h |
| Common header detected | Yes: moe_v2_common.h |
| Header addition strategy | Option 2 (headers in each Simt file) - USER SELECTED |
| User confirmation | Yes |

## 2. -g Compilation Option Check
| Item | Result |
|------|--------|
| -g detected | No (in ascendc_impl_build.py) |
| Compilation method | Used `--oom` flag which adds `-g --cce-enable-oom` |
| Build command | `bash build.sh --opkernel --soc=ascend950 --ops=moe_init_routing_v2 --oom` |
| Removed -g | No (permanent in OOM mode) |

## 3. Header Addition Record (MANDATORY)
| Header | Added | Position |
|--------|-------|----------|
| simt_api/asc_simt.h | Yes | Line 19 in each Simt file (outside namespace) |
| simt_api/asc_bf16.h | No | Not needed (no bf16 conversion) |
| simt_api/asc_fp16.h | No | Not needed (no fp16 conversion) |
| simt_api/device_functions.h | No | Not needed (no float↔int conversion) |
| simt_api/device_atomic_functions.h | No | Not needed (no atomic operations) |

**Headers added in 3 files**: moe_v2_expert_token_out_simt.h, moe_v2_src_to_dst_op_simt.h, moe_v2_src_to_dst_with_capacity_simt.h

## 4. Variable Name Conflict Check (MANDATORY)
| Variable | Conflict | Definition Line | Reference Lines |
|----------|----------|-----------------|-----------------|
| threadIdx | No | N/A | N/A |
| blockIdx | No | N/A | Only in non-Simt files |
| blockDim | No | N/A | N/A |
| gridDim | No | N/A | N/A |

**Note**: Simt files use `blockIdx_` (member variable with underscore suffix). Non-Simt files use `blockIdx` (no conflict).

## 5. Conversion Details
### API Mappings Summary
| C++ API | C API | Total Occurrences |
|---------|-------|------------------|
| Simt::GetThreadIdx() | threadIdx.x | 4 |
| Simt::GetThreadIdx<0>() | threadIdx.x | 1 |
| Simt::GetThreadIdx<1>() | threadIdx.y | 2 |
| Simt::GetThreadNum() | blockDim.x | 4 |
| Simt::GetThreadNum<0>() | blockDim.x | 1 |
| Simt::GetThreadNum<1>() | blockDim.y | 1 |
| Simt::Dim3{...} | dim3{...} | 5 |
| Simt::VF_CALL<...> | asc_vf_call<...> | 5 |

**Total conversions**: 22 API calls across 3 files

### File-wise Details
| File | APIs Converted | Header Added |
|------|---------------|--------------|
| moe_v2_expert_token_out_simt.h | 15 | Yes (Line 19) |
| moe_v2_src_to_dst_op_simt.h | 4 | Yes (Line 19) |
| moe_v2_src_to_dst_with_capacity_simt.h | 3 | Yes (Line 19) |

## 6. Backup Locations
| Stage | Path |
|-------|------|
| Before | /mnt/workspace/gitCode/cann/ops-transformer/moe/moe_init_routing_v2/before_o/ |
| After | /mnt/workspace/gitCode/cann/ops-transformer/moe/moe_init_routing_v2/after_o/ |

## 7. File Size Comparison
| File | Before Size | After Size | Difference |
|------|-------------|------------|------------|
| MoeInitRoutingV2_2035457016840893fb97c18d7d51c6ac.o | 514K (526,848 bytes) | 514K (526,848 bytes) | 0 bytes |
| MoeInitRoutingV2_4ef9d138d85173ada6c1b7fdd994e022.o | 518K (530,944 bytes) | 518K (530,944 bytes) | 0 bytes |
| MoeInitRoutingV2_8678960b3e91c65b9110de1d6dbbcf54.o | 518K (530,944 bytes) | 518K (530,944 bytes) | 0 bytes |
| MoeInitRoutingV2_ac22b736d69cd9bcecf5657f517fc2d7.o | 514K (526,848 bytes) | 514K (526,848 bytes) | 0 bytes |
| MoeInitRoutingV2_b2a7ca9572a6808f16dcaf76405b152e.o | 514K (526,848 bytes) | 514K (526,848 bytes) | 0 bytes |
| MoeInitRoutingV2_fb404a4d06c9095c111eef17fce7586c.o | 518K (530,944 bytes) | 518K (530,944 bytes) | 0 bytes |

**Total difference**: 0 bytes (no size change across 6 kernels)

## 8. Compilation Status
| Stage | Status | Notes |
|-------|--------|-------|
| Before compilation | Success | 6 kernels generated |
| After compilation | Success | 6 kernels generated |

## 9. Conversion Summary
- Operator: moe_init_routing_v2
- Simt files: 3
- Total API conversions: 22
- Headers added: 3 files (asc_simt.h in each)
- Variable conflicts: 0
- Compilation: Success (before and after)
- File size impact: No change
- Status: ✅ COMPLETE
