# MatmulReduceScatterV2 BF16 No-Bias Design

## Goal

Temporarily support BF16 computation on Ascend950 only when `bias == nullptr`, so the kernel does not instantiate the CATLASS Ascend950 BF16 Cube-bias path that is currently unsupported.

## Scope

- Keep FP16 computation and its existing bias support.
- Keep BF16 computation when the optional bias input is absent.
- Reject every BF16 computation with a present bias tensor.
- Do not implement BF16-to-FP32 bias conversion in this phase.
- Do not add or connect UT cases in this phase.

## Design

The ACLNN validation and graph-mode tiling validation both reject BF16 input when bias is present. The no-bias BF16 path uses `TILINGKEY_TPL_FP16` as a placeholder bias type because `TPL_ISBIAS` is false and the bias element type is never consumed.

The kernel template declaration enumerates only FP16 and FP32 bias types. The BF16 bias branches are removed from both the large-M and Small-M dispatch paths. This prevents OPC from precompiling unsupported `MmadPingpongBias<Arch::Ascend950, bfloat16_t>` combinations.

The operator definition and BF16 binary JSON keep their optional bias slots. Those declarations are needed for the BF16 no-bias binary signature; conditional presence restrictions are enforced by ACLNN and tiling validation.

## Error Handling

- ACLNN returns `ACLNN_ERR_PARAM_INVALID` for BF16 with a non-null bias.
- Graph-mode tiling returns `ge::GRAPH_FAILED` for BF16 with a present bias descriptor.

## Verification

The Ascend950 kernel build must complete without the CATLASS assertion `Bias does not support bfloat16 yet`. Static checks must also confirm that no kernel tiling selection or dispatch branch references `TILINGKEY_TPL_BF16`.
