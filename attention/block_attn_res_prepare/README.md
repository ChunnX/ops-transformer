# BlockAttnResPrepare

`BlockAttnResPrepare` implements Phase 1 of Attention Residuals on Ascend 950.

## Interface

Inputs:

- `block_residual`: FP32 ND tensor `[T, N, D]`.
- `effective_query`: FP32 ND tensor `[S, D]`.
- `valid_blocks`: INT32 ND tensor `[1]`; its value must be in `[1, N]`.
- `epsilon`: optional FP32 attribute, default `1e-6`, and must be greater than zero.

Outputs:

- `softmax_max`: FP32 `[S, T]`.
- `weighted_output`: FP32 `[S, T, D]`.
- `softmax_sum`: FP32 `[S, T]`.

## ACLNN and PyTorch interfaces

The ACLNN two-stage interface is `aclnnBlockAttnResPrepareGetWorkspaceSize` and
`aclnnBlockAttnResPrepare`; it is available only on Ascend 950 and validates the required tensor ranks, dtypes,
output shapes, and positive finite `epsilon` before dispatch.

PyTorch users can import `block_attn_res_prepare` from `cann_ops_transformer`.  The wrapper registers
`cann_ops_transformer::block_attn_res_prepare` and returns
`(softmax_max, weighted_output, softmax_sum)` in the same order as the ACLNN/operator interface.

Vector arithmetic is implemented with Ascend 950 VF programming, while tensor creation and GM/UB movement use
AscendC Tensor APIs. UB is assigned directly from tiling-derived byte offsets; the kernel does not create `TPipe`,
`TQue`, or `TBuf`. MTE2/V/MTE3 dependencies are protected by explicit paired `set_flag`/`wait_flag` instructions.
Both tiling keys include an `N=1` runtime path that skips max/exp/sum and weighted accumulation, directly copying
`V[:,0,:]` with `E=1`, `L=1`.

Host tiling expands `(T,S)` into independent AIV work units and selects `baseD` from `D/1024/512/256`. The
`qBufferNum`, `vBufferNum`, and `oBufferNum` fields are calculated from the actual loop counts: Q/O use one buffer
when `dTileNum == 1`, and V uses one buffer only when its complete static loop count is one. The released UB is used
to increase `vCacheRows`; the kernel caches all V rows only when runtime `valid_blocks[0] <= vCacheRows`, otherwise
it reloads V for the weighted-output stage.

The current single-VReg softmax implementation supports physical `N <= 64`. Network performance coverage is
`N=8`, `D=7168`, `S in {18,24}`, and `T in {1,2,4,32,254}`.

The arch35 kernel also contains the no-`TPipe` mixed AIC/AIV Cube template contract: direct UB/L1 address planning,
logical AIC/AIV work mapping, the runtime `N=1` bypass, and paired cross-core events for V ping/pong, MM1 dot, E, and
MM2 consumption. Host tiling evaluates the Cube template first for the supported prefill candidate range and falls back
to the Vector key when the shape or queried L1/L0/UB resources cannot satisfy the Cube plan. Cube MM1/E workspace uses
the aligned `nAlign` row pitch, including the padded N tail consumed by MM2. Cube tiling independently searches
`baseS` (full S first, then smaller M tiles) and `baseD`, and carries separate AIC/AIV counts, L1 buffer depths,
aligned element counts, and both `TCubeTiling` records.
