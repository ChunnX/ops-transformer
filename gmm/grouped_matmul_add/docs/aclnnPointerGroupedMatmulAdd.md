# aclnnPointerGroupedMatmulAdd

Ascend 950-only grouped weight-gradient accumulation for a packed token axis and
discrete FP32 destinations. This is a new ABI; `aclnnGroupedMatmulAddV2` is not
changed.

## Interface

```cpp
aclnnStatus aclnnPointerGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensorList *yRefs,
    bool transposeX,
    bool transposeWeight,
    int64_t groupType,
    int64_t groupListType,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnPointerGroupedMatmulAdd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

The fixed v1 operation is:

```text
yRefs[e] += x_e.T @ weight_e
```

`x` and `weight` are each one packed BF16 two-dimensional tensor. `groupList`
is a device INT64 vector containing the size of each group. `yRefs` is a
TensorList of two-dimensional FP32 ND tensors. The implementation reads the
device group counts directly; it never constructs a host TensorList from the
counts and performs no D2H synchronization.

## Constraints

- Ascend 950DT/950PR only.
- `transposeX=true`, `transposeWeight=false`, `groupType=2`,
  `groupListType=1`.
- `x.shape[0] == weight.shape[0]` and
  `yRefs[e].shape == [x.shape[1], weight.shape[1]]`.
- `1 <= len(yRefs) == groupList.shape[0] <= 128`.
- Inputs and destinations are ND and contiguous. No implicit format conversion,
  `Contiguous`, `ViewCopy`, or temporary destination is permitted.
- Every destination is distinct and must not overlap another destination or an
  input. The framework binding performs byte-range overlap validation before
  invoking ACLNN.
- A zero-sized group leaves its destination unchanged.
- Bias, quantization, graph capture, repeated destinations, and higher-order
  Autograd are unsupported in v1.

For MoonEP wgrad the MindSpeed binding passes `dY` as `x` and activation `X`
as `weight`, so each destination has the canonical Parameter shape
`[dY.size(1), X.size(1)]` without a transpose copy.
