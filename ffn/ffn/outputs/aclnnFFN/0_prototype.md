每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnFFNGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFFN"接口执行计算。

```cpp
aclnnStatus aclnnFFNGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *weight1,
  const aclTensor   *weight2,
  const aclIntArray *expertTokens,
  const aclTensor   *bias1,
  const aclTensor   *bias2,
  const aclTensor   *scale,
  const aclTensor   *offset,
  const aclTensor   *deqScale1,
  const aclTensor   *deqScale2,
  const aclTensor   *antiquantScale1,
  const aclTensor   *antiquantScale2,
  const aclTensor   *antiquantOffset1,
  const aclTensor   *antiquantOffset2,
  const char        *activation,
  int64_t            innerPrecise,
  const aclTensor   *y,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor
)
```

```cpp
aclnnStatus aclnnFFN(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream
)
```
