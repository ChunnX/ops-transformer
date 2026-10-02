# aclnnTurboQuantAttentionScore

安装算子包后包含 `aclnn_turbo_quant_attention_score.h`，链接 `libcust_opapi.so`。

```cpp
aclnnStatus aclnnTurboQuantAttentionScoreTensorGetWorkspaceSize(
    const aclTensor *queryRotated, const aclTensor *queryQjl,
    const aclTensor *keyCacheIdx, const aclTensor *keyCacheQjl,
    const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
    const aclTensor *blockTable, const aclTensor *seqLens,
    int64_t mseBits, const aclTensor *attnScores,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnTurboQuantAttentionScore(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

参数顺序与任务书的八个输入一致。调用方先按 `[T,Hq,max(seq_lens)]` 分配 FLOAT32 输出。
仓库生成器为 ValueDepend 输入额外生成 `aclnnTurboQuantAttentionScoreGetWorkspaceSize`，
该版本的 `seqLens` 类型是 `const aclIntArray *`，适用于 CPU 侧已有长度的调用者。
八个 Tensor 输入应调用带 `Tensor` 后缀的第一阶段接口；两个版本共用同一个执行接口。
为满足该生成器的 dtype 约定，长度额外支持 INT64，其他参数保持不变。
图模式形状推导声明对 `seq_lens` 的数据依赖，长度数据在 host 时推导真实最大长度。
aclnn 的 Tensor 版本中，设备长度不强制复制到 host；tiling 使用调用方提供的输出最后一维，
调用方必须保证该维等于 `max(seq_lens)`，不能用页表容量替代真实最大长度。
CPU 整数数组版本会检查输出最后一维与真实最大长度是否一致。
设备长度由内核用于掩码；负长度或超过调用方输出长度的请求防护为全负无穷，不会越界访问。

第一阶段返回 workspace 大小及 executor；第二阶段在调用方 stream 异步执行。
输入、输出和 workspace 在 stream 完成前必须保持有效。每次调用重新获取 executor。
内核自身不申请全局反量化 workspace；框架可因输入数据依赖和连续化需要返回额外 workspace。
实际返回值见自测报告。

完整输入约束见上级 README。无效形状、dtype、mse_bits 在 tiling 阶段报错；
长度值检查发生在数据位于 host 的场景。设备长度的合法性是调用方契约。
测试中的 `aclnn_runner.py` 使用 ctypes 创建真实 aclTensor，借用 torch 的 NPU 内存并调用上述接口。
