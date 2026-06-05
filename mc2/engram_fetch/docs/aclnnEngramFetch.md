# aclnnEngramFetch

## 功能描述

EngramFetch算子：根据全局索引从远端rank通过URMA获取token数据。

输入：
- bufferhandle：HCCL通信上下文标识（int32），绑定通信组
- indices：token全局索引（int32），shape [num_tokens]

对每个global_idx：
- owner_rank = global_idx / num_entries_per_rank
- local_entry = global_idx % num_entries_per_rank
- 本卡token → 直接拷贝，远端 → URMA拷到本卡

输出：
- fetched：shape [num_tokens, hidden_dim]，dtype由output_dtype决定

Attrs：
- num_ranks（REQUIRED）：rank总数
- num_entries_per_rank（REQUIRED）：每卡存储entry数
- hccl_buffer_size（默认0）：HCCL buffer大小
- output_dtype（默认1=BF16）：输出数据类型，1=BF16, 2=FP16, 3=FP32

## aclnnEngramFetchGetWorkspaceSize

- **参数**:
  - bufferhandle（aclTensor*）：HCCL上下文标识，int32
  - indices（aclTensor*）：全局索引，int32，shape [num_tokens]
  - numRanks（int64_t）：rank总数
  - numEntriesPerRank（int64_t）：每卡entry数
  - hcclBufferSize（int64_t）：HCCL buffer大小，默认0
  - outputDtype（int64_t）：输出dtype，1=BF16/2=FP16/3=FP32
  - fetched（aclTensor*）：输出tensor
  - workspaceSize / executor：标准输出

## aclnnEngramFetch

- **参数**: workspace, workspaceSize, executor, stream