#ifndef ATTENTION_FFN_CONTEXT_H
#define ATTENTION_FFN_CONTEXT_H

#include <cstdint>

namespace Mc2Aclnn {

struct AttentionFfnContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE];//TokenInfo、TokenData 等业务数据
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE];//必要的通信 handle
};

} // namespace Mc2Aclnn

#endif