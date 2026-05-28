#ifndef MOE_DISTRIBUTE_DISPATCH_ARCH35_OXC_H
#define MOE_DISTRIBUTE_DISPATCH_ARCH35_OXC_H

#include "arch35/moe_distribute_dispatch_arch35.h"

template <TemplateMoeDistributeDispatchA5TypeClass>
class MoeDistributeDispatchA5Oxc
    : public MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc> {
public:
    __aicore__ MoeDistributeDispatchA5Oxc() = default;
};

#endif
