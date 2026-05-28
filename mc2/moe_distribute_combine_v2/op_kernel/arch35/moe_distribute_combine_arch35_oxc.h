#ifndef MOE_DISTRIBUTE_COMBINE_ARCH35_OXC_H
#define MOE_DISTRIBUTE_COMBINE_ARCH35_OXC_H

#include "arch35/moe_distribute_combine_arch35.h"

template <TemplateMoeDistributeCombineA5TypeClass>
class MoeDistributeCombineA5Oxc
    : public MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc> {
public:
    __aicore__ MoeDistributeCombineA5Oxc() = default;
};

#endif
