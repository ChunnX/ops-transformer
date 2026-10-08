// Recording substitute for the library kernel only. The operator's real
// TensorApiBackend adapter executes and maps every serialized tiling field.
#pragma once
#include "kernel_operator.h"
#include "blaze/attention/block/grouped_rotate_quant_problem.h"
namespace Blaze::Attention::Kernel {
template<class Params> void RecordGroupedParams(const Params& params);
template<class Mmad, class Epilogue> struct GroupedRotateQuantKernel {
    struct Params {
        GM_ADDR xList=nullptr, rotationList=nullptr, yList=nullptr, scaleList=nullptr, alpha=nullptr, groupList=nullptr;
        GroupedRotateQuantConfig groups;
        uint32_t usedCoreNum=0;
        int64_t roundMode=4, scaleAlg=0;
        float dstTypeMax=0, invDstTypeMax=0;
    };
    void operator()(const Params& params) { RecordGroupedParams(params); }
};
}
