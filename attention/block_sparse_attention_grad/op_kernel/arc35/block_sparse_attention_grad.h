#include "kernel_operator.h"
using namespace AscendC;

namespace BSA {
class BlockSparseAttentionGradArch35 {
public:
    __aicore__ inline BlockSparseAttentionGradArch35() {};
    
    __aicore__ inline void Process(GM_ADDR dout,
                                    GM_ADDR q,
                                    GM_ADDR k,
                                    GM_ADDR v,
                                    GM_ADDR out,
                                    GM_ADDR softmaxLse,
                                    GM_ADDR blockSparseMask,
                                    GM_ADDR blockShape,
                                    GM_ADDR attentionMask,
                                    GM_ADDR actualQseqlen,
                                    GM_ADDR actualKvseqlen,
                                    GM_ADDR dq,
                                    GM_ADDR dk,
                                    GM_ADDR dv,
                                    GM_ADDR workspace, 
                                    GM_ADDR tiling)
    {
        printf("BlockSparseAttentionGrad LAUNCHED!\n");

        if ASCEND_IS_AIC {
            CubeProcess(dout, q, k, v, out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tiling);
        }
        if ASCEND_IS_AIV {
            VectorProcess(dout, q, k, v, out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tiling);
        }
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout,
                                        GM_ADDR q,
                                        GM_ADDR k,
                                        GM_ADDR v,
                                        GM_ADDR out,
                                        GM_ADDR softmaxLse,
                                        GM_ADDR blockSparseMask,
                                        GM_ADDR blockShape,
                                        GM_ADDR attentionMask,
                                        GM_ADDR actualQseqlen,
                                        GM_ADDR actualKvseqlen,
                                        GM_ADDR dq,
                                        GM_ADDR dk,
                                        GM_ADDR dv,
                                        GM_ADDR workspace, 
                                        GM_ADDR tiling)
    {
       printf("CubeProcess LAUNCHED!\n");
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout,
                                        GM_ADDR q,
                                        GM_ADDR k,
                                        GM_ADDR v,
                                        GM_ADDR out,
                                        GM_ADDR softmaxLse,
                                        GM_ADDR blockSparseMask,
                                        GM_ADDR blockShape,
                                        GM_ADDR attentionMask,
                                        GM_ADDR actualQseqlen,
                                        GM_ADDR actualKvseqlen,
                                        GM_ADDR dq,
                                        GM_ADDR dk,
                                        GM_ADDR dv,
                                        GM_ADDR workspace, 
                                        GM_ADDR tiling)
    {
       printf("VectorProcess LAUNCHED!\n");
    }
};
}
