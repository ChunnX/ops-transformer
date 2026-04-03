#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_functions.h"

using namespace AscendC;
namespace {
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t WARP_SIZE = 32;
};

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeGradFusedGradH2PreAndGradH2PostSimt(__ubuf__ T* fusedGradHPre2AndGradHPost2Local,
    __ubuf__ T* gradRsqrtLocal, __gm__ T* gradHcScaleGlobal, __gm__ T* gradHcBaseGlobal, __ubuf__ T* fusedGradHPreAndGradHPostLocal,  __ubuf__ T* fusedHPre2AndHPost2Local,
    __ubuf__ T* rsqrtLocal, __ubuf__ T* hcBaseLocal, __ubuf__ T* tmpLocal, const T hcScalePre, const T hcScalePost, int32_t taskCount, int32_t n)
{
    int32_t xTid = threadIdx.x; // 8
    int32_t yTid = threadIdx.y; // THREAD_NUM // 8 
    int32_t yBlockDim = blockDim.y;
    int32_t xBlockDim = blockDim.x;
    int32_t taskWidth = n * n + 2 * n;

    int32_t tid = yTid * xBlockDim + xTid;
    T times_2 = xTid / n == 0? 1.f : 2.f;

    T hcScaleVal = xTid / n == 0? hcScalePre : hcScalePost;
    T hcBaseVal = hcBaseLocal[xTid];     
    

    for (int32_t i = yTid; i < taskCount; i += yBlockDim) {
        int32_t hOffset = i * xBlockDim + xTid;
        T rsqrtVal = rsqrtLocal[i];
        T h2Val = fusedHPre2AndHPost2Local[hOffset];
        T h2ValNormed = h2Val * rsqrtVal;
        T gradHVal = fusedGradHPreAndGradHPostLocal[hOffset];

        // recompute weighted sum
        T sigmoidVal = 1.f / (1.f + expf(-(h2ValNormed * hcScaleVal + hcBaseVal)));

        // sigmoid grad
        T sigmoidGradVal = sigmoidVal * (1 - sigmoidVal) * gradHVal * times_2;

        // compute grad_rsqrt
        T gradRsqrt = sigmoidGradVal * hcScaleVal * h2Val;
        T gradRsqrtReduceSum = gradRsqrt + asc_shfl_xor(gradRsqrt, 1, 8);
        gradRsqrtReduceSum = gradRsqrtReduceSum + asc_shfl_xor(gradRsqrtReduceSum, 2, 8);
        gradRsqrtReduceSum = gradRsqrtReduceSum + asc_shfl_xor(gradRsqrtReduceSum, 4, 8);

        // 暂存
        if (i == yTid) {
            tmpLocal[tid] = sigmoidGradVal;
            fusedGradHPreAndGradHPostLocal[tid] = sigmoidGradVal * h2ValNormed;
        } else {
            Simt::AtomicAdd(tmpLocal + tid, sigmoidGradVal);
            Simt::AtomicAdd(fusedGradHPreAndGradHPostLocal + tid, sigmoidGradVal * h2ValNormed);
        }
        
        fusedGradHPre2AndGradHPost2Local[i * taskWidth + xTid] = sigmoidGradVal * hcScaleVal * rsqrtVal;
        if (xTid == 0) {
            gradRsqrtLocal[yTid] = gradRsqrtReduceSum;
        }
    }

    // ReduceSum
    int32_t s = (taskCount + 1) / 2;
    int32_t validTask = taskCount;
    for (; validTask > 1; validTask = (validTask + 1) / 2) {
        if (yTid + s < validTask) {
            int32_t otherOffset = (yTid + s) * xBlockDim + xTid;
            tmpLocal[tid] += tmpLocal[otherOffset];
            fusedGradHPreAndGradHPostLocal[tid] += fusedGradHPreAndGradHPostLocal[otherOffset];
            s = (s + 1) / 2;
        }
        __syncthreads();
    }

    T preHScaleVal = 0, postHScaleVal = 0;
    if (yTid == 0) {
        // copyout gradHcBase
        Simt::AtomicAdd(gradHcBaseGlobal + xTid, tmpLocal[xTid]);
        preHScaleVal = xTid / n == 0? fusedGradHPreAndGradHPostLocal[xTid] : 0;
        postHScaleVal = xTid / n == 1? fusedGradHPreAndGradHPostLocal[xTid] : 0;
    }

    preHScaleVal = asc_reduce_add(preHScaleVal);
    postHScaleVal = asc_reduce_add(postHScaleVal);
    
    if (tid == 0) {
        Simt::AtomicAdd(gradHcScaleGlobal, preHScaleVal);
        Simt::AtomicAdd(gradHcScaleGlobal + 1, postHScaleVal);
    }
}


template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SinkhornKnoppSimt(__ubuf__ T* gradHRes2Local, __ubuf__ T* gradRsqrtLocal, __gm__ T* gradHcScaleGlobal, 
    __gm__ T* gradHcBaseGlobal, __gm__ T* skRowNormGlobal, __gm__ T* skRowSumGlobal, __gm__ T* skColNormGlobal, __gm__ T* skColSumGlobal, __ubuf__ T* gradResLocal,
    __ubuf__ T* hRes2Local, __ubuf__ T* rsqrtLocal, __ubuf__ T* hcBaseLocal, __ubuf__ T* tmpLocal, const T hcScaleRes, const int32_t totalTasks, 
    const int32_t taskOffset, const int32_t taskCount, const int32_t skIterCount, const int32_t n, const T eps)
{
    int32_t xTid = threadIdx.x; // 16
    int32_t yTid = threadIdx.y; // THREAD_NUM // 16
    int32_t yBlockDim = blockDim.y;
    int32_t xBlockDim = blockDim.x;
    int32_t tid = yTid * xBlockDim + xTid;
    int32_t taskWidth = n * n + 2 * n;

    // sinkhorn
    int32_t baseOffset1 = 0, baseOffset2 = 0;
    for (int32_t iter = 0; iter < skIterCount; ++iter) {
        baseOffset1 = ((skIterCount - 1 - iter) * totalTasks + taskOffset) * n * n;
        baseOffset2 = ((skIterCount - 1 - iter) * totalTasks + taskOffset) * n;
        for (int32_t i = yTid; i < taskCount; i += yBlockDim) {
            int32_t hOffset = i * xBlockDim + xTid;
            int32_t offsetRow = baseOffset2 + i * n + xTid / n;
            int32_t offsetCol = baseOffset2 + i * n + xTid % n;

            T gradResVal = gradResLocal[hOffset];
            
            T skRowNormVal = skRowNormGlobal[hOffset + baseOffset1];
            T skRowSumVal = skRowSumGlobal[offsetRow];

            T skColNormVal = skColNormGlobal[hOffset + baseOffset1];
            T skColSumVal = skColSumGlobal[offsetCol];

            T grad1 = (gradResVal * skRowNormVal / powf(skColSumVal + eps, 2.0f));

            // reduce dim -2
            T colReduceSumVal = grad1 + asc_shfl_xor(grad1, 4, 16);
            colReduceSumVal = colReduceSumVal + asc_shfl_xor(colReduceSumVal, 8, 16);
            T gradRowNormed = gradResVal / (skColSumVal + eps) - colReduceSumVal;

            // reduce dim -1
            T grad2 = (gradRowNormed * (skRowNormVal * (skRowSumVal + eps)) / powf(skRowSumVal + eps, 2));
            T rowReduceSumVal = grad2 + asc_shfl_xor(grad2, 1, 16);
            rowReduceSumVal = rowReduceSumVal + asc_shfl_xor(rowReduceSumVal, 2, 16);

            T gradSinkhorn = gradRowNormed / (skRowSumVal + eps) - rowReduceSumVal;

            gradResLocal[hOffset] = gradSinkhorn;
        }
    }

    
    T hcScaleVal = hcScaleRes;
    T hcBaseVal = hcBaseLocal[xTid];

    for (int32_t i = yTid; i < taskCount; i += yBlockDim) {
        int32_t hOffset = i * xBlockDim + xTid;
        T rsqrtVal = rsqrtLocal[i];
        T h2Val = hRes2Local[hOffset];
        T h2ValNormed = h2Val * rsqrtVal;
        T gradSinkhorn = gradResLocal[hOffset];

        T hRes = h2ValNormed * hcScaleVal + hcBaseVal;

        T hRes0 = asc_shfl(hRes, 0, 4);
        T hRes1 = asc_shfl(hRes, 1, 4);
        T hRes2 = asc_shfl(hRes, 2, 4);
        T hRes3 = asc_shfl(hRes, 3, 4);

        int32_t maxIdx01 = hRes0 >= hRes1? 0 : 1;
        T hResMaxVal01 = fmaxf(hRes0, hRes1);

        int32_t maxIdx012 = hResMaxVal01 >= hRes2? maxIdx01 : 2;
        T hResMaxVal012 = fmaxf(hResMaxVal01, hRes2);

        int32_t maxIdx0123 = hResMaxVal012 >= hRes3? maxIdx012 : 3;
        T hResMaxVal0123 = fmaxf(hResMaxVal012, hRes3);

        T mask = maxIdx0123 != (xTid % n)? 1 : 0;
        T safeHRes = expf(hRes - hResMaxVal0123);
        T gradMax = safeHRes * mask * gradSinkhorn;

        T gradMaxReduceSum = gradMax + asc_shfl_xor(gradMax, 1, 16);
        gradMaxReduceSum = -(gradMaxReduceSum + asc_shfl_xor(gradMaxReduceSum, 2, 16));

        gradMaxReduceSum = mask == 1? safeHRes * gradSinkhorn : gradMaxReduceSum;

        // compute grad_rsqrt
        T gradRsqrt = gradMaxReduceSum * hcScaleVal * h2Val;
        T gradRsqrtReduceSum = gradRsqrt + asc_shfl_xor(gradRsqrt, 1, 16);
        gradRsqrtReduceSum = gradRsqrtReduceSum + asc_shfl_xor(gradRsqrtReduceSum, 2, 16);
        gradRsqrtReduceSum = gradRsqrtReduceSum + asc_shfl_xor(gradRsqrtReduceSum, 4, 16);
        gradRsqrtReduceSum = gradRsqrtReduceSum + asc_shfl_xor(gradRsqrtReduceSum, 8, 16);
        
        // 暂存
        if (i == yTid) {
            gradResLocal[tid] = gradMaxReduceSum;
            tmpLocal[tid] = gradMaxReduceSum * h2ValNormed;
        } else {
            Simt::AtomicAdd(gradResLocal + tid, gradMaxReduceSum);
            Simt::AtomicAdd(tmpLocal + tid, gradMaxReduceSum * h2ValNormed);
        }

        gradHRes2Local[i * taskWidth + n * 2 + xTid] = gradMaxReduceSum * hcScaleVal * rsqrtVal;
        
        if (xTid == 0) {
            Simt::AtomicAdd(gradRsqrtLocal + yTid, gradRsqrtReduceSum);
        }
    }

    __syncthreads();

    // ReduceSum
    int32_t s = (taskCount + 1) / 2;
    int32_t validTask = taskCount;
    for (; validTask > 1; validTask = (validTask + 1) / 2) {
        if (yTid + s < validTask) {
            int32_t otherOffset = (yTid + s) * xBlockDim + xTid;
            gradResLocal[tid] += gradResLocal[otherOffset];
            tmpLocal[tid] += tmpLocal[otherOffset];
            s = (s + 1) / 2;
        }
        __syncthreads();
    }

    T gradHScaleRes = 0;
    if (yTid == 0) {
        // copyout gradHcBase
        Simt::AtomicAdd(gradHcBaseGlobal + xTid, gradResLocal[xTid]);
        gradHScaleRes = tmpLocal[xTid];
    }

    gradHScaleRes = asc_reduce_add(gradHScaleRes);
    if (tid == 0) {
        Simt::AtomicAdd(gradHcScaleGlobal, gradHScaleRes);
    }
}


template<typename T, typename VECTOR_DTYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeGradX1Simt(__gm__ T* gradX1Global, __gm__ VECTOR_DTYPE* xGlobal,
    __ubuf__ T* gradRsqrtLocal, __ubuf__ T* rsqrtLocal, const int32_t taskCount, const int32_t n, const int32_t c)
{
    int32_t tid = threadIdx.x;
    int32_t tidInnerWarp = tid % 32;

    // 计算 factor
    for (int32_t i = tid; i < taskCount; i += THREAD_NUM) {
        T factor = - powf(rsqrtLocal[i], 3.0f) * gradRsqrtLocal[i] / (n * c * 1.0f);
        gradRsqrtLocal[i] = factor;
    }

    __syncthreads();

    for (int32_t i = tid * 4; i < n * taskCount * c; i += THREAD_NUM * 4) {
        int32_t taskIdx = i / (n * c);
        VECTOR_DTYPE xVec = xGlobal[i / 4];
        T factor = gradRsqrtLocal[taskIdx];

        Simt::AtomicAdd(gradX1Global + i, xVec.x * factor);
        Simt::AtomicAdd(gradX1Global + i + 1, xVec.y * factor);
        Simt::AtomicAdd(gradX1Global + i + 2, xVec.z * factor);
        Simt::AtomicAdd(gradX1Global + i + 3, xVec.w * factor);
    }
}