/*!
 * \file ccu_kernel.cc (dispatch_setup 最小版)
 * \brief CCU dispatch_setup —— 只做一件事: CCU 往状态区写标志位。
 *   用新 CCU API(ccu:: 自由函数)。WriteVariableWithNotify = 把本端变量(状态值)
 *   写到对端状态区 + 随路通知, 这正是"setup 写状态位"。只保留写状态位、不做数据搬运。
 *   函数名保持 CcuDispatchSetupKernel(注册处按此名绑定, 不改注册逻辑)。
 */
#include "ccu_kernel.h"
#include "utils.h"

namespace ops_dispatch_setup {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;

CcuResult CcuDispatchSetupKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgDispatchSetup *>(arg);

    DispatchSetupContext ctx;
    ctx.arg = kernelArg;

    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[DispatchSetupCcu] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 1. 取对端状态区的远端变量句柄
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    // 2. 加载参数(状态值)到本端 CCU 变量
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));

    // 3. dispatch_setup 的唯一动作: CCU 把状态值(标志位)写进对端状态区 + 随路通知
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
    }

    return CcuResult::CCU_SUCCESS;
}

}  // namespace ops_dispatch_setup
