/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_setup_ccu.h
 * \brief dispatch-ccu 的 CCU setup kernel —— hcomm::CcuKernel 子类（host 编译 C++，非 <<<>>> 核函数）
 *        设计文档：docs/CCU_DISPATCH_COMBINE_IMPL_DESIGN.md §3.1 §8
 *
 * 每个 CCU 算子定义 3 个类：
 *   ① CcuKernelArg 子类 —— 静态配置（channels / rank 拓扑），HcclCcuKernelRegister 注册期传入
 *   ② CcuTaskArg   子类 —— 动态地址（本次运行真实 GM 地址），HcclCcuKernelLaunch 启动期传入
 *   ③ CcuKernel    子类 —— kernel 本体，override Algorithm() / GeneArgs()
 */
#ifndef DISPATCH_SETUP_CCU_H
#define DISPATCH_SETUP_CCU_H

#include <vector>
#include <cstdint>
#include "ccu_kernel.h"            // hcomm::CcuKernel / CcuKernelArg / CcuTaskArg / CcuKernelSignature / KernelCreator
#include "hcomm_res_defs.h"        // ChannelHandle
#include "dispatch_ccu_common.h"

// ① 静态参数：注册期传入
class DispatchSetupKernelArg : public hcomm::CcuKernelArg {
public:
    DispatchSetupKernelArg(uint32_t worldSize, uint32_t myRank,
                           const std::vector<ChannelHandle>& chs, const DispatchCcuParam& p)
        : worldSize_(worldSize), myRank_(myRank), param_(p)
    {
        channels = chs;  // 基类 hcomm::CcuKernelArg 的成员
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override;

    uint32_t worldSize_;
    uint32_t myRank_;
    DispatchCcuParam param_;
};

// ② 动态参数：启动期传入
class DispatchSetupTaskArg : public hcomm::CcuTaskArg {
public:
    DispatchSetupTaskArg(uint64_t xAddr, uint64_t expertIdsAddr, uint64_t remoteVaTableAddr,
                         uint64_t tokenNum, uint64_t hidden)
        : xAddr_(xAddr), expertIdsAddr_(expertIdsAddr), remoteVaTableAddr_(remoteVaTableAddr),
          tokenNum_(tokenNum), hidden_(hidden) {}

    uint64_t xAddr_;             // 本卡 token GM 地址
    uint64_t expertIdsAddr_;     // 本卡 token 的 topk expert 路由 GM 地址
    uint64_t remoteVaTableAddr_; // device 侧 remote_va_table（rank → peer HBM 首地址）
    uint64_t tokenNum_;
    uint64_t hidden_;
};

// ③ CCU setup kernel 本体
class DispatchSetupCcu : public hcomm::CcuKernel {
public:
    explicit DispatchSetupCcu(const hcomm::CcuKernelArg& arg);

protected:
    // 通信算法主体：框架在注册期编译成 CcuInstr 微码。P0 步骤 4 实现（当前为骨架 stub）。
    HcclResult Algorithm() override;
    // 把 CcuTaskArg 的动态地址打成 uint64 数组，次序须与 Algorithm 内 Load 次序严格一致。
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    uint32_t worldSize_{0};
    uint32_t myRank_{0};
    DispatchCcuParam param_{};
};

#endif // DISPATCH_SETUP_CCU_H
