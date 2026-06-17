/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_padding.h
 * \brief 矩阵Padding（对齐补零）实现
 *
 * 当A矩阵的K维度或B矩阵的N维度不满足512B对齐要求时，AIC的DMA搬运效率会降低。
 * 本模块在AIV核心上将原始矩阵拷贝到workspace并对齐补零，确保AIC读取时满足对齐要求。
 *
 * Padding流程：
 *   1. AIV从原始GM地址读取A/B矩阵数据
 *   2. 写入workspace中对齐后的地址（不足512B的部分自动补零）
 *   3. 设置跨核同步flag，通知AIC padding完成
 *
 * TemplatePadder类：根据A/B矩阵的转置组合（4种情况）选择对应的布局策略
 * PaddingRunner类：对外提供统一接口，内部根据转置标志分发到TemplatePadder
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_PADDING_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_PADDING_H

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/cross_core_sync.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/padding_matmul.hpp"
#include "matmul_reduce_scatter_aiv_mode_util.h"

using namespace AscendC;
using namespace Catlass;
using namespace matmulReduceScatterV2_util;

// PaddingRunner的参数传递宏：将所有padding参数打包传递
#define PADDING_ARGS_FUN() \
    bool transA, bool transB, bool alignedA, bool alignedB, \
    uint32_t matrixAM, uint32_t matrixAK, uint32_t matrixBK, uint32_t matrixBN, \
    uint32_t matrixAMAlign, uint32_t matrixAKAlign, uint32_t matrixBKAlign, uint32_t matrixBNAlign, \
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmAAlign, GM_ADDR gmBAlign

namespace Catlass::Gemm::Kernel {
// TemplatePadder: 模板化矩阵Padding执行器
// 根据A/B矩阵的元素类型和布局，选择对应的PaddingMatrix实例进行对齐补零
// ArchTag_: 硬件架构标签, AType_: A矩阵类型, BType_: B矩阵类型
template <
    class ArchTag_,
    class AType_,
    class BType_
>
class TemplatePadder {
public:
    using ArchTag = ArchTag_;
    using ElementA = typename AType_::Element;   // A矩阵元素类型
    using LayoutA = typename AType_::Layout;     // A矩阵布局类型
    using ElementB = typename BType_::Element;   // B矩阵元素类型
    using LayoutB = typename BType_::Layout;     // B矩阵布局类型

    // A矩阵padding的计算长度：96KB / 元素大小，用于限制单次DMA搬运量
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    // B矩阵padding的计算长度：96KB / 元素大小
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR ptrA;          // A矩阵原始GM地址
        LayoutA layoutA;       // A矩阵原始布局
        GM_ADDR ptrB;          // B矩阵原始GM地址
        LayoutB layoutB;       // B矩阵原始布局
        GM_ADDR ptrWA;         // A矩阵padding地址
        LayoutA layoutWA;      // A矩阵padding布局
        GM_ADDR ptrWB;         // B矩阵padding地址
        LayoutB layoutWB;      // B矩阵padding布局
        bool alignA;           // A矩阵是否需要padding
        bool alignB;           // B矩阵是否需要padding

        // Methods
        // 默认构造函数
        CATLASS_HOST_DEVICE
        Params() {}

        // 带参数的构造函数：初始化所有padding参数
        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrWA_, LayoutA layoutWA_, GM_ADDR ptrWB_, LayoutB layoutWB_,
               bool alignA_, bool alignB_)
            : ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_),
              alignA(alignA_), alignB(alignB_) {}
    };

    // Methods
    // 默认构造函数
    CATLASS_DEVICE
    TemplatePadder() {}

    // operator()通用声明：根据核心类型分发到AIC或AIV实现
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // AIV核心的operator()特化：执行A/B矩阵的padding对齐和跨核同步
    // 1. 如果A矩阵需要对齐，从原始地址拷贝到对齐后的workspace地址
    // 2. 如果B矩阵需要对齐，从原始地址拷贝到对齐后的workspace地址
    // 3. 设置跨核同步flag，通知AIC padding完成
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if (params.alignA) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (params.alignB) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
        }
        // 0x0 synchronization control between AI Core
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
    }
private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;  // AIV完成padding的flag ID
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};  // 跨核同步flag实例
    Arch::Resource<ArchTag> resource;  // 硬件资源管理器（UB缓冲区分配等）
};

} // namespace Catlass::Gemm::Kernel

namespace padding {
// PaddingRunner: Padding操作的统一入口类
// 根据A/B矩阵的转置组合（4种情况：NT/NT, NT/T, T/NT, T/T）选择对应的布局策略
// InputType: A矩阵元素类型, WeightType: B矩阵元素类型
template <typename InputType, typename WeightType>
class PaddingRunner {
public:
    __aicore__ explicit PaddingRunner() = default;

    // Run: 执行padding操作，根据transA/transB的组合选择4种布局策略之一
    // 每种布局策略对应不同的对齐维度（A矩阵对齐K，B矩阵对齐N，转置时对齐维度不同）
    inline __aicore__ void Run(PADDING_ARGS_FUN())
    {
        using ArchTag = Arch::AtlasA2;
        using ElementA = InputType;
        using ElementB = WeightType;

        if (!transA && !transB) {
            // 情况1：A不转置(RowMajor) + B不转置(RowMajor)
            // A矩阵对齐K维度，B矩阵对齐N维度
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            // 根据是否转置
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBK, matrixBNAlign};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA, layoutA, gmB, layoutB,
                                                   gmAAlign, layoutWA, gmBAlign, layoutWB,
                                                   alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else if (!transA && transB) {
            // 情况2：A不转置(RowMajor) + B转置(ColumnMajor)
            // A矩阵对齐K维度，B矩阵对齐K维度（转置后K变为行方向）
            using LayoutA = layout::RowMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAM, matrixAKAlign};
            LayoutB layoutWB{matrixBKAlign, matrixBN};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA, layoutA, gmB, layoutB,
                                                   gmAAlign, layoutWA, gmBAlign, layoutWB,
                                                   alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else if (transA && !transB) {
            // 情况3：A转置(ColumnMajor) + B不转置(RowMajor)
            // A矩阵对齐M维度（转置后M变为行方向），B矩阵对齐N维度
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::RowMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBK, matrixBNAlign};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA, layoutA, gmB, layoutB,
                                                   gmAAlign, layoutWA, gmBAlign, layoutWB,
                                                   alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        } else {
            // 情况4：A转置(ColumnMajor) + B转置(ColumnMajor)
            // A矩阵对齐M维度，B矩阵对齐K维度
            using LayoutA = layout::ColumnMajor;
            using LayoutB = layout::ColumnMajor;
            using AType = Gemm::GemmType<ElementA, LayoutA>;
            using BType = Gemm::GemmType<ElementB, LayoutB>;
            LayoutA layoutA{matrixAM, matrixAK};
            LayoutB layoutB{matrixBK, matrixBN};
            LayoutA layoutWA{matrixAMAlign, matrixAK};
            LayoutB layoutWB{matrixBKAlign, matrixBN};

            using TemplatePadder = Gemm::Kernel::TemplatePadder<ArchTag, AType, BType>;
            typename TemplatePadder::Params params{gmA, layoutA, gmB, layoutB,
                                                   gmAAlign, layoutWA, gmBAlign, layoutWB,
                                                   alignedA, alignedB};
            TemplatePadder padder;
            padder(params);
        }
    }
};
}

#endif
