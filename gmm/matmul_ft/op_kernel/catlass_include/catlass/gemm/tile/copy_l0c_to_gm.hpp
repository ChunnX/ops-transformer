#ifndef FTSELF_CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define FTSELF_CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include_next "catlass/gemm/tile/copy_l0c_to_gm.hpp"

#include "catlass/gemv/helper.hpp"

namespace FTSelf::Gemm::Tile {

using Catlass::Gemm::Tile::CopyL0CToGmQuantMode;
using Catlass::Gemm::Tile::ScaleGranularity;

template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to L1, can not find the specialization.");
};
/*
template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
*/

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_>
struct CopyL0CToL1<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Catlass::Gemm::GemmType<ElementDst_, Catlass::layout::zN>,
                   FTSelf::Gemv::helper::FT_L02L1_TYPE::FIX_PIPE,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout,
        uint8_t unitFlag = 0, bool isChannelSplit=false)
    {

        AscendC::FixpipeParamsV220 intriParams;

        /*
                表3 Fixpipe搬运参数结构体说明
            参数名称            数据类型                        含义
            nSize               输入                源NZ矩阵在N方向上的大小。
                                                    不使能NZ2ND功能
                                                        若使能channelSplit功能，nSize必须为8的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                        若不使能channelSplit功能，nSize必须为16的倍数，
                                                        取值范围：nSize∈[1, 4095]。

                                                    使能NZ2ND功能: nSize取值范围 ∈[1, 4095]。

            mSize               输入                源NZ矩阵在M方向上的大小。
                                                    不使能NZ2ND功能
                                                        取值范围：mSize∈[1, 65535]。
                                                        使能NZ2ND功能
                                                        取值范围：mSize∈[1, 8192]。

            srcStride           输入                源NZ矩阵中相邻Z排布的起始地址偏移，
                                                    取值范围：srcStride∈[0, 65535]，
                                                    单位：C0_Size(16*sizeof(T)，T为srcLocal的数据类型)。

            dstStride           输入                不使能NZ2ND功能
                                                    目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0，
                                                    单位：datablock(32Bytes)。

                                                    使能NZ2ND功能
                                                    目的ND矩阵每一行中的元素个数，取值不为0 ，
                                                    单位：element。

            quantPre            输入                QuantMode_t是一个枚举类型，用于控制量化模式，
                                                    默认值为QuantMode_t::NoQuant，即不使能量化功能。
                                                    QuantMode_t取值如下：
                                                    NoQuant，不使能量化功能
                                                    F322F16，float量化成half
                                                    F322BF16，float量化成bfloat16_t
                                                    DEQF16，int32_t量化成half, scalar量化
                                                    VDEQF16， int32_t量化成half，tensor量化
                                                    QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化
                                                    VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化
                                                    REQ8，int32_t量化成uint8_t/int8_t，scalar量化
                                                    VREQ8，int32_t量化成uint8_t/int8_t，tensor量化

            deqScalar           输入                scalar量化参数，表示单个scale值，
                                                    quantPre量化模式为scalar量化时需要设置该参数。
                                                    支持的数据类型为uint64_t。

            ndNum               输入                源NZ矩阵的数目，也就是传输ND矩阵的数目，
                                                    取值范围：ndNum∈[1, 65535]

            srcNdStride         输入                不同NZ矩阵起始地址之间的间隔，
                                                    取值范围：srcNdStride∈[1, 512]，单位：1024B。
                                                    当ndNum配置为1时，srcNdStride配置为0即可，不生效。

            dstNdStride         输入                目的相邻ND矩阵起始地址之间的偏移，
                                                    取值范围：dstNdstride∈[1, 65535]，单位：element。
                                                    当ndNum配置为1时，dstNdStride配置为0即可，不生效。

            reluEn              输入                是否使能relu的开关，false：不使能relu功能；
                                                    true：使能relu功能。

            unitFlag            输入                预留参数，用户无需关心，使用默认值0即可。

            isChannelSplit      输入                是否使能通道拆分的功能。默认为false，不使能该功能。
                                                    仅在src和dst都为float时才能使能通道拆分，
                                                    且不能同时使能ChannelSplit和NZ2ND功能。
        */

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (Catlass::BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.isChannelSplit = isChannelSplit;

        /*
        template <typename DstT, typename SrcT,
         const FixpipeConfig& config = CFG_ROW_MAJOR>
        void Fixpipe(const LocalTensor<DstT>& dstLocal,
            const LocalTensor<SrcT>& srcLocal,
            const FixpipeParamsM300& intriParams)
        */

        // Call AscendC Fixpipe

        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};


/*
template <
    class ArchTag,
    class ElementAccumulator,
    class L1Type,
    FTSelf::Gemv::helper::FT_L02L1_TYPE CopyType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
*/

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_>
struct CopyL0CToL1<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Catlass::Gemm::GemmType<ElementDst_, Catlass::layout::zN>,
                   FTSelf::Gemv::helper::FT_L02L1_TYPE::CO12DST,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout,
        uint8_t unitFlag = 0, bool isChannelSplit=false,
        uint8_t reluPre=0,float scalar_for_deq=1.0)
    {

        AscendC::DataCopyCO12DstParams intriParams;

        /*
                表10 DataCopyCO12DstParams结构体参数定义（C0=16）
        参数名称                             含义
        nSize                       srcLocal横向方向的size大小。
                                    不使能NZ2ND功能，必须为C0的倍数，
                                    此时连续传输数据块的个数为nSize/C0。
                                    使能NZ2ND功能，不受限制。

        mSize                       srcLocal纵向方向的size大小。
                                    不使能NZ2ND功能，连续传输数据块的大小为
                                    mSize * C0个元素的长度。

                                    使能NZ2ND功能，NZ/ND矩阵的大小为mSize*nSize。

        dstStride                   不使能NZ2ND功能
                                    dstLocal相邻连续数据片段间隔
                                    （前面一个数据块的头与后面数据块的头的间隔），
                                    取值不为0。
                                    针对Atlas A2 训练系列产品/
                                    Atlas 800I A2 推理产品/
                                    A200I A2 Box 异构组件 ，
                                    单位为datablock(32Bytes)。

                                    针对Atlas A3 训练系列产品/Atlas A3 推理系列产品 ，
                                    单位为datablock(32Bytes)。

                                    使能NZ2ND功能
                                    dstLocal同一ND矩阵的相邻行的偏移(头与头)，
                                    取值不为0， 单位为元素。

        srcStride                   不使能NZ2ND功能：srcLocal相邻连续数据片段间隔
                                    （前面一个数据块的头与后面数据块的头的间隔），
                                    必须为16的倍数。
                                    取值范围：srcStride∈[0, 65535]，
                                    单位：C0_Size(C0*sizeof(U)，
                                    U为srcLocal的数据类型)。

                                    使能NZ2ND功能
                                    srcLocal同一NZ矩阵的相邻Z排布的偏移（头与头），
                                    必须为16的倍数，取值范围：
                                    srcStride∈[0, 65535]，单位C0_size。

        quantPre                    用于控制量化模式，QuantMode_t类型，
                                    具体定义如下。
                                    默认值为QuantMode_t::NoQuant，
                                    即不使能量化功能。
                                    配置为scalar量化时，
                                    需要调用SetFixpipePreQuantFlag接口来
                                    设置scalar量化参数；
                                    配置为tensor量化时，
                                    需要调用SetFixPipeConfig来设置tensor量化参数。

                                    enum QuantMode_t
                                    {
                                    NoQuant, // 不使能量化功能
                                    F322F16, // float量化成half, scalar量化
                                    F322BF16, // float量化成bfloat16_t, scalar量化
                                    DEQF16,  // int32_t量化成half, scalar量化
                                    VDEQF16, // int32_t量化成half，tensor量化
                                    QF322B8_PRE, // float量化成int8_t/uint8_t，scalar量化
                                    VQF322B8_PRE, // float量化成int8_t/uint8_t，tensor量化
                                    REQ8, // int32_t量化成int8_t/uint8_t，scalar量化
                                    VREQ8, // int32_t量化成int8_t/uint8_t，tensor量化
                                    };
        reluPre                    用于配置relu操作的模式，类型为uint8_t，取值如下：
                                    0：不使能relu
                                    1：Normal relu

        channelSplit                类型为bool，配置是否使能channel切分，
                                    对于float类型的dstLocal生效。
                                    false：不使能
                                    true：使能

        nz2ndEn                     类型为bool，配置是否使能NZ2ND的格式转换，
                                    仅在L0C->GM通路生效。
                                    如果要使能NZ2ND的功能需要同步调用SetFixpipeNz2ndFlag来
                                    设置格式转换的相关配置信息。
                                    false：不使能
                                    true：使能

        clipReluPre                 用于配置是否使能ClipRelu操作，
                                    参数类型为uint8_t，取值如下：0，不使能ClipRelu；
                                    1，使能ClipRelu，此时需要调用SetFixPipeClipRelu来设置clipRelu的最大值。
                                    该操作在随路量化后进行，quantPre配置后才能使用，
                                    当前支持的量化模式有F322F16/DEQF16/VDEQF16/QF322B8_PRE/VQF322B8_PRE/
                                    REQ8/VREQ8。
                                    该参数仅在Atlas 200I/500 A2 推理产品支持。

        eltWiseOp                   用于配置是否使能element-wise操作及操作模式。
                                    element-wise操作是指进行随路量化后，
                                    可以逐个元素加/减一个LocalTensor，
                                    大小为mSize * nSize，
                                    具体LocalTensor地址相关参数需要调用
                                    SetFixPipeAddr来设置。
                                    eltWiseOp参数类型为uint8_t，取值如下：
                                    0：不使能element-wise
                                    1：element-wise Addition
                                    2：element-wise Subtraction
                                    该参数仅在Atlas 200I/500 A2 推理产品支持。
        sid                         预留参数，为后续的功能做保留，开发者暂时无需关注。
        */

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (Catlass::BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        // if(reluEn){
        //     intriParams.reluPre = 1;
        // }else{
        //     intriParams.reluPre = 0;
        // }

        intriParams.reluPre = 0;

        // intriParams.unitFlag = unitFlag;
        intriParams.channelSplit = isChannelSplit;

        intriParams.nz2ndEn = false;

        float tmp = (float)scalar_for_deq;
        // 将float的tmp转换成uint64_t的deqScalar
        uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));

        // AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
        /*
        template <typename DstT, typename SrcT,
         const FixpipeConfig& config = CFG_ROW_MAJOR>
        void Fixpipe(const LocalTensor<DstT>& dstLocal,
            const LocalTensor<SrcT>& srcLocal,
            const FixpipeParamsM300& intriParams)
        */

        // Call AscendC Fixpipe

        /*
        template <typename T, typename U>
        __aicore__ inline void DataCopy(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<U>& srcLocal,
            const DataCopyCO12DstParams& intriParams);
        */
        // , AscendC::CFG_NZ
        AscendC::Fixpipe<ElementDst, ElementSrc>(dst, src, intriParams);
    }
};

} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
