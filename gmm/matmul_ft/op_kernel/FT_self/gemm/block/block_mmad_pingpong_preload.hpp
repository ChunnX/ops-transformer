
#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_SELF_PRELOAD
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP_SELF_PRELOAD

#include "catlass/catlass.hpp"
#include "FT_self/arch/resource_aiv.hpp"
#include "FT_self/arch/cross_core_sync_aiv.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "FT_self/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace FTSelf::Gemm::Block {
template<
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadPreload<
    Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    /*
    硬件架构： 分离架构
    分离架构将AI Core拆成矩阵计算（AI Cube，AIC）和向量计算（AI Vector，AIV）两个独立的核，
    每个核都有自己的Scalar单元，能独立加载自己的代码段，
    从而实现矩阵计算与向量计算的解耦，
    在系统软件的统一调度下互相配合达到计算效率优化的效果。
    AIV与AIC之间通过Global Memory进行数据传递。
    另外分离架构相比耦合架构，增加了两个Buffer：BT Buffer(BiasTable Buffer，存放Bias)和
    FP Buffer(Fixpipe Buffer，存放量化参数、Relu参数等)。

    1. AIC架构 （矩阵计算，Cube 核）
        a. 包含4个并行执行单元（搬运单元和计算单元）：MTE1，MTE2，Cube，Scalar，FixPipe
        b. 包含7个内存单元：GM（核外），L1，L0A，L0B，L0C，BiasTable Buffer，Fixpipe Buffer

    2. AIV架构 (向量计算， Vector 核)
        a. 包含4个并行执行单元：MTE2，MTE3，Vector，Scalar
        b. 包含2个内存单元：GM（核外），UB

    3. 典型计算数据流
        a. Vector计算：GM-UB-[Vector]-UB-GM
        b. Cube计算：
            1) GM-L1-L0A/L0B-[Cube]-L0C-FixPipe-GM
            2) GM-L1-L0A/L0B-[Cube]-L0C-FixPipe-L1 (进行后续的relu/scale等运算)

    注意：实际上，在分离架构的 L1 Buffer 中，A1/B1/C1 等均是逻辑上的概念，不存在独立的空间
    而是上述的存储TPosition 共享 L1 Buffer 空间，所以这里在L1 上统一使用了 A1。
    */
    // Type Aliases
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;

    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using TileMmad = TileMmad_;

    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA,ElementB>::ElementAccumulator;

    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;

    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;

    using LayoutCInL0 = Catlass::layout::zN;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::K * L1TileShape::N * sizeof(ElementB);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, Catlass::layout::RowMajor>, "LayoutC only support RowMajor yet!");

    // Check L1TileShape: 统一通过 A1 存储传输，所以要相加小于整体size
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);

    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    /*
    同步控制简介
    TQueSync类提供同步控制接口，开发者可以使用这类API来自行完成同步控制。
    需要注意的是，通常情况下，开发者基于编程模型中介绍的编程模型和范式进行编程时不需要关注同步，
    编程模型帮助开发者完成了同步控制；使用编程模型和范式是我们推荐的编程方式，
    自行同步控制可能会带来一定的编程复杂度，不建议开发者使用。
    TQueSync类接口和核间同步中提供的同步控制接口的区别在于，
    核间同步中的接口标注为ISASI类别，
    不能保证跨硬件版本兼容；TQueSync类接口可以保证跨硬件版本兼容。
    */

    /*
    I. 同步控制简介
    介绍同步控制之前需要先回顾一下抽象硬件架构。
    1. AI Core内部的异步并行计算过程：
        a. Scalar计算单元读取指令序列，
        并把向量计算、矩阵计算、数据搬运指令发射给对应单元的指令队列，
        向量计算单元、矩阵计算单元、数据搬运单元异步的并行执行接收到的指令。
        该过程可以表示为指令流。

        b. 不同的指令间有可能存在依赖关系，
        为了保证不同指令队列间的指令按照正确的逻辑关系执行，
        Scalar计算单元也会给对应单元下发同步指令。
        Scalar与向量计算单元、矩阵计算单元、数据搬运单元之间的同步过程
        可以表示为同步信号流。

        c.AI Core内部数据处理的基本过程：
        DMA搬入单元把数据搬运到Local Memory，
        Vector/Cube计算单元完成数据计算，
        并把计算结果写回Local Memory，
        DMA搬出单元把处理好的数据搬运回Global Memory。
        该过程可以抽象为数据流。

    AI Core内部的执行单元是异步并行的，
    在读写Local Memory内存时，可能存在依赖，需要进行同步控制。

    2. 典型示例：
    描述了一个常见的Vector计算数据流：
    1) 先通过DMA执行单元将数据从Global Memory搬入到Local Memory；
    2) 进行计算；
    3) 然后再通过DMA执行单元将计算结果从Local Memory搬出到Global Memory。

    四个执行单元Scalar、Vector、DMA(VECIN)、DMA(VECOUT)并行执行，
    若访问同一片Local Memory，需要同步机制来控制他们的访问时序：
    保证先搬入Local Memory后再计算，计算完成后再搬出。

    II. 硬件流水类型
    AI Core内部并行的指令流水类型和解释如下所示，
    硬件流水的详细介绍请参考基本架构章节。

                        表1 指令流水类型和相关说明
    流水类型                    含义
    PIPE_S              标量流水线，使用Tensor GetValue函数时为此流水

    PIPE_V              矢量计算流水及L0C->UB 数据搬运流水

    PIPE_M              矩阵计算流水

    PIPE_MTE1           L1->L0A、L1->L0B数据搬运流水

    PIPE_MTE2           GM->L1、GM->L0A、GM->L0B、GM->UB 数据搬运流水

    PIPE_MTE3           UB->GM、UB->L1数据搬运流水

    PIPE_FIX            L0C->GM、L0C->L1数据搬运流水，
                        当前版本暂不支持该流水类型。

    III. 同步控制分类
    对上述并行流水的同步控制分为两种：
    1. 多流水同步：通过TQueSync的SetFlag/WaitFlag或者SetFlag/WaitFlag(ISASI)接口进行不同流水线间的同步控制。
        a. SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，
        并将硬件中的对应标志位设置为1。
        b. WaitFlag：当执行到该指令时，如果发现对应标志位为0，该队列的后续指令将一直被阻塞；
        如果发现对应标志位为1，则将对应标志位设置为0，同时后续指令开始执行。
    2. 单流水同步：通过PipeBarrier(ISASI)完成同一流水线内的同步控制，
    用于在同一流水线内部约束执行顺序。其作用是，保证前序指令中所有数据的读写工作全部完成，
    后序指令才能执行。

    IV. 什么时候需要开发者手动插入同步
    1. Vector计算单元
        a. 单流水同步：PIPE_V由编译器自动完成同步插入，PIPE_MTE2/PIPE_MTE3在
            搬运地址有重叠的情况下需要开发者插入同步（具体示例请参考注意事项）。
        b. 多流水同步：PIPE_V、PIPE_MTE2、PIPE_MTE3、PIPE_S之间的多流水同步，都是双向的，
        PIPE_S 与 PIPE_V/PIPE_MTE2/PIPE_MTE3之间的同步由编译器自动完成同步插入，
        剩余的同步（上述其余流水内部的同步）由Ascend C框架完成。

    2. Cube计算单元: Cube侧所有流水同步都由Ascend C框架完成，不需要算子开发者插入同步。
    */
    /// Construct
    CATLASS_DEVICE
    BlockMmadPreload(Catlass::Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // block 上 L1 Cache 中 A1 和 B1 的起始地址偏移
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;

        // Init buffers
        for(uint32_t i=0; i < STAGES; i++) {
            // Catlass::Arch::Resource 封装了不同level Cache 上的 LocalTensorBuffer
            // Assign L1/L0A/L0B space for each stages
            /*
            LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
            */
            /*
            这里 L1 的存储共用 A1 位置，所以要注意起始地址要分隔开来，顺序访问
            */
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);

            /*
            L0 Cache 上面，分别使用 A2, B2 两个position，所以两个tensor的L0 Cache 均从0开始
            */
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES; // 保证同步时间编号不会冲突
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadPreload()
    {
        for(uint32_t i=0; i < STAGES; i++){
            // 等待相关内存事件完成后再结束运行
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }


    /*
    SetFlag/WaitFlag(ISASI)
    I. 功能说明:
    同一核内不同流水线之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。
    1. SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，
                并将硬件中的对应标志位设置为EventId。
    2. WaitFlag：当执行到该指令时，如果发现对应标志位为0，该队列的后续指令将一直被阻塞；
                 如果发现对应标志位为EventId，则将对应标志位设置为0，同时后续指令开始执行。

    II. 函数原型
    a.SetFlag：
    template <HardEvent event>
        __aicore__ inline void SetFlag(int32_t eventID)

    b.WaitFlag:
    template <HardEvent event>
    __aicore__ inline void WaitFlag(int32_t eventID)

    III.参数说明
                                    表1 参数说明
    参数名              输入/输出               描述
    event                 输入                模板参数。
                                            同步事件，数据类型为HardEvent。
                                            详细内容参考下文中的同步类型说明。

    eventID               输入              事件ID。数据类型为int32_t类型。其定义如下：
                                            eventID需要通过
                                            AllocEventID或者FetchEventID来获取。

                                            Atlas 训练系列产品，数据范围为：0-3
                                            Atlas 推理系列产品AI Core，数据范围为：0-7

                                            Atlas A2 训练系列产品/
                                            Atlas 800I A2 推理产品/
                                            A200I A2 Box 异构组件，数据范围为：0-7

                                            Atlas A3 训练系列产品/Atlas A3 推理系列产品，
                                            数据范围为：0-7
    */

    /*
    同步类型说明如下:

    enum class HardEvent : uint8_t {
        // 名称（源流水_目标流水），例如MTE2_V，代表PIPE_MTE2为源流水，PIPE_V为目标流水。
        // 标识从PIPE_MTE2到PIPE_V的同步，PIPE_V等待PIPE_MTE2。
        MTE2_MTE1
    MTE1_MTE2
    MTE1_M
    M_MTE1
    MTE2_V
    V_MTE2
    MTE3_V
    V_MTE3
    M_V
    V_M
    V_V
    MTE3_MTE1
    MTE1_MTE3
    MTE1_V
    MTE2_M
    M_MTE2
    V_MTE1
    M_FIX
    FIX_M
    MTE3_MTE2
    MTE2_MTE3
    S_V
    V_S
    S_MTE2
    MTE2_S
    S_MTE3
    MTE3_S
    MTE2_FIX
    FIX_MTE2
    FIX_S
    M_S
    FIX_MTE3
    */

    /*
    4. 返回值： 无

    5. 约束说明
    1. SetFlag/WaitFlag必须成对出现。
    2. 禁止用户在使用SetFlag和WaitFlag时，自行指定eventID，容易与框架同步事件冲突，导致卡死问题。eventID需要
       通过AllocEventID或者FetchEventID来获取。
    */

    /*
    AllocEventID
    I. 功能说明
    用于申请HardEvent（硬件类型同步事件）的TEventID，必须与ReleaseEventID搭配使用，
    调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。

    II. 函数原型
    template <HardEvent evt>
    __aicore__ inline TEventID TPipe::AllocEventID()

    III. 参数说明
                                    表1 模板参数说明
    参数名                                      描述
    evt                 HardEvent硬件同步类型。该类型的具体说明
                        请参考SetFlag/WaitFlag(ISASI)中同步类型的说明。

    IV. 约束说明
    TEventID有数量限制，使用结束应该立刻调用ReleaseEventID释放，防止TEventID耗尽。

    V. 返回值
    TEventID

    VI. 调用示例
    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //需要插scalar等vector的同步，申请对应的HardEvent的ID
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
    ......
    ......
    ......
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //释放scalar等vector的同步HardEvent的ID
    ......
    */

    /*
    FetchEventID
    I. 功能说明
    根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，
    仅提供可用的TEventID。

    II. 函数原型
    template <HardEvent evt>
    __aicore__ inline TEventID TPipe::FetchEventID()
    __aicore__ inline TEventID TPipe::FetchEventID(HardEvent evt)

    III. 参数说明
    参数名                          输入/输出                       含义
    evt                               输入              HardEvent类型，硬件同步类型。
                                                        该类型的具体说明请参考
                                                        SetFlag/WaitFlag(ISASI)中
                                                        同步类型的说明。

    IV. 约束说明
    相比于AllocEventID，FetchEventID适用于临时使用ID的场景，获取ID后，不会对ID进行占用。
    在一些复杂的使用场景下，需要开发者自行保证使用正确。
    比如相同流水连续调用SetFlag/WaitFlag，如果两次传入的ID都是使用FetchEventID获取的，
    因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用AllocEventID。

    V. 返回值：TEventID

    VI. 调用示例
    AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //需要插scalar等vector的同步，申请对应的HardEvent的ID
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);

    */

    /*
    ReleaseEventID
    I. 功能说明
    用于释放HardEvent（硬件类型同步事件）的TEventID，通常与AllocEventID搭配使用。

    II. 函数原型
    template <HardEvent evt>
    __aicore__ inline void ReleaseEventID(TEventID id)

    III. 参数说明
                                表1 模板参数说明
        参数名                          描述
        evt                 HardEvent硬件同步类型。
                        该类型的具体说明请参考SetFlag/WaitFlag(ISASI)中同步类型的说明。

                                表2 参数说明
        参数名          输入/输出                   描述
        id                 输入             TEventID类型，
                                        调用AllocEventID申请获得的TEventID。

    IV. 约束说明
    AllocEventID、ReleaseEventID需成对出现，
    ReleaseEventID传入的TEventID需由对应的AllocEventID申请而来。

    V. 返回值：无

    VI. 调用示例
    AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //需要插scalar等vector的同步，申请对应的HardEvent的ID
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
    ......
    ......
    ......
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //释放scalar等vector的同步HardEvent的ID
    ......
    */




    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const & gmA,
        AscendC::GlobalTensor<ElementA> const & gmNextA,
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const & gmB,
        AscendC::GlobalTensor<ElementB> const & gmNextB,
        LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const & gmC, LayoutC const &layoutC,
        Catlass::GemmCoord const &actualShape, Catlass::GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock
    )
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t mRoundNext = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShapeNext.m());
        uint32_t nRoundNext = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShapeNext.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);

        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRound, nRound));
        auto layoutInL0CNext = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(mRoundNext, nRoundNext));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);


        if(isFirstBlock){
            // load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            // 设定Tile 在global memory 中的layout, 即将一个Tile作为一个矩阵中的一部分，
            // 将其中的元素重新组织为与layoutA相同类型的的layout，其中shape为Tile规模
            // 但是每个元素/分形 行和列之间的 stride 还是按照原来整体layout的 shape/stride 来进行组织
            // 因为这里每个block是只输入并处理一个L1 Tile
            auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(),kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActual,actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }


        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        for(uint32_t kLoopIdx=0; kLoopIdx < kTileCount; kLoopIdx++)
        {
            // 下一阶段执行的stage id
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES)? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // 流水线，提前将下一阶段的数据从 GM 加载到 L1 中与计算overlap
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1){
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);

                // Get L1 Tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];

                // Get GM tile for next stage
                Catlass::MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                Catlass::MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};

                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            } else if (hasNextBlock){
                uint32_t kLoopIdxNext = 0;
                // 下一阶段 若非最后一个 loop，那么执行一个L1TileShape::K,否则执行剩余的数据
                kActualNext = min(actualShapeNext.k(), L1TileShape::K);

                // Get L1 Tensor for next stage
                auto l1ANextTensor = l1ATensorList[l1ListIdNext];
                auto l1BNextTensor = l1BTensorList[l1ListIdNext];

                // Get GM tile for next stage
                Catlass::MatrixCoord gmTileNextAOffset{0, kLoopIdxNext * L1TileShape::K};
                Catlass::MatrixCoord gmTileNextBOffset{kLoopIdxNext * L1TileShape::K, 0};

                auto gmTileNextA = gmNextA[layoutA.GetOffset(gmTileNextAOffset)];
                auto gmTileNextB = gmNextB[layoutB.GetOffset(gmTileNextBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileNextA = layoutA.GetTileLayout(Catlass::MakeCoord(actualShapeNext.m(), kActualNext));
                copyGmToL1A(l1ANextTensor, gmTileNextA, layoutAInL1, layoutTileNextA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileNextB = layoutB.GetTileLayout(Catlass::MakeCoord(kActualNext, actualShapeNext.n()));
                copyGmToL1B(l1BNextTensor, gmTileNextB, layoutBInL1, layoutTileNextB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 Tensor for current usage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            for(int mPartIdx=0; mPartIdx < mPartLoop; mPartIdx++){
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                for(int kPartIdx=0; kPartIdx < kPartLoop; kPartIdx++){
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual,kPartActual);

                    // Locate the current tile of matrix A on L1
                    Catlass::MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};

                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    // 等待现在的L0阶段中之前在L0上的数据消费已经完成，即相应 MMAD 计算完成
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if((mPartIdx == 0) && (kPartIdx == 0)){
                        // 若为当前stage第一次迭代，需要等待到第一批数据，
                        // 即在当前stage涉及的迭代前已经preload 完的A数据成功preload 到 L1 上才可进行A数据向 L0 上写
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1))
                    {
                        // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                        // 即当前stage涉及到现有 A1的数据的操作已经完成
                        // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 A1 数据中写入新的数据了
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for(int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++){
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutBInL0 layoutBInL0 =
                            LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);

                        // Locate the current tile of matrix B on L1
                        Catlass::MatrixCoord l1BOffset{kPartIdx * L0TileShape::K,
                            nPartIdx * L0TileShape::N};

                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        // 等待当前阶段L0 数据消费完成，即相应 MMAD 计算完成
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if((nPartIdx == 0) && (kPartIdx == 0)){
                            // 若为当前stage第一次迭代，需要等待到第一批数据，
                            // 即在当前stage涉及的迭代前已经preload 完的B数据成功
                            // preload 到 L1 上才可进行B数据向 L0 上写
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            // 若这是当前stage 最后一次的迭代，则需要设置Flag，
                            // 即当前stage涉及到现有 L0 B的数据的操作已经完成
                            // 之后再进行PING-PONG preload时，可以向该阶段对应的L1 B 数据中写入新的数据了
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }

                        // Notify to do mmad
                        // 标记L0 Cache 中 A2 和 B2 的数据写入已经完成，可以进行 MMAD 运算了
                        // 同时也标记当前开始进行 L0 的 CO1 Cache 了。
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        Catlass::MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        // 获取当前局部输出
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        // 等待 L0 上 A2 和 B2 的写入完成，当源数据写入完成后，即可进行矩阵运算了。
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        // 当前 M,N Tile 的第一个 K 时，需要初始化输出的C矩阵为0

                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual,
                            nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        // 标记计算完成，即当前已经完成了一个l0 tile 的运算，可以加载下一个l0 tile了
                        // 这里最内层为 B 矩阵
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    // 交替进行阶段，实现 L1 与 L0 之间数据传输与 MMAD 计算的PING-PANG
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }

            // 交替进行阶段，实现L1 与 Global 之间数据传输 与 MMAD计算 的PING-PANG
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        // 将最终结果从 L0 的 CO1 输出到GM即可
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            // 标记开始写入cGM 数据
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            // 等待允许写入开始
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            // 写入数据
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            // 标记数据写入 cGM 已经完成
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];

    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];

    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];

    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    // 指示当前所处的pipeline 中的阶段（双阶段PING-PONG）
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
} // namespace FTSelf::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_HPP
