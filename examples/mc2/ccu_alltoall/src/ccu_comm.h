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
 * \file ccu_comm.h
 * \brief CCU 通信公共件——事件契约、LoopGroup 搬运引擎、host 侧 CCU 资源管理
 *
 * 内容三块（可被任意 CCU 通信消费方复用）：
 *   契约   CcuMappedAddrs（CKE 事件槽位的 MMIO 地址映射，非 GM 内存）、
 *          A2aGoSize（本地拷贝 loop 参数四元组）、通信参数 GM staging 通路
 *   引擎   LoopGroup 本地拷贝（GM → CCU Buffer → GM）、loop 参数位域编码、CalGoSize
 *   资源   CCU 实例创建（跨 DIE 资源池）、P2P 通道获取（host 侧）
 *
 * 命名规则（词根=功能域；新增函数/类型按此对号，防演化漂移）：
 *   Ccu      = CCU 传输原语——只碰寄存器/实例/通道语义（wake、轮询 done、销毁实例），
 *              与具体集合通信算法无关；出现第二个 CCU 服务时，此类可上提本文件共享
 *   AllToAll = 服务函数——语义绑定 AllToAll 算法（轮参数布局、轮结构资源、服务
 *              流程），定义于本文件 §2~§4 节
 *   存量类型名未按此对齐（CcuCommTilingData 实为 AllToAll 服务 tiling、
 *   CcuKernelArgAllToAllMesh1D 双词根混用），待后续统一
 *
 * 物理本质：Variable = XN 寄存器、Event = CKE 完成事件寄存器（CCU 片上，非 GM 内存）；
 * AIC 经 WriteGmByPassDCache / ReadGmByPassDCache 直写/轮询（MMIO uncached 语义，
 * 寄存器不走 DCache 一致性协议）。预约随 HcommCcuInsDestroy 统一解映射还池。
 */
#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>

#include <acl/acl_rt.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>
#include <hccl/hcomm_primitives.h>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>

#include "hccl_ccu_res.h"
// asc-comm 正式版 host launch 头（CANN 9.2.0+）：<<<>>> CCU kernel 语法依赖
// 其 asccomm_ccu_schd / HcommCcuHostKernelLaunch 声明
#include "ccu_host_launch.h"
#include "ccu_res.h"

// 条件 include 服务于本文件的 device 侧函数（engine）；消费方的 AIC kernel 入口
// 亦依赖（InitSocState 等），各自头文件自带同款条件 include
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#ifdef ACLCHECK
#undef ACLCHECK
#endif

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(call)                                                                          \
    do {                                                                                         \
        auto ret = (call);                                                                       \
        int retCode = static_cast<int>(ret);                                                     \
        HcclResult hcclRet = static_cast<HcclResult>(ret);                                       \
        if (retCode >= static_cast<int>(CCU_E_DRV_START) &&                                      \
            retCode <= static_cast<int>(CCU_E_RESERVED)) {                                       \
            hcclRet = HCCL_E_INTERNAL;                                                           \
        }                                                                                        \
        if (hcclRet != HCCL_SUCCESS) {                                                           \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__,        \
                retCode);                                                                        \
            return hcclRet;                                                                      \
        }                                                                                        \
    } while (0)

#define CHK_PTR_NULL(ptr)       \
    do {                        \
        if ((ptr) == nullptr) { \
            return HCCL_E_PTR;  \
        }                       \
    } while (0)

namespace ccu = ::AscendC::ccu;

#define CCU_CHK_RET(call)                                \
    do {                                                 \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) {                     \
            return ccuRet;                               \
        }                                                \
    } while (0)

#define CCU_CHK_VOID_RET(call)                           \
    do {                                                 \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) {                     \
            return;                                      \
        }                                                \
    } while (0)

// ---- 本地拷贝 LoopGroup 机器参数与资源配额 ----
constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_MEMSLICES_PER_LOOP = 8;
constexpr uint32_t CCU_LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint64_t CCU_MAX_RANK_SIZE = 16;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t CCU_DIE_NUM = 2;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS = 400;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP = 16;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF = 128;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE = 400;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT = 48;
constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD = 2;

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

// ---- 通信参数通路 ----
// 参数直达：host 每次调用把 CcuCommTilingData 整体写入 GM staging（stageAddr），
// CCU kernel 唤醒后自读并按轮闭式派生（LoadRoundArgs）；AIC 不再经手通信参数
// （历史设计：AIC 逐槽 MMIO 推送 55 个寄存器槽位 + CCU 每轮重绑，已退役）
constexpr uint32_t MAX_COMMTURN = 16;

// CKE 事件槽位到 MMIO 映射地址的映射表（AIC 清扫/commit/逐 tile 轮询用）
struct CcuMappedAddrs {
    // dataCommTurn+1 个 Event：[0]=wake，[t+1]=data 轮 t done（轮 0 含搭载的 scale 整块）
    uint64_t eventVa[MAX_COMMTURN + 2];
};

// CCU 本地拷贝 loop 参数四元组（CalGoSize 输出，经 tiling 传至 AIC；字段名与 device 侧
// GroupOpSizeVars 保持一致，POD 可经 <<<>>> 传值）
struct A2aGoSize {
    uint64_t addrOffset;
    uint64_t loopIterNum;
    uint64_t loopExtendNum;
    uint64_t tailSize;
};

// ---- LoopGroup 引擎结构 ----
struct LoopGroupConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount;
    uint32_t bufCount;
};

// CCU 本地拷贝 loop 参数四元组（Variable 版）——字段名与 host 侧 A2aGoSize 保持一致
struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopIterNum;
    ccu::Variable loopExtendNum;
    ccu::Variable tailSize;
};

struct CcuLoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParam[2];
};

// CCU kernel 随参携带的通信参数块（host 每次调用刷新）。按值嵌入 kernelArg——
// CCU kernel 只读参数块本体，不 chase 二级指针（launch 参数块是 CCU 侧唯一确定
// 可解引用的载体）；字段语义与消费方服务 tiling（本文件 §2 ccu_all_to_all_tiling 节）共用此定义
struct CcuCommStageParams {
    // data 轮闭式派生源标量（src=t*subBytes、dst=rankId*blockSize+t*subBytes、slice=subBytes）
    uint64_t blockSize{0};
    uint64_t subBytes{0};
    uint64_t srcStride{0};
    // scale 整块（随 data 轮 0 搭载，不拆分）闭式派生源标量：dst = 对端发布地址
    // (recvBuf) + scaleDstOffset——scaleDstOffset 由 host 烘焙 region 偏移
    // （recvScale 紧邻 recvData：recvScale = recvBuf + inputSize，workspace 布局
    // 全组一致）+ rankId*scaleSliceSize，CCU 侧从发布值闭式派生（单类发布）
    uint64_t scaleSliceSize{0};
    uint64_t scaleSrcStride{0};
    uint64_t scaleDstOffset{0};
    // 通信地址（帧可变：输入/接收 buffer 可能换址）。recvScale 基址不再单独下发——
    // scale 目的地址由发布值 + scaleDstOffset 派生
    uint64_t x1Addr{0};
    uint64_t x1ScaleAddr{0};
    uint64_t recvBufAddr{0};
    // CCU 内存访问令牌（data / scale 两 buffer 源侧各一）。对端发布的仅为 token：
    // 远端写 data 与 scale 两个目的区共用同一令牌（实证令牌对远端写目的 region
    // 宽松——现状两类令牌本就都用于写对端同一 workspace）
    uint64_t token{0};
    uint64_t scaleToken{0};
    // 本地拷贝 loop 参数（data / scale 各一）
    A2aGoSize goSize{};
    A2aGoSize scaleGoSize{};
};

// CCU kernel 参数块（对齐 CANN CcuKernelArgAlltoAllMesh1D；通信参数随参按值直达）
struct CcuKernelArgAllToAllMesh1D {
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t channelCount;
    uint64_t rankSize;
    uint32_t rankId;
    CcuEventHandle eventHandle;
    uint32_t eventNum;
    uint32_t commTurn;  // CCU kernel 内部循环次数 = dataCommTurn + 1
    CcuCommStageParams stage;  // 通信参数（host 每次调用刷新，CCU 唤醒后自读）
};

// 对端路由：每对端 2 个通道槽位句柄（output+token 成对；单类发布，全帧一组值，
// scale 目的地址由对端从 output 值闭式派生，槽位不再中途覆写）
struct PeerRoute {
    ccu::Variable output;
    ccu::Variable token;
};

// 单货类换块传输的轮参数（data 轮 / scale 块共用同一传输模式；LoadRoundArgs 从
// stage 闭式派生后 imm 赋值进寄存器）
struct AllToAllBlockParams {
    ccu::Variable input;       // 发送侧基址（x1Addr / x1ScaleAddr）
    ccu::Variable srcOffset;   // 轮级源偏移
    ccu::Variable srcToken;    // 发送侧访问令牌（x1 / x1Scale 各自的源侧 token）
    ccu::Variable dstOffset;   // 目的偏移（相对对端发布地址；scale 已含 region 偏移）
    ccu::Variable sliceSize;   // 本轮子段字节数
    ccu::Variable srcStride;   // 块间距字节数（均匀 AllToAll 紧邻切分）
    GroupOpSizeVars goSize;    // 本地拷贝 loop 参数
};

// CCU kernel 工作上下文（传输引擎的运行态）
struct AllToAllMesh1DContext {
    const CcuKernelArgAllToAllMesh1D *arg;

    // 本 rank 轮参数（data 轮 t；scale 整块仅随轮 0 搭载，其余轮不装载不消费）
    AllToAllBlockParams data;
    AllToAllBlockParams scale;
    // 本 rank self output/token（帧首单类发布一次：recvBuf + token；localDst 同源）
    ccu::Variable selfOutput;
    ccu::Variable selfToken;
    // 对端路由（live 绑定通道槽位——对端发布后即为本帧值；沿用 vector 而非定长数组：
    // Variable 默认构造即分配寄存器槽，定长数组会为未用 rank 空配 16×2 槽）
    std::vector<PeerRoute> routing;
    // Write 完成事件：完成位仅由 ccu::Write 置位收集；轮内不再有 Event 掩码屏障
    // （round-done 的落地确认走 PostSync 通道握手，见 _kernel.asc）
    ccu::Event event;

    LoopGroupConfig moConfig;
    LoopGroupResource moRes;
    bool resourceAllocated = false;

    std::map<std::string, CcuLoopEntity> loopMap;

    void CreateLoopEntity(std::string loopStr)
    {
        loopMap.emplace(loopStr, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(std::string loopStr)
    {
        return loopMap.count(loopStr) != 0;
    }
};

// ---- CCU 实例资源描述（默认配额）----
struct CcuDefaultRes {
    HcommCcuResType type;
    uint32_t num;
};

static constexpr CcuDefaultRes CCU_DEFAULT_RES[] = {
    {HCOMM_CCU_RES_TYPE_ADDRESS,    CCU_DEFAULT_RES_FRACTION_ADDRESS},
    {HCOMM_CCU_RES_TYPE_LOOP,       CCU_DEFAULT_RES_FRACTION_LOOP},
    {HCOMM_CCU_RES_TYPE_CCU_BUF,    CCU_DEFAULT_RES_FRACTION_CCU_BUF},
    {HCOMM_CCU_RES_TYPE_VARIABLE,   CCU_DEFAULT_RES_FRACTION_VARIABLE},
    {HCOMM_CCU_RES_TYPE_EVENT,      CCU_DEFAULT_RES_FRACTION_EVENT},
    {HCOMM_CCU_RES_TYPE_CCU_THREAD, CCU_DEFAULT_RES_FRACTION_CCU_THREAD},
};

// ---- CCU loop/offset 参数的 64-bit 寄存器位域编码工具 ----

// 生成低 (end+1) 位全 1 的掩码
static constexpr uint64_t SetBits(uint16_t end)
{
    return ((uint64_t(1) << (end + 1)) - uint64_t(1));
}

static uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return SetBits(loopNumBitNum);
}

// 编码 loopCtxId(8b) + gsaOffset(32b) + loopIterNum(13b)
static uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdBitNum = 8;
    constexpr uint16_t ctxIdShiftBit = 45;
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 13;
    constexpr uint16_t loopNumBitNum = 13;
    constexpr uint16_t loopNumShiftBit = 0;
    return ((loopCtxId & SetBits(ctxIdBitNum)) << ctxIdShiftBit)
           | ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
           | ((loopIterNum & SetBits(loopNumBitNum)) << loopNumShiftBit);
}

// 编码 repeatNum(7b) + repeatLoopIndex(7b) + totalLoopNum(7b)
static uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum = 7;
    constexpr uint16_t repeatNumShiftBit = 55;
    constexpr uint16_t repeatLoopBitNum = 7;
    constexpr uint16_t repeatLoopShiftBit = 48;
    constexpr uint16_t totalLoopBitNum = 7;
    constexpr uint16_t totalLoopShiftBit = 41;
    return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
           | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
           | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
}

// 编码 gsaOffset(32b) + msOffset(11b) + ckeOffset(10b)
static uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t gsaBitNum = 32;
    constexpr uint16_t gsaShiftBit = 21;
    constexpr uint16_t msBitNum = 11;
    constexpr uint16_t msShiftBit = 10;
    constexpr uint16_t ckeBitNum = 10;
    constexpr uint16_t ckeShiftBit = 0;
    return ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
           | ((msOffset & SetBits(msBitNum)) << msShiftBit)
           | ((ckeOffset & SetBits(ckeBitNum)) << ckeShiftBit);
}

// 本地拷贝 LoopGroup 机器配置的单一装配出处（host 侧与 device 侧共用）
static LoopGroupConfig MakeLocalCopyConfig()
{
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_MEMSLICES_PER_LOOP;
    return config;
}

// 计算本地拷贝的 loop 参数四元组：sliceSize 超过单次 LoopGroup 容量（256KB）时分段
static A2aGoSize CalGoSize(uint64_t size, const LoopGroupConfig &config)
{
    uint64_t loopSize = config.loopCount * config.memSlice;
    uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t m = size / loopSize;
    uint64_t n = (size - m * loopSize) / config.memSlice;
    uint64_t p = size - m * loopSize - n * config.memSlice;

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = config.loopCount - 1;
        p = config.memSlice;
    }

    A2aGoSize goSize{};
    goSize.addrOffset = config.memSlice * config.loopCount * m;
    goSize.loopIterNum = m;

    constexpr uint64_t loopNumTwo = 2;
    if (n == 0 && p == 0) {
        goSize.loopExtendNum = 0;
        goSize.tailSize = 0;
    } else if (n != 0 && p == 0) {
        goSize.loopExtendNum = GetParallelParam(n - 1, 0, 1);
        goSize.tailSize = config.memSlice;
    } else if (n == 0 && p != 0) {
        goSize.loopExtendNum = GetParallelParam(0, 0, 1);
        goSize.tailSize = p;
    } else {
        goSize.loopExtendNum = GetParallelParam(n - 1, 1, loopNumTwo);
        goSize.tailSize = p;
    }

    return goSize;
}

// 初始化本地拷贝的 Loop 资源：首次调用分配 Buffer/Event；每次调用重新创建 Loop Entity
// （按值捕获当前地址——只创建一次会复用旧地址快照导致数据错位）
static void InitLocalCopyResources(AllToAllMesh1DContext &ctx, ccu::LocalAddr *loopSrc,
                                   ccu::LocalAddr *loopDst, ccu::Variable *loopLen)
{
    if (!ctx.resourceAllocated) {
        ctx.moConfig = MakeLocalCopyConfig();

        ctx.moRes.eventCount = ctx.moConfig.loopCount;
        ctx.moRes.completedEvent = ccu::Array<ccu::Event>(ctx.moRes.eventCount);

        ctx.moRes.bufCount = ctx.moConfig.loopCount * ctx.moConfig.msInterleave;
        ctx.moRes.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.moRes.bufCount);

        ctx.resourceAllocated = true;
    }

    {
        std::string loopType = "localcopy";
        if (ctx.IsLoopEntityRegistered(loopType)) {
            ctx.loopMap.erase(loopType);
        }
        ctx.CreateLoopEntity(loopType);
        auto &entity = ctx.loopMap[loopType];
        for (uint32_t index = 0; index < 2; index++) {
            uint32_t bufBase = index * ctx.moConfig.msInterleave;
            ccu::Event loopEvt = ctx.moRes.completedEvent[index];
            entity.body[index].reset(new ccu::Func(
                [&ctx, index, bufBase, loopEvt, loopSrc, loopDst, loopLen]() {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase], loopSrc[index], loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                    ccu::LocalCopy(loopDst[index], ctx.moRes.ccuBuf[bufBase], loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }));
            entity.loops[index].reset(
                new ccu::Loop(entity.loopParam[index], *entity.body[index]));
        }
    }
}

// CCU 本地数据拷贝（GM → CCU Buffer → GM，通过 LoopGroup 并行执行）
static CcuResult GroupCopy(AllToAllMesh1DContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src,
                           GroupOpSizeVars goSize)
{
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable loopLen[2];

    InitLocalCopyResources(ctx, loopSrc, loopDst, loopLen);

    auto &loops = ctx.loopMap["localcopy"];

    CCU_IF(goSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam += goSize.loopIterNum;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        loopSrc[0].addr = src.addr;
        loopSrc[0].token = src.token;
        loopDst[0].addr = dst.addr;
        loopDst[0].token = dst.token;
        loopLen[0] = sliceSize;

        loops.loopParam[0] = loopParam;
        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(goSize.loopExtendNum != 0)
    {
        src.addr += goSize.addrOffset;
        dst.addr += goSize.addrOffset;

        loopSrc[0].addr = src.addr;
        loopSrc[0].token = src.token;
        loopDst[0].addr = dst.addr;
        loopDst[0].token = dst.token;
        loopLen[0] = goSize.tailSize;

        src.addr += goSize.tailSize;
        dst.addr += goSize.tailSize;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        loopSrc[1].addr = src.addr;
        loopSrc[1].token = src.token;
        loopDst[1].addr = dst.addr;
        loopDst[1].token = dst.token;
        loopLen[1] = sliceSize;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(goSize.loopExtendNum, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

// ---- host 侧 CCU 资源管理（错误码转换 / 实例创建 / P2P 通道获取）----

static HcclResult ConvertCcuToHccl(CcuResult ccuResult)
{
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default: return HCCL_E_INTERNAL;
    }
}

static void DestroyAllDescs(std::vector<HcommCcuResDescHandle> &descs)
{
    for (auto &desc : descs) {
        (void)HcommCcuInsResDescDestroy(desc);
    }
}

// 创建 CCU 实例（跨 DIE 资源池）
static HcclResult CreateCcuInsWithEnabledDies(CcuInsHandle &insHandle)
{
    std::vector<HcommCcuResDescHandle> resDescs;
    for (uint32_t dieId = 0; dieId < CCU_DIE_NUM; dieId++) {
        HcommCcuResDescHandle desc = 0;
        CcuResult descCreateRet = HcommCcuInsResDescCreate(dieId, &desc);
        if (descCreateRet != CCU_SUCCESS) {
            (void)HcommCcuInsResDescDestroy(desc);
            DestroyAllDescs(resDescs);
            return ConvertCcuToHccl(descCreateRet);
        }

        CcuResult probeRet = HcommCcuQueryRemainResDesc(desc);
        if (probeRet == CCU_E_UNAVAIL) {
            (void)HcommCcuInsResDescDestroy(desc);
            continue;
        }
        if (probeRet != CCU_SUCCESS) {
            (void)HcommCcuInsResDescDestroy(desc);
            DestroyAllDescs(resDescs);
            return ConvertCcuToHccl(probeRet);
        }

        for (const auto &res : CCU_DEFAULT_RES) {
            CcuResult setRet = HcommCcuInsResDescSetNum(desc, res.type, res.num);
            if (setRet != CCU_SUCCESS) {
                (void)HcommCcuInsResDescDestroy(desc);
                DestroyAllDescs(resDescs);
                return ConvertCcuToHccl(setRet);
            }
        }
        resDescs.push_back(desc);
    }

    if (resDescs.empty()) {
        return HCCL_E_INTERNAL;
    }

    CcuResult createRet = HcommCcuInsCreate(resDescs.data(), static_cast<uint32_t>(resDescs.size()), &insHandle);
    DestroyAllDescs(resDescs);
    return ConvertCcuToHccl(createRet);
}

// 为每个远端 rank 获取一条 CCU 通信 channel；dieId 由首个 channel 的本地端点
// 查询 ENDPOINT_ATTR_DIE_ID 获得（供资源分配与 schd.phyDieMask 使用）
static HcclResult AcquireCcuChannels(HcclComm comm, uint32_t rankId, uint32_t rankSize,
                                     std::vector<ChannelHandle> &kernelChannels, uint32_t &dieId)
{
    kernelChannels.resize(rankSize - 1);
    dieId = 0;

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rankId) {
            continue;
        }

        uint32_t netLayer = 0;
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        HCCLCHECK(HcclRankGraphGetLinks(comm, netLayer, rankId, remoteRank, &linkList, &listSize));

        HcclChannelDesc desc;
        HCCLCHECK(HcclChannelDescInit(&desc, 1));
        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        bool protocolExists = false;
        for (uint32_t idx = 0; idx < listSize; idx++) {
            CommLink link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                desc.remoteRank = remoteRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolExists = true;
                break;
            }
        }
        if (!protocolExists) {
            return HCCL_E_NOT_FOUND;
        }
        HCCLCHECK(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1,
            &kernelChannels[channelIdx]));
        if (channelIdx == 0) {
            EndpointAttrDieId dieIdValue = 0;
            HCCLCHECK(HcclRankGraphGetEndpointInfo(
                comm, rankId, &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId), &dieIdValue));
            dieId = dieIdValue;
        }
        channelIdx++;
    }

    return HCCL_SUCCESS;
}
