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
 * \file mc2_log.cpp
 * \brief
 */

#include "mc2_log.h"

namespace Mc2Log {

static void PrintTCubeTilingDataSecondPart(const std::string &opName, ::TCubeTiling &tiling)
{
    OP_LOGD(opName, " tiling.deptchAL1CacheUB %d", tiling.depthAL1CacheUB);
    OP_LOGD(opName, " tiling.deptchBL1CacheUB %d", tiling.depthBL1CacheUB);
    OP_LOGD(opName, " tiling.get_dbL0A %d", tiling.dbL0A);
    OP_LOGD(opName, " tiling.get_dbL0B %d", tiling.dbL0B);
    OP_LOGD(opName, " tiling.get_dbL0C %d", tiling.dbL0C);
    OP_LOGD(opName, " tiling.ALayoutInfoB %d", tiling.ALayoutInfoB);
    OP_LOGD(opName, " tiling.ALayoutInfoS %d", tiling.ALayoutInfoS);
    OP_LOGD(opName, " tiling.ALayoutInfoN %d", tiling.ALayoutInfoN);
    OP_LOGD(opName, " tiling.ALayoutInfoG %d", tiling.ALayoutInfoG);
    OP_LOGD(opName, " tiling.ALayoutInfoD %d", tiling.ALayoutInfoD);
    OP_LOGD(opName, " tiling.BLayoutInfoB %d", tiling.BLayoutInfoB);
    OP_LOGD(opName, " tiling.BLayoutInfoS %d", tiling.BLayoutInfoS);
    OP_LOGD(opName, " tiling.BLayoutInfoN %d", tiling.BLayoutInfoN);
    OP_LOGD(opName, " tiling.BLayoutInfoG %d", tiling.BLayoutInfoG);
    OP_LOGD(opName, " tiling.BLayoutInfoD %d", tiling.BLayoutInfoD);
    OP_LOGD(opName, " tiling.CLayoutInfoB %d", tiling.CLayoutInfoB);
    OP_LOGD(opName, " tiling.CLayoutInfoS1 %d", tiling.CLayoutInfoS1);
    OP_LOGD(opName, " tiling.CLayoutInfoN %d", tiling.CLayoutInfoN);
    OP_LOGD(opName, " tiling.CLayoutInfoG %d", tiling.CLayoutInfoG);
    OP_LOGD(opName, " tiling.CLayoutInfoS2 %d", tiling.CLayoutInfoS2);
    OP_LOGD(opName, " tiling.BatchNum %d", tiling.BatchNum);
    OP_LOGD(opName, " tiling.get_mxTypePara %d", tiling.mxTypePara);
}

static void PrintTCubeTilingDataSecondPart(const std::string &opName,
                                           optiling::TCubeTiling &tiling) {
  OP_LOGD(opName, " tiling.depthAL1CacheUB %d", tiling.get_depthAL1CacheUB());
  OP_LOGD(opName, " tiling.depthBL1CacheUB %d", tiling.get_depthBL1CacheUB());
  OP_LOGD(opName, " tiling.get_dbL0A %d", tiling.get_dbL0A());
  OP_LOGD(opName, " tiling.get_dbL0B %d", tiling.get_dbL0B());
  OP_LOGD(opName, " tiling.get_dbL0C %d", tiling.get_dbL0C());
  OP_LOGD(opName, " tiling.ALayoutInfoB %d", tiling.get_ALayoutInfoB());
  OP_LOGD(opName, " tiling.ALayoutInfoS %d", tiling.get_ALayoutInfoS());
  OP_LOGD(opName, " tiling.ALayoutInfoN %d", tiling.get_ALayoutInfoN());
  OP_LOGD(opName, " tiling.ALayoutInfoG %d", tiling.get_ALayoutInfoG());
  OP_LOGD(opName, " tiling.ALayoutInfoD %d", tiling.get_ALayoutInfoD());
  OP_LOGD(opName, " tiling.BLayoutInfoB %d", tiling.get_BLayoutInfoB());
  OP_LOGD(opName, " tiling.BLayoutInfoS %d", tiling.get_BLayoutInfoS());
  OP_LOGD(opName, " tiling.BLayoutInfoN %d", tiling.get_BLayoutInfoN());
  OP_LOGD(opName, " tiling.BLayoutInfoG %d", tiling.get_BLayoutInfoG());
  OP_LOGD(opName, " tiling.BLayoutInfoD %d", tiling.get_BLayoutInfoD());
  OP_LOGD(opName, " tiling.CLayoutInfoB %d", tiling.get_CLayoutInfoB());
  OP_LOGD(opName, " tiling.CLayoutInfoS1 %d", tiling.get_CLayoutInfoS1());
  OP_LOGD(opName, " tiling.CLayoutInfoN %d", tiling.get_CLayoutInfoN());
  OP_LOGD(opName, " tiling.CLayoutInfoG %d", tiling.get_CLayoutInfoG());
  OP_LOGD(opName, " tiling.CLayoutInfoS2 %d", tiling.get_CLayoutInfoS2());
  OP_LOGD(opName, " tiling.BatchNum %d", tiling.get_BatchNum());
  OP_LOGD(opName, " tiling.get_mxTypePara %d", tiling.get_mxTypePara());
}

void PrintMMV3TilingData(const std::string &opName,
                         optiling::MC2MatmulV3TilingData &tiling) {
  PrintTCubeTilingData(opName, tiling.matmulTiling);
  OP_LOGD(opName, " mTailCnt %d", tiling.get_mTailCnt());
  OP_LOGD(opName, " nTailCnt %d", tiling.get_nTailCnt());
  OP_LOGD(opName, " kTailCnt %d", tiling.get_kTailCnt());
  OP_LOGD(opName, " mBaseTailSplitCnt %d", tiling.get_mBaseTailSplitCnt());
  OP_LOGD(opName, " nBaseTailSplitCnt %d", tiling.get_nBaseTailSplitCnt());
  OP_LOGD(opName, " mTailMain %d", tiling.get_mTailMain());
  OP_LOGD(opName, " nTailMain %d", tiling.get_nTailMain());
  OP_LOGD(opName, " isHf32 %d", tiling.get_isHf32());
}

void PrintMMV3TilingData(const std::string &opName,
                         Mc2Tiling::MC2MatmulV3TilingData &tiling) {
  PrintTCubeTilingData(opName, tiling.matmulTiling);
  OP_LOGD(opName, " mTailCnt %d", tiling.mTailCnt);
  OP_LOGD(opName, " nTailCnt %d", tiling.nTailCnt);
  OP_LOGD(opName, " kTailCnt %d", tiling.kTailCnt);
  OP_LOGD(opName, " mBaseTailSplitCnt %d", tiling.mBaseTailSplitCnt);
  OP_LOGD(opName, " nBaseTailSplitCnt %d", tiling.nBaseTailSplitCnt);
  OP_LOGD(opName, " mTailMain %d", tiling.mTailMain);
  OP_LOGD(opName, " nTailMain %d", tiling.nTailMain);
  OP_LOGD(opName, " isHf32 %d", tiling.isHf32);
}

void PrintTCubeTilingData(const std::string &opName, ::TCubeTiling &tiling)
{
    OP_LOGD(opName, " tiling.usedCoreNum %d", tiling.usedCoreNum);
    OP_LOGD(opName, " tiling.M %d", tiling.M);
    OP_LOGD(opName, " tiling.N %d", tiling.N);
    OP_LOGD(opName, " tiling.Ka %d", tiling.Ka);
    OP_LOGD(opName, " tiling.Kb %d", tiling.Kb);
    OP_LOGD(opName, " tiling.singleCoreM %d", tiling.singleCoreM);
    OP_LOGD(opName, " tiling.singleCoreN %d", tiling.singleCoreN);
    OP_LOGD(opName, " tiling.singleCoreK %d", tiling.singleCoreK);
    OP_LOGD(opName, " tiling.baseM %d", tiling.baseM);
    OP_LOGD(opName, " tiling.baseN %d", tiling.baseN);
    OP_LOGD(opName, " tiling.baseK %d", tiling.baseK);
    OP_LOGD(opName, " tiling.depthA1 %d", tiling.depthA1);
    OP_LOGD(opName, " tiling.depthB1 %d", tiling.depthB1);
    OP_LOGD(opName, " tiling.stepM %d", tiling.stepM);
    OP_LOGD(opName, " tiling.stepN %d", tiling.stepN);
    OP_LOGD(opName, " tiling.isBias %d", tiling.isBias);
    OP_LOGD(opName, " tiling.transLength %d", tiling.transLength);
    OP_LOGD(opName, " tiling.iterateOrder %d", tiling.iterateOrder);
    OP_LOGD(opName, " tiling.shareMode %d", tiling.shareMode);
    OP_LOGD(opName, " tiling.usedL1Size %d", tiling.shareL1Size);
    OP_LOGD(opName, " tiling.usedL0CSize %d", tiling.shareL0CSize);
    OP_LOGD(opName, " tiling.shareUBSize %d", tiling.shareUbSize);
    OP_LOGD(opName, " tiling.batchM %d", tiling.batchM);
    OP_LOGD(opName, " tiling.batchN %d", tiling.batchN);
    OP_LOGD(opName, " tiling.singleBatchM %d", tiling.singleBatchM);
    OP_LOGD(opName, " tiling.singleBatchN %d", tiling.singleBatchN);
    OP_LOGD(opName, " tiling.stepKa %d", tiling.stepKa);
    OP_LOGD(opName, " tiling.stepKb %d", tiling.stepKb);
    // for cleancode, make sure func less than 50 lines
    PrintTCubeTilingDataSecondPart(opName,tiling);
}

void PrintTCubeTilingData(const std::string &opName,
                          optiling::TCubeTiling &tiling) {
  OP_LOGD(opName, " tiling.usedCoreNum %d", tiling.get_usedCoreNum());
  OP_LOGD(opName, " tiling.M %d", tiling.get_M());
  OP_LOGD(opName, " tiling.N %d", tiling.get_N());
  OP_LOGD(opName, " tiling.Ka %d", tiling.get_Ka());
  OP_LOGD(opName, " tiling.Kb %d", tiling.get_Kb());
  OP_LOGD(opName, " tiling.singleCoreM %d", tiling.get_singleCoreM());
  OP_LOGD(opName, " tiling.singleCoreN %d", tiling.get_singleCoreN());
  OP_LOGD(opName, " tiling.singleCoreK %d", tiling.get_singleCoreK());
  OP_LOGD(opName, " tiling.baseM %d", tiling.get_baseM());
  OP_LOGD(opName, " tiling.baseN %d", tiling.get_baseN());
  OP_LOGD(opName, " tiling.baseK %d", tiling.get_baseK());
  OP_LOGD(opName, " tiling.depthA1 %d", tiling.get_depthA1());
  OP_LOGD(opName, " tiling.depthB1 %d", tiling.get_depthB1());
  OP_LOGD(opName, " tiling.stepM %d", tiling.get_stepM());
  OP_LOGD(opName, " tiling.stepN %d", tiling.get_stepN());
  OP_LOGD(opName, " tiling.isBias %d", tiling.get_isBias());
  OP_LOGD(opName, " tiling.transLength %d", tiling.get_transLength());
  OP_LOGD(opName, " tiling.iterateOrder %d", tiling.get_iterateOrder());
  OP_LOGD(opName, " tiling.shareMode %d", tiling.get_shareMode());
  OP_LOGD(opName, " tiling.usedL1Size %d", tiling.get_shareL1Size());
  OP_LOGD(opName, " tiling.usedL0CSize %d", tiling.get_shareL0CSize());
  OP_LOGD(opName, " tiling.shareUBSize %d", tiling.get_shareUbSize());
  OP_LOGD(opName, " tiling.batchM %d", tiling.get_batchM());
  OP_LOGD(opName, " tiling.batchN %d", tiling.get_batchN());
  OP_LOGD(opName, " tiling.singleBatchM %d", tiling.get_singleBatchM());
  OP_LOGD(opName, " tiling.singleBatchN %d", tiling.get_singleBatchN());
  OP_LOGD(opName, " tiling.stepKa %d", tiling.get_stepKa());
  OP_LOGD(opName, " tiling.stepKb %d", tiling.get_stepKb());
  // for cleancode, make sure func less than 50 lines
  PrintTCubeTilingDataSecondPart(opName, tiling);
}

void PrintRCSTilingData(const std::string &opName,
                        optiling::RCSTiling &rcsTiling) {
  OP_LOGD(opName, " rcsTiling.rankDim %u", rcsTiling.get_rankDim());
  OP_LOGD(opName, " rcsTiling.rankID %u", rcsTiling.get_rankID());
  OP_LOGD(opName, " rcsTiling.commtype %u", rcsTiling.get_commtype());
  OP_LOGD(opName, " rcsTiling.subtype %u", rcsTiling.get_subtype());
  OP_LOGD(opName, " rcsTiling.tileCnt %u", rcsTiling.get_tileCnt());
  OP_LOGD(opName, " rcsTiling.tailM %u", rcsTiling.get_tailM());
  OP_LOGD(opName, " rcsTiling.tailCnt %u", rcsTiling.get_tailCnt());
  OP_LOGD(opName, " rcsTiling.biasLen %u", rcsTiling.get_biasLen());
  OP_LOGD(opName, " rcsTiling.isAdd %u", rcsTiling.get_isAdd());
  OP_LOGD(opName, " rcsTiling.rankM %u", rcsTiling.get_rankM());
  OP_LOGD(opName, " rcsTiling.rankN %u", rcsTiling.get_rankN());
  OP_LOGD(opName, " rcsTiling.rankK %u", rcsTiling.get_rankK());
  OP_LOGD(opName, " rcsTiling.gatherIndex %u", rcsTiling.get_gatherIndex());
  OP_LOGD(opName, " rcsTiling.isTransA %u", rcsTiling.get_isTransposeA());
  OP_LOGD(opName, " rcsTiling.isTransB %u", rcsTiling.get_isTransposeB());
  OP_LOGD(opName, " rcsTiling.storageGather %u", rcsTiling.get_storageGather());
  OP_LOGD(opName, " rcsTiling.nd2NzWorkLen %lu", rcsTiling.get_nd2NzWorkLen());
  OP_LOGD(opName, " rcsTiling.cToFloatLen %lu", rcsTiling.get_cToFloatLen());
  OP_LOGD(opName, " rcsTiling.gatherLen %lu", rcsTiling.get_gatherLen());
  OP_LOGD(opName, " rcsTiling.workspaceAddr4 %u",
          rcsTiling.get_workspaceAddr4());
  OP_LOGD(opName, " rcsTiling.aicCoreNum %u", rcsTiling.get_aicCoreNum());
  OP_LOGD(opName, " rcsTiling.needUbBuffer %u", rcsTiling.get_needUbBuffer());
  OP_LOGD(opName, " rcsTiling.addX3UbCnt %u", rcsTiling.get_addX3UbCnt());
}

void PrintMc2MsgData(const std::string &opName, optiling::Mc2Msg &msg) {
  OP_LOGD(opName, " msg.sendOff %lu", msg.get_sendOff());
  OP_LOGD(opName, " msg.recvOff %lu", msg.get_recvOff());
  OP_LOGD(opName, " msg.tailSendOff %lu", msg.get_tailSendOff());
  OP_LOGD(opName, " msg.tailRecvOff %lu", msg.get_tailRecvOff());
  OP_LOGD(opName, " msg.sendCnt %lu", msg.get_sendCnt());
  OP_LOGD(opName, " msg.recvCnt %lu", msg.get_recvCnt());
  OP_LOGD(opName, " msg.tailSendCnt %lu", msg.get_tailSendCnt());
  OP_LOGD(opName, " msg.tailRecvCnt %lu", msg.get_tailRecvCnt());
  OP_LOGD(opName, " msg.totalCnt %lu", msg.get_totalCnt());
  OP_LOGD(opName, " msg.turnNum %u", msg.get_turnNum());
  OP_LOGD(opName, " msg.tailNum %u", msg.get_tailNum());
  OP_LOGD(opName, " msg.stride %u", msg.get_stride());
  OP_LOGD(opName, " msg.workspaceOff %u", msg.get_workspaceOff());
  OP_LOGD(opName, " msg.notifyOff %u", msg.get_notifyOff());

  OP_LOGD(opName, " msg.notifyBeginCnt %u", msg.get_notifyBeginCnt());
  OP_LOGD(opName, " msg.notifyEndCnt %u", msg.get_notifyEndCnt());
  OP_LOGD(opName, " msg.useBufferType %u", msg.get_useBufferType());
  OP_LOGD(opName, " msg.funID %u", msg.get_funID());
  OP_LOGD(opName, " msg.dataType %u", msg.get_dataType());
  OP_LOGD(opName, " msg.groupNum %u", msg.get_groupNum());

  OP_LOGD(opName, " msg.reuseMode %u", msg.get_reuseMode());
  OP_LOGD(opName, " msg.commType %u", msg.get_commType());
  OP_LOGD(opName, " msg.reduceOp %u", msg.get_reduceOp());
  OP_LOGD(opName, " msg.commOrder %u", msg.get_commOrder());
  OP_LOGD(opName, " msg.waitPolicy %u", msg.get_waitPolicy());
  OP_LOGD(opName, " msg.rspPolicy %u", msg.get_rspPolicy());

  OP_LOGD(opName, " msg.exitPolicy %u", msg.get_exitPolicy());

  OP_LOGD(opName, " msg.commAlg %u", msg.get_commAlg());
  OP_LOGD(opName, " msg.taskType %u", msg.get_taskType());
  OP_LOGD(opName, " msg.preparePosition %u", msg.get_preparePosition());
}

void PrintTileL2TilingData(const std::string &opName,
                           optiling::TileL2Tiling &tileL2Tiling) {
  OP_LOGD(opName, " tileL2Tiling.mL2TileCnt %u", tileL2Tiling.get_mL2TileCnt());
  OP_LOGD(opName, " tileL2Tiling.nL2TileCnt %u", tileL2Tiling.get_nL2TileCnt());
  OP_LOGD(opName, " tileL2Tiling.mTileBlocks %u",
          tileL2Tiling.get_mTileBlocks());
  OP_LOGD(opName, " tileL2Tiling.nTileBlocks %u",
          tileL2Tiling.get_nTileBlocks());
  OP_LOGD(opName, " tileL2Tiling.mTailBlocks %u",
          tileL2Tiling.get_mTailBlocks());
  OP_LOGD(opName, " tileL2Tiling.nTailBlocks %u",
          tileL2Tiling.get_nTailBlocks());
  OP_LOGD(opName, " tileL2Tiling.rankTileNum %u",
          tileL2Tiling.get_rankTileNum());
  OP_LOGD(opName, " tileL2Tiling.calcOrder %u", tileL2Tiling.get_calcOrder());
  OP_LOGD(opName, " tileL2Tiling.enableL2Tile %u",
          tileL2Tiling.get_enableL2Tile());
}

void PrintRCSTilingData(const std::string &opName, Mc2Tiling::RCSTiling &rcsTiling)
{
    OP_LOGD(opName, " rcsTiling.rankDim %u", rcsTiling.rankDim);
    OP_LOGD(opName, " rcsTiling.rankID %u", rcsTiling.rankID);
    OP_LOGD(opName, " rcsTiling.commtype %u", rcsTiling.commtype);
    OP_LOGD(opName, " rcsTiling.subtype %u", rcsTiling.subtype);
    OP_LOGD(opName, " rcsTiling.tileCnt %u", rcsTiling.tileCnt);
    OP_LOGD(opName, " rcsTiling.tailM %u", rcsTiling.tailM);
    OP_LOGD(opName, " rcsTiling.tailCnt %u", rcsTiling.tailCnt);
    OP_LOGD(opName, " rcsTiling.biasLen %u", rcsTiling.biasLen);
    OP_LOGD(opName, " rcsTiling.isAdd %u", rcsTiling.isAdd);
    OP_LOGD(opName, " rcsTiling.rankM %u", rcsTiling.rankM);
    OP_LOGD(opName, " rcsTiling.rankN %u", rcsTiling.rankN);
    OP_LOGD(opName, " rcsTiling.rankK %u", rcsTiling.rankK);
    OP_LOGD(opName, " rcsTiling.gatherIndex %u", rcsTiling.gatherIndex);
    OP_LOGD(opName, " rcsTiling.isTransA %u", rcsTiling.isTransposeA);
    OP_LOGD(opName, " rcsTiling.isTransB %u", rcsTiling.isTransposeB);
    OP_LOGD(opName, " rcsTiling.storageGather %u", rcsTiling.storageGather);
    OP_LOGD(opName, " rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OP_LOGD(opName, " rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OP_LOGD(opName, " rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
    OP_LOGD(opName, " rcsTiling.workspaceAddr4 %u", rcsTiling.workspaceAddr4);
    OP_LOGD(opName, " rcsTiling.aicCoreNum %u", rcsTiling.aicCoreNum);
    OP_LOGD(opName, " rcsTiling.needUbBuffer %u", rcsTiling.needUbBuffer);
    OP_LOGD(opName, " rcsTiling.addX3UbCnt %u", rcsTiling.addX3UbCnt);
}

void PrintTileL2TilingData(const std::string &opName, Mc2Tiling::TileL2Tiling &tileL2Tiling)
{
    OP_LOGD(opName, " tileL2Tiling.mL2TileCnt %u", tileL2Tiling.mL2TileCnt);
    OP_LOGD(opName, " tileL2Tiling.nL2TileCnt %u", tileL2Tiling.nL2TileCnt);
    OP_LOGD(opName, " tileL2Tiling.mTileBlocks %u", tileL2Tiling.mTileBlocks);
    OP_LOGD(opName, " tileL2Tiling.nTileBlocks %u", tileL2Tiling.nTileBlocks);
    OP_LOGD(opName, " tileL2Tiling.mTailBlocks %u", tileL2Tiling.mTailBlocks);
    OP_LOGD(opName, " tileL2Tiling.nTailBlocks %u", tileL2Tiling.nTailBlocks);
    OP_LOGD(opName, " tileL2Tiling.rankTileNum %u", tileL2Tiling.rankTileNum);
    OP_LOGD(opName, " tileL2Tiling.calcOrder %u", tileL2Tiling.calcOrder);
    OP_LOGD(opName, " tileL2Tiling.enableL2Tile %u", tileL2Tiling.enableL2Tile);
}

void PrintMc2MsgData(const std::string &opName, Mc2Tiling::Mc2Msg &msg)
{
    OP_LOGD(opName, " msg.sendOff %lu", msg.sendOff);
    OP_LOGD(opName, " msg.recvOff %lu", msg.recvOff);
    OP_LOGD(opName, " msg.tailSendOff %lu", msg.tailSendOff);
    OP_LOGD(opName, " msg.tailRecvOff %lu", msg.tailRecvOff);
    OP_LOGD(opName, " msg.sendCnt %lu", msg.sendCnt);
    OP_LOGD(opName, " msg.recvCnt %lu", msg.recvCnt);
    OP_LOGD(opName, " msg.tailSendCnt %lu", msg.tailSendCnt);
    OP_LOGD(opName, " msg.tailRecvCnt %lu", msg.tailRecvCnt);
    OP_LOGD(opName, " msg.totalCnt %lu", msg.totalCnt);
    OP_LOGD(opName, " msg.turnNum %u", msg.turnNum);
    OP_LOGD(opName, " msg.tailNum %u", msg.tailNum);
    OP_LOGD(opName, " msg.stride %u", msg.stride);
    OP_LOGD(opName, " msg.workspaceOff %u", msg.workspaceOff);
    OP_LOGD(opName, " msg.notifyOff %u", msg.notifyOff);

    OP_LOGD(opName, " msg.notifyBeginCnt %u", msg.notifyBeginCnt);
    OP_LOGD(opName, " msg.notifyEndCnt %u", msg.notifyEndCnt);
    OP_LOGD(opName, " msg.useBufferType %u", msg.useBufferType);
    OP_LOGD(opName, " msg.funID %u", msg.funID);
    OP_LOGD(opName, " msg.dataType %u", msg.dataType);
    OP_LOGD(opName, " msg.groupNum %u", msg.groupNum);

    OP_LOGD(opName, " msg.reuseMode %u", msg.reuseMode);
    OP_LOGD(opName, " msg.commType %u", msg.commType);
    OP_LOGD(opName, " msg.reduceOp %u", msg.reduceOp);
    OP_LOGD(opName, " msg.commOrder %u", msg.commOrder);
    OP_LOGD(opName, " msg.waitPolicy %u", msg.waitPolicy);
    OP_LOGD(opName, " msg.rspPolicy %u", msg.rspPolicy);

    OP_LOGD(opName, " msg.exitPolicy %u", msg.exitPolicy);

    OP_LOGD(opName, " msg.commAlg %u", msg.commAlg);
    OP_LOGD(opName, " msg.taskType %u", msg.taskType);
    OP_LOGD(opName, " msg.preparePosition %u", msg.preparePosition);
}

void PrintTCubeTilingParams(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3DataParams &tiling)
{
    OP_LOGD(opName, " tiling.batchA %d", tiling.batchA);
    OP_LOGD(opName, " tiling.batchB %d", tiling.batchB);
    OP_LOGD(opName, " tiling.batchC %d", tiling.batchC);
    OP_LOGD(opName, " tiling.batchA1 %d", tiling.batchA1);
    OP_LOGD(opName, " tiling.batchA2 %d", tiling.batchA2);
    OP_LOGD(opName, " tiling.batchA3 %d", tiling.batchA3);
    OP_LOGD(opName, " tiling.batchA4 %d", tiling.batchA4);
    OP_LOGD(opName, " tiling.batchB1 %d", tiling.batchB1);
    OP_LOGD(opName, " tiling.batchB2 %d", tiling.batchB2);
    OP_LOGD(opName, " tiling.batchB3 %d", tiling.batchB3);
    OP_LOGD(opName, " tiling.batchB4 %d", tiling.batchB4);
    OP_LOGD(opName, " tiling.batchC1 %d", tiling.batchC1);
    OP_LOGD(opName, " tiling.batchC2 %d", tiling.batchC2);
    OP_LOGD(opName, " tiling.batchC3 %d", tiling.batchC3);
    OP_LOGD(opName, " tiling.batchC4 %d", tiling.batchC4);
    OP_LOGD(opName, " tiling.singleCoreBatch %d", tiling.singleCoreBatch);
    OP_LOGD(opName, " tiling.isPerTensor %d", tiling.isPerTensor);
    OP_LOGD(opName, " tiling.isPertoken %d", tiling.isPertoken);
    OP_LOGD(opName, " tiling.isDoubleScale %d", tiling.isDoubleScale);
    OP_LOGD(opName, " tiling.biasThreeDim %d", tiling.biasThreeDim);
    OP_LOGD(opName, " tiling.ubCalcM %d", tiling.ubCalcM);
    OP_LOGD(opName, " tiling.ubCalcN %d", tiling.ubCalcN);
    OP_LOGD(opName, " tiling.needUbBuffer %d", tiling.needUbBuffer);
    OP_LOGD(opName, " tiling.realSingleCoreM %d", tiling.realSingleCoreM);
    OP_LOGD(opName, " tiling.realSingleCoreN %d", tiling.realSingleCoreN);
    OP_LOGD(opName, " tiling.biasDtype %d", tiling.biasDtype);
    OP_LOGD(opName, " tiling.ubSize %d", tiling.ubSize);
    OP_LOGD(opName, " tiling.isMClash %d", tiling.isMClash);
    OP_LOGD(opName, " tiling.isNClash %d", tiling.isNClash);
    OP_LOGD(opName, " tiling.groupSizeM %d", tiling.groupSizeM);
    OP_LOGD(opName, " tiling.groupSizeN %d", tiling.groupSizeN);
    OP_LOGD(opName, " tiling.groupSizeK %d", tiling.groupSizeK);
}

void PrintTCubeTilingL2cache(const std::string &opName, DequantBmm::Mc2L2cacheTileParams &tiling)
{
    OP_LOGD(opName, " tiling.mTileCntL2 %d", tiling.mTileCntL2);
    OP_LOGD(opName, " tiling.nTileCntL2 %d", tiling.nTileCntL2);
    OP_LOGD(opName, " tiling.mTileBlock %d", tiling.mTileBlock);
    OP_LOGD(opName, " tiling.nTileBlock %d", tiling.nTileBlock);
    OP_LOGD(opName, " tiling.calOrder %d", tiling.calOrder);
    OP_LOGD(opName, " tiling.isBasicTiling %d", tiling.isBasicTiling);
}

void PrintTCubeTilingWindowParam(const std::string &opName, DequantBmm::Mc2SlidingWindowParams &tiling)
{
    OP_LOGD(opName, " tiling.mTailTile %d", tiling.mTailTile);
    OP_LOGD(opName, " tiling.nTailTile %d", tiling.nTailTile);
}

void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling) 
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mBaseTailSpiltCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSpiltCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

}  // namespace Mc2Log
