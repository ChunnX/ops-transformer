/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file fit_balance_tiling.h
 * \brief fit-balance tiling 家族唯一公共文件（三层结构, 两分支共居, 迁移方案 §2.2/§4.9）
 *
 * 当前仅含自由长度分支: HCCL AICPU/CCU 引擎, 每轮开销大 → 反解最优轮长压轮数。
 * 消费者为旧通算融合算子的 fit-balance 路径（950 标卡 4P/8P; 引擎仅支持 950, 后续芯片扩展时更新此注释）。
 *
 * 历史存档: URMA wave-packing 拓展分支（v2 aiv_urma 的 estTime k 搜索方案）曾在此文件,
 * 2026-09-15 评审裁定废弃——与核利用率优先切分大部分结果相同, 少量差异定位为
 * 核利用率优先的实现问题(修复走独立事项)。相关标定方法论与实测数据仍有效,
 * 存档于 tiling-plan/ALLTO_ALL_MATMUL_V2_FIT_BALANCE_*。
 *
 * 布局(§4.39 拆分, 2026-09-28): 按功能分层为 fit_balance_tiling/ 目录下多个头文件,
 *       本文件为对外总接口(伞形头)——消费方 include 路径不变, 详见下方拆分清单
 *
 * 零特有代码架构(§4.24/§4.38 演进): 钩子机制已废除; 引擎不区分通信原语——原语语义
 * (通信份数口径已融进标定斜率 k / 轴向与 mm 倍数在 CommProfile / 流水线结构由 LocalMMPosition
 * 表达)全部终结在标定层; 键控维度仅 (LocalMMPosition 本地块位置, CommComputeOrder 通算顺序,
 * TilingCommEngine 通信引擎, dtype)。算子侧 = CALIBRATION 实例 + 枚举入参 + 一行调用。
 *
 * 统一引擎分派（三引擎接入结构完全一致, 见 TilingCommEngine/ParseCommEngine）:
 *   同时支持多通信模式的算子按同一模式接入——CCU/AICPU 走自由长度分支（各引擎拟合表）;
 *   AIV_URMA 当前无求解分支(原 wave-packing 已废弃), 枚举值保留供未来方案接入;
 *   每引擎一组标定常量 + 一张表 + 一个分派点, 无任何引擎特殊路径。
 *
 * 等价性基线: 自由长度分支 2026-09-14 P0 提取经 162 用例 bit 级验证
 * (与老类逐字段比对), 消费方 agmm_v2 双包构建通过。
 */
// ============================================================================
// §4.39 文件拆分(2026-09-28): 引擎按功能分层(fit_balance_tiling/ 目录), 消费方
// include 本伞形头即可(路径与拆分前完全一致):
//   fit_balance_types.h  零: 公共类型/枚举/常量
//   comm_model.h         一.1: 通信模型(y=k·x+N·b+c 三口径, 零份数乘子)
//   matmul_model.h       一.2: 计算模型(利用率双口径/梯度/正反解)
//   cut_planner.h        二: 离散切分引擎(CutPlanner 七步)
//   calibration.h        三.1: 客户标定数据总集(CommProfile + k/b/c + 算力档)
//   solver.h             三.2/四: 求解层(ctx/InitCtx/公共片段/Schedule)
// ============================================================================

#ifndef __APACE_FIT_BALANCE_TILING_H__
#define __APACE_FIT_BALANCE_TILING_H__

#pragma once

#include "fit_balance_tiling/fit_balance_types.h"
#include "fit_balance_tiling/comm_model.h"
#include "fit_balance_tiling/matmul_model.h"
#include "fit_balance_tiling/cut_planner.h"
#include "fit_balance_tiling/calibration.h"
#include "fit_balance_tiling/solver.h"

#endif // __APACE_FIT_BALANCE_TILING_H__
