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
 * \file allto_all_tiling_factory.h
 * \brief AllToAll Tiling 工厂类，根据架构和拓扑类型创建合适的 tiling 实例
 */
#ifndef __ALLTO_ALL_TILING_FACTORY_H__
#define __ALLTO_ALL_TILING_FACTORY_H__

#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/op_tiling/formulaic_tiling_datatype.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../common/matmul_allto_all_util_tiling.h"

namespace MC2Tiling {

/**
 * @brief AllToAll Tiling 工厂类
 *
 */
class AlltoAllTilingFactory {
public:
    /**
     * @brief 创建 AllToAll tiling 实例
     *
     * @param args tiling 参数
     * @param kernelType 算子通信类型
     * @param socVersion SOC 版本
     * @param npuArch NPU 架构
     * @param quantMode 量化模式
     * @return CutResult tiling 结果
     */
    static CutResult CreateTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                  SocVersion socVersion = SocVersion::SOC950, NpuArch npuArch = NpuArch::DAV_3510,
                                  QuantMode quantMode = QuantMode::NON_QUANT);
};

} // namespace MC2Tiling

#endif // __ALLTO_ALL_TILING_FACTORY_H__
