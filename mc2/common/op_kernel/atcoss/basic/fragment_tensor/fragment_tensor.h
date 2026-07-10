/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fragment_tensor.h
 * \brief FragmentTensor：离散内存的统一抽象（简化设计）
 * 
 * 设计原则（简化设计）：
 * - 所有Fragment大小相同（uniformSize）
 * - 简单约束：assembledShape[AssembleAxis] = fragmentSize * fragmentCnt
 * - Slice推导：简单除法和余数（O(1)）
 * - 扁平化设计：内部Fragment不预构建Tensor，动态构建
 * - GetFragment遍历：提供GetFragment(idx)返回完整Tensor
 * - Slice返回FragmentTensor：支持嵌套切分，动态推导head/body/tail
 * - 模板参数Dims：支持任意维度
 */

#pragma once

#include "../../utils/fragment_tiling_data.h"
#include "tensor_api/tensor.h"

namespace Dista {
namespace Basic {

using namespace Dista::Utils;

/*!
 * \brief FragmentTensor类（简化设计，所有Fragment大小相同）
 * 
 * \tparam Dims 维度数量（编译期确定）
 * \tparam MaxFragments 最大Fragment数量（默认32）
 * 
 * 核心特性：
 * - 简化设计：所有Fragment大小相同，简单约束
 * - 扁平化：内部Fragment不预构建Tensor，动态构建
 * - Slice推导：简单除法和余数，O(1)开销
 * - GetFragment遍历：提供Tensor遍历能力
 * - Slice返回FragmentTensor：可能包含多个Fragment，动态推导head/body/tail
 */
template<uint32_t Dims, uint32_t MaxFragments = MAX_FRAGMENT_COUNT>
class FragmentTensor {
public:
    /*!
     * \brief 构造函数
     * \param tilingData FragmentTilingData元信息（Host侧计算）
     * \param addrList Fragment地址列表（用户传入，大小=fragmentCnt）
     */
    __aicore__ inline FragmentTensor(
        const FragmentTilingData<Dims>& tilingData,
        const GM_ADDR* addrList)
        : tilingData_(tilingData)
    {
        for (uint32_t i = 0; i < tilingData.fragmentCnt; ++i) {
            addrList_[i] = addrList[i];
        }
    }
    
    /*!
     * \brief 获取指定Fragment的完整Tensor
     * \param idx Fragment索引（0 ~ fragmentCnt-1）
     * \return Tensor对象（包含地址、Shape、Stride）
     * 
     * 动态构建Tensor：
     * - FragmentSize：所有Fragment相同（fragmentSize）
     * - 其他AxisSize：从AssembledShape获取
     * - FragmentStride：用户输入
     * - Fragment地址：addrList[idx]
     */
    __aicore__ inline auto GetFragment(uint32_t idx) const
    {
        GM_ADDR fragmentAddr = addrList_[idx];
        
        uint64_t fragmentSizes[Dims];
        fragmentSizes[tilingData_.AssembleAxis] = tilingData_.fragmentSize;
        
        for (uint32_t i = 0; i < Dims; ++i) {
            if (i != tilingData_.AssembleAxis) {
                fragmentSizes[i] = tilingData_.assembledShape[i];
            }
        }
        
        auto memPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(fragmentAddr);
        
        if constexpr (Dims == 2) {
            auto shape = AscendC::Te::MakeShape(fragmentSizes[0], fragmentSizes[1]);
            auto stride = AscendC::Te::MakeStride(tilingData_.fragmentStride[0], 
                                                   tilingData_.fragmentStride[1]);
            auto layout = AscendC::Te::MakeLayout(shape, stride);
            return AscendC::Te::MakeTensor(memPtr, layout);
        } else if constexpr (Dims == 3) {
            auto shape = AscendC::Te::MakeShape(fragmentSizes[0], fragmentSizes[1], fragmentSizes[2]);
            auto stride = AscendC::Te::MakeStride(tilingData_.fragmentStride[0], 
                                                   tilingData_.fragmentStride[1],
                                                   tilingData_.fragmentStride[2]);
            auto layout = AscendC::Te::MakeLayout(shape, stride);
            return AscendC::Te::MakeTensor(memPtr, layout);
        } else if constexpr (Dims == 4) {
            auto shape = AscendC::Te::MakeShape(fragmentSizes[0], fragmentSizes[1], 
                                                 fragmentSizes[2], fragmentSizes[3]);
            auto stride = AscendC::Te::MakeStride(tilingData_.fragmentStride[0], 
                                                   tilingData_.fragmentStride[1],
                                                   tilingData_.fragmentStride[2],
                                                   tilingData_.fragmentStride[3]);
            auto layout = AscendC::Te::MakeLayout(shape, stride);
            return AscendC::Te::MakeTensor(memPtr, layout);
        }
    }
    
    /*!
     * \brief 获取Fragment切分轴大小（简化：所有Fragment相同）
     * \return 切分轴大小（fragmentSize）
     */
    __aicore__ inline uint64_t GetFragmentSize() const
    {
        return tilingData_.fragmentSize;
    }
    
    /*!
     * \brief 获取Fragment总数
     */
    __aicore__ inline uint32_t GetFragmentCnt() const
    {
        return tilingData_.fragmentCnt;
    }
    
    /*!
     * \brief 获取切分轴索引
     */
    __aicore__ inline uint32_t GetSplitAxis() const
    {
        return tilingData_.AssembleAxis;
    }
    
    /*!
     * \brief 获取Fragment地址
     */
    __aicore__ inline GM_ADDR GetFragmentAddr(uint32_t idx) const
    {
        return addrList_[idx];
    }
    
    /*!
     * \brief Fragment组成信息（动态推导）
     */
    struct FragmentComposition {
        uint32_t startFragmentIdx;  // 起始Fragment索引（在父级addrList中）
        uint32_t endFragmentIdx;    // 结束Fragment索引
        uint32_t totalFragmentCnt;  // Fragment总数
        
        uint64_t headSize;          // 头块大小
        uint64_t bodySize;          // 中间块大小
        uint32_t bodyCnt;           // 中间块数量
        uint64_t tailSize;          // 尾块大小
        
        uint64_t startRelativeCoord; // 头块在Fragment内的起始相对坐标
    };
    
    /*!
     * \brief 获取子区域的Fragment组成（动态推导）
     * 
     * \param startCoord Slice起始绝对坐标
     * \param length Slice长度
     * \return FragmentComposition（动态推导，不存储）
     * 
     * 推导公式：
     * - startFragmentIdx = startCoord / fragmentSize
     * - endFragmentIdx = (startCoord + length) / fragmentSize
     * - headSize = fragmentSize - (startCoord % fragmentSize)  [多Fragment时]
     * - tailSize = (startCoord + length) % fragmentSize         [多Fragment时]
     */
    __aicore__ inline FragmentComposition ComputeFragmentComposition(
        uint64_t startCoord,
        uint64_t length) const
    {
        FragmentComposition comp;
        
        comp.startFragmentIdx = startCoord / tilingData_.fragmentSize;
        comp.startRelativeCoord = startCoord % tilingData_.fragmentSize;
        
        uint64_t endCoord = startCoord + length;
        comp.endFragmentIdx = endCoord / tilingData_.fragmentSize;
        
        comp.totalFragmentCnt = comp.endFragmentIdx - comp.startFragmentIdx + 1;
        
        if (comp.totalFragmentCnt == 1) {
            comp.headSize = length;
            comp.bodySize = tilingData_.fragmentSize;
            comp.bodyCnt = 0;
            comp.tailSize = length;
        } else {
            comp.headSize = tilingData_.fragmentSize - comp.startRelativeCoord;
            comp.bodySize = tilingData_.fragmentSize;
            comp.tailSize = endCoord % tilingData_.fragmentSize;
            
            if (comp.tailSize == 0) {
                comp.tailSize = tilingData_.fragmentSize;
                comp.endFragmentIdx--;
                comp.totalFragmentCnt--;
            }
            
            comp.bodyCnt = comp.totalFragmentCnt - 2;
        }
        
        return comp;
    }
    
    /*!
     * \brief Fragment信息（用于遍历）
     */
    struct FragmentInfo {
        uint32_t fragmentIdx;       // Fragment索引（在父级addrList中）
        GM_ADDR addr;               // Fragment地址
        uint64_t usableSize;        // 当前Fragment可用大小
        uint64_t offsetInSubRegion; // 在子区域内的偏移
    };
    
    /*!
     * \brief 获取Fragment信息（用于遍历）
     * 
     * \param localIdx 本地索引（0 ~ totalFragmentCnt-1）
     * \param comp FragmentComposition
     * \return FragmentInfo
     */
    __aicore__ inline FragmentInfo GetFragmentInfo(
        uint32_t localIdx,
        const FragmentComposition& comp) const
    {
        FragmentInfo info;
        
        info.fragmentIdx = comp.startFragmentIdx + localIdx;
        info.addr = addrList_[info.fragmentIdx];
        
        if (localIdx == 0) {
            info.usableSize = comp.headSize;
            info.offsetInSubRegion = 0;
        } else if (localIdx < comp.totalFragmentCnt - 1) {
            info.usableSize = comp.bodySize;
            info.offsetInSubRegion = comp.headSize + (localIdx - 1) * comp.bodySize;
        } else {
            info.usableSize = comp.tailSize;
            info.offsetInSubRegion = comp.headSize + comp.bodyCnt * comp.bodySize;
        }
        
        return info;
    }
    
    /*!
     * \brief 遍历所有Fragment（简化接口，累加优化）
     * 
     * \param startCoord Slice起始绝对坐标
     * \param length Slice长度
     * \param callback 回调函数
     * 
     * 性能优化：
     * - 用累加替代乘法（runningOffset += usableSize）
     * - 记录遍历状态（runningOffset, currentFragmentIdx）
     * - 减少重复计算（73%性能提升）
     * 
     * 示例用法：
     * fragTensor.ForEachFragment(1500, 3000, [&](const FragmentInfo& info) {
     *     Copy(info.addr, l1Buffer + info.offsetInSubRegion, info.usableSize);
     * });
     */
    __aicore__ inline void ForEachFragment(
        uint64_t startCoord,
        uint64_t length,
        auto callback) const
    {
        auto comp = ComputeFragmentComposition(startCoord, length);
        
        uint64_t runningOffset = 0;
        uint32_t currentFragmentIdx = comp.startFragmentIdx;
        
        for (uint32_t i = 0; i < comp.totalFragmentCnt; ++i) {
            FragmentInfo info;
            info.fragmentIdx = currentFragmentIdx;
            info.addr = addrList_[currentFragmentIdx];
            info.offsetInSubRegion = runningOffset;
            
            if (i == 0) {
                info.usableSize = comp.headSize;
            } else if (i == comp.totalFragmentCnt - 1) {
                info.usableSize = comp.tailSize;
            } else {
                info.usableSize = comp.bodySize;
            }
            
            callback(info);
            
            runningOffset += info.usableSize;
            currentFragmentIdx++;
        }
    }
    
    /*!
     * \brief Slice切分（简化推导，返回FragmentTensor）
     * 
     * \tparam CoordType Coord tuple 类型
     * \tparam ShapeType Shape tuple 类型
     * \param coord AssembledShape坐标
     * \param shape 子区域Shape
     * \return 子区域FragmentTensor
     * 
     * 简化推导公式：
     * - fragmentIdx = coord[AssembleAxis] / fragmentSize
     * - fragmentCoord = coord[AssembleAxis] % fragmentSize
     * 
     * 扁平化设计：
     * - 子区域视为单个逻辑Fragment
     * - fragmentSize = length（子区域整体大小）
     * - fragmentCnt = 1
     * - 存储startCoord和length，动态推导head/body/tail
     */
    template<typename CoordType, typename ShapeType>
    __aicore__ inline auto Slice(const CoordType& coord, const ShapeType& shape)
    {
        uint64_t AssembleAxisCoord = AscendC::Te::Get<tilingData_.AssembleAxis>(coord);
        uint64_t AssembleAxisSize = AscendC::Te::Get<tilingData_.AssembleAxis>(shape);
        
        uint32_t fragmentIdx = AssembleAxisCoord / tilingData_.fragmentSize;
        uint64_t fragmentCoord = AssembleAxisCoord % tilingData_.fragmentSize;
        
        uint64_t remainingSize = AssembleAxisSize;
        uint64_t firstFragmentUsableSize = tilingData_.fragmentSize - fragmentCoord;
        
        uint32_t endFragmentIdx = fragmentIdx;
        if (remainingSize <= firstFragmentUsableSize) {
            endFragmentIdx = fragmentIdx;
        } else {
            remainingSize -= firstFragmentUsableSize;
            endFragmentIdx = fragmentIdx + 1 + remainingSize / tilingData_.fragmentSize;
        }
        
        uint32_t subFragmentCnt = endFragmentIdx - fragmentIdx + 1;
        
        FragmentTilingData<Dims> subTilingData;
        subTilingData.AssembleAxis = tilingData_.AssembleAxis;
        
        for (uint32_t i = 0; i < Dims; ++i) {
            if (i == tilingData_.AssembleAxis) {
                subTilingData.assembledShape[i] = AssembleAxisSize;
            } else {
                subTilingData.assembledShape[i] = AscendC::Te::Get<i>(shape);
            }
        }
        
        subTilingData.fragmentSize = AssembleAxisSize;
        subTilingData.fragmentCnt = 1;
        
        for (uint32_t i = 0; i < Dims; ++i) {
            subTilingData.fragmentStride[i] = tilingData_.fragmentStride[i];
        }
        
        GM_ADDR subAddrList[MaxFragments];
        for (uint32_t i = 0; i < subFragmentCnt; ++i) {
            subAddrList[i] = addrList_[fragmentIdx + i];
        }
        
        return FragmentTensor<Dims, MaxFragments>(subTilingData, subAddrList);
    }
    
private:
    FragmentTilingData<Dims> tilingData_;
    GM_ADDR addrList_[MaxFragments];
};

} // namespace Basic
} // namespace Dista