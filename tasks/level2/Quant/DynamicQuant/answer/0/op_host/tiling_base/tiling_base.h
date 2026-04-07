/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Bench：从 RmsNormQuant 等任务拷贝的 Tiling 基类；`log.h` 改为 `common/log.h`。
 */
#pragma once

#include <memory>
#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../common/log.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

namespace Ops {
namespace NN {
namespace Optiling {

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t blockDim = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

class TilingBaseClass {
public:
    explicit TilingBaseClass(gert::TilingContext* context) : context_(context) {}

    virtual ~TilingBaseClass() = default;

    ge::graphStatus DoTiling()
    {
        if (context_ == nullptr) {
            OP_LOGE("tiling", "context_ cannot be null");
            return ge::GRAPH_FAILED;
        }
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = DoLibApiTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = PostTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context_->SetTilingKey(GetTilingKey());
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }

    virtual void Reset(gert::TilingContext* context)
    {
        if (context == nullptr) {
            OP_LOGD("tiling", "context is null");
        }
        context_ = context;
    }

protected:
    virtual bool IsCapable() = 0;
    virtual ge::graphStatus GetPlatformInfo() = 0;
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    virtual ge::graphStatus DoOpTiling() = 0;
    virtual ge::graphStatus DoLibApiTiling() = 0;
    [[nodiscard]] virtual uint64_t GetTilingKey() const = 0;
    virtual ge::graphStatus GetWorkspaceSize() = 0;
    virtual ge::graphStatus PostTiling() = 0;
    virtual void DumpTilingInfo() {}

    static uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
    {
        uint32_t ration;
        if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
            return sliceNum;
        }
        ration = aivCoreNum / aicCoreNum;
        return (sliceNum + (ration - 1)) / ration;
    }

    template <typename T>
    [[nodiscard]] std::string GetShapeDebugStr(const T& shape) const
    {
        std::ostringstream oss;
        oss << "[";
        if (shape.GetDimNum() > 0) {
            for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
                oss << shape.GetDim(i) << ", ";
            }
            oss << shape.GetDim(shape.GetDimNum() - 1);
        }
        oss << "]";
        return oss.str();
    }

protected:
    gert::TilingContext* context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_;
};

} // namespace Optiling
} // namespace NN
} // namespace Ops
