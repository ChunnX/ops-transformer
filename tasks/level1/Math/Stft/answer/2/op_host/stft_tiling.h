/*!
 * \file stft_tiling.h
 * \brief
 */
#ifndef STFT_TILING_H_
#define STFT_TILING_H_

#pragma once
#include "register/op_def_registry.h"

namespace optiling {

struct STFTCompileInfo {
    uint64_t coreNum;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t sysWorkspaceSize;
};
} // namespace optiling
#endif