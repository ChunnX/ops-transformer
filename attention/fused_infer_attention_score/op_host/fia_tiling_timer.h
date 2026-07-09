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
 * \file fia_tiling_timer.h
 * \brief Tiling 阶段耗时埋点工具：全局单例 + RAII scope 计时 + OP_LOGE_WITHOUT_REPORT 输出
 */

#ifndef FIA_TILING_TIMER_H
#define FIA_TILING_TIMER_H

#include <vector>
#include <string>
#include <cstdint>
#include <time.h>
#include "log/log.h"

namespace optiling {

struct FiaTimingRecord {
    std::string name;
    int64_t startNs;
    int64_t endNs;
    int depth;
};

class FiaTilingTimer {
public:
    static FiaTilingTimer& GetInstance()
    {
        static FiaTilingTimer instance;
        return instance;
    }

    void Reset()
    {
        records_.clear();
        depth_ = 0;
    }

    int AcquireDepth()
    {
        return depth_++;
    }

    void ReleaseDepth()
    {
        if (depth_ > 0) {
            depth_--;
        }
    }

    void AddRecord(const std::string& name, int64_t startNs, int64_t endNs, int depth)
    {
        records_.push_back({name, startNs, endNs, depth});
    }

    const std::vector<FiaTimingRecord>& GetRecords() const
    {
        return records_;
    }

    static int64_t GetNsTs()
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
    }

    void LogReport(const char* opName) const
    {
        if (records_.empty()) {
            return;
        }
        OP_LOGE_WITHOUT_REPORT(opName, "[FIA_TILING_TIME] === Timing Report Start (records=%zu) ===",
                               records_.size());
        for (size_t i = 0; i < records_.size(); i++) {
            const FiaTimingRecord& r = records_[i];
            int64_t dur = r.endNs - r.startNs;
            int64_t startSec = r.startNs / 1000000000LL;
            int64_t startNsPart = r.startNs % 1000000000LL;
            if (startNsPart < 0) {
                startNsPart += 1000000000LL;
                startSec -= 1;
            }
            int indent = r.depth * 2;
            if (indent > 40) {
                indent = 40;
            }
            OP_LOGE_WITHOUT_REPORT(opName,
                "[FIA_TILING_TIME] [%03zu] %*s%s | start=%lld.%09lld dur=%lldns(%lldus)",
                i, indent, "", r.name.c_str(),
                static_cast<long long>(startSec), static_cast<long long>(startNsPart),
                static_cast<long long>(dur), static_cast<long long>(dur / 1000LL));
        }
        OP_LOGE_WITHOUT_REPORT(opName, "[FIA_TILING_TIME] === Timing Report End ===");
    }

private:
    FiaTilingTimer() = default;
    std::vector<FiaTimingRecord> records_;
    int depth_ = 0;
};

class FiaTilingScopeTimer {
public:
    explicit FiaTilingScopeTimer(const char* name) : name_(name)
    {
        startNs_ = FiaTilingTimer::GetNsTs();
        depth_ = FiaTilingTimer::GetInstance().AcquireDepth();
    }
    ~FiaTilingScopeTimer()
    {
        int64_t endNs = FiaTilingTimer::GetNsTs();
        FiaTilingTimer::GetInstance().AddRecord(name_, startNs_, endNs, depth_);
        FiaTilingTimer::GetInstance().ReleaseDepth();
    }
    FiaTilingScopeTimer(const FiaTilingScopeTimer&) = delete;
    FiaTilingScopeTimer& operator=(const FiaTilingScopeTimer&) = delete;

private:
    const char* name_;
    int64_t startNs_ = 0;
    int depth_ = 0;
};

class FiaTilingReportGuard {
public:
    explicit FiaTilingReportGuard(const char* opName) : opName_(opName)
    {
        FiaTilingTimer::GetInstance().Reset();
    }
    ~FiaTilingReportGuard()
    {
        FiaTilingTimer::GetInstance().LogReport(opName_);
    }
    FiaTilingReportGuard(const FiaTilingReportGuard&) = delete;
    FiaTilingReportGuard& operator=(const FiaTilingReportGuard&) = delete;

private:
    const char* opName_;
};

} // namespace optiling

#define FIA_TIMING_CONCAT_IMPL(a, b) a##b
#define FIA_TIMING_CONCAT(a, b) FIA_TIMING_CONCAT_IMPL(a, b)
#define FIA_TIMING_SCOPE(name) \
    optiling::FiaTilingScopeTimer FIA_TIMING_CONCAT(_fiaTimer_, __LINE__)(name)

#endif // FIA_TILING_TIMER_H
