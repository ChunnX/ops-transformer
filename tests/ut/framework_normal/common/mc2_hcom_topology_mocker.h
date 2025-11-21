/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HCOM_TOPOLOGY_MOCKER_H
#define MC2_HCOM_TOPOLOGY_MOCKER_H

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

namespace Mc2Hcom {
using MockValues = std::vector<std::pair<const char*, uint32_t>>;

class MC2HcomTopologyMocker {
public:
    static MC2HcomTopologyMocker& GetInstance();
    void SetValue(const char* key, uint32_t value);
    void SetValues(const MockValues& values);
    uint32_t GetValue(const char* key, uint32_t defaultValue) const;
    void Reset();

private:
    MC2HcomTopologyMocker() = default;
    std::unordered_map<std::string, uint32_t> mockValue_;
};

}

#endif // MC2_HCOM_TOPOLOGY_MOCKER_H