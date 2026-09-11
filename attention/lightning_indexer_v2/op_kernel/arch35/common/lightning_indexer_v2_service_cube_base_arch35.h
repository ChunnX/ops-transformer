/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_SERVICE_CUBE_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_CUBE_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

template <typename Derived>
class LightningIndexerV2ServiceCubeBase {
public:
    __aicore__ inline void AllocEventID()
    {
        AscendC::SetMMLayoutTransform(true);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 2);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::QUERY_MTE1_MTE2_EVENT + 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::QUERY_MTE1_MTE2_EVENT + 1);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Derived::M_MTE1_EVENT + 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(Derived::M_MTE1_EVENT + 1);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Derived::FIX_M_EVENT + 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(Derived::FIX_M_EVENT + 1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::SetMMLayoutTransform(false);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::KEY_MTE1_MTE2_EVENT + 2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::QUERY_MTE1_MTE2_EVENT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Derived::QUERY_MTE1_MTE2_EVENT + 1);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Derived::M_MTE1_EVENT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(Derived::M_MTE1_EVENT + 1);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Derived::FIX_M_EVENT + 0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(Derived::FIX_M_EVENT + 1);
    }

protected:
    template <typename K, typename Q>
    __aicore__ inline void BindCubeTensors(const AscendC::GlobalTensor<int32_t> &blockTable,
        const AscendC::GlobalTensor<K> &key, const AscendC::GlobalTensor<Q> &query)
    {
        auto &service = static_cast<Derived &>(*this);
        service.blkTableGm_ = blockTable;
        service.keyGm_ = key;
        service.queryGm_ = query;
    }
};

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_SERVICE_CUBE_BASE_ARCH35_H
