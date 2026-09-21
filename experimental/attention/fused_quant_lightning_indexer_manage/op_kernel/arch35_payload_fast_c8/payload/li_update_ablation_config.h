/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H
#define VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H

// P5 is the production path from ops_li_update_a5@0362e7e: construct the
// survivor payload, classify hit/miss entries, select victims, and commit the
// request-pool cache state.  Diagnostic P0-P4 builds are intentionally not
// exposed by this standalone nano-vLLM operator project.
#define LI_UPDATE_ABLATION_MODE 5

#endif // VLLM_FAST_LI_UPDATE_ABLATION_CONFIG_H
