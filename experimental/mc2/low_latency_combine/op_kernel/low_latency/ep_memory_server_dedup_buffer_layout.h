// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
// Existing ascendEP dispatch controls end before 764 KiB. Keep its 2 MiB
// data boundary and fixed 4 GiB bank stride; do not import upstream offsets.
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET (764ULL * 1024ULL)
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET 0ULL
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET 16384ULL
#define COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET (2ULL * 1024ULL * 1024ULL)
