// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapted from DeepEP; original copyright and license notices follow.

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "runtime/session.hpp"
#include <cstring>
#include <cstdlib>
#include <vector>
#include "kernels/udma_layout.hpp"
#include <mutex>
#include <stdexcept>
#include "host/init/shmem_host_init.h"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_cc.h"

namespace deepep {
namespace {
std::mutex runtime_mutex;
bool occupied = false;
void check(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed (status=" + std::to_string(status) + ")");
    }
}
} // namespace

bool Session::busy()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    return occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED;
}

Session::Session(int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout, bool udma,
                 bool megamoe)
{
    constexpr uint64_t alignment = 2ULL * 1024 * 1024;
    if (size < 2 || size > (megamoe ? 128 : 256) || (!megamoe && (size & (size - 1))) || rank < 0 || rank >= size ||
        bytes < alignment || bytes > (megamoe ? 1ULL : 32ULL) * 1024 * 1024 * 1024 || bytes % alignment != 0 ||
        timeout < 1 || timeout > 300 || (megamoe && udma) || endpoint.empty() ||
        endpoint.size() >= ACLSHMEM_MAX_IP_PORT_LEN || endpoint.find('\0') != std::string::npos) {
        throw std::invalid_argument("Invalid rank, capacity, timeout or bootstrap endpoint");
    }
    // BF16 and FP8 use the same peer-table transport at every supported EP size.
    peer_table_ = udma;
    if (peer_table_ && (size < 2 || size > 256 || (size & (size - 1)))) {
        throw std::invalid_argument("peer-table requires EP2/4/8/16/32/64/128/256");
    }
    std::lock_guard<std::mutex> guard(runtime_mutex);
    if (occupied || aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) {
        throw std::runtime_error("A SHMEM runtime already exists in this process");
    }
    check(aclrtGetCurrentContext(&context_), "aclrtGetCurrentContext");
    if (context_ == nullptr) {
        throw std::runtime_error("Initialize the framework NPU context before creating a runtime");
    }
    const char* soc = aclrtGetSocName();
    if (soc == nullptr || std::string(soc).rfind("Ascend950", 0) != 0) {
        throw std::runtime_error("This runtime build is qualified only for Ascend950");
    }
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t uid{};
    attr.my_pe = rank;
    attr.n_pes = size;
    attr.local_mem_size = bytes;
    std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
    attr.comm_args = &uid;
    attr.option_attr.data_op_engine_type =
        udma ? static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_UDMA) : ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = timeout;
    attr.option_attr.shm_create_timeout = timeout;
    attr.option_attr.control_operation_timeout = timeout;
    // A failed initialization can leave SDK state behind. Keep the process
    // reserved until a new process is launched; do not finalize foreign state.
    occupied = true;
    if (udma) {
        // Combine owns pairs 0..15; peer-table Dispatch uses pairs 0..7.
        // Serialized operators reuse the bank, not concurrent SQ writers.
        // The pinned SDK shares Clos SQs across peers; MESH routes are separate.
        check(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, ascend_deepep::udma_layout::kQpCount),
              "aclshmemx_set_qp_num");
        check(aclshmemx_set_shared_jetty(ACLSHMEM_DATA_OP_UDMA, true), "aclshmemx_set_shared_jetty");
    }
    check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
    try {
        // Peer-table aggregate packets require physical 512B boundaries, not
        // merely 512B-relative offsets within a possibly 16B-aligned allocation.
        workspace_ = udma ? aclshmem_align(512U, bytes) : aclshmem_malloc(bytes);
        if (workspace_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc failed");
        }
        check(aclrtMemset(workspace_, bytes, 0, bytes), "aclrtMemset");
        bytes_ = bytes;
        rank_ = rank;
        world_ = size;
        udma_ = udma;
        closed_ = false;
    } catch (...) {
        if (workspace_ != nullptr) {
            aclshmem_free(workspace_);
            workspace_ = nullptr;
        }
        if (aclshmem_finalize() == 0) {
            occupied = false;
        }
        throw;
    }
}

void Session::check_context() const
{
    if (poisoned_) {
        throw std::runtime_error("Runtime cleanup failed; restart the distributed job");
    }
    if (closed_) {
        throw std::runtime_error("Runtime is closed");
    }
    aclrtContext current = nullptr;
    check(aclrtGetCurrentContext(&current), "aclrtGetCurrentContext");
    if (current != context_) {
        throw std::runtime_error("Use the runtime on its creating NPU context");
    }
}

uintptr_t Session::workspace_address() const
{
    check_context();
    return reinterpret_cast<uintptr_t>(workspace_);
}

void Session::barrier()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    check_context();
    aclshmem_barrier_all();
}

void Session::close()
{
    std::lock_guard<std::mutex> guard(runtime_mutex);
    if (closed_) {
        return;
    }
    check_context();
    aclshmem_barrier_all();
    if (workspace_ != nullptr) {
        aclshmem_free(workspace_);
        workspace_ = nullptr;
    }
    // On failure retain the reservation and never claim successful cleanup.
    const int status = aclshmem_finalize();
    if (status != 0) {
        poisoned_ = true;
        check(status, "aclshmem_finalize (restart the distributed job)");
    }
    occupied = false;
    closed_ = true;
}
} // namespace deepep
