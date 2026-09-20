// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include "framework_stubs.h"
#include "../../op_kernel_aicpu/quant_block_sparse_attn_metadata_aicpu.h"
#include "../../op_host/quant_block_sparse_attn_metadata_check.h"
#define __aicore__
#include "../../../quant_block_sparse_attn/op_kernel/arch35/common/util_regbase_mx.h"

namespace {
void Require(bool condition, const char *message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Case {
    int64_t batch, heads, rows;
    std::vector<int32_t> counts;
    std::vector<uint32_t> output;
    aicpu::Tensor sparse, metadata;
    aicpu::CpuKernelContext ctx;
    Case(int64_t b, int64_t n, int64_t qb, int cores, int block)
        : batch(b),
          heads(n),
          rows(qb),
          counts(b * n * qb),
          output(8 + b * n * 36 * 8 + 72 * 8 + 16, 0xCDCDCDCD),
          sparse({b, n, qb}, counts.data()),
          metadata({static_cast<int64_t>(output.size() - 16)}, output.data())
    {
        ctx.inputs = {&sparse, nullptr, nullptr, nullptr, nullptr};
        ctx.outputs = {&metadata};
        ctx.attrs = {{"batch_size", b},
                     {"num_heads_q", n},
                     {"num_heads_kv", int64_t{1}},
                     {"head_dim", int64_t{128}},
                     {"sparse_block_size_q", int64_t{block}},
                     {"sparse_block_size_k", int64_t{block}},
                     {"quant_mode", int64_t{2}},
                     {"mask_mode", int64_t{3}},
                     {"layout_q", "TND"},
                     {"layout_kv", "PA_BNBD"},
                     {"layout_sparse_indices", "B_N_Qb_Kb"},
                     {"soc_version", "Ascend950"},
                     {"aic_core_num", int64_t{cores}},
                     {"aiv_core_num", int64_t{cores * 2}}};
    }
    uint32_t Run(const char *layout)
    {
        ctx.attrs["layout_q"] = layout;
        std::fill(output.begin(), output.end(), 0xCDCDCDCD);
        aicpu::QuantBlockSparseAttnMetadataCpuKernel kernel;
        const uint32_t status = kernel.Compute(ctx);
        Require(std::all_of(output.end() - 16, output.end(), [](uint32_t x) { return x == 0xCDCDCDCD; }),
                "metadata wrote beyond output capacity");
        return status;
    }
    aclnnStatus Host(const char *layout, const aclTensor *cu = nullptr, const aclTensor *used = nullptr, int mode = 2,
                     const char *kv = "PA_BNBD")
    {
        return ParamsCheck(&sparse, cu, nullptr, used, nullptr, batch, heads, 1, 128, 128, 128, mode, 3, layout, kv,
                           "B_N_Qb_Kb", 36, 72, "Ascend950", &metadata);
    }
};

// Decode the production metadata as the MX consumer does. Count all physical
// Q rows, including empty sparse rows, across every core and section.
void CheckCoverage(const Case &test, uint32_t sq, uint32_t block)
{
    const uint32_t validRows = sq / block + (sq % block != 0);
    std::vector<int> visits(test.batch * test.heads * validRows, 0);
    const auto &data = test.output;
    Require(data[0] > 0 && data[0] <= test.batch * test.heads, "invalid section count");
    for (uint32_t sec = 0; sec < data[0]; ++sec) {
        for (uint32_t core = 0; core < 36; ++core) {
            const auto *entry = data.data() + 8 + (sec * 36 + core) * 8;
            if (entry[0] == 0) {
                continue;
            }
            const uint32_t end = entry[4] + (entry[5] != 0);
            Require(end <= test.batch * test.heads, "metadata BN1 end out of bounds");
            for (uint32_t bn = entry[1]; bn < end; ++bn) {
                const uint32_t first = bn == entry[1] ? entry[2] : 0;
                const uint32_t last = regbasemx::MxS1LoopEnd(sq, block, bn == end - 1 ? entry[5] : 0);
                Require(last <= validRows, "metadata reads beyond physical query rows");
                for (uint32_t row = first; row < last; ++row) {
                    visits.at(bn * validRows + row)++;
                }
            }
        }
    }
    Require(std::all_of(visits.begin(), visits.end(), [](int x) { return x == 1; }),
            "a physical Q row was omitted or scheduled more than once");
}

void TestScheduling()
{
    std::mt19937 rng(11704);
    size_t runs = 0;
    for (const int block : {64, 128}) {
        for (const int sq : {1, 17, 63, 64, 65, 127, 128, 129, 257}) {
            for (const int cores : {1, 2, 7, 36}) {
                for (const int extraRows : {0, 5}) {
                    for (int pattern = 0; pattern < 5; ++pattern) {
                        Case test(2, 4, (sq + block - 1) / block + extraRows, cores, block);
                        for (size_t i = 0; i < test.counts.size(); ++i) {
                            if (pattern == 0) {
                                test.counts[i] = 0;
                            }
                            if (pattern == 1) {
                                test.counts[i] = 4;
                            }
                            if (pattern == 2) {
                                test.counts[i] = rng() % 9;
                            }
                            if (pattern == 3) {
                                test.counts[i] = i < test.counts.size() / 2 ? 3 : 0;
                            }
                            if (pattern == 4) {
                                test.counts[i] = i < test.counts.size() / 2 ? 0 : 3;
                            }
                        }
                        Require(test.Run("TND") == 0, "TND scheduling failed");
                        const auto baseline = test.output;
                        for (const char *layout : {"BSND", "BNSD"}) {
                            Require(test.Run(layout) == 0, "padded scheduling failed");
                            Require(test.output == baseline, "layout changed logical metadata");
                            CheckCoverage(test, sq, block);
                            ++runs;
                        }
                    }
                }
            }
        }
    }
    Case multiSection(2, 4, 5000, 36, 128);
    std::fill(multiSection.counts.begin(), multiSection.counts.end(), 8);
    for (const char *layout : {"TND", "BSND", "BNSD"}) {
        Require(multiSection.Run(layout) == 0 && multiSection.output[0] > 1, "multiple sections not exercised");
        CheckCoverage(multiSection, 5000 * 128 - 1, 128);
    }
    std::cout << "PASS: " << runs << " padded schedule/coverage cases plus 3 multi-section cases\n";
}

void TestContracts()
{
    Case test(2, 4, 3, 36, 128);
    int32_t values[] = {0, 129, 258};
    aicpu::Tensor lengths({3}, values), empty({0}, nullptr);
    for (const char *layout : {"BSND", "BNSD"}) {
        Require(test.Host(layout) == 0, "host rejected valid padded metadata");
        Require(test.Host(layout, &lengths) != 0 && test.Host(layout, nullptr, &lengths) != 0,
                "host accepted explicit query lengths");
        Require(test.Host(layout, &empty) != 0 && test.Host(layout, nullptr, &empty) != 0,
                "host accepted empty tensors instead of nullptr");
        Require(test.Host(layout, nullptr, nullptr, 1) != 0, "host accepted padded FP8");
        test.ctx.inputs[1] = &lengths;
        Require(test.Run(layout) != 0, "AICPU accepted query prefix lengths");
        test.ctx.inputs[1] = nullptr;
        test.ctx.inputs[3] = &lengths;
        Require(test.Run(layout) != 0, "AICPU accepted used Q lengths");
        test.ctx.inputs[3] = nullptr;
        test.sparse.shape.dims = {2, 3};
        Require(test.Host(layout) != 0 && test.Run(layout) != 0, "padded metadata accepted legacy 2D sparse rows");
        test.sparse.shape.dims = {2, 4, 3};
    }
    Require(test.Host("TND") == 0 && test.Host("NTD", nullptr, nullptr, 1) == 0, "packed layouts regressed");
    Require(test.Host("NTD") != 0 && test.Host("invalid") != 0 && test.Host(nullptr) != 0,
            "host accepted invalid layout/mode");
    Require(test.Host("BSND", nullptr, nullptr, 2, "BSND") != 0, "host accepted continuous KV");
    test.ctx.attrs["quant_mode"] = int64_t{1};
    Require(test.Run("BSND") != 0 && test.Run("BNSD") != 0, "AICPU accepted padded FP8");
    Require(test.Run("NTD") == 0, "FP8 NTD regressed");
    test.ctx.attrs["quant_mode"] = int64_t{2};
    Require(test.Run("NTD") != 0 && test.Run("invalid") != 0, "AICPU accepted invalid layout");
    test.ctx.inputs[1] = &empty; // GE's representation of an absent optional input.
    Require(test.Run("BSND") == 0, "AICPU rejected absent-input placeholder");
    test.ctx.inputs[1] = nullptr;
    // ACLNN preserves an absent optional input as a Tensor with a null data
    // address and no shape dimensions. NumElements() is 1 for that shape.
    aicpu::Tensor absentOptional({}, nullptr), explicitScalar({}, values);
    Require(absentOptional.NumElements() == 1, "absent scalar-shaped placeholder was not exercised");
    for (const char *layout : {"BSND", "BNSD"}) {
        for (const uint32_t inputIndex : {1U, 3U}) {
            test.ctx.inputs[inputIndex] = &absentOptional;
            Require(test.Run(layout) == 0, "AICPU rejected ACLNN null-data optional placeholder");
            test.ctx.inputs[inputIndex] = &explicitScalar;
            Require(test.Run(layout) != 0, "AICPU accepted explicit scalar Q lengths");
            test.ctx.inputs[inputIndex] = nullptr;
        }
        test.ctx.inputs[1] = &absentOptional;
        test.ctx.inputs[3] = &absentOptional;
        Require(test.Run(layout) == 0, "AICPU rejected both absent Q length inputs");
        test.ctx.inputs[1] = nullptr;
        test.ctx.inputs[3] = nullptr;
    }
    test.sparse.shape.dims = {2, 3, 3};
    Require(test.Host("BSND") != 0 && test.Run("BSND") != 0, "head mismatch accepted");
    test.sparse.shape.dims = {3, 4, 3};
    Require(test.Host("BNSD") != 0 && test.Run("BNSD") != 0, "batch mismatch accepted");
    test.sparse.shape.dims = {2, 4, 3};
    test.sparse.dtype = aicpu::DT_FLOAT;
    Require(test.Run("BSND") != 0, "AICPU accepted wrong sparse dtype");
    test.sparse.dtype = aicpu::DT_INT32;
    test.metadata.shape.dims = {8};
    Require(test.Run("BSND") != 0, "AICPU accepted undersized output");
    Require(std::all_of(test.output.begin(), test.output.end(), [](uint32_t x) { return x == 0xCDCDCDCD; }),
            "invalid output was modified");
    std::cout << "PASS: ACLNN/AICPU layout, null-input, shape, dtype and output-capacity contracts\n";
}

template <QBSALayout layout>
void TestAddresses()
{
    constexpr uint32_t bSize = 3, nSize = 8, sq = 129;
    std::vector<uint64_t> physical;
    for (uint32_t b = 0; b < bSize; ++b) {
        for (uint32_t i = 0; i < (layout == QBSALayout::BNSD ? nSize : sq); ++i) {
            for (uint32_t j = 0; j < (layout == QBSALayout::BNSD ? sq : nSize); ++j) {
                const auto n = layout == QBSALayout::BNSD ? i : j;
                const auto s = layout == QBSALayout::BNSD ? j : i;
                physical.push_back((b * nSize + n) * sq + s);
            }
        }
    }
    for (uint32_t b = 0; b < bSize; ++b) {
        for (uint32_t n = 0; n < nSize; ++n) {
            for (uint32_t s = 0; s < sq; ++s) {
                const auto slot = regbasemx::MxQuerySlot<layout>(b * sq, s, n, nSize, sq);
                Require(physical.at(slot) == (b * nSize + n) * sq + s, "Q/scale/output address mismatch");
                Require(regbasemx::MxLseSlot<layout>(b * sq, s, n, nSize, sq) == (b * nSize + n) * sq + s,
                        "BNS LSE address mismatch");
            }
        }
    }
}
} // namespace

int main()
{
    try {
        TestContracts();
        TestScheduling();
        TestAddresses<QBSALayout::BSND>();
        TestAddresses<QBSALayout::BNSD>();
        const uint64_t base = uint64_t{1} << 30;
        Require(regbasemx::MxQuerySlot<QBSALayout::BNSD>(base, 7, 3, 8, 129) == base * 8 + 3 * 129 + 7,
                "64-bit address truncated");
        Require(regbasemx::MxS1LoopEnd(UINT32_MAX, 128, 0) == 33554432U, "row count overflow");
        std::cout << "PASS: padded Q/QScale/output/LSE addresses and 32-bit boundary\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
