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
 * \file test_geir_moe_tutel_combine_gates.cpp
 * \brief MoeTutelCombineGates GE graph-mode (GEIR) end-to-end example.
 *
 *        Builds a single-operator compute graph through the operator IR
 *            REG_OP(MoeTutelCombineGates)  (op_graph/moe_tutel_combine_gates_proto.h)
 *        feeds x / y_grad / indices / locations through ge::Session::RunGraph and
 *        compares gates_grad with an independent CPU float64 oracle of
 *            gates_grad[b, s] = sum_h x[s, h] * y_grad[e * C + p, h],
 *            e = indices[b, s], p = locations[b, s],
 *        with the out-of-range routing rule (e < 0 / e >= B / p < 0 / p >= C -> exactly 0).
 *        Cases cover all three supported dtype combinations (bfloat16 / float16 / float32).
 *
 *        Build and run with:
 *            bash build.sh --run_example moe_tutel_combine_gates graph --soc=ascend950
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api_types.h"
#include "ge_api.h"
// ops_proto_legacy.h declares the ge::op::Data operator (successor of the
// deprecated array_ops.h); moe_tutel_combine_gates_proto.h is included next and
// registers MoeTutelCombineGates in the same ge::op namespace.
#include "ops_proto_legacy.h"

#include "../op_graph/moe_tutel_combine_gates_proto.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// Element access, generic over the three supported dtypes. All host buffers are
// byte vectors and every access goes through StoreElem / LoadElem, so one harness
// serves bfloat16, float16 and float32.
// ---------------------------------------------------------------------------
enum class DType {
    BF16,
    FP16,
    FP32
};

size_t ElemSize(DType dt)
{
    return (dt == DType::FP32) ? sizeof(float) : sizeof(uint16_t);
}

ge::DataType GeType(DType dt)
{
    switch (dt) {
        case DType::BF16:
            return ge::DT_BF16;
        case DType::FP16:
            return ge::DT_FLOAT16;
        default:
            return ge::DT_FLOAT;
    }
}

// Round half to even, which is what both the oracle's single final rounding and
// the kernel's CAST_ROUND produce.
uint16_t FloatToBf16Bits(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

uint16_t FloatToFp16Bits(float f)
{
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFFu) == 0xFFu) { // NaN / Inf
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) { // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) { // subnormal / underflow
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) {
            ++half;
        }
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = ((static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        ++half; // may carry into the exponent, which is correct
    }
    return static_cast<uint16_t>(sign | half);
}

void StoreElem(DType dt, std::vector<uint8_t> &buf, size_t idx, double v)
{
    const float f = static_cast<float>(v);
    if (dt == DType::FP32) {
        std::memcpy(buf.data() + idx * sizeof(float), &f, sizeof(float));
    } else if (dt == DType::BF16) {
        const uint16_t bits = FloatToBf16Bits(f);
        std::memcpy(buf.data() + idx * sizeof(uint16_t), &bits, sizeof(bits));
    } else {
        const uint16_t bits = FloatToFp16Bits(f);
        std::memcpy(buf.data() + idx * sizeof(uint16_t), &bits, sizeof(bits));
    }
}

double LoadElem(DType dt, const std::vector<uint8_t> &buf, size_t idx)
{
    if (dt == DType::FP32) {
        float f = 0.0f;
        std::memcpy(&f, buf.data() + idx * sizeof(float), sizeof(f));
        return static_cast<double>(f);
    }
    uint16_t bits = 0;
    std::memcpy(&bits, buf.data() + idx * sizeof(uint16_t), sizeof(bits));
    if (dt == DType::BF16) {
        const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
        float f = 0.0f;
        std::memcpy(&f, &f32, sizeof(f));
        return static_cast<double>(f);
    }
    // float16 bit pattern -> float, exact for every finite value.
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t mant = bits & 0x3FFu;
    uint32_t f32 = 0;
    if (exp == 0) {
        if (mant != 0) { // subnormal
            uint32_t m = mant;
            int32_t e = -14;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            f32 = sign | (static_cast<uint32_t>(e + 127) << 23) | ((m & 0x3FFu) << 13);
        } else {
            f32 = sign;
        }
    } else if (exp == 0x1Fu) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &f32, sizeof(f));
    return static_cast<double>(f);
}

// ---------------------------------------------------------------------------
// Deterministic xorshift64 so a failing case can be reproduced bit for bit.
// ---------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(uint64_t seed)
        : state_(seed | 1ULL)
    {}

    uint32_t Next()
    {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return static_cast<uint32_t>(state_ >> 32);
    }

    // [-1, 1) with 2^-31 granularity: keeps H-long inner products inside the
    // normal range of all three dtypes.
    double NextSigned()
    {
        return static_cast<double>(static_cast<int32_t>(Next())) * (1.0 / 2147483648.0);
    }

private:
    uint64_t state_;
};

struct Shape {
    int64_t S = 0; // samples  = x.shape[0]      = indices.shape[1]
    int64_t H = 0; // hidden   = x.shape[1]      = y_grad.shape[2]
    int64_t B = 0; // batch    = y_grad.shape[0] = indices.shape[0]
    int64_t C = 0; // capacity = y_grad.shape[1]
    int64_t TaskNum() const
    {
        return S * B;
    }
};

struct HostInputs {
    std::vector<uint8_t> x;
    std::vector<uint8_t> y;
    std::vector<int32_t> idx;
    std::vector<int32_t> loc;
};

// Illegal-routing design points are injected through `illegal`: index i of the
// vector selects the routing defective for the i-th task, cycled over the 4
// out-of-range classes so that all of them are exercised. -1 means "legal".
HostInputs BuildInputs(const Shape &sh, DType dt, uint64_t seed, const std::vector<int32_t> &illegal)
{
    HostInputs in;
    Rng rng(seed);
    const size_t esz = ElemSize(dt);
    in.x.assign(static_cast<size_t>(sh.S * sh.H) * esz, 0);
    for (size_t i = 0; i < static_cast<size_t>(sh.S * sh.H); ++i) {
        StoreElem(dt, in.x, i, rng.NextSigned());
    }
    in.y.assign(static_cast<size_t>(sh.B * sh.C * sh.H) * esz, 0);
    for (size_t i = 0; i < static_cast<size_t>(sh.B * sh.C * sh.H); ++i) {
        StoreElem(dt, in.y, i, rng.NextSigned());
    }
    const int64_t taskNum = sh.TaskNum();
    in.idx.assign(static_cast<size_t>(taskNum), 0);
    in.loc.assign(static_cast<size_t>(taskNum), 0);
    for (int64_t t = 0; t < taskNum; ++t) {
        const int32_t e = (sh.B > 0) ? static_cast<int32_t>(rng.Next() % static_cast<uint32_t>(sh.B)) : 0;
        const int32_t p = (sh.C > 0) ? static_cast<int32_t>(rng.Next() % static_cast<uint32_t>(sh.C)) : 0;
        in.idx[static_cast<size_t>(t)] = e;
        in.loc[static_cast<size_t>(t)] = p;
        if (!illegal.empty()) {
            switch (illegal[static_cast<size_t>(t) % illegal.size()]) {
                case 0:
                    in.idx[static_cast<size_t>(t)] = -1;
                    break; // e < 0
                case 1:
                    in.idx[static_cast<size_t>(t)] = static_cast<int32_t>(sh.B);
                    break; // e >= B
                case 2:
                    in.loc[static_cast<size_t>(t)] = -1;
                    break; // p < 0
                case 3:
                    in.loc[static_cast<size_t>(t)] = static_cast<int32_t>(sh.C);
                    break; // p >= C
                default:
                    break;
            }
        }
    }
    return in;
}

// Independent CPU float64 oracle; it never touches GE, the tiling or the kernel.
void ComputeGolden(const Shape &sh, DType dt, const HostInputs &in, std::vector<double> &golden)
{
    const int64_t taskNum = sh.TaskNum();
    golden.assign(static_cast<size_t>(taskNum), 0.0);
    if (sh.S <= 0 || sh.B <= 0 || sh.H <= 0 || sh.C <= 0 || taskNum <= 0) {
        return; // empty inner product / no readable slot -> exactly 0 everywhere
    }
    for (int64_t t = 0; t < taskNum; ++t) {
        const int64_t e = in.idx[static_cast<size_t>(t)];
        const int64_t p = in.loc[static_cast<size_t>(t)];
        if (!((e >= 0) && (e < sh.B) && (p >= 0) && (p < sh.C))) {
            continue; // out-of-range routing -> exactly 0
        }
        const int64_t token = t % sh.S;
        const int64_t slot = e * sh.C + p;
        double acc = 0.0;
        for (int64_t h = 0; h < sh.H; ++h) {
            acc += LoadElem(dt, in.x, static_cast<size_t>(token * sh.H + h)) *
                   LoadElem(dt, in.y, static_cast<size_t>(slot * sh.H + h));
        }
        golden[static_cast<size_t>(t)] = acc;
    }
}

// Wrap host bytes in a ge::Tensor whose desc matches the Data node of the graph.
ge::Tensor MakeGeTensor(const std::vector<int64_t> &dims, ge::DataType dt, const uint8_t *data, size_t bytes)
{
    ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, dt);
    desc.SetRealDimCnt(dims.size());
    // Keep the placement aligned with the Data node desc (kPlacementHost): GE
    // inserts the host-to-device transfer itself before the kernel runs.
    desc.SetPlacement(ge::kPlacementHost);
    return ge::Tensor(desc, const_cast<uint8_t *>(data), bytes);
}

// ---------------------------------------------------------------------------
// RunCase — one end-to-end case: build the single-op graph, feed the four
// inputs, run the graph and compare gates_grad with the oracle.
// ---------------------------------------------------------------------------
int RunCase(ge::Session &session, uint32_t graphId, const char *name, const Shape &sh, DType dt, uint64_t seed,
            const std::vector<int32_t> &illegal, double rtol, double atol)
{
    const HostInputs in = BuildInputs(sh, dt, seed, illegal);
    std::vector<double> golden;
    ComputeGolden(sh, dt, in, golden);

    // --- build the graph: Data(x) Data(y_grad) Data(indices) Data(locations)
    //     -> MoeTutelCombineGates -> gates_grad (order strictly matches proto.h)
    ge::Graph graph(name);
    auto dataX = ge::op::Data("x").set_attr_index(0);
    auto dataY = ge::op::Data("y_grad").set_attr_index(1);
    auto dataIdx = ge::op::Data("indices").set_attr_index(2);
    auto dataLoc = ge::op::Data("locations").set_attr_index(3);

    ge::TensorDesc descX(ge::Shape({sh.S, sh.H}), ge::FORMAT_ND, GeType(dt));
    ge::TensorDesc descY(ge::Shape({sh.B, sh.C, sh.H}), ge::FORMAT_ND, GeType(dt));
    ge::TensorDesc descIdx(ge::Shape({sh.B, sh.S}), ge::FORMAT_ND, ge::DT_INT32);
    descX.SetPlacement(ge::kPlacementHost);
    descY.SetPlacement(ge::kPlacementHost);
    descIdx.SetPlacement(ge::kPlacementHost);
    dataX.update_input_desc_x(descX);
    dataY.update_input_desc_x(descY);
    dataIdx.update_input_desc_x(descIdx);
    dataLoc.update_input_desc_x(descIdx);

    auto op = ge::op::MoeTutelCombineGates(name);
    op.set_input_x(dataX);
    op.set_input_y_grad(dataY);
    op.set_input_indices(dataIdx);
    op.set_input_locations(dataLoc);

    ge::TensorDesc descOut(ge::Shape({sh.B, sh.S}), ge::FORMAT_ND, GeType(dt));
    op.update_output_desc_gates_grad(descOut);

    graph.AddOp(dataX);
    graph.AddOp(dataY);
    graph.AddOp(dataIdx);
    graph.AddOp(dataLoc);
    std::vector<ge::Operator> graphInputs = {dataX, dataY, dataIdx, dataLoc};
    std::vector<ge::Operator> graphOutputs = {op};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    CHECK_RET(session.AddGraph(graphId, graph) == ge::GRAPH_SUCCESS,
              LOG_PRINT("[%s] FAILED: session AddGraph\n", name);
              return -1);

    // --- feed the four inputs (order strictly matches the graph inputs)
    const size_t idxBytes = in.idx.size() * sizeof(int32_t);
    std::vector<ge::Tensor> inputs;
    inputs.push_back(MakeGeTensor({sh.S, sh.H}, GeType(dt), in.x.data(), in.x.size()));
    inputs.push_back(MakeGeTensor({sh.B, sh.C, sh.H}, GeType(dt), in.y.data(), in.y.size()));
    inputs.push_back(MakeGeTensor({sh.B, sh.S}, ge::DT_INT32, reinterpret_cast<const uint8_t *>(in.idx.data()),
                                   idxBytes));
    inputs.push_back(MakeGeTensor({sh.B, sh.S}, ge::DT_INT32, reinterpret_cast<const uint8_t *>(in.loc.data()),
                                   idxBytes));

    std::vector<ge::Tensor> outputs;
    CHECK_RET(session.RunGraph(graphId, inputs, outputs) == ge::GRAPH_SUCCESS,
              LOG_PRINT("[%s] FAILED: RunGraph (tiling/kernel error)\n", name);
              return -1);
    CHECK_RET(outputs.size() == 1, LOG_PRINT("[%s] FAILED: unexpected output count %zu\n", name, outputs.size());
              return -1);

    // --- compare every task against the oracle
    const int64_t taskNum = sh.TaskNum();
    const size_t esz = ElemSize(dt);
    const uint8_t *outData = outputs[0].GetData();
    CHECK_RET(taskNum == 0 || outData != nullptr, LOG_PRINT("[%s] FAILED: null output data\n", name); return -1);
    // Copy into a byte vector so the same LoadElem comparator serves all dtypes.
    std::vector<uint8_t> outBytes;
    if (outData != nullptr) {
        outBytes.assign(outData, outData + static_cast<size_t>(taskNum) * esz);
    }

    int failed = 0;
    double worst = 0.0;
    for (int64_t t = 0; t < taskNum; ++t) {
        const double got = LoadElem(dt, outBytes, static_cast<size_t>(t));
        const double want = golden[static_cast<size_t>(t)];
        const double diff = std::fabs(got - want);
        worst = (diff > worst) ? diff : worst;
        const bool isExactZero = (illegal.size() == 4); // every task is out of range
        const bool pass = isExactZero ? (got == 0.0) : (diff <= atol + rtol * std::fabs(want));
        if (!pass && failed < 3) {
            LOG_PRINT("[%s] mismatch at task %ld: got %.10g, want %.10g\n", name, static_cast<long>(t), got, want);
            ++failed;
        }
    }

    if (failed > 0) {
        LOG_PRINT("[%s] FAILED (S=%ld H=%ld B=%ld C=%ld, worst |diff| = %.6g)\n", name, static_cast<long>(sh.S),
                  static_cast<long>(sh.H), static_cast<long>(sh.B), static_cast<long>(sh.C), worst);
        return -1;
    }
    LOG_PRINT("[%s] PASSED (S=%ld H=%ld B=%ld C=%ld taskNum=%ld, worst |diff| = %.6g)\n", name, static_cast<long>(sh.S),
              static_cast<long>(sh.H), static_cast<long>(sh.B), static_cast<long>(sh.C),
              static_cast<long>(taskNum), worst);
    return 0;
}

} // namespace

int main()
{
    LOG_PRINT("Initialize ge using ge global options\n");
    std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                  {"ge.graphRunMode", "1"}};
    CHECK_RET(ge::GEInitialize(globalOptions) == ge::GRAPH_SUCCESS, LOG_PRINT("GEInitialize failed\n"); return -1);

    int ret = 0;
    {
        const std::map<ge::AscendString, ge::AscendString> buildOptions = {};
        ge::Session session(buildOptions);

        const std::vector<int32_t> legal;
        const std::vector<int32_t> allIllegal = {0, 1, 2, 3}; // e<0 / e>=B / p<0 / p>=C, cycled
        uint32_t graphId = 0;

        // float32 (tilingKey 1): rtol = atol = 1e-6 — the reduction is a compensated fp32 sum.
        ret |= RunCase(session, graphId++, "fp32_TC01", {500, 64, 2, 500}, DType::FP32, 0x5A5A1234ULL, legal, 1e-6,
                       1e-6);
        ret |= RunCase(session, graphId++, "fp32_TC06_illegal", {353, 512, 3, 64}, DType::FP32, 0x12345678ULL,
                       allIllegal, 1e-6, 1e-6);
        ret |= RunCase(session, graphId++, "fp32_H1", {4, 1, 2, 3}, DType::FP32, 0x0F0F0F0FULL, legal, 1e-6, 1e-6);
        ret |= RunCase(session, graphId++, "fp32_H0_zero", {4, 0, 2, 3}, DType::FP32, 0x2468ACE0ULL, legal, 1e-6,
                       1e-6);
        // bfloat16 (tilingKey 0): rtol = atol = 1e-2.
        ret |= RunCase(session, graphId++, "bf16_TC03", {1024, 1024, 8, 256}, DType::BF16, 0x11112222ULL, legal, 1e-2,
                       1e-2);
        ret |= RunCase(session, graphId++, "bf16_TC19_unaligned", {128, 513, 2, 64}, DType::BF16, 0x33334444ULL,
                       legal, 1e-2, 1e-2);
        // float16 (tilingKey 2): rtol = atol = 1e-3.
        ret |= RunCase(session, graphId++, "fp16_unaligned", {96, 130, 3, 17}, DType::FP16, 0x55556666ULL, legal,
                       1e-3, 1e-3);
        ret |= RunCase(session, graphId++, "fp16_S0_empty", {0, 8, 2, 3}, DType::FP16, 0x77778888ULL, legal, 1e-3,
                       1e-3);
    }

    CHECK_RET(ge::GEFinalize() == ge::GRAPH_SUCCESS, LOG_PRINT("GEFinalize failed\n"); return -1);
    LOG_PRINT("test_geir_moe_tutel_combine_gates %s\n", (ret == 0) ? "PASSED" : "FAILED");
    return ret;
}
