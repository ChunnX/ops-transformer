/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.  Licensed under Apache-2.0.
 *
 *  test_combine_teardown.cpp —— CombineTeardown(mxfp8 多核) 单卡 aclnn 测试驱动。
 *  用法: test_combine_teardown <bs> <topk> <h> <E> <R> <datadir>
 *  mxfp8：combine_data=fp8(e4m3fn,1B)；combine_scale=e8m0(1B,[GE*bs,h/32])；x 输出 bf16。
 *  输入 bin： combine_data.bin(fp8,[GE*bs,h]) combine_scale.bin(e8m0,[GE*bs,h/32]) expert_scales.bin(fp32,[bs,topk])
 *            expand_idx.bin(int32,[bs,topk]) expert_ids.bin(int32,[bs,topk])
 *  输出 bin： x.npu.bin(bf16,[bs,h])
 */
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_combine_teardown.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif
#define ACLCHECK(expr)                                                                 \
    do { aclError __r = (expr);                                                        \
        if (__r != ACL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); return -1; } \
    } while (0)

static bool ReadFile(const std::string &path, void *buf, size_t bytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("[ERR] open read %s\n", path.c_str()); return false; }
    f.read(reinterpret_cast<char *>(buf), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(f.gcount()) != bytes) { printf("[ERR] short read %s\n", path.c_str()); return false; }
    return true;
}
static bool WriteFile(const std::string &path, const void *buf, size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { printf("[ERR] open write %s\n", path.c_str()); return false; }
    f.write(reinterpret_cast<const char *>(buf), static_cast<std::streamsize>(bytes));
    return true;
}
static aclTensor *MakeTensor(void *dev, const std::vector<int64_t> &shape, aclDataType dt)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), dev);
}

int main(int argc, char **argv)
{
    if (argc < 7) { printf("usage: %s bs topk h E R datadir\n", argv[0]); return -1; }
    const uint32_t bs = (uint32_t)atoi(argv[1]);
    const uint32_t topk = (uint32_t)atoi(argv[2]);
    const uint32_t h = (uint32_t)atoi(argv[3]);
    const uint32_t E = (uint32_t)atoi(argv[4]);
    const uint32_t R = (uint32_t)atoi(argv[5]);
    const std::string dir = argv[6];
    const uint32_t GE = R * E;
    const uint32_t scaleH = h / 32U;

    const size_t cdB = (size_t)GE * bs * h;           // fp8 payload bytes
    const size_t csB = (size_t)GE * bs * scaleH;      // e8m0 scale bytes
    const size_t idsElems = (size_t)bs * topk;
    const size_t xElems = (size_t)bs * h;
    const size_t bf16B = sizeof(uint16_t);

    printf("[cfg] bs=%u topk=%u h=%u E=%u R=%u GE=%u scaleH=%u (mxfp8)\n", bs, topk, h, E, R, GE, scaleH);

    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));

    std::vector<uint8_t> hCd(cdB), hCs(csB);
    std::vector<float> hSc(idsElems);
    std::vector<int32_t> hIdx(idsElems), hId(idsElems);
    if (!ReadFile(dir + "/combine_data.bin", hCd.data(), cdB)) return -1;
    if (!ReadFile(dir + "/combine_scale.bin", hCs.data(), csB)) return -1;
    if (!ReadFile(dir + "/expert_scales.bin", hSc.data(), idsElems * sizeof(float))) return -1;
    if (!ReadFile(dir + "/expand_idx.bin", hIdx.data(), idsElems * sizeof(int32_t))) return -1;
    if (!ReadFile(dir + "/expert_ids.bin", hId.data(), idsElems * sizeof(int32_t))) return -1;

    void *dCd = nullptr, *dCs = nullptr, *dSc = nullptr, *dIdx = nullptr, *dId = nullptr, *dX = nullptr;
    ACLCHECK(aclrtMalloc(&dCd, cdB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dCs, csB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dSc, idsElems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dIdx, idsElems * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dId, idsElems * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dX, xElems * bf16B, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMemcpy(dCd, cdB, hCd.data(), cdB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dCs, csB, hCs.data(), csB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dSc, idsElems * sizeof(float), hSc.data(), idsElems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dIdx, idsElems * sizeof(int32_t), hIdx.data(), idsElems * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dId, idsElems * sizeof(int32_t), hId.data(), idsElems * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(dX, xElems * bf16B, 0, xElems * bf16B));

    aclTensor *cdT = MakeTensor(dCd, {(int64_t)(GE * bs), (int64_t)h}, ACL_FLOAT8_E4M3FN);
    aclTensor *csT = MakeTensor(dCs, {(int64_t)(GE * bs), (int64_t)scaleH}, ACL_FLOAT8_E8M0);
    aclTensor *scT = MakeTensor(dSc, {(int64_t)bs, (int64_t)topk}, ACL_FLOAT);
    aclTensor *idxT = MakeTensor(dIdx, {(int64_t)bs, (int64_t)topk}, ACL_INT32);
    aclTensor *idT = MakeTensor(dId, {(int64_t)bs, (int64_t)topk}, ACL_INT32);
    aclTensor *xT = MakeTensor(dX, {(int64_t)bs, (int64_t)h}, ACL_BF16);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = aclnnCombineTeardownGetWorkspaceSize(cdT, csT, scT, idxT, idT,
        (int64_t)R, (int64_t)E, xT, &wsSize, &exec);
    if (s != ACLNN_SUCCESS) { printf("[ERR] GetWorkspaceSize ret=%d\n", s); return -1; }
    void *ws = nullptr;
    if (wsSize > 0) { ACLCHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST)); }
    s = aclnnCombineTeardown(ws, wsSize, exec, stream);
    if (s != ACLNN_SUCCESS) { printf("[ERR] aclnnCombineTeardown ret=%d\n", s); return -1; }
    ACLCHECK(aclrtSynchronizeStream(stream));
    printf("[ok] combine_teardown executed\n");

    std::vector<uint16_t> hX(xElems);
    ACLCHECK(aclrtMemcpy(hX.data(), xElems * bf16B, dX, xElems * bf16B, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(dir + "/x.npu.bin", hX.data(), xElems * bf16B);
    printf("[ok] outputs written to %s\n", dir.c_str());

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
