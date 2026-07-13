/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.  Licensed under Apache-2.0.
 *
 *  test_dispatch_teardown.cpp —— DispatchTeardown(mxfp8 多核) 单卡 aclnn 测试驱动。
 *  纯本地 AIV、无通信 → 单卡单进程：读 golden 造的输入 bin → H2D → aclnn 两段式 → D2H → 写 npu 输出 bin。
 *  用法： test_dispatch_teardown <bs> <topk> <h> <E> <R> <datadir>
 *  mxfp8：token_data/expand_x = fp8(e4m3fn,1B)；token_scale/expand_scale = e8m0(1B)，scaleH=h/32。
 *  输入 bin：token_data.bin(fp8,[R*E*bs,h]) token_scale.bin(e8m0,[R*E*bs,h/32]) wish_cnt.bin(int32,[R*E]) expert_ids.bin(int32,[bs,topk])
 *  输出 bin：expand_x.npu.bin(fp8,[A,h]) expand_scale.npu.bin(e8m0,[A,h/32]) send_counts.npu.bin(int32,[R*E]) expand_idx.npu.bin(int32,[bs,topk]) expert_token_nums.npu.bin(int32,[E])
 */
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_dispatch_teardown.h"

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
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) { strides[i] = strides[i + 1] * shape[i + 1]; }
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
    const uint32_t minKE = (topk < E) ? topk : E;
    const uint32_t A = R * bs * minKE;
    const uint32_t scaleH = h / 32U;

    const size_t tokB = (size_t)GE * bs * h;             // fp8 payload bytes
    const size_t tokScB = (size_t)GE * bs * scaleH;      // e8m0 scale bytes
    const size_t idsElems = (size_t)bs * topk;
    const size_t exB = (size_t)A * h;
    const size_t exScB = (size_t)A * scaleH;

    printf("[cfg] bs=%u topk=%u h=%u E=%u R=%u GE=%u A=%u scaleH=%u (mxfp8)\n", bs, topk, h, E, R, GE, A, scaleH);

    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));

    std::vector<uint8_t> hTok(tokB), hTokSc(tokScB);
    std::vector<int32_t> hWish(GE), hIds(idsElems);
    if (!ReadFile(dir + "/token_data.bin", hTok.data(), tokB)) return -1;
    if (!ReadFile(dir + "/token_scale.bin", hTokSc.data(), tokScB)) return -1;
    if (!ReadFile(dir + "/wish_cnt.bin", hWish.data(), GE * sizeof(int32_t))) return -1;
    if (!ReadFile(dir + "/expert_ids.bin", hIds.data(), idsElems * sizeof(int32_t))) return -1;

    void *dTok = nullptr, *dTokSc = nullptr, *dWish = nullptr, *dIds = nullptr;
    void *dEx = nullptr, *dExSc = nullptr, *dSc = nullptr, *dIdx = nullptr, *dEtn = nullptr;
    ACLCHECK(aclrtMalloc(&dTok, tokB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dTokSc, tokScB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dWish, GE * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dIds, idsElems * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dEx, exB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dExSc, exScB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dSc, GE * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dIdx, idsElems * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&dEtn, E * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMemcpy(dTok, tokB, hTok.data(), tokB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dTokSc, tokScB, hTokSc.data(), tokScB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dWish, GE * sizeof(int32_t), hWish.data(), GE * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(dIds, idsElems * sizeof(int32_t), hIds.data(), idsElems * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(dEx, exB, 0, exB));            // 尾部行保持 0，与 golden zero-pad 对齐
    ACLCHECK(aclrtMemset(dExSc, exScB, 0, exScB));
    ACLCHECK(aclrtMemset(dSc, GE * sizeof(int32_t), 0, GE * sizeof(int32_t)));
    ACLCHECK(aclrtMemset(dIdx, idsElems * sizeof(int32_t), 0, idsElems * sizeof(int32_t)));
    ACLCHECK(aclrtMemset(dEtn, E * sizeof(int32_t), 0, E * sizeof(int32_t)));

    aclTensor *tokenT = MakeTensor(dTok, {(int64_t)(GE * bs), (int64_t)h}, ACL_FLOAT8_E4M3FN);
    aclTensor *tokScT = MakeTensor(dTokSc, {(int64_t)(GE * bs), (int64_t)scaleH}, ACL_FLOAT8_E8M0);
    aclTensor *wishT = MakeTensor(dWish, {(int64_t)GE}, ACL_INT32);
    aclTensor *idsT = MakeTensor(dIds, {(int64_t)bs, (int64_t)topk}, ACL_INT32);
    aclTensor *exT = MakeTensor(dEx, {(int64_t)A, (int64_t)h}, ACL_FLOAT8_E4M3FN);
    aclTensor *exScT = MakeTensor(dExSc, {(int64_t)A, (int64_t)scaleH}, ACL_FLOAT8_E8M0);
    aclTensor *scT = MakeTensor(dSc, {(int64_t)GE}, ACL_INT32);
    aclTensor *idxT = MakeTensor(dIdx, {(int64_t)bs, (int64_t)topk}, ACL_INT32);
    aclTensor *etnT = MakeTensor(dEtn, {(int64_t)E}, ACL_INT32);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = aclnnDispatchTeardownGetWorkspaceSize(tokenT, tokScT, wishT, idsT,
        (int64_t)R, (int64_t)E, exT, exScT, scT, idxT, etnT, &wsSize, &exec);
    if (s != ACLNN_SUCCESS) { printf("[ERR] GetWorkspaceSize ret=%d\n", s); return -1; }
    void *ws = nullptr;
    if (wsSize > 0) { ACLCHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST)); }
    aclrtEvent evS = nullptr, evE = nullptr;
    ACLCHECK(aclrtCreateEvent(&evS)); ACLCHECK(aclrtCreateEvent(&evE));
    ACLCHECK(aclrtRecordEvent(evS, stream));
    s = aclnnDispatchTeardown(ws, wsSize, exec, stream);
    if (s != ACLNN_SUCCESS) { printf("[ERR] aclnnDispatchTeardown ret=%d\n", s); return -1; }
    ACLCHECK(aclrtRecordEvent(evE, stream));
    ACLCHECK(aclrtSynchronizeStream(stream));
    float kms = 0.0f; ACLCHECK(aclrtEventElapsedTime(&kms, evS, evE));
    printf("[ok] dispatch_teardown executed\n");
    printf("[time] kernel = %.1f us (h=%d)\n", kms * 1000.0f, (int)h);

    std::vector<uint8_t> hEx(exB), hExSc(exScB);
    std::vector<int32_t> hSc(GE), hIdx(idsElems), hEtn(E);
    ACLCHECK(aclrtMemcpy(hEx.data(), exB, dEx, exB, ACL_MEMCPY_DEVICE_TO_HOST));
    ACLCHECK(aclrtMemcpy(hExSc.data(), exScB, dExSc, exScB, ACL_MEMCPY_DEVICE_TO_HOST));
    ACLCHECK(aclrtMemcpy(hSc.data(), GE * sizeof(int32_t), dSc, GE * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    ACLCHECK(aclrtMemcpy(hIdx.data(), idsElems * sizeof(int32_t), dIdx, idsElems * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    ACLCHECK(aclrtMemcpy(hEtn.data(), E * sizeof(int32_t), dEtn, E * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(dir + "/expand_x.npu.bin", hEx.data(), exB);
    WriteFile(dir + "/expand_scale.npu.bin", hExSc.data(), exScB);
    WriteFile(dir + "/send_counts.npu.bin", hSc.data(), GE * sizeof(int32_t));
    WriteFile(dir + "/expand_idx.npu.bin", hIdx.data(), idsElems * sizeof(int32_t));
    WriteFile(dir + "/expert_token_nums.npu.bin", hEtn.data(), E * sizeof(int32_t));
    printf("[ok] outputs written to %s\n", dir.c_str());

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
