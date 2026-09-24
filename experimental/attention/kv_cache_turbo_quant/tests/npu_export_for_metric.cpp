/**
 * 跑 NPU Encode，导出 got_{idx,qjl,norm,gamma} 供 §3.2 精度门禁。
 *
 * 用法：
 *   ./npu_export_for_metric /path/to/tests/data/npu_cases [case|all]
 * 写出到 <case>/npu_got/{idx,qjl,norm_bf16,gamma_bf16}.bin
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_kv_cache_turbo_quant.h"

#define CHECK_ACL(expr, msg)                                                                                           \
    do {                                                                                                               \
        auto _st = (expr);                                                                                             \
        if (_st != ACL_SUCCESS) {                                                                                      \
            std::cerr << "[FATAL] " << msg << " ret=" << _st << std::endl;                                             \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

constexpr int kHeadDim = 128;

struct CaseMeta {
    int64_t tokens{0};
    int64_t heads{0};
    int bits{3};
};

bool LoadMeta(const std::string &path, CaseMeta *m)
{
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("tokens=", 0) == 0) {
            m->tokens = std::stoll(line.substr(7));
        } else if (line.rfind("heads=", 0) == 0) {
            m->heads = std::stoll(line.substr(6));
        } else if (line.rfind("bits=", 0) == 0) {
            m->bits = std::stoi(line.substr(5));
        }
    }
    return m->tokens > 0 && m->heads > 0;
}

bool LoadBytes(const std::string &path, std::vector<uint8_t> *out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const auto n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    out->assign(n, 0);
    ifs.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(n));
    return true;
}

bool SaveBytes(const std::string &path, const void *data, size_t n)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n));
    return static_cast<bool>(ofs);
}

bool ExportOne(const std::string &caseDir, const std::string &name, aclrtStream stream)
{
    CaseMeta meta;
    if (!LoadMeta(caseDir + "/meta.txt", &meta)) {
        std::cerr << "FAIL " << name << " meta\n";
        return false;
    }
    const int64_t nVec = meta.tokens * meta.heads;
    const int64_t idxBytes = (kHeadDim * meta.bits + 7) / 8;
    const int64_t qjlBytes = (kHeadDim + 7) / 8;

    std::vector<uint8_t> kvHost;
    std::vector<uint8_t> rotHost;
    std::vector<uint8_t> qjlHost;
    if (!LoadBytes(caseDir + "/kv_bf16.bin", &kvHost) || !LoadBytes(caseDir + "/rotation_matrix.bin", &rotHost) ||
        !LoadBytes(caseDir + "/qjl_matrix.bin", &qjlHost)) {
        std::cerr << "FAIL " << name << " load\n";
        return false;
    }

    void *kvDev = nullptr;
    void *rotDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlOutDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    const size_t idxSz = static_cast<size_t>(nVec * idxBytes);
    const size_t qjlSz = static_cast<size_t>(nVec * qjlBytes);
    const size_t scSz = static_cast<size_t>(nVec) * 2;
    CHECK_ACL(aclrtMalloc(&kvDev, kvHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "kv");
    CHECK_ACL(aclrtMalloc(&rotDev, rotHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "rot");
    CHECK_ACL(aclrtMalloc(&qjlDev, qjlHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "qjl");
    CHECK_ACL(aclrtMalloc(&idxDev, idxSz, ACL_MEM_MALLOC_HUGE_FIRST), "idx");
    CHECK_ACL(aclrtMalloc(&qjlOutDev, qjlSz, ACL_MEM_MALLOC_HUGE_FIRST), "qjlOut");
    CHECK_ACL(aclrtMalloc(&normDev, scSz, ACL_MEM_MALLOC_HUGE_FIRST), "norm");
    CHECK_ACL(aclrtMalloc(&gammaDev, scSz, ACL_MEM_MALLOC_HUGE_FIRST), "gamma");

    CHECK_ACL(aclrtMemcpy(kvDev, kvHost.size(), kvHost.data(), kvHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    CHECK_ACL(aclrtMemcpy(rotDev, rotHost.size(), rotHost.data(), rotHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    CHECK_ACL(aclrtMemcpy(qjlDev, qjlHost.size(), qjlHost.data(), qjlHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");

    int64_t kvShape[3] = {meta.tokens, meta.heads, kHeadDim};
    int64_t matShape[2] = {kHeadDim, kHeadDim};
    int64_t idxShape[3] = {meta.tokens, meta.heads, idxBytes};
    int64_t qjlShape[3] = {meta.tokens, meta.heads, qjlBytes};
    int64_t scShape[2] = {meta.tokens, meta.heads};

    aclTensor *kvT = aclCreateTensor(kvShape, 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, kvShape, 3, kvDev);
    aclTensor *rotT = aclCreateTensor(matShape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, matShape, 2, rotDev);
    aclTensor *qjlT = aclCreateTensor(matShape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, matShape, 2, qjlDev);
    aclTensor *idxT = aclCreateTensor(idxShape, 3, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, idxShape, 3, idxDev);
    aclTensor *qjlOutT = aclCreateTensor(qjlShape, 3, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, qjlShape, 3, qjlOutDev);
    aclTensor *normT = aclCreateTensor(scShape, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, scShape, 2, normDev);
    aclTensor *gammaT = aclCreateTensor(scShape, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, scShape, 2, gammaDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnKvCacheTurboQuantGetWorkspaceSize(kvT, rotT, qjlT, meta.bits, idxT, qjlOutT, normT, gammaT,
                                                            &workspaceSize, &executor);
    if (st != ACL_SUCCESS) {
        std::cerr << "FAIL " << name << " GetWS " << st << "\n";
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "ws");
    }
    st = aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
    if (st != ACL_SUCCESS) {
        std::cerr << "FAIL " << name << " aclnn " << st << "\n";
        return false;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream), "sync");

    std::vector<uint8_t> gotIdx(idxSz);
    std::vector<uint8_t> gotQjl(qjlSz);
    std::vector<uint8_t> gotNorm(scSz);
    std::vector<uint8_t> gotGamma(scSz);
    CHECK_ACL(aclrtMemcpy(gotIdx.data(), idxSz, idxDev, idxSz, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    CHECK_ACL(aclrtMemcpy(gotQjl.data(), qjlSz, qjlOutDev, qjlSz, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    CHECK_ACL(aclrtMemcpy(gotNorm.data(), scSz, normDev, scSz, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    CHECK_ACL(aclrtMemcpy(gotGamma.data(), scSz, gammaDev, scSz, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");

    const std::string outDir = caseDir + "/npu_got";
    // mkdir via system — portable enough in this env
    const std::string mk = "mkdir -p \"" + outDir + "\"";
    if (std::system(mk.c_str()) != 0) {
        std::cerr << "FAIL " << name << " mkdir\n";
        return false;
    }
    if (!SaveBytes(outDir + "/idx.bin", gotIdx.data(), gotIdx.size()) ||
        !SaveBytes(outDir + "/qjl.bin", gotQjl.data(), gotQjl.size()) ||
        !SaveBytes(outDir + "/norm_bf16.bin", gotNorm.data(), gotNorm.size()) ||
        !SaveBytes(outDir + "/gamma_bf16.bin", gotGamma.data(), gotGamma.size())) {
        std::cerr << "FAIL " << name << " write\n";
        return false;
    }
    std::cout << "PASS export " << name << " -> " << outDir << "\n";

    aclDestroyTensor(kvT);
    aclDestroyTensor(rotT);
    aclDestroyTensor(qjlT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(qjlOutT);
    aclDestroyTensor(normT);
    aclDestroyTensor(gammaT);
    aclrtFree(kvDev);
    aclrtFree(rotDev);
    aclrtFree(qjlDev);
    aclrtFree(idxDev);
    aclrtFree(qjlOutDev);
    aclrtFree(normDev);
    aclrtFree(gammaDev);
    if (workspace) {
        aclrtFree(workspace);
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    std::string root = "data/npu_cases";
    std::string mode = "all";
    if (argc >= 2) {
        root = argv[1];
    }
    if (argc >= 3) {
        mode = argv[2];
    }
    std::vector<std::string> cases;
    if (mode == "all") {
        cases = {"gqa_decode_b1", "gqa_decode_b64", "gqa_prefill_t2048", "gqa_mse2_t2048", "gqa_mse4_t2048"};
    } else {
        cases = {mode};
    }

    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        return 1;
    }
    int fail = 0;
    for (const auto &name : cases) {
        if (!ExportOne(root + "/" + name, name, stream)) {
            ++fail;
        }
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    if (fail) {
        std::cout << "ALL FAIL export fail=" << fail << "\n";
        return 1;
    }
    std::cout << "ALL PASS npu_export_for_metric\n";
    return 0;
}
