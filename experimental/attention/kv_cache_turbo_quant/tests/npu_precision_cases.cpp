/**
 * NPU 五档 case 打包对拍：读 tests/data/npu_cases/<name>/，调 aclnn，比 idx/qjl 字节。
 *
 * 用法：
 *   source set_env.sh && source .../custom_transformer/bin/set_env.bash
 *   ./npu_precision_cases /path/to/tests/data/npu_cases
 */
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
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

bool BytesEq(const uint8_t *a, const uint8_t *b, size_t n)
{
    return std::memcmp(a, b, n) == 0;
}

bool RunOne(const std::string &caseDir, const std::string &name, aclrtStream stream)
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
    std::vector<uint8_t> expIdx;
    std::vector<uint8_t> expQjl;
    if (!LoadBytes(caseDir + "/kv_bf16.bin", &kvHost) || !LoadBytes(caseDir + "/rotation_matrix.bin", &rotHost) ||
        !LoadBytes(caseDir + "/qjl_matrix.bin", &qjlHost) || !LoadBytes(caseDir + "/expect_idx.bin", &expIdx) ||
        !LoadBytes(caseDir + "/expect_qjl.bin", &expQjl)) {
        std::cerr << "FAIL " << name << " load bins\n";
        return false;
    }
    if (kvHost.size() != static_cast<size_t>(nVec * kHeadDim * 2) ||
        expIdx.size() != static_cast<size_t>(nVec * idxBytes) ||
        expQjl.size() != static_cast<size_t>(nVec * qjlBytes)) {
        std::cerr << "FAIL " << name << " size mismatch\n";
        return false;
    }

    void *kvDev = nullptr;
    void *rotDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlOutDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    CHECK_ACL(aclrtMalloc(&kvDev, kvHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc kv");
    CHECK_ACL(aclrtMalloc(&rotDev, rotHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc rot");
    CHECK_ACL(aclrtMalloc(&qjlDev, qjlHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc qjl");
    CHECK_ACL(aclrtMalloc(&idxDev, expIdx.size(), ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc idx");
    CHECK_ACL(aclrtMalloc(&qjlOutDev, expQjl.size(), ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc qjlOut");
    CHECK_ACL(aclrtMalloc(&normDev, static_cast<size_t>(nVec) * 2, ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc norm");
    CHECK_ACL(aclrtMalloc(&gammaDev, static_cast<size_t>(nVec) * 2, ACL_MEM_MALLOC_HUGE_FIRST), name + " malloc gamma");

    std::vector<uint8_t> poison(expIdx.size(), 0xA5);
    CHECK_ACL(aclrtMemcpy(idxDev, poison.size(), poison.data(), poison.size(), ACL_MEMCPY_HOST_TO_DEVICE), "poison");
    CHECK_ACL(aclrtMemcpy(kvDev, kvHost.size(), kvHost.data(), kvHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D kv");
    CHECK_ACL(aclrtMemcpy(rotDev, rotHost.size(), rotHost.data(), rotHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D rot");
    CHECK_ACL(aclrtMemcpy(qjlDev, qjlHost.size(), qjlHost.data(), qjlHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D qjl");

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
        std::cerr << "FAIL " << name << " GetWorkspaceSize ret=" << st << "\n";
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "ws");
    }
    st = aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
    if (st != ACL_SUCCESS) {
        std::cerr << "FAIL " << name << " aclnn ret=" << st << "\n";
        return false;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream), "sync");

    std::vector<uint8_t> gotIdx(expIdx.size());
    std::vector<uint8_t> gotQjl(expQjl.size());
    CHECK_ACL(aclrtMemcpy(gotIdx.data(), gotIdx.size(), idxDev, gotIdx.size(), ACL_MEMCPY_DEVICE_TO_HOST), "D2H idx");
    CHECK_ACL(aclrtMemcpy(gotQjl.data(), gotQjl.size(), qjlOutDev, gotQjl.size(), ACL_MEMCPY_DEVICE_TO_HOST), "D2H qjl");

    const bool ok = BytesEq(gotIdx.data(), expIdx.data(), expIdx.size()) &&
                    BytesEq(gotQjl.data(), expQjl.data(), expQjl.size());
    if (ok) {
        std::cout << "PASS " << name << " idx/qjl exact T=" << meta.tokens << " H=" << meta.heads
                  << " bits=" << meta.bits << "\n";
    } else {
        std::cout << "FAIL " << name << " idx/qjl mismatch\n";
        size_t poison = 0, mism = 0, first = SIZE_MAX;
        for (size_t i = 0; i < gotIdx.size(); ++i) {
            if (gotIdx[i] == 0xA5) ++poison;
            if (gotIdx[i] != expIdx[i]) {
                ++mism;
                if (first == SIZE_MAX) first = i;
            }
        }
        std::cout << "  idx mism=" << mism << "/" << gotIdx.size() << " poison_bytes=" << poison;
        if (first != SIZE_MAX) {
            std::cout << " first@" << first << " got=" << int(gotIdx[first]) << " exp=" << int(expIdx[first]);
        }
        std::cout << "\n";
        size_t qpoison=0,qmism=0,qfirst=SIZE_MAX;
        for (size_t i = 0; i < gotQjl.size(); ++i) {
            if (gotQjl[i] == 0xA5) ++qpoison;
            if (gotQjl[i] != expQjl[i]) { ++qmism; if (qfirst==SIZE_MAX) qfirst=i; }
        }
        std::cout << "  qjl mism=" << qmism << "/" << gotQjl.size() << " poison_bytes=" << qpoison;
        if (qfirst != SIZE_MAX) {
            std::cout << " first@" << qfirst << " got=" << int(gotQjl[qfirst]) << " exp=" << int(expQjl[qfirst]);
        }
        std::cout << "\n";
    }

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
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    std::string root = ".";
    if (argc >= 2) {
        root = argv[1];
    }
    // 默认：bisect + decode；传 all 再加三档 prefill（标量 Kernel 可能很慢）
    // 传 decode 仅两档 decode；传 bisect 仅 smoke/t1h2/t2h1/t1h8
    std::string mode = (argc >= 3) ? argv[2] : "default";
    std::vector<std::string> cases;
    if (mode == "all") {
        cases = {"smoke_1x1", "t1h2", "t2h1", "t1h8", "gqa_decode_b1", "gqa_decode_b64",
                 "gqa_prefill_t2048", "gqa_mse2_t2048", "gqa_mse4_t2048"};
    } else if (mode == "decode") {
        cases = {"gqa_decode_b1", "gqa_decode_b64"};
    } else if (mode == "bisect") {
        cases = {"smoke_1x1", "t1h2", "t2h1", "t1h8"};
    } else if (mode == "default") {
        cases = {"smoke_1x1", "t1h2", "t2h1", "t1h8", "gqa_decode_b1", "gqa_decode_b64"};
    } else {
        // 任意单 case 名：便于定位
        cases = {mode};
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit fail\n";
        return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "setDevice fail\n";
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "createStream fail\n";
        return 1;
    }

    int fail = 0;
    for (const auto &name : cases) {
        if (!RunOne(root + "/" + name, name, stream)) {
            ++fail;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (fail) {
        std::cout << "ALL FAIL npu_precision_cases fail=" << fail << "\n";
        return 1;
    }
    std::cout << "ALL PASS npu_precision_cases (" << cases.size() << " cases)\n";
    return 0;
}
