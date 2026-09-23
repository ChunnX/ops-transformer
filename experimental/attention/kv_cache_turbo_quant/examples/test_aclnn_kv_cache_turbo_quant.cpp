/*
 * KvCacheTurboQuant aclnn 示例 / 自测驱动器。
 *
 * 从裸二进制读输入，调用 aclnn 算子（两段式接口），把四个输出写回裸二进制，并打印每轮计时
 * （一行一个样本，供 python 侧统计）。输入字节原样透传，因此可直接喂给任务包 golden.py
 * 做逐字节对拍。
 *
 * usage: test_aclnn_kv_cache_turbo_quant --kv kv.bin --rot rot.bin --qjl qjl.bin --out outdir
 *        --tokens T --heads H --bits B [--warmup W] [--repeat R] [--device D] [--label NAME]
 *
 * outputs written into outdir: quant_idx.bin, quant_qjl.bin, quant_norm.bin, quant_gamma.bin
 *
 * 构建与完整自测步骤见同目录 build_and_run.sh 与 ../tests/README.md。
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_kv_cache_turbo_quant.h"

namespace {

constexpr int64_t kHeadDim = 128;

struct Args {
    std::string kv;
    std::string rot;
    std::string qjl;
    std::string out;
    std::string label = "case";
    int64_t tokens = 0;
    int64_t heads = 0;
    int64_t bits = 3;
    int64_t warmup = 3;
    int64_t repeat = 10;
    int32_t device = 0;
};

bool ParseArgs(int argc, char **argv, Args *args)
{
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const char *value = (i + 1 < argc) ? argv[i + 1] : nullptr;
        auto need = [&](const char *name) {
            if (value == nullptr) {
                printf("missing value for %s\n", name);
            }
            return value != nullptr;
        };
        if (key == "--kv") {
            if (!need("--kv")) return false;
            args->kv = value;
            ++i;
        } else if (key == "--rot") {
            if (!need("--rot")) return false;
            args->rot = value;
            ++i;
        } else if (key == "--qjl") {
            if (!need("--qjl")) return false;
            args->qjl = value;
            ++i;
        } else if (key == "--out") {
            if (!need("--out")) return false;
            args->out = value;
            ++i;
        } else if (key == "--label") {
            if (!need("--label")) return false;
            args->label = value;
            ++i;
        } else if (key == "--tokens") {
            if (!need("--tokens")) return false;
            args->tokens = std::atoll(value);
            ++i;
        } else if (key == "--heads") {
            if (!need("--heads")) return false;
            args->heads = std::atoll(value);
            ++i;
        } else if (key == "--bits") {
            if (!need("--bits")) return false;
            args->bits = std::atoll(value);
            ++i;
        } else if (key == "--warmup") {
            if (!need("--warmup")) return false;
            args->warmup = std::atoll(value);
            ++i;
        } else if (key == "--repeat") {
            if (!need("--repeat")) return false;
            args->repeat = std::atoll(value);
            ++i;
        } else if (key == "--device") {
            if (!need("--device")) return false;
            args->device = static_cast<int32_t>(std::atoll(value));
            ++i;
        } else {
            printf("unknown argument: %s\n", key.c_str());
            return false;
        }
    }
    return args->tokens > 0 && args->heads > 0 && !args->kv.empty() && !args->rot.empty() &&
           !args->qjl.empty() && !args->out.empty();
}

bool ReadFile(const std::string &path, std::vector<uint8_t> *buf)
{
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        printf("cannot open %s\n", path.c_str());
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    const long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    buf->resize(static_cast<size_t>(size));
    const size_t read = std::fread(buf->data(), 1, buf->size(), fp);
    std::fclose(fp);
    return read == buf->size();
}

bool WriteFile(const std::string &path, const void *data, size_t size)
{
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        printf("cannot write %s\n", path.c_str());
        return false;
    }
    const size_t written = std::fwrite(data, 1, size, fp);
    std::fclose(fp);
    return written == size;
}

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t d : shape) {
        size *= d;
    }
    return size;
}

aclTensor *MakeTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), deviceAddr);
}

bool ToDevice(const std::vector<uint8_t> &host, size_t bytes, void **deviceAddr)
{
    if (aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        printf("aclrtMalloc(%zu) failed\n", bytes);
        return false;
    }
    if (!host.empty() &&
        aclrtMemcpy(*deviceAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        printf("aclrtMemcpy H2D failed\n");
        return false;
    }
    return true;
}

double Median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

} // namespace

int main(int argc, char **argv)
{
    Args args;
    if (!ParseArgs(argc, argv, &args)) {
        printf("usage: %s --kv kv.bin --rot rot.bin --qjl qjl.bin --out dir --tokens T --heads H --bits B\n",
               argv[0]);
        return 2;
    }
    const int64_t rows = args.tokens * args.heads;
    const int64_t idxBytes = args.bits * kHeadDim / 8;
    const int64_t qjlBytes = kHeadDim / 8;

    std::vector<uint8_t> kvHost;
    std::vector<uint8_t> rotHost;
    std::vector<uint8_t> qjlHost;
    if (!ReadFile(args.kv, &kvHost) || !ReadFile(args.rot, &rotHost) || !ReadFile(args.qjl, &qjlHost)) {
        return 3;
    }
    if (kvHost.size() != static_cast<size_t>(rows) * kHeadDim * sizeof(uint16_t)) {
        printf("kv file size mismatch: got %zu expect %zu\n", kvHost.size(),
               static_cast<size_t>(rows) * kHeadDim * sizeof(uint16_t));
        return 3;
    }

    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(args.device) != ACL_SUCCESS) {
        printf("acl init failed: %s\n", aclGetRecentErrMsg());
        return 4;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        printf("aclrtCreateStream failed: %s\n", aclGetRecentErrMsg());
        return 4;
    }

    const std::vector<int64_t> kvShape = {args.tokens, args.heads, kHeadDim};
    const std::vector<int64_t> matShape = {kHeadDim, kHeadDim};
    const std::vector<int64_t> idxShape = {args.tokens, args.heads, idxBytes};
    const std::vector<int64_t> qjlShape = {args.tokens, args.heads, qjlBytes};
    const std::vector<int64_t> scalarShape = {args.tokens, args.heads};

    void *kvDev = nullptr;
    void *rotDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlOutDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    const size_t idxBytesTotal = static_cast<size_t>(ShapeSize(idxShape));
    const size_t qjlBytesTotal = static_cast<size_t>(ShapeSize(qjlShape));
    const size_t scalarBytes = static_cast<size_t>(ShapeSize(scalarShape)) * sizeof(uint16_t);
    const std::vector<uint8_t> zeros(idxBytesTotal > qjlBytesTotal ? idxBytesTotal : qjlBytesTotal, 0);
    const std::vector<uint8_t> zerosScalar(scalarBytes, 0);
    bool ok = ToDevice(kvHost, kvHost.size(), &kvDev) && ToDevice(rotHost, rotHost.size(), &rotDev) &&
              ToDevice(qjlHost, qjlHost.size(), &qjlDev) && ToDevice(zeros, idxBytesTotal, &idxDev) &&
              ToDevice(zeros, qjlBytesTotal, &qjlOutDev) && ToDevice(zerosScalar, scalarBytes, &normDev) &&
              ToDevice(zerosScalar, scalarBytes, &gammaDev);
    if (!ok) {
        return 5;
    }

    aclTensor *kvTensor = MakeTensor(kvShape, ACL_BF16, kvDev);
    aclTensor *rotTensor = MakeTensor(matShape, ACL_FLOAT, rotDev);
    aclTensor *qjlTensor = MakeTensor(matShape, ACL_FLOAT, qjlDev);
    aclTensor *idxTensor = MakeTensor(idxShape, ACL_UINT8, idxDev);
    aclTensor *qjlOutTensor = MakeTensor(qjlShape, ACL_UINT8, qjlOutDev);
    aclTensor *normTensor = MakeTensor(scalarShape, ACL_BF16, normDev);
    aclTensor *gammaTensor = MakeTensor(scalarShape, ACL_BF16, gammaDev);

    std::vector<double> fullMs;
    std::vector<double> execMs;
    const int64_t total = args.warmup + args.repeat;
    bool failed = false;
    for (int64_t iter = 0; iter < total; ++iter) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        const aclnnStatus ret = aclnnKvCacheTurboQuantGetWorkspaceSize(
            kvTensor, rotTensor, qjlTensor, args.bits, idxTensor, qjlOutTensor, normTensor, gammaTensor,
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            printf("%s GetWorkspaceSize failed: %d %s\n", args.label.c_str(), ret, aclGetRecentErrMsg());
            failed = true;
            break;
        }
        void *workspace = nullptr;
        if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("%s workspace malloc failed\n", args.label.c_str());
            failed = true;
            break;
        }
        if (aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream) != ACL_SUCCESS ||
            aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            printf("%s execute failed: %s\n", args.label.c_str(), aclGetRecentErrMsg());
            failed = true;
            break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double full = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (iter >= args.warmup) {
            fullMs.push_back(full);
            execMs.push_back(full); // execute is the only thing measured; sync is included above
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
    }

    if (!failed) {
        // pull the outputs back and store them for the python comparison
        std::vector<uint8_t> idxHost(idxBytesTotal);
        std::vector<uint8_t> qjlHost(qjlBytesTotal);
        std::vector<uint8_t> normHost(scalarBytes);
        std::vector<uint8_t> gammaHost(scalarBytes);
        aclrtMemcpy(idxHost.data(), idxHost.size(), idxDev, idxHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(qjlHost.data(), qjlHost.size(), qjlOutDev, qjlHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(normHost.data(), normHost.size(), normDev, normHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(gammaHost.data(), gammaHost.size(), gammaDev, gammaHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = WriteFile(args.out + "/quant_idx.bin", idxHost.data(), idxHost.size()) &&
             WriteFile(args.out + "/quant_qjl.bin", qjlHost.data(), qjlHost.size()) &&
             WriteFile(args.out + "/quant_norm.bin", normHost.data(), normHost.size()) &&
             WriteFile(args.out + "/quant_gamma.bin", gammaHost.data(), gammaHost.size());
    }

    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    printf("label: %s tokens=%ld heads=%ld rows=%ld bits=%ld warmup=%ld repeat=%ld\n", args.label.c_str(),
           args.tokens, args.heads, rows, args.bits, args.warmup, args.repeat);
    if (!failed) {
        printf("latency_ms_full seq: ");
        for (double v : fullMs) {
            printf("%.4f ", v);
        }
        printf("\n");
        printf("latency_ms_median: %.4f\n", Median(fullMs));
        printf("latency_ms_min: %.4f\n", fullMs.empty() ? 0.0 : *std::min_element(fullMs.begin(), fullMs.end()));
    }

    aclDestroyTensor(kvTensor);
    aclDestroyTensor(rotTensor);
    aclDestroyTensor(qjlTensor);
    aclDestroyTensor(idxTensor);
    aclDestroyTensor(qjlOutTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(gammaTensor);
    aclrtFree(kvDev);
    aclrtFree(rotDev);
    aclrtFree(qjlDev);
    aclrtFree(idxDev);
    aclrtFree(qjlOutDev);
    aclrtFree(normDev);
    aclrtFree(gammaDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(args.device);
    aclFinalize();
    return failed ? 1 : 0;
}
