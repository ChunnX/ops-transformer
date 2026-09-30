/*
 * KvCacheTurboQuant aclnn 示例 / 自测驱动器。
 *
 * 从裸二进制读输入，调用 aclnn 算子（两段式接口），把四个输出写回裸二进制，并打印每轮计时
 * （一行一个样本，供 python 侧统计）。输入字节原样透传，因此可直接喂给任务包 golden.py
 * 做逐字节对拍。
 *
 * usage: test_aclnn_kv_cache_turbo_quant --kv kv.bin --rot rot.bin --qjl qjl.bin --out outdir
 *        --tokens T --heads H --bits B [--warmup W] [--repeat R] [--device D] [--label NAME]
 *        [--matrix-dtype fp32|bf16] [--meta-dtype bf16|fp16]
 *
 * --matrix-dtype bf16 / --meta-dtype fp16 对应任务书 §2.3 允许的接口扩展：矩阵按 bf16 传入
 * （驱动器按 RNE 转换成 bf16 并把实际使用的矩阵存为 rot_used.bin / qjl_used.bin），
 * metadata 输出为 fp16。默认值（fp32 / bf16）即正式五用例的口径。
 *
 * outputs written into outdir: quant_idx.bin, quant_qjl.bin, quant_norm.bin, quant_gamma.bin
 *                             (+ rot_used.bin / qjl_used.bin)
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
    std::string neg;  // 非法输入自测模式（见 tests/invalid_cases.py）；非空时期望被拒绝
    std::string matrixDtype = "fp32";  // fp32（正式口径）| bf16（接口扩展）
    std::string metaDtype = "bf16";    // bf16（正式口径）| fp16（接口扩展）
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
        } else if (key == "--neg") {
            if (!need("--neg")) return false;
            args->neg = value;
            ++i;
        } else if (key == "--matrix-dtype") {
            if (!need("--matrix-dtype")) return false;
            args->matrixDtype = value;
            ++i;
        } else if (key == "--meta-dtype") {
            if (!need("--meta-dtype")) return false;
            args->metaDtype = value;
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
           !args->qjl.empty() && !args->out.empty() &&
           (args->matrixDtype == "fp32" || args->matrixDtype == "bf16") &&
           (args->metaDtype == "bf16" || args->metaDtype == "fp16");
}

// fp32 -> bf16，舍入到最近的偶数（与 numpy / ml_dtypes 的 bfloat16 转换一致）。
// bf16 在 fp32 中可精确表示，所以参考实现（golden.py 的 _npu_fp32）读 bf16 矩阵时得到的就是这些值。
std::vector<uint8_t> ToBf16(const std::vector<uint8_t> &fp32Bytes)
{
    const size_t count = fp32Bytes.size() / sizeof(float);
    const float *src = reinterpret_cast<const float *>(fp32Bytes.data());
    std::vector<uint8_t> out(count * sizeof(uint16_t));
    uint16_t *dst = reinterpret_cast<uint16_t *>(out.data());
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, src + i, sizeof(bits));
        const uint32_t lsb = (bits >> 16) & 1U;
        dst[i] = static_cast<uint16_t>((bits + 0x7FFFU + lsb) >> 16);
    }
    return out;
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
        printf("       [--matrix-dtype fp32|bf16] [--meta-dtype bf16|fp16]\n");
        return 2;
    }
    // ---- 非法输入自测（--neg <mode>）：只改属性/张量描述符，期望 GetWorkspaceSize 非 0 拒绝 ----
    if (args.neg == "mse-bits-5") {
        args.bits = 5;
    } else if (args.neg == "mse-bits-1") {
        args.bits = 1;
    } else if (!args.neg.empty() && args.neg != "kv-rank2" && args.neg != "head-dim-64" &&
               args.neg != "heads-2" && args.neg != "heads-64" && args.neg != "kv-fp32" &&
               args.neg != "rot-64" && args.neg != "rot-bf16" && args.neg != "qjl-bf16" &&
               args.neg != "empty") {
        printf("unknown --neg mode: %s\n", args.neg.c_str());
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
    // 接口扩展：矩阵按 bf16 传入（驱动器负责转换，并把实际使用的矩阵落盘供对拍）
    const bool matrixBf16 = (args.matrixDtype == "bf16");
    const bool metaFp16 = (args.metaDtype == "fp16");
    if (matrixBf16) {
        rotHost = ToBf16(rotHost);
        qjlHost = ToBf16(qjlHost);
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

    // 非法输入自测：改张量描述符/ dtype（缓冲区大小不变；负例不会进入执行阶段）
    std::vector<int64_t> kvShapeT = kvShape;
    std::vector<int64_t> matShapeT = matShape;
    std::vector<int64_t> qjlShapeT = matShape;
    aclDataType kvDtype = ACL_BF16;
    aclDataType rotDtype = matrixBf16 ? ACL_BF16 : ACL_FLOAT;
    aclDataType qjlDtype = matrixBf16 ? ACL_BF16 : ACL_FLOAT;
    if (args.neg == "kv-rank2") {
        kvShapeT = {args.tokens * args.heads, kHeadDim};
    } else if (args.neg == "head-dim-64") {
        kvShapeT = {args.tokens, args.heads, 64};
    } else if (args.neg == "heads-2") {
        kvShapeT = {args.tokens, 2, kHeadDim};
    } else if (args.neg == "heads-64") {
        kvShapeT = {args.tokens, 64, kHeadDim};
    } else if (args.neg == "empty") {
        kvShapeT = {0, args.heads, kHeadDim};
    } else if (args.neg == "kv-fp32") {
        kvDtype = ACL_FLOAT;
    } else if (args.neg == "rot-64") {
        matShapeT = {64, 64};
    } else if (args.neg == "rot-bf16") {
        rotDtype = ACL_BF16;
    } else if (args.neg == "qjl-bf16") {
        qjlDtype = ACL_BF16;
    }

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

    aclTensor *kvTensor = MakeTensor(kvShapeT, kvDtype, kvDev);
    aclTensor *rotTensor = MakeTensor(matShapeT, rotDtype, rotDev);
    aclTensor *qjlTensor = MakeTensor(qjlShapeT, qjlDtype, qjlDev);
    aclTensor *idxTensor = MakeTensor(idxShape, ACL_UINT8, idxDev);
    aclTensor *qjlOutTensor = MakeTensor(qjlShape, ACL_UINT8, qjlOutDev);
    const aclDataType metaType = metaFp16 ? ACL_FLOAT16 : ACL_BF16;
    aclTensor *normTensor = MakeTensor(scalarShape, metaType, normDev);
    aclTensor *gammaTensor = MakeTensor(scalarShape, metaType, gammaDev);

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
        if (!args.neg.empty()) {  // 非法输入自测：期望被拒绝，被接受则记为失败
            printf("%s NEG NOT REJECTED (mode=%s)\n", args.label.c_str(), args.neg.c_str());
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
        // 实际送进算子的矩阵（bf16 转换后 / fp32 原样），供对拍脚本使用
        // （注意：必须在下面重新声明 qjlHost 之前写，否则会被输出的同名局部变量遮蔽）
        ok = WriteFile(args.out + "/rot_used.bin", rotHost.data(), rotHost.size()) &&
             WriteFile(args.out + "/qjl_used.bin", qjlHost.data(), qjlHost.size());
        // pull the outputs back and store them for the python comparison
        std::vector<uint8_t> idxHost(idxBytesTotal);
        std::vector<uint8_t> qjlHost(qjlBytesTotal);
        std::vector<uint8_t> normHost(scalarBytes);
        std::vector<uint8_t> gammaHost(scalarBytes);
        aclrtMemcpy(idxHost.data(), idxHost.size(), idxDev, idxHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(qjlHost.data(), qjlHost.size(), qjlOutDev, qjlHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(normHost.data(), normHost.size(), normDev, normHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(gammaHost.data(), gammaHost.size(), gammaDev, gammaHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        ok = ok && WriteFile(args.out + "/quant_idx.bin", idxHost.data(), idxHost.size()) &&
             WriteFile(args.out + "/quant_qjl.bin", qjlHost.data(), qjlHost.size()) &&
             WriteFile(args.out + "/quant_norm.bin", normHost.data(), normHost.size()) &&
             WriteFile(args.out + "/quant_gamma.bin", gammaHost.data(), gammaHost.size());
        if (!ok) {
            printf("failed to write outputs into %s\n", args.out.c_str());
            failed = true;
        }
    }

    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    printf("label: %s tokens=%ld heads=%ld rows=%ld bits=%ld matrix_dtype=%s meta_dtype=%s warmup=%ld repeat=%ld\n",
           args.label.c_str(), args.tokens, args.heads, rows, args.bits, args.matrixDtype.c_str(),
           args.metaDtype.c_str(), args.warmup, args.repeat);
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
