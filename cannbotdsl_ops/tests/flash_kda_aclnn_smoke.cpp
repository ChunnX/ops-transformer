#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "aclnn_flash_kda.h"
#include "flash_kda_variant_policy.h"

namespace {

struct TensorResource {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    void *device = nullptr;
    size_t bytes = 0;
    aclTensor *tensor = nullptr;
};

bool Check(aclError status, const char *expression)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %d\n", expression, status);
    return false;
}

size_t ElementSize(aclDataType dtype)
{
    return dtype == ACL_FLOAT ? sizeof(float) : sizeof(uint16_t);
}

bool CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, TensorResource &resource)
{
    resource.shape = shape;
    resource.strides.resize(shape.size());
    int64_t stride = 1;
    for (size_t index = shape.size(); index-- > 0;) {
        resource.strides[index] = stride;
        stride *= shape[index];
    }
    resource.bytes = static_cast<size_t>(stride) * ElementSize(dtype);
    if (!Check(aclrtMalloc(&resource.device, resource.bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ||
        !Check(aclrtMemset(resource.device, resource.bytes, 0, resource.bytes), "aclrtMemset")) {
        return false;
    }
    resource.tensor = aclCreateTensor(resource.shape.data(), resource.shape.size(), dtype, resource.strides.data(), 0,
                                      ACL_FORMAT_ND, resource.shape.data(), resource.shape.size(), resource.device);
    if (resource.tensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        return false;
    }
    return true;
}

uint32_t NextRandom(uint32_t &state)
{
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

float RandomSigned(uint32_t &state)
{
    constexpr float inverse = 1.0F / 16777216.0F;
    return static_cast<float>(NextRandom(state) >> 8U) * (2.0F * inverse) - 1.0F;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

bool FillBf16(TensorResource &resource, float scale, uint32_t seed)
{
    std::vector<uint16_t> host(resource.bytes / sizeof(uint16_t));
    for (auto &value : host) {
        value = FloatToBf16(RandomSigned(seed) * scale);
    }
    return Check(aclrtMemcpy(resource.device, resource.bytes, host.data(), resource.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "fill BF16 tensor");
}

bool FillNormalizedBf16(TensorResource &resource, size_t vectorSize, uint32_t seed)
{
    std::vector<uint16_t> host(resource.bytes / sizeof(uint16_t));
    std::vector<float> values(vectorSize);
    for (size_t offset = 0; offset < host.size(); offset += vectorSize) {
        float squaredNorm = 0.0F;
        for (auto &value : values) {
            value = RandomSigned(seed);
            squaredNorm += value * value;
        }
        const float inverseNorm = 1.0F / std::sqrt(squaredNorm);
        for (size_t index = 0; index < vectorSize; ++index) {
            host[offset + index] = FloatToBf16(values[index] * inverseNorm);
        }
    }
    return Check(aclrtMemcpy(resource.device, resource.bytes, host.data(),
                            resource.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "fill normalized BF16 tensor");
}

bool FillFloat(TensorResource &resource, float center, float scale, uint32_t seed)
{
    std::vector<float> host(resource.bytes / sizeof(float));
    for (auto &value : host) {
        value = center + RandomSigned(seed) * scale;
    }
    return Check(aclrtMemcpy(resource.device, resource.bytes, host.data(), resource.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                 "fill float tensor");
}

void DestroyTensor(TensorResource &resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.device != nullptr) {
        aclrtFree(resource.device);
    }
}

bool IsFiniteBf16(const TensorResource &resource)
{
    std::vector<uint16_t> host(resource.bytes / sizeof(uint16_t));
    if (!Check(aclrtMemcpy(host.data(), resource.bytes, resource.device, resource.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy")) {
        return false;
    }
    return std::all_of(host.begin(), host.end(), [](uint16_t value) { return (value & 0x7f80U) != 0x7f80U; });
}

bool IsFiniteFloat(const TensorResource &resource)
{
    std::vector<float> host(resource.bytes / sizeof(float));
    if (!Check(aclrtMemcpy(host.data(), resource.bytes, resource.device, resource.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy")) {
        return false;
    }
    return std::all_of(host.begin(), host.end(), [](float value) { return std::isfinite(value); });
}

bool DumpTensor(const TensorResource &resource, const std::string &path)
{
    std::vector<uint8_t> host(resource.bytes);
    if (!Check(aclrtMemcpy(host.data(), host.size(), resource.device, resource.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "dump aclrtMemcpy")) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::fprintf(stderr, "open dump file failed: %s\n", path.c_str());
        return false;
    }
    output.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(host.size()));
    return output.good();
}

constexpr int64_t kDefaultBatch = 1;
constexpr int64_t kDefaultHeads = 3;
constexpr int64_t kDefaultSequence = 8192;
constexpr int64_t kHeadDim = 128;
constexpr double kScale = 0.0883883476;
constexpr double kLowerBound = -1.0;
constexpr aclnnStatus kAclnnSuccess = 0;
constexpr aclnnStatus kAclnnFailure = -1;

struct ShapeConfig {
    int64_t batch = kDefaultBatch;
    int64_t heads = kDefaultHeads;
    int64_t sequence = kDefaultSequence;
};

bool IsSupportedShape(const ShapeConfig &shape)
{
    return cannbotdsl::generated::MatchFlashKdaVariantPolicy(
               shape.batch, shape.heads, shape.sequence) != nullptr;
}

struct Invocation {
    Invocation() : resources(10) {}

    std::vector<TensorResource> resources;
    void *workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclrtStream stream = nullptr;
    aclnnStatus status = kAclnnFailure;
};

bool CopyTensorToHost(const TensorResource &resource, std::vector<uint8_t> &host)
{
    host.resize(resource.bytes);
    return Check(aclrtMemcpy(host.data(), host.size(), resource.device, resource.bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                 "copy tensor to host");
}

bool DumpInvocationInputs(const Invocation &invocation, const std::string &dumpDir)
{
    const char *names[] = {
        "k.bf16", "v.bf16", "q.bf16", "beta.bf16",
        "g.bf16", "a_log.f32", "dt_bias.f32", "initial_state.f32",
    };
    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
        if (!DumpTensor(invocation.resources[index], dumpDir + "/" + names[index])) {
            return false;
        }
    }
    return true;
}

bool DumpInvocationMetadata(const ShapeConfig &shape, const std::string &dumpDir)
{
    std::ofstream output(dumpDir + "/metadata.txt");
    if (!output) {
        std::fprintf(stderr, "open dump metadata failed: %s\n", dumpDir.c_str());
        return false;
    }
    output << "schema=1\n"
           << "batch=" << shape.batch << "\n"
           << "heads=" << shape.heads << "\n"
           << "sequence=" << shape.sequence << "\n"
           << "head_dim=" << kHeadDim << "\n"
           << std::setprecision(17)
           << "scale=" << kScale << "\n"
           << "lower_bound=" << kLowerBound << "\n";
    return output.good();
}

bool DumpInvocationOutputs(const Invocation &invocation, const std::string &dumpDir)
{
    return DumpTensor(invocation.resources[8], dumpDir + "/out.bf16") &&
           DumpTensor(invocation.resources[9], dumpDir + "/final_state.f32");
}

bool PrepareInvocation(Invocation &invocation, const ShapeConfig &shape,
                       uint32_t seedOffset, size_t wave, size_t streamIndex,
                       bool checkNegative)
{
    if (!Check(aclrtCreateStream(&invocation.stream), "aclrtCreateStream")) {
        return false;
    }
    auto &resources = invocation.resources;
    const std::vector<int64_t> activation{shape.batch, shape.heads, shape.sequence, kHeadDim};
    const std::vector<int64_t> betaShape{shape.batch, shape.heads, shape.sequence, 1};
    const std::vector<int64_t> state{shape.batch, shape.heads, kHeadDim, kHeadDim};
    bool created =
        CreateTensor(activation, ACL_BF16, resources[0]) && CreateTensor(activation, ACL_BF16, resources[1]) &&
        CreateTensor(activation, ACL_BF16, resources[2]) && CreateTensor(betaShape, ACL_BF16, resources[3]) &&
        CreateTensor(activation, ACL_BF16, resources[4]) && CreateTensor({shape.heads}, ACL_FLOAT, resources[5]) &&
        CreateTensor({shape.heads, kHeadDim}, ACL_FLOAT, resources[6]) && CreateTensor(state, ACL_FLOAT, resources[7]) &&
        CreateTensor(activation, ACL_BF16, resources[8]) && CreateTensor(state, ACL_FLOAT, resources[9]);
    created = created && FillNormalizedBf16(resources[0], kHeadDim, seedOffset + 1U) &&
              FillBf16(resources[1], 1.0F, seedOffset + 2U) &&
              FillNormalizedBf16(resources[2], kHeadDim, seedOffset + 3U) &&
              FillBf16(resources[3], 1.0F, seedOffset + 4U) &&
              FillBf16(resources[4], 1.0F, seedOffset + 5U) &&
              FillFloat(resources[5], -0.25F, 0.75F, seedOffset + 6U) &&
              FillFloat(resources[6], 0.0F, 1.0F, seedOffset + 7U) &&
              FillFloat(resources[7], 0.05F, 0.05F, seedOffset + 8U);
    if (!created) {
        return false;
    }

    if (checkNegative) {
        auto expectRejected = [&](const char *label, const aclTensor *rejectedK,
                                  const aclTensor *rejectedV, double scale,
                                  double lowerBound) {
            uint64_t rejectedWorkspaceSize = 123U;
            aclOpExecutor *rejectedExecutor =
                reinterpret_cast<aclOpExecutor *>(static_cast<uintptr_t>(1U));
            const aclnnStatus rejectedStatus = aclnnFlashKdaGetWorkspaceSize(
                rejectedK, rejectedV, resources[2].tensor,
                resources[3].tensor, resources[4].tensor,
                resources[5].tensor, resources[6].tensor,
                resources[7].tensor, scale, lowerBound,
                resources[8].tensor, resources[9].tensor,
                &rejectedWorkspaceSize, &rejectedExecutor);
            std::printf(
                "negative_case=%s status=%d workspace_size=%llu executor=%p\n",
                label, rejectedStatus,
                static_cast<unsigned long long>(rejectedWorkspaceSize),
                static_cast<void *>(rejectedExecutor));
            return rejectedStatus != kAclnnSuccess &&
                   rejectedWorkspaceSize == 0U && rejectedExecutor == nullptr;
        };
        const bool rejectedScalars =
            expectRejected("lower_bound_below", resources[0].tensor,
                           resources[1].tensor, kScale, -5.01) &&
            expectRejected("lower_bound_above", resources[0].tensor,
                           resources[1].tensor, kScale, 0.01) &&
            expectRejected("lower_bound_nan", resources[0].tensor,
                           resources[1].tensor, kScale,
                           std::numeric_limits<double>::quiet_NaN()) &&
            expectRejected("scale_nan", resources[0].tensor,
                           resources[1].tensor,
                           std::numeric_limits<double>::quiet_NaN(),
                           kLowerBound) &&
            expectRejected("scale_float_overflow", resources[0].tensor,
                           resources[1].tensor,
                           std::numeric_limits<double>::max(), kLowerBound);
        const bool rejectedTensors =
            expectRejected("null_k", nullptr, resources[1].tensor,
                           kScale, kLowerBound) &&
            expectRejected("invalid_k_contract", resources[5].tensor,
                           resources[1].tensor, kScale, kLowerBound);
        const std::vector<int64_t> unsupportedShape{
            shape.batch * shape.heads, 1, shape.sequence, kHeadDim};
        const std::vector<int64_t> unsupportedStrides{
            shape.sequence * kHeadDim, shape.sequence * kHeadDim,
            kHeadDim, 1};
        aclTensor *unsupportedV = aclCreateTensor(
            unsupportedShape.data(), unsupportedShape.size(), ACL_BF16,
            unsupportedStrides.data(), 0, ACL_FORMAT_ND,
            unsupportedShape.data(), unsupportedShape.size(),
            resources[1].device);
        const bool rejectedUnsupported = unsupportedV != nullptr &&
            expectRejected("unsupported_variant", resources[0].tensor,
                           unsupportedV, kScale, kLowerBound);
        if (unsupportedV != nullptr) {
            aclDestroyTensor(unsupportedV);
        }
        uint64_t nullOutputWorkspaceSize = 123U;
        aclOpExecutor *nullOutputExecutor =
            reinterpret_cast<aclOpExecutor *>(static_cast<uintptr_t>(1U));
        const aclnnStatus nullWorkspaceOutputStatus =
            aclnnFlashKdaGetWorkspaceSize(
                resources[0].tensor, resources[1].tensor,
                resources[2].tensor, resources[3].tensor,
                resources[4].tensor, resources[5].tensor,
                resources[6].tensor, resources[7].tensor, kScale,
                kLowerBound, resources[8].tensor, resources[9].tensor,
                nullptr, &nullOutputExecutor);
        const aclnnStatus nullExecutorOutputStatus =
            aclnnFlashKdaGetWorkspaceSize(
                resources[0].tensor, resources[1].tensor,
                resources[2].tensor, resources[3].tensor,
                resources[4].tensor, resources[5].tensor,
                resources[6].tensor, resources[7].tensor, kScale,
                kLowerBound, resources[8].tensor, resources[9].tensor,
                &nullOutputWorkspaceSize, nullptr);
        std::printf(
            "negative_case=null_outputs workspace_status=%d executor_status=%d "
            "workspace_size=%llu executor=%p\n",
            nullWorkspaceOutputStatus, nullExecutorOutputStatus,
            static_cast<unsigned long long>(nullOutputWorkspaceSize),
            static_cast<void *>(nullOutputExecutor));
        if (!rejectedScalars || !rejectedTensors || !rejectedUnsupported ||
            nullWorkspaceOutputStatus == kAclnnSuccess ||
            nullExecutorOutputStatus == kAclnnSuccess ||
            nullOutputWorkspaceSize != 0U || nullOutputExecutor != nullptr) {
            return false;
        }
    }

    invocation.status = aclnnFlashKdaGetWorkspaceSize(
        resources[0].tensor, resources[1].tensor, resources[2].tensor,
        resources[3].tensor, resources[4].tensor, resources[5].tensor,
        resources[6].tensor, resources[7].tensor, kScale, kLowerBound,
        resources[8].tensor, resources[9].tensor, &invocation.workspaceSize,
        &invocation.executor);
    std::printf(
        "wave=%zu stream=%zu workspace_size=%llu get_workspace_status=%d\n",
        wave, streamIndex,
        static_cast<unsigned long long>(invocation.workspaceSize),
        invocation.status);
    if (invocation.status != kAclnnSuccess) {
        return false;
    }
    if (invocation.workspaceSize != 0 &&
        !Check(aclrtMalloc(&invocation.workspace, invocation.workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "workspace aclrtMalloc")) {
        invocation.status = kAclnnFailure;
        return false;
    }
    return true;
}

bool LaunchInvocation(Invocation &invocation, size_t wave, size_t streamIndex)
{
    invocation.status = aclnnFlashKda(
        invocation.workspace, invocation.workspaceSize, invocation.executor,
        invocation.stream);
    std::printf("wave=%zu stream=%zu execute_status=%d\n",
                wave, streamIndex, invocation.status);
    return invocation.status == kAclnnSuccess;
}

bool SynchronizeInvocation(Invocation &invocation)
{
    if (invocation.status != kAclnnSuccess ||
        !Check(aclrtSynchronizeStream(invocation.stream),
               "aclrtSynchronizeStream")) {
        invocation.status = kAclnnFailure;
        return false;
    }
    return true;
}

bool ValidateInvocation(Invocation &invocation, size_t wave, size_t streamIndex)
{
    const bool outFinite = IsFiniteBf16(invocation.resources[8]);
    const bool stateFinite = IsFiniteFloat(invocation.resources[9]);
    std::printf(
        "wave=%zu stream=%zu output_finite=%s final_state_finite=%s\n",
        wave, streamIndex, outFinite ? "true" : "false",
        stateFinite ? "true" : "false");
    return outFinite && stateFinite;
}

void DestroyInvocation(Invocation &invocation)
{
    if (invocation.workspace != nullptr) {
        aclrtFree(invocation.workspace);
    }
    for (auto &resource : invocation.resources) {
        DestroyTensor(resource);
    }
    if (invocation.stream != nullptr) {
        aclrtDestroyStream(invocation.stream);
    }
}

} // namespace

int main(int argc, char **argv)
{
    bool dump = false;
    bool stress = false;
    bool negative = false;
    int64_t requestedWaves = 0;
    std::string dumpDir;
    ShapeConfig shape;
    auto parsePositive = [](const char *text, int64_t &value) {
        char *end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(text, &end, 10);
        if (errno == ERANGE || end == text || *end != '\0' || parsed <= 0) {
            return false;
        }
        value = static_cast<int64_t>(parsed);
        return true;
    };
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--dump-dir" && index + 1 < argc) {
            dump = true;
            dumpDir = argv[++index];
        } else if (option == "--stress") {
            stress = true;
        } else if (option == "--negative") {
            negative = true;
        } else if (option == "--waves" && index + 1 < argc &&
                   parsePositive(argv[index + 1], requestedWaves)) {
            ++index;
        } else if (option == "--batch" && index + 1 < argc &&
                   parsePositive(argv[index + 1], shape.batch)) {
            ++index;
        } else if (option == "--heads" && index + 1 < argc &&
                   parsePositive(argv[index + 1], shape.heads)) {
            ++index;
        } else if (option == "--sequence" && index + 1 < argc &&
                   parsePositive(argv[index + 1], shape.sequence)) {
            ++index;
        } else {
            std::fprintf(
                stderr,
                "usage: %s [--dump-dir DIR] [--stress] [--negative] "
                "[--waves N] [--batch B] [--heads N] [--sequence S]\n",
                argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (!IsSupportedShape(shape)) {
        std::fprintf(stderr, "unsupported provider shape: B%lld/N%lld/S%lld\n",
                     static_cast<long long>(shape.batch),
                     static_cast<long long>(shape.heads),
                     static_cast<long long>(shape.sequence));
        return EXIT_FAILURE;
    }
    if (requestedWaves > 10000) {
        std::fprintf(stderr, "--waves must not exceed 10000\n");
        return EXIT_FAILURE;
    }

    std::printf("shape=B%lld/N%lld/S%lld\n",
                static_cast<long long>(shape.batch),
                static_cast<long long>(shape.heads),
                static_cast<long long>(shape.sequence));

    if (!Check(aclInit(nullptr), "aclInit") ||
        !Check(aclrtSetDevice(0), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }

    const size_t streamCount = stress ? 2U : 1U;
    const size_t waveCount = requestedWaves > 0
        ? static_cast<size_t>(requestedWaves)
        : (stress ? 2U : 1U);
    std::vector<std::vector<uint8_t>> outputBaseline(streamCount);
    std::vector<std::vector<uint8_t>> stateBaseline(streamCount);
    std::vector<double> waveElapsedMilliseconds;
    bool success = true;
    for (size_t wave = 0; wave < waveCount && success; ++wave) {
        std::vector<Invocation> invocations(streamCount);
        for (size_t streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
            const uint32_t seedOffset = static_cast<uint32_t>(streamIndex * 16U);
            if (!PrepareInvocation(invocations[streamIndex], shape, seedOffset,
                                   wave, streamIndex,
                                   negative && wave == 0U && streamIndex == 0U)) {
                success = false;
                break;
            }
        }
        if (success && dump && wave == 0U &&
            (!DumpInvocationMetadata(shape, dumpDir) ||
             !DumpInvocationInputs(invocations[0], dumpDir))) {
            success = false;
        }
        const auto waveStart = std::chrono::steady_clock::now();
        for (size_t streamIndex = 0;
             streamIndex < streamCount && success; ++streamIndex) {
            success = LaunchInvocation(
                invocations[streamIndex], wave, streamIndex);
        }
        for (size_t streamIndex = 0;
             streamIndex < streamCount && success; ++streamIndex) {
            success = SynchronizeInvocation(invocations[streamIndex]);
        }
        if (success) {
            const auto waveEnd = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double, std::milli>(
                waveEnd - waveStart).count();
            waveElapsedMilliseconds.push_back(elapsed);
            std::printf("wave=%zu launch_sync_ms=%.3f\n", wave, elapsed);
        }
        for (size_t streamIndex = 0;
             streamIndex < streamCount && success; ++streamIndex) {
            success = ValidateInvocation(
                invocations[streamIndex], wave, streamIndex);
        }
        for (size_t streamIndex = 0;
             streamIndex < streamCount && success; ++streamIndex) {
            std::vector<uint8_t> output;
            std::vector<uint8_t> state;
            success = CopyTensorToHost(
                          invocations[streamIndex].resources[8], output) &&
                      CopyTensorToHost(
                          invocations[streamIndex].resources[9], state);
            if (!success) {
                break;
            }
            if (wave == 0U) {
                outputBaseline[streamIndex] = std::move(output);
                stateBaseline[streamIndex] = std::move(state);
            } else if (output != outputBaseline[streamIndex] ||
                       state != stateBaseline[streamIndex]) {
                std::fprintf(
                    stderr,
                    "repeat mismatch for stream %zu on wave %zu\n",
                    streamIndex, wave);
                success = false;
            }
        }
        if (success && dump && wave == 0U &&
            !DumpInvocationOutputs(invocations[0], dumpDir)) {
            success = false;
        }
        for (auto &invocation : invocations) {
            DestroyInvocation(invocation);
        }
    }

    if (success && !waveElapsedMilliseconds.empty()) {
        const auto bounds = std::minmax_element(
            waveElapsedMilliseconds.begin(), waveElapsedMilliseconds.end());
        double total = 0.0;
        for (const double elapsed : waveElapsedMilliseconds) {
            total += elapsed;
        }
        std::printf(
            "timing streams=%zu waves=%zu launch_sync_ms_min=%.3f "
            "avg=%.3f max=%.3f\n",
            streamCount, waveElapsedMilliseconds.size(), *bounds.first,
            total / static_cast<double>(waveElapsedMilliseconds.size()),
            *bounds.second);
    }

    aclrtResetDevice(0);
    aclFinalize();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
