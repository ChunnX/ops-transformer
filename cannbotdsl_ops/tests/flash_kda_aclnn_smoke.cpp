#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "aclnn_flash_kda.h"

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

} // namespace

int main()
{
    if (!Check(aclInit(nullptr), "aclInit") || !Check(aclrtSetDevice(0), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        return EXIT_FAILURE;
    }

    constexpr int64_t batch = 1;
    constexpr int64_t heads = 3;
    constexpr int64_t sequence = 8192;
    constexpr int64_t headDim = 128;
    std::vector<TensorResource> resources(10);
    const std::vector<int64_t> activation{batch, heads, sequence, headDim};
    const std::vector<int64_t> betaShape{batch, heads, sequence, 1};
    const std::vector<int64_t> state{batch, heads, headDim, headDim};
    bool created =
        CreateTensor(activation, ACL_BF16, resources[0]) && CreateTensor(activation, ACL_BF16, resources[1]) &&
        CreateTensor(activation, ACL_BF16, resources[2]) && CreateTensor(betaShape, ACL_BF16, resources[3]) &&
        CreateTensor(activation, ACL_BF16, resources[4]) && CreateTensor({heads}, ACL_FLOAT, resources[5]) &&
        CreateTensor({heads, headDim}, ACL_FLOAT, resources[6]) && CreateTensor(state, ACL_FLOAT, resources[7]) &&
        CreateTensor(activation, ACL_BF16, resources[8]) && CreateTensor(state, ACL_FLOAT, resources[9]);
    created = created && FillBf16(resources[0], 0.1F, 1U) && FillBf16(resources[1], 1.0F, 2U) &&
              FillBf16(resources[2], 0.1F, 3U) && FillBf16(resources[3], 1.0F, 4U) &&
              FillBf16(resources[4], 1.0F, 5U) && FillFloat(resources[5], -0.25F, 0.75F, 6U) &&
              FillFloat(resources[6], 0.0F, 1.0F, 7U) && FillFloat(resources[7], 0.05F, 0.05F, 8U);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    constexpr aclnnStatus kAclnnSuccess = 0;
    constexpr aclnnStatus kAclnnFailure = -1;
    aclnnStatus status = kAclnnFailure;
    if (created) {
        status = aclnnFlashKdaGetWorkspaceSize(resources[0].tensor, resources[1].tensor, resources[2].tensor,
                                               resources[3].tensor, resources[4].tensor, resources[5].tensor,
                                               resources[6].tensor, resources[7].tensor, 0.0883883476, -5.0,
                                               resources[8].tensor, resources[9].tensor, &workspaceSize, &executor);
        std::printf("workspace_size=%llu get_workspace_status=%d\n", static_cast<unsigned long long>(workspaceSize),
                    status);
    }

    void *workspace = nullptr;
    if (status == kAclnnSuccess && workspaceSize != 0 &&
        !Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace aclrtMalloc")) {
        status = kAclnnFailure;
    }
    if (status == kAclnnSuccess) {
        status = aclnnFlashKda(workspace, workspaceSize, executor, stream);
        std::printf("execute_status=%d\n", status);
    }
    if (status == kAclnnSuccess && !Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        status = kAclnnFailure;
    }
    const bool outFinite = status == kAclnnSuccess && IsFiniteBf16(resources[8]);
    const bool stateFinite = status == kAclnnSuccess && IsFiniteFloat(resources[9]);
    std::printf("output_finite=%s final_state_finite=%s\n", outFinite ? "true" : "false",
                stateFinite ? "true" : "false");

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    for (auto &resource : resources) {
        DestroyTensor(resource);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return status == kAclnnSuccess && outFinite && stateFinite ? EXIT_SUCCESS : EXIT_FAILURE;
}
