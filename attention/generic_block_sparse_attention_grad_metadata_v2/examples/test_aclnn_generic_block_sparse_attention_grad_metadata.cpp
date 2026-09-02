#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

static int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape)
        size *= dim;
    return size;
}

template <typename T>
static aclTensor *MakeTensor(const std::vector<T> &host, const std::vector<int64_t> &shape, aclDataType dtype,
                             void **device)
{
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(T);
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        return nullptr;
    if (aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
        return nullptr;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
        strides[i] = shape[i + 1] * strides[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), *device);
}

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
        return 1;
    // BNSD layout with conforming shapes:
    //   idx:   [B, N2, ceilDiv(maxKvSeqlen, blockShapeY), maxQSeqlen]
    //   count: [B, N2, ceilDiv(maxKvSeqlen, blockShapeY)]
    const int64_t batch = 1;
    const int64_t numQHeads = 8;
    const int64_t numKvHeads = 8;
    const int64_t headDim = 128;
    const int64_t maxQSeqlen = 256;
    const int64_t maxKvSeqlen = 256;
    const int64_t blockShapeX = 1;
    const int64_t blockShapeY = 256;
    const int64_t kvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
    void *idxDev = nullptr, *countDev = nullptr, *metaDev = nullptr;
    std::vector<int32_t> idxHost(static_cast<size_t>(batch * numKvHeads * kvBlockNum * maxQSeqlen), 0);
    std::vector<int32_t> countHost(static_cast<size_t>(batch * numKvHeads * kvBlockNum), 0);
    std::vector<int64_t> metadata(528, 0);
    aclTensor *idx = MakeTensor(idxHost, {batch, numKvHeads, kvBlockNum, maxQSeqlen}, ACL_INT32, &idxDev);
    aclTensor *count = MakeTensor(countHost, {batch, numKvHeads, kvBlockNum}, ACL_INT32, &countDev);
    aclTensor *meta = MakeTensor(metadata, {528}, ACL_INT64, &metaDev);
    int64_t blockShapeData[] = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char layoutQ[] = "BNSD";
    char layoutKv[] = "BNSD";
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, count, nullptr, nullptr, nullptr, nullptr, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim,
        blockShape, 1, layoutQ, layoutKv, 1, 0, -1, -1, meta, &workspaceSize, &executor);
    if (ret == ACL_SUCCESS)
        ret = aclnnGenericBlockSparseAttentionGradMetadata(nullptr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS)
        ret = aclrtSynchronizeStream(stream);
    std::printf("aclnnGenericBlockSparseAttentionGradMetadata: %d, workspace=%lu\n", ret, workspaceSize);
    aclDestroyIntArray(blockShape);
    aclDestroyTensor(idx);
    aclDestroyTensor(count);
    aclDestroyTensor(meta);
    aclrtFree(idxDev);
    aclrtFree(countDev);
    aclrtFree(metaDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return ret == ACL_SUCCESS ? 0 : 1;
}
