#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn_bsa_select_block_mask.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// ===== Utility: Parse key=value text file =====
std::map<std::string, std::string> ParseParamsFile(const std::string& filepath) {
    std::map<std::string, std::string> params;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG_PRINT("Failed to open params file: %s\n", filepath.c_str());
        return params;
    }
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            // Trim whitespace
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t' || key.back() == '\r'))
                key.pop_back();
            while (!value.empty() && (value.back() == ' ' || value.back() == '\r'))
                value.pop_back();
            params[key] = value;
        }
    }
    return params;
}

std::vector<int64_t> ParseShape(const std::string& shapeStr) {
    std::vector<int64_t> shape;
    std::stringstream ss(shapeStr);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoll(item));
    }
    return shape;
}

// ===== Utility: Read/Write raw binary files =====
template <typename T>
std::vector<T> ReadBinFile(const std::string& filepath, int64_t expectedSize) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOG_PRINT("Failed to open bin file: %s\n", filepath.c_str());
        return {};
    }
    std::vector<T> data(expectedSize);
    file.read(reinterpret_cast<char*>(data.data()), expectedSize * sizeof(T));
    int64_t bytesRead = file.gcount();
    if (bytesRead != expectedSize * sizeof(T)) {
        LOG_PRINT("Warning: read %lld bytes, expected %lld from %s\n",
                  bytesRead, (int64_t)(expectedSize * sizeof(T)), filepath.c_str());
        data.resize(bytesRead / sizeof(T));
    }
    return data;
}

template <typename T>
void WriteBinFile(const std::string& filepath, const std::vector<T>& data) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOG_PRINT("Failed to write bin file: %s\n", filepath.c_str());
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

// ===== ACL Initialization =====
int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// ===== Create ACL Tensor from host data =====
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return -1);
    return 0;
}

// ===== Create ACL output tensor (device-only, zero-initialized) =====
int CreateAclOutTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                       aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclOutTensor: ERROR - shape is empty\n");
        return -1;
    }

    // Determine element size based on data type
    int64_t elemSize = 0;
    if (dataType == ACL_FLOAT16) elemSize = sizeof(op::fp16_t);
    else if (dataType == ACL_FLOAT) elemSize = sizeof(float);
    else if (dataType == ACL_INT8) elemSize = sizeof(int8_t);
    else if (dataType == ACL_INT32) elemSize = sizeof(int32_t);
    else if (dataType == ACL_INT64) elemSize = sizeof(int64_t);
    else {
        LOG_PRINT("CreateAclOutTensor: unsupported dataType\n");
        return -1;
    }

    auto totalSize = GetShapeSize(shape) * elemSize;

    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for output failed. ERROR: %d\n", ret); return ret);

    // Zero-initialize output memory
    ret = aclrtMemset(*deviceAddr, totalSize, 0, totalSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor for output failed\n"); return -1);
    return 0;
}

int main(int argc, char* argv[]) {
    LOG_PRINT("=== BSASelectBlockMask ACLNN Test (Bin-file mode) ===\n");

    // Parse command line: <input_dir> <output_dir> <device_id>
    std::string inputDir = ".";
    std::string outputDir = "npu";
    int32_t deviceId = 0;

    if (argc >= 2) inputDir = argv[1];
    if (argc >= 3) outputDir = argv[2];
    if (argc >= 4) deviceId = std::stoi(argv[3]);

    LOG_PRINT("  inputDir: %s\n", inputDir.c_str());
    LOG_PRINT("  outputDir: %s\n", outputDir.c_str());
    LOG_PRINT("  deviceId: %d\n", deviceId);

    // ===== Load parameters from input_params.txt =====
    auto params = ParseParamsFile(inputDir + "/input_params.txt");
    if (params.empty()) {
        LOG_PRINT("Failed to load params\n");
        return -1;
    }

    int64_t B = std::stoll(params["B"]);
    int64_t N = std::stoll(params["N"]);
    int64_t S1 = std::stoll(params["S1"]);
    int64_t S2 = std::stoll(params["S2"]);
    int64_t D = std::stoll(params["D"]);
    int64_t bx = std::stoll(params["bx"]);
    int64_t by = std::stoll(params["by"]);
    int64_t xBlocks = std::stoll(params["xBlocks"]);
    int64_t yBlocks = std::stoll(params["yBlocks"]);
    double sparsity = std::stod(params["sparsity"]);
    double scaleValue = std::stod(params["scaleValue"]);
    int64_t numKeyValueHeads = std::stoll(params["numKeyValueHeads"]);
    std::string layout = params["layout"];
    bool useSeqlen = std::stoll(params["use_seqlen"]) != 0;
    bool useQBlockLen = std::stoll(params["use_q_block_len"]) != 0;
    bool useKBlockLen = std::stoll(params["use_k_block_len"]) != 0;

    LOG_PRINT("Parameters: B=%ld N=%ld S1=%ld S2=%ld D=%ld bx=%ld by=%ld\n",
              B, N, S1, S2, D, bx, by);
    LOG_PRINT("  layout=%s sparsity=%.4f scale=%.6f\n", layout.c_str(), sparsity, scaleValue);
    LOG_PRINT("  use_seqlen=%d use_q_block_len=%d use_k_block_len=%d\n",
              useSeqlen, useQBlockLen, useKBlockLen);

    // ===== Initialize ACL =====
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // ===== Read input data from bin files =====
    auto queryShape = ParseShape(params["shape_query"]);
    auto keyShape = ParseShape(params["shape_key"]);

    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);

    LOG_PRINT("Reading input tensors...\n");

    // Read query (FP16)
    auto queryHostData = ReadBinFile<op::fp16_t>(inputDir + "/input_query.bin", querySize);
    CHECK_RET(!queryHostData.empty(), LOG_PRINT("Failed to read query data\n"); return -1);

    // Read key (FP16)
    auto keyHostData = ReadBinFile<op::fp16_t>(inputDir + "/input_key.bin", keySize);
    CHECK_RET(!keyHostData.empty(), LOG_PRINT("Failed to read key data\n"); return -1);

    // Read block shape (INT64)
    auto blockShapeData = ReadBinFile<int64_t>(inputDir + "/input_block_shape.bin", 2);
    CHECK_RET(!blockShapeData.empty(), LOG_PRINT("Failed to read block_shape data\n"); return -1);

    // ===== Create ACL input tensors =====
    LOG_PRINT("Creating ACL input tensors...\n");

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;

    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);

    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);

    // blockShape IntArray
    aclIntArray* blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape\n"); return -1);

    // ===== Create optional input arrays =====
    aclIntArray* seqlenQArray = nullptr;
    aclIntArray* seqlenKvArray = nullptr;
    aclIntArray* blockLenQArray = nullptr;
    aclIntArray* blockLenKArray = nullptr;

    if (useSeqlen) {
        int64_t seqlenQLen = std::stoll(params["seqlen_q_len"]);
        int64_t seqlenKvLen = std::stoll(params["seqlen_kv_len"]);

        auto seqlenQData = ReadBinFile<int32_t>(inputDir + "/input_seqlen_q.bin", seqlenQLen);
        std::vector<int64_t> seqlenQInt64(seqlenQData.begin(), seqlenQData.end());
        seqlenQArray = aclCreateIntArray(seqlenQInt64.data(), seqlenQInt64.size());
        CHECK_RET(seqlenQArray != nullptr, LOG_PRINT("Failed to create seqlenQ array\n"); return -1);

        auto seqlenKvData = ReadBinFile<int32_t>(inputDir + "/input_seqlen_kv.bin", seqlenKvLen);
        std::vector<int64_t> seqlenKvInt64(seqlenKvData.begin(), seqlenKvData.end());
        seqlenKvArray = aclCreateIntArray(seqlenKvInt64.data(), seqlenKvInt64.size());
        CHECK_RET(seqlenKvArray != nullptr, LOG_PRINT("Failed to create seqlenKv array\n"); return -1);
    }

    if (useQBlockLen) {
        int64_t blockLenQLen = std::stoll(params["block_len_q_len"]);
        auto blockLenQData = ReadBinFile<int32_t>(inputDir + "/input_block_len_q.bin", blockLenQLen);
        std::vector<int64_t> blockLenQInt64(blockLenQData.begin(), blockLenQData.end());
        blockLenQArray = aclCreateIntArray(blockLenQInt64.data(), blockLenQInt64.size());
        CHECK_RET(blockLenQArray != nullptr, LOG_PRINT("Failed to create blockLenQ array\n"); return -1);
    }

    if (useKBlockLen) {
        int64_t blockLenKLen = std::stoll(params["block_len_k_len"]);
        auto blockLenKData = ReadBinFile<int32_t>(inputDir + "/input_block_len_k.bin", blockLenKLen);
        std::vector<int64_t> blockLenKInt64(blockLenKData.begin(), blockLenKData.end());
        blockLenKArray = aclCreateIntArray(blockLenKInt64.data(), blockLenKInt64.size());
        CHECK_RET(blockLenKArray != nullptr, LOG_PRINT("Failed to create blockLenK array\n"); return -1);
    }

    // ===== Create output tensors =====
    LOG_PRINT("Creating output tensors...\n");

    // Output shapes (consistent with golden data format):
    //   blockSparseMaskOut: INT8,  [B, N, Xblocks, Yblocks]
    //   qCompOut:          FP32,  [B, N, Xblocks, D]
    //   kCompOut:          FP32,  [B, N, Yblocks, D]
    //   scoreOut:          FP32,  [B, N, Xblocks, Yblocks]
    //   softmaxOut:        FP16,  [B, N, Xblocks, Yblocks]
    std::vector<int64_t> maskOutShape = {B, N, xBlocks, yBlocks};
    std::vector<int64_t> qCompOutShape = {B, N, xBlocks, D};
    std::vector<int64_t> kCompOutShape = {B, N, yBlocks, D};
    std::vector<int64_t> scoreOutShape = {B, N, xBlocks, yBlocks};
    std::vector<int64_t> softmaxOutShape = {B, N, xBlocks, yBlocks};

    aclTensor* maskOutTensor = nullptr;
    aclTensor* qCompOutTensor = nullptr;
    aclTensor* kCompOutTensor = nullptr;
    aclTensor* scoreOutTensor = nullptr;
    aclTensor* softmaxOutTensor = nullptr;

    void* maskOutDeviceAddr = nullptr;
    void* qCompOutDeviceAddr = nullptr;
    void* kCompOutDeviceAddr = nullptr;
    void* scoreOutDeviceAddr = nullptr;
    void* softmaxOutDeviceAddr = nullptr;

    ret = CreateAclOutTensor(maskOutShape, &maskOutDeviceAddr, ACL_INT8, &maskOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create maskOut tensor\n"); return ret);

    ret = CreateAclOutTensor(qCompOutShape, &qCompOutDeviceAddr, ACL_FLOAT, &qCompOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create qCompOut tensor\n"); return ret);

    ret = CreateAclOutTensor(kCompOutShape, &kCompOutDeviceAddr, ACL_FLOAT, &kCompOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create kCompOut tensor\n"); return ret);

    ret = CreateAclOutTensor(scoreOutShape, &scoreOutDeviceAddr, ACL_FLOAT, &scoreOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create scoreOut tensor\n"); return ret);

    ret = CreateAclOutTensor(softmaxOutShape, &softmaxOutDeviceAddr, ACL_FLOAT16, &softmaxOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create softmaxOut tensor\n"); return ret);

    // ===== Layout strings =====
    char qLayoutBuffer[16];
    char kvLayoutBuffer[16];
    strncpy(qLayoutBuffer, layout.c_str(), 15);
    strncpy(kvLayoutBuffer, layout.c_str(), 15);
    qLayoutBuffer[15] = '\0';
    kvLayoutBuffer[15] = '\0';

    // ===== Call GetWorkspaceSize =====
    LOG_PRINT("Calling aclnnBSASelectBlockMaskGetWorkspaceSize...\n");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
        queryTensor,
        keyTensor,
        blockShape,
        nullptr,                // postBlockShapeOptional
        seqlenQArray,           // actualSeqLengthsOptional
        seqlenKvArray,          // actualSeqLengthsKvOptional
        blockLenQArray,         // actualBlockLenQueryOptional
        blockLenKArray,         // actualBlockLenKeyOptional
        qLayoutBuffer,
        kvLayoutBuffer,
        numKeyValueHeads,
        scaleValue,
        sparsity,
        maskOutTensor,          // blockSparseMaskOut
        qCompOutTensor,         // qCompOut
        kCompOutTensor,         // kCompOut
        scoreOutTensor,         // scoreOut
        softmaxOutTensor,       // softmaxOut
        &workspaceSize,
        &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMaskGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null\n"); return -1);

    LOG_PRINT("workspaceSize = %lu bytes (%.2f MB)\n", workspaceSize, workspaceSize / 1024.0 / 1024.0);

    // Allocate workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // ===== Execute =====
    LOG_PRINT("Calling aclnnBSASelectBlockMask...\n");

    ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMask failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Synchronizing stream...\n");
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // ===== Copy results to host and save =====
    LOG_PRINT("Copying results to host and saving...\n");

    // Create or clean output directory
    // If dir exists -> clear contents; if not exists -> create
    std::string prepareDirCmd = "if [ -d \"" + outputDir + "\" ]; then rm -rf " + outputDir + "/*; else mkdir -p " + outputDir + "; fi";
    system(prepareDirCmd.c_str());

    // --- mask ---
    int64_t maskOutSize = GetShapeSize(maskOutShape);
    std::vector<int8_t> maskResult(maskOutSize);
    ret = aclrtMemcpy(maskResult.data(), maskOutSize * sizeof(int8_t), maskOutDeviceAddr,
                      maskOutSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy maskOut failed\n"); return ret);
    WriteBinFile(outputDir + "/npu_mask.bin", maskResult);

    // --- q_comp ---
    int64_t qCompSize = GetShapeSize(qCompOutShape);
    std::vector<float> qCompResult(qCompSize);
    ret = aclrtMemcpy(qCompResult.data(), qCompSize * sizeof(float), qCompOutDeviceAddr,
                      qCompSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy qCompOut failed\n"); return ret);
    WriteBinFile(outputDir + "/npu_q_comp.bin", qCompResult);

    // --- k_comp ---
    int64_t kCompSize = GetShapeSize(kCompOutShape);
    std::vector<float> kCompResult(kCompSize);
    ret = aclrtMemcpy(kCompResult.data(), kCompSize * sizeof(float), kCompOutDeviceAddr,
                      kCompSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy kCompOut failed\n"); return ret);
    WriteBinFile(outputDir + "/npu_k_comp.bin", kCompResult);

    // --- score ---
    int64_t scoreSize = GetShapeSize(scoreOutShape);
    std::vector<float> scoreResult(scoreSize);
    ret = aclrtMemcpy(scoreResult.data(), scoreSize * sizeof(float), scoreOutDeviceAddr,
                      scoreSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scoreOut failed\n"); return ret);
    WriteBinFile(outputDir + "/npu_score.bin", scoreResult);

    // --- softmax (attn_score) ---
    int64_t softmaxSize = GetShapeSize(softmaxOutShape);
    std::vector<op::fp16_t> softmaxResult(softmaxSize);
    ret = aclrtMemcpy(softmaxResult.data(), softmaxSize * sizeof(op::fp16_t), softmaxOutDeviceAddr,
                      softmaxSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy softmaxOut failed\n"); return ret);
    WriteBinFile(outputDir + "/npu_attn_score.bin", softmaxResult);

    // --- Save NPU output metadata ---
    {
        std::ofstream metaFile(outputDir + "/npu_meta.txt");
        metaFile << "shape_mask=" << B << "," << N << "," << xBlocks << "," << yBlocks << "\n";
        metaFile << "shape_q_comp=" << B << "," << N << "," << xBlocks << "," << D << "\n";
        metaFile << "shape_k_comp=" << B << "," << N << "," << yBlocks << "," << D << "\n";
        metaFile << "shape_score=" << B << "," << N << "," << xBlocks << "," << yBlocks << "\n";
        metaFile << "shape_attn_score=" << B << "," << N << "," << xBlocks << "," << yBlocks << "\n";
        metaFile << "dtype_mask=INT8\n";
        metaFile << "dtype_q_comp=FP32\n";
        metaFile << "dtype_k_comp=FP32\n";
        metaFile << "dtype_score=FP32\n";
        metaFile << "dtype_attn_score=FP16\n";
    }

    LOG_PRINT("\n=== NPU Results ===\n");
    LOG_PRINT("  maskOut:      shape=[%ld,%ld,%ld,%ld]\n", B, N, xBlocks, yBlocks);
    LOG_PRINT("  qCompOut:     shape=[%ld,%ld,%ld,%ld]\n", B, N, xBlocks, D);
    LOG_PRINT("  kCompOut:     shape=[%ld,%ld,%ld,%ld]\n", B, N, yBlocks, D);
    LOG_PRINT("  scoreOut:     shape=[%ld,%ld,%ld,%ld]\n", B, N, xBlocks, yBlocks);
    LOG_PRINT("  softmaxOut:   shape=[%ld,%ld,%ld,%ld]\n", B, N, xBlocks, yBlocks);
    LOG_PRINT("  Output saved to: %s\n", outputDir.c_str());

    // ===== Cleanup =====
    if (queryTensor) aclDestroyTensor(queryTensor);
    if (keyTensor) aclDestroyTensor(keyTensor);
    if (maskOutTensor) aclDestroyTensor(maskOutTensor);
    if (qCompOutTensor) aclDestroyTensor(qCompOutTensor);
    if (kCompOutTensor) aclDestroyTensor(kCompOutTensor);
    if (scoreOutTensor) aclDestroyTensor(scoreOutTensor);
    if (softmaxOutTensor) aclDestroyTensor(softmaxOutTensor);

    if (blockShape) aclDestroyIntArray(blockShape);
    if (seqlenQArray) aclDestroyIntArray(seqlenQArray);
    if (seqlenKvArray) aclDestroyIntArray(seqlenKvArray);
    if (blockLenQArray) aclDestroyIntArray(blockLenQArray);
    if (blockLenKArray) aclDestroyIntArray(blockLenKArray);

    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (maskOutDeviceAddr) aclrtFree(maskOutDeviceAddr);
    if (qCompOutDeviceAddr) aclrtFree(qCompOutDeviceAddr);
    if (kCompOutDeviceAddr) aclrtFree(kCompOutDeviceAddr);
    if (scoreOutDeviceAddr) aclrtFree(scoreOutDeviceAddr);
    if (softmaxOutDeviceAddr) aclrtFree(softmaxOutDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n=== Test completed successfully! ===\n");
    return 0;
}