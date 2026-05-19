#ifndef RULE_TEST_CASES_H
#define RULE_TEST_CASES_H

#ifdef __cplusplus
extern "C" {
#endif

// rule_01: 采用统一的命名前缀
// 通过用例
void aclrtMalloc(void** devPtr, size_t size, int policy);
void acldvppCreateChannel(void* channelDesc);
void aclblasMatmul(void* matrixA, void* matrixB, void* matrixC);
void HcclGetRankSize(unsigned int* rankSize);
void HcommEndpointCreate(void* endpoint, void* endpointHandle);
void aclrtMemcpy2d(void* dst, const void* src, size_t pitch, size_t width, size_t height);
void aclrtLaunchKernelV2(void* kernel, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, void** args);

// 失败用例（未以合法前缀开头）
void Malloc(void** devPtr, size_t size);
void GetDevice(int* deviceId);
void CreateChannel(void* channelDesc);

// rule_02: 不能使用不常见的缩略语
// 通过用例
void aclrtHostRegisterV2(void* hostPtr, size_t size, unsigned int flags);
void aclrtMemsetD32Async(void* devPtr, int value, size_t num);
void aclrtCreateStreamV2(void* stream, unsigned int flags);
void aclrtMemcpyBatchAsyncV2(void** dst, const void** src, size_t* pitch, size_t count);
void aclrtDeviceGetP2PAtomicCapabilities(int* canAccessPeer, int* atomicAccessFlags);

// 失败用例（包含不常见缩写）
void aclrtDevGetBareTgid(int deviceId, unsigned int* tgid);
void aclrtGetV(int* value);

// rule_03: 非公开目录中的函数，不应使用导出宏
// 通过用例（无导出宏）
void aclrtInternalHelper(int mode);
void internalFunc(void);

// 失败用例（有导出宏但不在公开目录）
// 注意：这些函数如果放在非公开目录中会被检测为违规
// ACL_FUNC_VISIBILITY void aclrtExportedHelper(int mode);

// rule_04: 公开目录中的函数，必须使用合法导出宏
// 通过用例（有导出宏）
// ACL_FUNC_VISIBILITY aclError aclrtMalloc(void** devPtr, size_t size, int policy);
// ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void* dst, const void* src, size_t size, int kind);

// 失败用例（无导出宏但应该在公开目录）
void aclrtInternalFunc(void);
// aclError aclrtNoMacroFunc(void);

// rule_05: 可选参数必须有默认值
// 通过用例（有默认值）
void ReadOpModel(const char* path, void* config);

// 失败用例（参数被标记为可选但无默认值）
void AtraceStackcoreParse(const char* filePath, int len);

// rule_06: 参数数量不超过5个
// 通过用例
void aclrtMallocParams(void** devPtr, size_t size, int policy);
void aclrtCreateStreamParams(void* stream, int priority, int flags);

// 失败用例（超过5个参数）
void CreateComplexOp(int a, int b, int c, int d, int e, int f, int g);

// rule_21: 前缀+动词命名结构
// 通过用例
void aclrtBinarySetExceptionCallback(int callbackType, void* callbackFunc);
void aclrtCtxGetSysParamOpt(int paramId, void* value);
void aclrtEventElapsedTime(float* ms, void* start, void* end);
void aclrtHostUnregister(void* hostPtr);
void aclrtStreamGetId(unsigned int* streamId, void* stream);
void aclrtResetDeviceForce(int deviceId);
void HcommMemGrant(void* endpointHandle, void* grantInfo);
void HcclGetRankSizeFunc(unsigned int* rankSize);

// 失败用例
void aclrtInvalidFunc(void* param);
void abcInvalidFunc(void* param);

// rule_22: 禁止用Ex后缀
// 通过用例
void aclrtCreateStreamV2(void* stream, int priority);
void aclrtMemcpyAsyncV3(void* dst, const void* src, size_t count, int kind);
void aclnnAddLoraV2(void* op, void* loraParams);

// 失败用例
void aclrtOldFuncEx(int mode);
void aclnnAddLoraEx(void* op);

// rule_23: 首字母大写
// 通过用例
void aclrtMalloc(void** ptr, size_t size);
void aclrtCreateStream(void* stream);
void aclnnChannelShuffle(void* op);

// 失败用例
void aclnnAddlora(void* op);
void aclrtgetDevice(int* id);

// rule_24: 异步标识
// 通过用例
void aclrtMemsetAsync(void* devPtr, int value, size_t count);
void aclrtMemcpyAsync(void* dst, const void* src, size_t count, int kind);

// 失败用例
void aclrtMemset(void* devPtr, int value, size_t count);

#ifdef __cplusplus
}
#endif

#endif  // RULE_TEST_CASES_H