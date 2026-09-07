// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>
#include "aclnn_FlashKda.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(int result, const char* operation) {
    if (result != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
struct Tensor {
    aclTensor* tensor = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    Tensor(std::vector<int64_t> shape, aclDataType dtype, const std::string& input = "") {
        std::vector<int64_t> strides(shape.size());
        int64_t elements = 1;
        for (size_t i = shape.size(); i-- > 0;) { strides[i] = elements; elements *= shape[i]; }
        bytes = elements * (dtype == ACL_FLOAT ? 4 : 2);
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc tensor");
        if (!input.empty()) {
            std::vector<char> data(bytes);
            std::ifstream file(input, std::ios::binary);
            file.read(data.data(), bytes);
            if (file.gcount() != static_cast<std::streamsize>(bytes)) throw std::runtime_error("bad tensor file: " + input);
            Check(aclrtMemcpy(device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), device);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); if (device) aclrtFree(device); }
    void Dump(const std::string& path) const {
        std::vector<char> data(bytes);
        Check(aclrtMemcpy(data.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        std::ofstream file(path, std::ios::binary);
        file.write(data.data(), bytes);
        if (!file) throw std::runtime_error("write failed: " + path);
    }
};

int main(int argc, char** argv) {
    if (argc != 9) { std::cerr << "Usage: validate DIR B Nk Nv S DEVICE LOWER_BOUND SCALE\n"; return 2; }
    try {
        const std::string dir = argv[1];
        const int64_t b = std::stoll(argv[2]), nk = std::stoll(argv[3]), nv = std::stoll(argv[4]), s = std::stoll(argv[5]);
        const int device = std::stoi(argv[6]);
        const double lower_bound = std::stod(argv[7]), scale = std::stod(argv[8]);
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "aclrtSetDevice");
        Check(aclrtSetOpExecuteTimeOutWithMs(20000), "set kernel timeout");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        {
            Tensor q({b,nk,s,128}, ACL_BF16, dir + "/q.bin"), k({b,nk,s,128}, ACL_BF16, dir + "/k.bin");
            Tensor v({b,nv,s,128}, ACL_BF16, dir + "/v.bin"), g({b,nv,s,128}, ACL_BF16, dir + "/g.bin");
            Tensor beta({b,nv,s}, ACL_BF16, dir + "/beta.bin");
            Tensor initial({b,nv,128,128}, ACL_FLOAT, dir + "/initial_state.bin");
            Tensor a_log({nv}, ACL_FLOAT, dir + "/a_log.bin"), bias({nv,128}, ACL_FLOAT, dir + "/dt_bias.bin");
            Tensor output({b,nv,s,128}, ACL_BF16), state({b,nv,128,128}, ACL_FLOAT);
            for (int iteration = 0; iteration < 2; ++iteration) {
                uint64_t bytes = 0;
                aclOpExecutor* executor = nullptr;
                Check(aclnnFlashKdaGetWorkspaceSize(q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                    initial.tensor, a_log.tensor, bias.tensor, output.tensor, state.tensor,
                    lower_bound, scale, &bytes, &executor), "FlashKda GetWorkspaceSize");
                void* workspace = nullptr;
                if (bytes) Check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
                Check(aclnnFlashKda(workspace, bytes, executor, stream), "FlashKda Execute");
                Check(aclrtSynchronizeStream(stream), "synchronize");
                output.Dump(dir + "/output_" + std::to_string(iteration) + ".bin");
                state.Dump(dir + "/state_" + std::to_string(iteration) + ".bin");
                if (workspace) aclrtFree(workspace);
                std::cout << "ACLNN iteration " << iteration << " workspace=" << bytes << '\n';
            }
            initial.Dump(dir + "/initial_state_after.bin");
        }
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(device), "reset device");
        Check(aclFinalize(), "finalize");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
