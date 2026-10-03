/**
 * \file sim_main.cpp
 * \brief SageAttention2 设备核 camodel 仿真 harness 宿主侧（bisheng --run-mode=sim 混编）。
 *
 * 启动约定（2026-10-03 探针实证，详见同目录 README.md）：
 * - bisheng --run-mode=sim -xasc 把 __global__ 设备核连同 fatbin 注册构造器编入
 *   宿主可执行，宿主侧以 chevron 语法 `kernel<<<blockDim, nullptr, stream>>>(args)`
 *   发射；lowering 落到 aclrtBinaryLoadFromData / aclrtBinaryGetFunctionByEntry /
 *   aclrtLaunchKernelWithHostArgs 链路，由 camodel 栈（libascendcl 桩 +
 *   libruntime_camodel.so 的 *Impl 符号）承接。
 * - KERNEL_TYPE_MIX_AIC_1_2 下 blockDim = AIC 任务数（core_num_used）；每个任务
 *   1 AIC + 2 AIV，AIV 侧 GetBlockIdx = 2*aic_blk + sub、GetSubBlockIdx = sub，
 *   与内核 Init 的 core_idx_ = GetBlockIdx()/2 约定一致。
 * - 运行环境：LD_LIBRARY_PATH 指向 cann-9.2.0-beta.1 camodel 栈目录（见
 *   run_sim_e2e.sh），CAMODEL_LOG_PATH 指向日志目录。
 *
 * 数据流：读 sim_reference.cpp 产出的输入/tiling bin → rtMalloc/rtMemcpy 上卡 →
 * chevron 发射 9 参（query,key,value,keyMean,lseCorrection,attentionOut,softmaxLse,
 * workspace,tiling，严格对齐 sageattention2.cpp 入口契约）→ 同步回读 → 写设备侧
 * O/LSE bin 供 compare_outputs.py 比对。
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "sim_case_common.hpp"

// 与 op_kernel/sageattention2.cpp 的 canonical 入口同名（小写下划线，C 链接）
extern "C" __global__ __aicore__ void sage_attention2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *keyMean, __gm__ uint8_t *lseCorrection,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling);

// ---- camodel runtime（libruntime_camodel.so）最小 C ABI 声明 ----
extern "C" {
int rtSetDevice(int32_t devId);
int rtMalloc(void **devPtr, uint64_t size, int memType, uint16_t moduleId);
int rtFree(void *devPtr);
int rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, int kind);
int rtMemset(void *devPtr, uint64_t count, int32_t value);
int rtStreamCreate(void **stream, int32_t priority);
int rtStreamSynchronize(void *stream);
int rtStreamDestroy(void *stream);
int rtDeviceReset(int32_t devId);
}

namespace {

constexpr int kMemcpyH2D = 1;  // RT_MEMCPY_HOST_TO_DEVICE
constexpr int kMemcpyD2H = 2;  // RT_MEMCPY_DEVICE_TO_HOST

struct DevBuffer {
    uint8_t *ptr = nullptr;
    uint64_t bytes = 0;
};

bool AllocAndUpload(DevBuffer &buf, const void *host, uint64_t bytes) {
    buf.bytes = bytes;
    if (rtMalloc(reinterpret_cast<void **>(&buf.ptr), bytes, 2, 0) != 0) {
        std::fprintf(stderr, "[FAIL] rtMalloc(%llu) 失败\n", static_cast<unsigned long long>(bytes));
        return false;
    }
    if (rtMemcpy(buf.ptr, bytes, host, bytes, kMemcpyH2D) != 0) {
        std::fprintf(stderr, "[FAIL] rtMemcpy H2D(%llu) 失败\n", static_cast<unsigned long long>(bytes));
        return false;
    }
    return true;
}

bool DownloadAndSave(const DevBuffer &buf, const std::string &path) {
    std::vector<uint8_t> host(buf.bytes);
    if (rtMemcpy(host.data(), buf.bytes, buf.ptr, buf.bytes, kMemcpyD2H) != 0) {
        std::fprintf(stderr, "[FAIL] rtMemcpy D2H(%s) 失败\n", path.c_str());
        return false;
    }
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const size_t wrote = std::fwrite(host.data(), 1, buf.bytes, f);
    std::fclose(f);
    return wrote == buf.bytes;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string dir = (argc > 1) ? argv[1] : std::string(".");
    const auto P = [&dir](const char *name) { return dir + "/" + name; };
    using namespace sage_attn_sim;

    // 1. 读参考侧产物（输入/tiling 与 golden 严格同源）
    std::vector<uint16_t> q_bits, k_bits, v_bits;
    std::vector<float> km;
    std::vector<uint8_t> tiling_bytes;
    if (!ReadBin<uint16_t>(P(kQBin), q_bits, static_cast<size_t>(kB) * kSq * kHq * kDOg) ||
        !ReadBin<uint16_t>(P(kKBin), k_bits, static_cast<size_t>(kB) * kSk * kHkv * kDOg) ||
        !ReadBin<uint16_t>(P(kVBin), v_bits, static_cast<size_t>(kB) * kSk * kHkv * kDOg) ||
        !ReadBin<float>(P(kKmBin), km, static_cast<size_t>(kB) * kHkv * 128u) ||
        !ReadBin<uint8_t>(P(kTilingBin), tiling_bytes, sizeof(sage_attn::SageAttention2TilingData))) {
        std::fprintf(stderr, "[FAIL] 读输入/tiling bin 失败（先跑 sim_reference）\n");
        return 1;
    }
    const auto *tiling = reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling_bytes.data());

    // 防漂移交叉验证：本端经同一 ComputeHostTiling 单点重算，与参考侧落盘的
    // 160 字节逐位比对（两侧 tiling 不同源即拒绝运行）。
    {
        sage_attn::SageAttention2TilingData tiling_check{};
        uint64_t ws_check = 0;
        if (!BuildTiling(tiling_check, ws_check) ||
            std::memcmp(&tiling_check, tiling_bytes.data(), sizeof(tiling_check)) != 0) {
            std::fprintf(stderr, "[FAIL] tiling bin 与本端重算不一致（单点被破坏？）\n");
            return 1;
        }
    }

    // workspace 台账与 op_host 同源：ComputeHostTiling 的 ws_bytes 已含 LSE 暂存与
    // STAGE_CORE_STRIDE × core_num_used 的 V0→V1 scale 交接条带
    sage_attn::SageAttention2TilingData tiling_re{};
    uint64_t ws_bytes = 0;
    if (!BuildTiling(tiling_re, ws_bytes)) {
        std::fprintf(stderr, "[FAIL] workspace 台账计算被拒\n");
        return 1;
    }

    // 2. 输出尺寸（O fp16 NHD、LSE fp32 [b*hq*sq]）
    const uint64_t o_bytes = static_cast<uint64_t>(kB) * kSq * kHq * kDOg * sizeof(uint16_t);
    const uint64_t lse_bytes = static_cast<uint64_t>(kB) * kHq * kSq * sizeof(float);

    if (rtSetDevice(0) != 0) {
        std::fprintf(stderr, "[FAIL] rtSetDevice 失败\n");
        return 1;
    }

    // 3. GM 分配与上传（lseCorrection 直读通道设备分支未消费，恒 nullptr；
    //    workspace 走 rtMemset 清零后仅驻卡，host 侧无需回读）
    DevBuffer q_d, k_d, v_d, km_d, o_d, lse_d, ws_d, tiling_d;
    void *stream = nullptr;
    bool ok = AllocAndUpload(q_d, q_bits.data(), q_bits.size() * sizeof(uint16_t)) &&
              AllocAndUpload(k_d, k_bits.data(), k_bits.size() * sizeof(uint16_t)) &&
              AllocAndUpload(v_d, v_bits.data(), v_bits.size() * sizeof(uint16_t)) &&
              AllocAndUpload(km_d, km.data(), km.size() * sizeof(float)) &&
              AllocAndUpload(tiling_d, tiling_bytes.data(), tiling_bytes.size());
    ok = ok && AllocAndUpload(o_d, q_bits.data(), o_bytes) &&  // O 先用垃圾占位，长度正确即可
         AllocAndUpload(lse_d, km.data(), lse_bytes);
    ok = ok && rtMalloc(reinterpret_cast<void **>(&ws_d.ptr), ws_bytes, 2, 0) == 0 &&
         rtMemset(ws_d.ptr, ws_bytes, 0) == 0;
    ok = ok && rtStreamCreate(&stream, 0) == 0;
    if (!ok) {
        std::fprintf(stderr, "[FAIL] GM 资源准备失败\n");
        return 1;
    }

    // 4. 发射：blockDim = core_num_used（MIX_AIC_1_2 的 AIC 任务数）
    const uint32_t blockDim = tiling->core_num_used;
    std::printf("[INFO] launch blockDim=%u ws_bytes=%llu\n", blockDim,
                static_cast<unsigned long long>(ws_bytes));
    std::fflush(stdout);
    sage_attention2<<<blockDim, nullptr, stream>>>(
        q_d.ptr, k_d.ptr, v_d.ptr, km_d.ptr,
        static_cast<__gm__ uint8_t *>(nullptr),   // lseCorrection：设备分支未消费
        o_d.ptr, lse_d.ptr, ws_d.ptr, tiling_d.ptr);
    if (rtStreamSynchronize(stream) != 0) {
        std::fprintf(stderr, "[FAIL] rtStreamSynchronize 失败（内核异常/死锁？查 CAMODEL_LOG_PATH）\n");
        return 1;
    }

    // 5. 回读设备输出
    if (!DownloadAndSave(o_d, P(kDeviceOBin)) || !DownloadAndSave(lse_d, P(kDeviceLseBin))) {
        std::fprintf(stderr, "[FAIL] 写设备侧输出 bin 失败\n");
        return 1;
    }
    std::printf("[PASS] 设备侧输出已落盘：%s / %s\n", kDeviceOBin, kDeviceLseBin);

    rtFree(q_d.ptr);
    rtFree(k_d.ptr);
    rtFree(v_d.ptr);
    rtFree(km_d.ptr);
    rtFree(o_d.ptr);
    rtFree(lse_d.ptr);
    rtFree(ws_d.ptr);
    rtFree(tiling_d.ptr);
    rtStreamDestroy(stream);
    rtDeviceReset(0);
    return 0;
}
