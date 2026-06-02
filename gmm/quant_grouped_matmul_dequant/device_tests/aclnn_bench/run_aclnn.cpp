// Standalone aclnn benchmark for QuantGroupedMatmulDequant.
//
// Reads raw .bin tensors produced by device_tests/generate_data.py and
// dispatches aclnnQuantGroupedMatmulDequant directly (no torch_npu). Each
// torch_npu run pays ~23s of Python+PyTorch init for a few-µs kernel;
// this harness reuses a single ACL context across N timed iterations.
//
// Usage:
//   run_aclnn --datadir DIR --variant {dynamic|dynamic_u64|dynamic_smooth}
//             [--iterations N] [--warmup W] [--device-id D]
//             [--output-json F]
//
// Adapted from QBMv3 device_tests/aclnn_bench/run_aclnn.cpp (2026-05-21).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_quant_grouped_matmul_dequant.h"
#include "aclnn_quant_grouped_matmul_dequant_weight_nz.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif


// --------------------------------------------------------------------
// CHECK macros
// --------------------------------------------------------------------
#define CHECK_ACL(expr)                                                   \
    do {                                                                  \
        auto _ret = (expr);                                               \
        if (_ret != ACL_SUCCESS) {                                        \
            fprintf(stderr, "[ACL FAIL] %s = %d at %s:%d\n",              \
                    #expr, _ret, __FILE__, __LINE__);                     \
            return _ret;                                                  \
        }                                                                 \
    } while (0)

#define CHECK_ACLNN(expr)                                                 \
    do {                                                                  \
        auto _ret = (expr);                                               \
        if (_ret != ACLNN_SUCCESS) {                                      \
            fprintf(stderr, "[ACLNN FAIL] %s = %d at %s:%d\n",            \
                    #expr, _ret, __FILE__, __LINE__);                     \
            return _ret;                                                  \
        }                                                                 \
    } while (0)

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "[FAIL] %s at %s:%d\n", (msg),                \
                    __FILE__, __LINE__);                                  \
            return 1;                                                     \
        }                                                                 \
    } while (0)

// --------------------------------------------------------------------
// Read raw .bin file
// --------------------------------------------------------------------
static int read_bin(const std::string &path, std::vector<uint8_t> &buf) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[FAIL] cannot open %s\n", path.c_str());
        return 1;
    }
    auto sz = f.tellg();
    f.seekg(0);
    buf.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return 0;
}

// --------------------------------------------------------------------
// Parse config_shared.txt for shape (E, M, K, N)
// --------------------------------------------------------------------
struct ShapeCfg {
    int64_t E = 0, M = 0, K = 0, N = 0;
};

static std::map<std::string, std::string> parse_kv(const std::string &path) {
    std::map<std::string, std::string> m;
    std::ifstream f(path);
    std::string token;
    while (f >> token) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        m[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return m;
}

static int load_cfg(const std::string &datadir, ShapeCfg &cfg) {
    // qgmmd config.txt format:
    //   E=2 M=32 K=128 N=128 seed=42
    //   dynamic_quant=True smooth_on=False
    //   active_experts=2
    auto kv = parse_kv(datadir + "/config.txt");
    if (kv.empty()) {
        fprintf(stderr, "[FAIL] cannot read %s/config.txt\n",
                datadir.c_str());
        return 1;
    }
    cfg.E = std::stoll(kv["E"]);
    cfg.M = std::stoll(kv["M"]);
    cfg.K = std::stoll(kv["K"]);
    cfg.N = std::stoll(kv["N"]);
    return 0;
}

// --------------------------------------------------------------------
// Allocate device buffer + copy host bytes; create aclTensor (ND).
// --------------------------------------------------------------------
struct DevTensor {
    aclTensor *t = nullptr;
    void *dev = nullptr;
    size_t bytes = 0;
    std::vector<int64_t> shape;
    aclDataType dtype = ACL_DT_UNDEFINED;
};

static int upload(const std::vector<uint8_t> &host_bytes,
                  const std::vector<int64_t> &shape,
                  aclDataType dtype, DevTensor &out) {
    size_t bytes = host_bytes.size();
    CHECK_ACL(aclrtMalloc(&out.dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(out.dev, bytes, host_bytes.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    out.t = aclCreateTensor(shape.data(), shape.size(), dtype,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), out.dev);
    if (out.t == nullptr) {
        fprintf(stderr, "[FAIL] aclCreateTensor returned nullptr\n");
        return 1;
    }
    out.bytes = bytes;
    out.shape = shape;
    out.dtype = dtype;
    return 0;
}

// Output tensor: device-only, no host copy needed initially.
static int alloc_output(const std::vector<int64_t> &shape,
                        aclDataType dtype, DevTensor &out) {
    size_t elem = 0;
    switch (dtype) {
    case ACL_FLOAT16: elem = 2; break;
    case ACL_FLOAT: elem = 4; break;
    case ACL_INT32: elem = 4; break;
    default: elem = 1; break;
    }
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    size_t bytes = static_cast<size_t>(numel) * elem;
    CHECK_ACL(aclrtMalloc(&out.dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    out.t = aclCreateTensor(shape.data(), shape.size(), dtype,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), out.dev);
    if (out.t == nullptr) {
        fprintf(stderr, "[FAIL] aclCreateTensor (out) returned nullptr\n");
        return 1;
    }
    out.bytes = bytes;
    out.shape = shape;
    out.dtype = dtype;
    return 0;
}

// --------------------------------------------------------------------
// Cosine similarity (host-side, fp16 result vs fp16 golden)
// --------------------------------------------------------------------
static float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 0x1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t b;
    if (e == 0) {
        if (m == 0) { b = s << 31; }
        else {
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff;
            e++;
            b = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 0x1f) {
        b = (s << 31) | 0x7f800000 | (m << 13);
    } else {
        b = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static double cos_sim(const uint16_t *a, const uint16_t *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float x = fp16_to_fp32(a[i]);
        float y = fp16_to_fp32(b[i]);
        dot += (double)x * y;
        na += (double)x * x;
        nb += (double)y * y;
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// --------------------------------------------------------------------
// main
// --------------------------------------------------------------------
int main(int argc, char **argv) {
    std::string datadir;
    std::string variant = "dynamic";  // dynamic | dynamic_u64 | dynamic_smooth
    int64_t iterations = 20;
    int64_t warmup = 3;
    int device_id = 0;
    std::string output_json;
    bool verify = true;

    int64_t arg_E = 0, arg_M = 0, arg_K = 0, arg_N = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--datadir" && i + 1 < argc) { datadir = argv[++i]; }
        else if (a == "--variant" && i + 1 < argc) { variant = argv[++i]; }
        else if (a == "--iterations" && i + 1 < argc) { iterations = std::stoll(argv[++i]); }
        else if (a == "--warmup" && i + 1 < argc) { warmup = std::stoll(argv[++i]); }
        else if (a == "--device-id" && i + 1 < argc) { device_id = std::stoi(argv[++i]); }
        else if (a == "--output-json" && i + 1 < argc) { output_json = argv[++i]; }
        else if (a == "--no-verify") { verify = false; }
        else if (a == "--E" && i + 1 < argc) { arg_E = std::stoll(argv[++i]); }
        else if (a == "--M" && i + 1 < argc) { arg_M = std::stoll(argv[++i]); }
        else if (a == "--K" && i + 1 < argc) { arg_K = std::stoll(argv[++i]); }
        else if (a == "--N" && i + 1 < argc) { arg_N = std::stoll(argv[++i]); }
        else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    CHECK(!datadir.empty(), "--datadir required");

    const bool use_u64 = (variant == "dynamic_u64");
    const bool use_smooth = (variant == "dynamic_smooth");

    // Shape priority: explicit --E/--M/--K/--N args override config.txt.
    // Allows callers (run_one.py) to derive shape from .pth metadata and
    // bypass data_stage.py's stripped config.txt (which only stores the
    // shape NAME, not the dims).
    ShapeCfg cfg;
    if (arg_E > 0 && arg_M > 0 && arg_K > 0 && arg_N > 0) {
        cfg.E = arg_E; cfg.M = arg_M; cfg.K = arg_K; cfg.N = arg_N;
    } else {
        CHECK(load_cfg(datadir, cfg) == 0, "config load failed (and no --E/--M/--K/--N override)");
    }
    const int64_t E = cfg.E, M = cfg.M, K = cfg.K, N = cfg.N;

    // Vendor discovery: caller must ensure ASCEND_CUSTOM_OPP_PATH points
    // at custom_transformer AND `opp/vendors/config.ini` is removed/doesn't
    // list a competing vendor (else the runtime loads the wrong vendor's
    // op_proto/op_master and dispatch fails with 361001).
    // See aclnn_bench/README setup notes.

    // Signal to qgmmd tiling that we're uploading COMPACT blocked-ZN bytes
    // (E*K*N total, last block has actualBlockNG < blockN1) rather than the
    // PADDED layout that aclnn's ND→TransData entry produces. Tiling reads
    // this in CheckInOutShapes for ZN_BLOCKED to override fracN. See
    // tiling.cpp QGMMD_WEIGHT_COMPACT block.
    setenv("QGMMD_WEIGHT_COMPACT", "1", /*overwrite=*/1);

    // ---------------- Init ACL ----------------
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ---------------- Load .bin files ----------------
    std::vector<uint8_t> x_bytes, weight_bytes, scale_bytes, gl_bytes, smooth_bytes, xscale_bytes;
    CHECK(read_bin(datadir + "/x.bin", x_bytes) == 0, "x.bin");
    // UT passes x_scale ({m} fp32) even in pertoken mode — match that.
    (void)read_bin(datadir + "/x_scale.bin", xscale_bytes);  // optional
    // Use pre-converted FRACTAL_NZ weight (storage [E, K/32, N/16, 16, 32])
    // and dispatch via aclnnQuantGroupedMatmulDequantWeightNZ — skips the
    // ND→NZ TransData step that the ND entry runs internally.
    CHECK(read_bin(datadir + "/weight_int8_zn.bin", weight_bytes) == 0,
          "weight_int8_zn.bin");
    // qgmmd kernel expects fp32 weight_scale per the run_one.py workaround
    // ("expected DT_FLOAT but found DT_INT64" rejection on int64 path).
    // u64 variant uses scale.view(int32)[lower].view(fp32) — applied here too.
    std::string scale_name = use_u64 ? "weight_scale_u64" : "weight_scale";
    CHECK(read_bin(datadir + "/" + scale_name + ".bin", scale_bytes) == 0,
          "weight_scale.bin");
    CHECK(read_bin(datadir + "/group_list.bin", gl_bytes) == 0,
          "group_list.bin");
    if (use_smooth) {
        CHECK(read_bin(datadir + "/smooth_scale.bin", smooth_bytes) == 0,
              "smooth_scale.bin");
    }

    // ---------------- Upload tensors ----------------
    // x: fp16 [M, K] ND.
    DevTensor x, weight, scale, gl, smooth, out;
    CHECK(upload(x_bytes, {M, K}, ACL_FLOAT16, x) == 0, "x");

    // weight: FRACTAL_NZ pre-converted. Bytes copied AS IS (continuous
    // array, single memcpy of E*K*N bytes). generate_data's nd_to_zn_int8_
    // dispatch produces:
    //   N <= 65280: standard ZN layout (5-D shape {E, K/32, N/16, 16, 32}).
    //   N >  65280: COMPACT blocked ZN — total bytes = E*K*N (no padding),
    //               last block uses actualBlockNG = (N - (num_blocks-1)*
    //               65280) / 16 instead of BLOCKED_ZN_NG=4080. Kernel's
    //               PrefetchWeightToL1 BlockedZN branch handles this via
    //               actualBlockNG (utils.h:128-188); offset to block i is
    //               blockIdx * (K1 * blockN1 * 512) which lands on the
    //               compact data for blocks 0..num_blocks-2 (each full)
    //               and the partial last block starts at the same byte
    //               offset since prior blocks ARE full.
    // Storage-shape descriptor: just tells aclnn which tiling branch to
    // take + which weightBlockN1 to pass to the kernel.
    {
        const int64_t K_FRACTAL = 32;
        const int64_t N_FRACTAL = 16;
        const int64_t BLOCKED_ZN_NG = 4080;
        const int64_t BLOCKED_ZN_N_ELEMS = BLOCKED_ZN_NG * N_FRACTAL;  // 65280
        const int64_t K1 = (K + K_FRACTAL - 1) / K_FRACTAL;
        const size_t bytes = weight_bytes.size();
        CHECK_ACL(aclrtMalloc(&weight.dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(weight.dev, bytes, weight_bytes.data(), bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));

        std::vector<int64_t> shape;
        if (N <= BLOCKED_ZN_N_ELEMS) {
            const int64_t N1 = (N + N_FRACTAL - 1) / N_FRACTAL;
            shape = {E, K1, N1, N_FRACTAL, K_FRACTAL};
        } else {
            const int64_t num_blocks = (N + BLOCKED_ZN_N_ELEMS - 1) / BLOCKED_ZN_N_ELEMS;
            shape = {E, num_blocks, K1, BLOCKED_ZN_NG, N_FRACTAL, K_FRACTAL};
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        weight.t = aclCreateTensor(shape.data(), shape.size(), ACL_INT8,
                                   strides.data(), 0,
                                   aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                   shape.data(), shape.size(),
                                   weight.dev);
        CHECK(weight.t != nullptr, "weight aclCreateTensor (FRACTAL_NZ)");
        weight.bytes = bytes;
        weight.shape = shape;
        weight.dtype = ACL_INT8;
    }

    // weight_scale: fp32 [E, N]. For u64 variant, apply the host-side
    // lower-32-bit-view workaround to get fp32 bytes from int64 storage.
    if (use_u64) {
        // Original scale_bytes is int64 [E, N]. Lower 32 bits hold the
        // FP32 bit pattern (VDEQ16 encoding: uint64 = 0x4000_0000 << 32
        // | fp32_bits). Strip upper 32 bits.
        size_t en = static_cast<size_t>(E) * static_cast<size_t>(N);
        CHECK(scale_bytes.size() == en * 8, "weight_scale_u64 size mismatch");
        std::vector<uint8_t> fp32_bytes(en * 4);
        for (size_t i = 0; i < en; ++i) {
            std::memcpy(&fp32_bytes[i * 4], &scale_bytes[i * 8], 4);
        }
        CHECK(upload(fp32_bytes, {E, N}, ACL_FLOAT, scale) == 0,
              "weight_scale (u64→fp32 view)");
    } else {
        CHECK(upload(scale_bytes, {E, N}, ACL_FLOAT, scale) == 0,
              "weight_scale");
    }

    // group_list: int64 [E].
    CHECK(upload(gl_bytes, {E}, ACL_INT64, gl) == 0, "group_list");

    // smooth_scale: fp16 [K], optional.
    if (use_smooth) {
        CHECK(upload(smooth_bytes, {K}, ACL_FLOAT16, smooth) == 0,
              "smooth_scale");
    }

    // out: fp16 [M, N].
    CHECK(alloc_output({M, N}, ACL_FLOAT16, out) == 0, "out");

    // ---------------- Warmup + timed iterations ----------------
    aclrtEvent evt_start = nullptr, evt_end = nullptr;
    CHECK_ACL(aclrtCreateEvent(&evt_start));
    CHECK_ACL(aclrtCreateEvent(&evt_end));

    std::vector<float> times_us;
    times_us.reserve(iterations);

    char xQuantMode[] = "pertoken";  // dynamic quant
    // op-plugin source (StructKernelNpuOpApi.cpp): `auto trans = true;`
    // aclnn entry rejects transposeWeight=false. Weight tensor must be
    // pre-transposed to [E, N, K] in caller, transposeWeight=true.
    const bool transposeWeight = true;

    for (int64_t iter = -warmup; iter < iterations; ++iter) {
        uint64_t ws_size = 0;
        aclOpExecutor *exec = nullptr;
        CHECK_ACLNN(aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(
            x.t,
            weight.t,
            scale.t,
            gl.t,
            /*biasOptional=*/nullptr,
            /*xScaleOptional=*/nullptr,
            /*xOffsetOptional=*/nullptr,
            use_smooth ? smooth.t : nullptr,
            xQuantMode,
            transposeWeight,
            out.t,
            &ws_size, &exec));

        void *ws = nullptr;
        if (ws_size > 0) {
            CHECK_ACL(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        if (iter >= 0) CHECK_ACL(aclrtRecordEvent(evt_start, stream));
        CHECK_ACLNN(aclnnQuantGroupedMatmulDequantWeightNZ(ws, ws_size, exec, stream));
        if (iter >= 0) {
            CHECK_ACL(aclrtRecordEvent(evt_end, stream));
            CHECK_ACL(aclrtSynchronizeStream(stream));
            float ms = 0.0f;
            CHECK_ACL(aclrtEventElapsedTime(&ms, evt_start, evt_end));
            times_us.push_back(ms * 1000.0f);
        } else {
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }

        if (ws) aclrtFree(ws);
    }

    // ---------------- Verify cosine vs golden ----------------
    double cos = 1.0;
    if (verify) {
        std::string golden_name = use_smooth ? "golden_smooth" : "golden";
        std::vector<uint8_t> golden_bytes;
        if (read_bin(datadir + "/" + golden_name + ".bin", golden_bytes) == 0) {
            std::vector<uint8_t> result(out.bytes);
            CHECK_ACL(aclrtMemcpy(result.data(), result.size(), out.dev,
                                  out.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            size_t n_elem = out.bytes / 2;
            cos = cos_sim(reinterpret_cast<uint16_t *>(result.data()),
                          reinterpret_cast<uint16_t *>(golden_bytes.data()),
                          n_elem);
        } else {
            cos = -1.0;  // golden missing
        }
    }

    // ---------------- Stats ----------------
    std::sort(times_us.begin(), times_us.end());
    float t_min = times_us.empty() ? 0.0f : times_us.front();
    float t_max = times_us.empty() ? 0.0f : times_us.back();
    float t_med = times_us.empty() ? 0.0f : times_us[times_us.size() / 2];
    float t_mean = times_us.empty() ? 0.0f :
        std::accumulate(times_us.begin(), times_us.end(), 0.0f) /
        times_us.size();

    fprintf(stdout,
            "variant=%s E=%ld M=%ld K=%ld N=%ld | iters=%ld "
            "min=%.2fus med=%.2fus mean=%.2fus max=%.2fus | cos=%.6f\n",
            variant.c_str(), E, M, K, N, iterations, t_min, t_med, t_mean,
            t_max, cos);

    if (!output_json.empty()) {
        std::ofstream jf(output_json);
        jf << "{\n";
        jf << "  \"variant\": \"" << variant << "\",\n";
        jf << "  \"E\": " << E << ", \"M\": " << M
           << ", \"K\": " << K << ", \"N\": " << N << ",\n";
        jf << "  \"iterations\": " << iterations
           << ", \"warmup\": " << warmup << ",\n";
        jf << "  \"time_us\": { \"min\": " << t_min
           << ", \"med\": " << t_med
           << ", \"mean\": " << t_mean
           << ", \"max\": " << t_max << " },\n";
        jf << "  \"cos\": " << cos << ",\n";
        jf << "  \"samples\": [";
        for (size_t i = 0; i < times_us.size(); ++i) {
            jf << times_us[i] << (i + 1 < times_us.size() ? ", " : "");
        }
        jf << "]\n}\n";
    }

    // ---------------- Cleanup ----------------
    aclrtDestroyEvent(evt_start);
    aclrtDestroyEvent(evt_end);
    aclDestroyTensor(x.t);
    aclDestroyTensor(weight.t);
    aclDestroyTensor(scale.t);
    aclDestroyTensor(gl.t);
    if (use_smooth) aclDestroyTensor(smooth.t);
    aclDestroyTensor(out.t);
    aclrtFree(x.dev);
    aclrtFree(weight.dev);
    aclrtFree(scale.dev);
    aclrtFree(gl.dev);
    if (use_smooth) aclrtFree(smooth.dev);
    aclrtFree(out.dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}
