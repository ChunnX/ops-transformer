// sa2_host_e2e.cpp — 宿主机差分桥（M2a 内核差分测试）：单线程全链执行
// KV 预处理 → 注意力主循环 → epilogue，输出 q/k/v/out/lse 原始字节，
// 由 tests/test_host_e2e_bridge.py 载入后与 sageattn_reference 逐配置对拍。
//
// 覆盖目标（宿主可执行的正确性闭环，todo §四 M2a「内核宿主差分桥」）：
// 1. 全链管线接线：planner 解码 → KV prep → QQuant/QK/softmax/PV/epilogue；
// 2. P0-1 回归：奇偶双 Die 全 KV head 覆盖（odd_hk 用例 hk=3 奇数，die1 尾槽无效占位）；
// 3. GQA 组映射与 corr 广播（gqa 用例 Hq/Hk=4/2）；
// 4. pad 几何（d=48→64、d=96→128；sq=200→qp=256 尾块 8 有效行）与 causal 掩码；
// 5. TC-05 数值化（todo §四）：pad48/pad96 用例经内核 pad 路径（48→64、96→128）+ epilogue 裁剪后，
//    由 pytest 读回同一份输入字节送 sageattn_reference 直算 D=48/96 对拍（同输入字节级一致，强于同 seed）。
//
// 已知口径差（对拍用容差而非逐位的原因，todo §一.1 骨架层遗留口径缺口）：
// - per_thread 档内核 S 组合域为 FP32（无 FP16 舍入点），参考实现统一 score.half() 一次舍入；
// - 内核 qScale/kScale 以 FP16 存储，参考实现全程 FP32 scale。
// 两者均为 L-A golden 反推收口项（M4.3），不在此预先改齐。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../sim/sa2_sim_engine.h"  // 宿主仿真引擎（sim 模块独立，零依赖 op_kernel 硬件实现）

namespace {

constexpr uint64_t kLcgMul = 6364136223846793005ULL;
constexpr uint64_t kLcgAdd = 1442695040888963407ULL;

uint32_t NextRandom(uint64_t &state) {
    state = state * kLcgMul + kLcgAdd;
    return static_cast<uint32_t>(state >> 33);
}

// 确定性输入：[-2, 2) 均匀网格值（LCG 可跨语言复现由 pytest 直接读回本二进制导出的输入，无需对齐 RNG）
float Sample(uint64_t &state) {
    const uint32_t r = NextRandom(state) & 0xFFFFFu;
    return static_cast<float>(r) / 1048576.0f * 4.0f - 2.0f;
}

struct CaseConfig {
    const char *name;
    int64_t b, hq, hk, sq, sk, d;
    bool granPerThread;
    bool isCausal;
    int64_t accum;  // PvAccum 载体
};

void WriteBytes(const std::string &path, const void *data, size_t bytes) {
    FILE *f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "cannot open %s for write\n", path.c_str());
        exit(2);
    }
    if (bytes != 0 && fwrite(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "short write %s\n", path.c_str());
        exit(2);
    }
    fclose(f);
}

template <bool GranPerThread, bool IsCausal>
int RunCase(const CaseConfig &cfg, const std::string &prefix) {
    using sa2::DeriveNumericPolicy;
    using sa2::Gran;
    using sa2::PvAccum;
    using sa2::Sa2Params;
    using sa2::Sa2TilingData;

    const Gran gran = GranPerThread ? Gran::kPerThread : Gran::kPerWarp;
    const PvAccum accum = static_cast<PvAccum>(cfg.accum);
    const float smScale = 1.0f / std::sqrt(static_cast<float>(cfg.d));
    const auto policy = DeriveNumericPolicy(gran, accum, cfg.sq, cfg.sk, cfg.d, smScale);
    if (policy.dp == 0) {
        fprintf(stderr, "bad head_dim %lld\n", static_cast<long long>(cfg.d));
        return 1;
    }

    Sa2Params p{};
    p.b = cfg.b;
    p.hq = cfg.hq;
    p.hk = cfg.hk;
    p.sq = cfg.sq;
    p.sk = cfg.sk;
    p.d = cfg.d;
    p.dp = policy.dp;
    p.qp = policy.qp;
    p.kp = policy.kp;
    p.qStride[0] = cfg.hq * cfg.sq * cfg.d;
    p.qStride[1] = cfg.sq * cfg.d;
    p.qStride[2] = cfg.d;
    p.kStride[0] = cfg.hk * cfg.sk * cfg.d;
    p.kStride[1] = cfg.sk * cfg.d;
    p.kStride[2] = cfg.d;
    p.vStride[0] = p.kStride[0];
    p.vStride[1] = p.kStride[1];
    p.vStride[2] = cfg.d;
    p.causal = cfg.isCausal;
    p.smoothK = true;
    p.smoothV = false;
    p.returnLse = true;
    p.gran = static_cast<int64_t>(gran);
    p.accum = cfg.accum;
    p.smScale = smScale;
    p.qFactor = policy.qFactor;
    p.kFactor = policy.kFactor;
    // κ 两域字段（与 op_host/sa2_tiling.cpp 契约填充逐字对齐）：θ=ln(κ) 驱动 D-R1 rescale-skip，
    // 漏填将使差分轨退化为 θ=0（skip 判据失真、+lnκ 类 LSE 漂移不可检出，迭代 10 突变验证发现）
    p.pDomain = policy.pDomain;
    p.lnPDomain = policy.lnPDomain;
    p.segWidth = policy.segWidth;
    p.cores = 1;  // 目标形态：单 AIC（P0-1 回归口径）

    // workspace 六分区（512B 对齐，公式与 KvHeadView 寻址一致）
    const int64_t bh = cfg.b * cfg.hk;
    const auto align512 = [](int64_t x) { return (x + 511) / 512 * 512; };
    const int64_t kMeanSize = bh * policy.dp * 4;
    const int64_t vMeanSize = kMeanSize;
    const int64_t k8Size = bh * policy.kp * policy.dp;
    const int64_t kScaleSize = bh * policy.kp * policy.segWidth * 2;
    const int64_t v8Size = k8Size;
    const int64_t vScaleSize = bh * (policy.kp / 64) * policy.dp * 2;
    Sa2TilingData td{};
    td.params = p;
    int64_t off = 0;
    td.kMean = off; off = align512(off + kMeanSize);
    td.vMean = off; off = align512(off + vMeanSize);
    td.k8 = off; off = align512(off + k8Size);
    td.kScale = off; off = align512(off + kScaleSize);
    td.v8 = off; off = align512(off + v8Size);
    td.vScale = off; off = align512(off + vScaleSize);
    const int64_t wsBytes = off;

    std::vector<uint8_t> ws(wsBytes, 0);
    std::vector<uint16_t> q(static_cast<size_t>(cfg.b * cfg.hq * cfg.sq * cfg.d));
    std::vector<uint16_t> k(static_cast<size_t>(cfg.b * cfg.hk * cfg.sk * cfg.d));
    std::vector<uint16_t> v(k.size());
    std::vector<uint16_t> out(q.size());
    std::vector<float> lse(static_cast<size_t>(cfg.b * cfg.hq * cfg.sq), 0.0f);

    uint64_t rng = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(cfg.d) << 13 ^
                   static_cast<uint64_t>(cfg.sk) << 3;
    for (auto &x : q) x = sa2::hostbits::HalfF32(Sample(rng)).val;
    for (auto &x : k) x = sa2::hostbits::HalfF32(Sample(rng)).val;
    for (auto &x : v) x = sa2::hostbits::HalfF32(Sample(rng)).val;

    auto simEngine = sa2::sim::Sa2SimulationEngine<sa2::hostbits::HalfF32, GranPerThread, IsCausal>();
    simEngine.Init(reinterpret_cast<uint8_t *>(q.data()), reinterpret_cast<uint8_t *>(k.data()),
                   reinterpret_cast<uint8_t *>(v.data()), reinterpret_cast<uint8_t *>(out.data()),
                   reinterpret_cast<uint8_t *>(lse.data()), ws.data(), &td);
    simEngine.Process();

    WriteBytes(prefix + ".q.bin", q.data(), q.size() * 2);
    WriteBytes(prefix + ".k.bin", k.data(), k.size() * 2);
    WriteBytes(prefix + ".v.bin", v.data(), v.size() * 2);
    WriteBytes(prefix + ".out.bin", out.data(), out.size() * 2);
    WriteBytes(prefix + ".lse.bin", lse.data(), lse.size() * 4);
    FILE *meta = fopen((prefix + ".meta.txt").c_str(), "w");
    if (meta != nullptr) {
        fprintf(meta, "b=%lld hq=%lld hk=%lld sq=%lld sk=%lld d=%lld gran=%s causal=%d accum=%lld\n",
                static_cast<long long>(cfg.b), static_cast<long long>(cfg.hq), static_cast<long long>(cfg.hk),
                static_cast<long long>(cfg.sq), static_cast<long long>(cfg.sk), static_cast<long long>(cfg.d),
                cfg.granPerThread ? "per_thread" : "per_warp", cfg.isCausal ? 1 : 0,
                static_cast<long long>(cfg.accum));
        fclose(meta);
    }
    printf("[e2e] %s done: out=%zu fp16, lse=%zu fp32\n", cfg.name, out.size(), lse.size());
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <output-prefix>\n", argv[0]);
        return 1;
    }
    const std::string prefix = argv[1];
    // 用例 1：GQA 4/2 + per_thread + fp32+fp16（默认验收配置），非因果
    const CaseConfig gqa{"gqa_thread_fp16acc", 1, 4, 2, 128, 256, 128, true, false, 2};
    // 用例 2：奇 hk=3（双 Die 尾槽）+ per_warp + fp32 + causal + pad 48→64 + 尾块 200→256
    const CaseConfig odd{"odd_warp_causal", 1, 3, 3, 200, 200, 48, false, true, 0};
    // 用例 3（TC-05 数值化）：pad 48→64 + per_thread + fp32+fp16（默认验收档）+ GQA 4/2，
    // sq=96 → qp=128 尾块 32 有效行，pad 列几何经 Q 量化分段（dp=64 → 4 段）与 epilogue 裁剪全链
    const CaseConfig pad48{"pad48_thread_tc05", 1, 4, 2, 96, 128, 48, true, false, 2};
    // 用例 4（TC-05 数值化）：pad 96→128 + per_warp + fp32，sk=160 → kp=256（KV 尾块 32 有效行）
    const CaseConfig pad96{"pad96_warp_tc05", 1, 2, 2, 64, 160, 96, false, false, 0};
    if (RunCase<true, false>(gqa, prefix + "_gqa") != 0) return 1;
    if (RunCase<false, true>(odd, prefix + "_odd") != 0) return 1;
    if (RunCase<true, false>(pad48, prefix + "_pad48") != 0) return 1;
    if (RunCase<false, false>(pad96, prefix + "_pad96") != 0) return 1;
    return 0;
}
