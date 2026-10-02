// 用途：ccec dav-c310-vec 后端行为探针矩阵（迭代 2 实测；非交付内核，禁入任何构建目标）——
// 钉死 AIV pass 拒绝判别条件 R1/R2（全文结论见 op_kernel/CMakeLists.txt 头注），
// 编译法：bisheng -xcce -std=c++17 -O3 --cce-aicore-only --cce-auto-sync --cce-mask-opt
//         -D__DAV_C310__ -D__AICORE__ --cce-aicore-arch=dav-c310-vec <CANN asc includes> -c 本文件。
// 实测结论（容器 cann_dev_950，CANN 9.2.0-beta.1，2026-10-02）：
//   p01 整数标量 + VF 直调                              PASS
//   p02/p03/p04/p07 __global__ 内标量 fdiv/fadd/fsub/fmul/fcmp/expf  FAIL R1 "Unsupported scalar instruction in AIV loop"
//   p05/p12/p24~p27 __ubuf__ 裸标量解引用（int/half/f32/u8）         FAIL "error pointer address space cast"
//       —— 仅 inttoptr(GM 形参) 派生指针如此；TBuf/GetPhyAddr 派生指针标量解引用全形态
//          合法（p23/p29/p30/p31/p32），真机内核 UB 指针均为后者，不构成交付阻塞
//   p13/p14 __gm__ 裸标量解引用（含标量 fdiv）            PASS（GM 标量访存不触发 R1）
//   p08 noinline __aicore__ 被调函数内标量 fdiv          FAIL R1（外联不豁免）
//   p09 VF 体标量 fdiv                                   FAIL R1（VF 体同单元）
//   p10a/p16 VF 体常量偏移 __ubuf__ 指针算术/解引用       FAIL 地址空间类错误（运行时下标循环形态 p20 PASS）
//   p17/p18 三参结构体 / 第二 LoadAlign                   PASS
//   p19 VF 体常量偏移 GEP(+64)                            FAIL 地址空间类错误
//   p20 VF 体行循环 + 运行时下标指针步进                  PASS
//   p21 noinline 被调函数内 __ubuf__ 标量拷贝             FAIL 地址空间类错误（inttoptr 派生）
//   p23 TBuf 派生指针 + wrapper 标量 half 拷贝 + 全向量 VF 状态数学 PASS（softmax 改写蓝本）
//   p28 inttoptr 派生 + wrapper 标量 u32/f32 拷贝 + VF    FAIL 地址空间类错误（对照 p29）
//   p29 TBuf 派生 + wrapper 标量 u32/f32 拷贝 + VF        PASS
//   p33b 循环内寄存器 Duplicate 广播                      PASS
//   p33c 循环内 Muls 标量参（memcpy 位强转源）            FAIL "Unsupported Inst must be hoisted"
//   p33d 循环内 Muls 循环不变标量参                       PASS
//   p33e 循环内 Muls 标量参（直接 u32 装载 + bitcast(load)）PASS
//   p33f/p33g 循环变标量的分支位运算/移位位强转→float     FAIL ISel "Cannot select: f32 = bitcast"
// 结论：判别条件与「入口是否 __global__ / 调用链深度 / 显隐式实例化」无关（旧注记两条根因
// 均否证；旧 PASS 探针 j_probe_nong 无 __global__ 引用，属死代码伪 PASS）。
#include "kernel_operator.h"
#include <cstdint>
#include <cstring>

namespace vapi = AscendC::MicroAPI;

namespace sa2probe {

constexpr vapi::CastTrait kProbeB162F32Zero = {
    vapi::RegLayout::ZERO, vapi::SatMode::UNKNOWN, vapi::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// 基线 VF 叶（全向量，无任何标量浮点/标量位强转）。
__simd_vf__ inline void VfLeaf(__ubuf__ float *in, __ubuf__ float *out)
{
    vapi::RegTensor<float> r;
    vapi::LoadAlign(r, in);
    vapi::MaskReg m = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    vapi::Add(r, r, r, m);
    vapi::StoreAlign(out, r, m);
}

struct RowArgs {
    __ubuf__ float *score;
    __ubuf__ float *out;
    float scale;      // 循环不变标量参（p33d 形态）
    int64_t rows;
};

// p33d/p33b 合法形态：循环内 Muls(循环不变标量参) + 寄存器 Duplicate 广播。
__simd_vf__ inline void VfRowInvariant(RowArgs a)
{
    vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    for (int64_t r = 0; r < a.rows; ++r) {
        vapi::RegTensor<float> s;
        vapi::LoadAlign(s, a.score + r * 64);
        vapi::Muls(s, s, a.scale, m64);
        vapi::RegTensor<float> red;
        vapi::ReduceMax(red, s, m64);
        vapi::RegTensor<float> dup;
        vapi::Duplicate(dup, red, m64);
        vapi::Add(s, s, dup, m64);
        vapi::StoreAlign(a.out + r * 64, s, m64);
    }
}

// p33e 合法形态：循环变标量 = 直接 u32 装载 + bitcast(load)（ISel 可选中）。
__simd_vf__ inline void VfRowBitsLoad(RowArgs a)
{
    vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    for (int64_t r = 0; r < a.rows; ++r) {
        vapi::RegTensor<float> s;
        vapi::LoadAlign(s, a.score + r * 64);
        uint32_t bits = reinterpret_cast<const __ubuf__ uint32_t *>(a.score)[256 + r];
        float sc;
        std::memcpy(&sc, &bits, sizeof(sc));
        vapi::Muls(s, s, sc, m64);
        vapi::StoreAlign(a.out + r * 64, s, m64);
    }
}

}  // namespace sa2probe

// p01：基线（整数标量 + VF 直调）——PASS 锚点。
__global__ __aicore__ void sa2_probe_p01_baseline(__gm__ float *in, __gm__ float *out, __gm__ uint8_t *tiling)
{
    (void)tiling;
    sa2probe::VfLeaf(reinterpret_cast<__ubuf__ float *>(reinterpret_cast<uint64_t>(in)),
                     reinterpret_cast<__ubuf__ float *>(reinterpret_cast<uint64_t>(out)));
}

#if !defined(SA2_PROBE_FAILING_VARIANTS)
// 逐条 FAIL 变体以 SA2_PROBE_FAILING_VARIANTS 门控（默认只编译 PASS 锚点，保证本文件
// 单独 -c 恒可编译；复测某 FAIL 变体时加该宏并只保留目标入口）。
#else
// p02：__global__ 内标量 fdiv —— FAIL R1。
__global__ __aicore__ void sa2_probe_p02_g_fdiv(__gm__ float *in, __gm__ float *out, __gm__ uint8_t *tiling)
{
    (void)tiling;
    float a = in[0];
    float b = in[1];
    out[64] = a / b;
}
// p04：__global__ 内标量 fcmp —— FAIL R1。
__global__ __aicore__ void sa2_probe_p04_g_fcmp(__gm__ float *in, __gm__ float *out, __gm__ uint8_t *tiling)
{
    (void)tiling;
    float a = in[0];
    float b = in[1];
    out[64] = (a > b) ? a : b;
}
#endif
