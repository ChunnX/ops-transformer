/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * MX-FP8 tiling 引擎，从 QuantMatmulTilingSwat 提取关键计算逻辑，
 * 适配 MC2 框架（用 context_->GetPlatformInfo() 代替 platform_ascendc.h）
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include <cstdio>

struct Mc2MxTilingResult {
    uint64_t baseM = 0, baseN = 0, baseK = 0;
    uint64_t kL1 = 0, scaleKL1 = 0;
    uint64_t stepK = 0;
    uint64_t mBlockCnt = 0, nBlockCnt = 0;
    uint32_t dbL0c = 1;
    uint64_t mTailTile = 1, nTailTile = 1;
    uint64_t mBaseTailSplitCnt = 1, nBaseTailSplitCnt = 1;
    uint64_t mTailMain = 0, nTailMain = 0;
    bool valid = false;
};

class Mc2MxTiling {
public:
    Mc2MxTiling(uint64_t fullM, uint64_t fullN, uint64_t fullK,
                uint64_t l1Size, uint64_t l0cSize, uint64_t aicNum, uint32_t aBytes, uint32_t bBytes)
        : m_(fullM), n_(fullN), k_(fullK), l1_(l1Size), l0c_(l0cSize),
          aicNum_(aicNum), aB_(aBytes), bB_(bBytes) {}

    Mc2MxTilingResult Compute()
    {
        Mc2MxTilingResult r;
        CalcBasicBlock();
        if (baseM_ == 0 || baseN_ == 0 || baseK_ == 0) return r;

        CalcPathSpecificL1();
        scaleKL1_ = CalcScaleKL1();

        r.baseM             = baseM_;
        r.baseN             = baseN_;
        r.baseK             = baseK_;
        r.stepK             = std::min(stepKa_, stepKb_);
        r.kL1               = r.stepK * baseK_;
        r.scaleKL1          = scaleKL1_;
        r.mBlockCnt         = mBlockCnt_;
        r.nBlockCnt         = nBlockCnt_;
        r.dbL0c             = dbL0c_;
        r.mTailTile         = 1;
        r.nTailTile         = 1;
        r.mBaseTailSplitCnt = 1;
        r.nBaseTailSplitCnt = 1;
        r.mTailMain         = 0;
        r.nTailMain         = 0;
        printf("[Mc2MxTiling] input: m=%lu n=%lu k=%lu l1=%lu l0c=%lu aicNum=%lu aB=%u bB=%u\n",
               m_, n_, k_, l1_, l0c_, aicNum_, (unsigned)aB_, (unsigned)bB_);
        printf("[Mc2MxTiling] block: baseM=%lu baseN=%lu baseK=%lu dbL0c=%u\n",
               baseM_, baseN_, baseK_, dbL0c_);
        printf("[Mc2MxTiling] L1 depth: depthA1=%lu depthB1=%lu stepKa=%lu stepKb=%lu\n",
               depthA1_, depthB1_, stepKa_, stepKb_);
        printf("[Mc2MxTiling] scale: scaleFactorA=%lu scaleFactorB=%lu\n",
               scaleFactorA_, scaleFactorB_);
        printf("[Mc2MxTiling] output: kL1=%lu(r.stepK*baseK=%lu*%lu) scaleKL1=%lu stepK=%lu\n",
               r.kL1, r.stepK, baseK_, r.scaleKL1, r.stepK);

        r.valid             = true;
        return r;
    }

private:
    uint64_t m_, n_, k_, l1_, l0c_, aicNum_, aB_, bB_;
    uint64_t baseM_ = 0, baseN_ = 0, baseK_ = 0;
    uint64_t depthA1_ = 0, depthB1_ = 0;
    uint64_t stepKa_ = 0, stepKb_ = 0;
    uint64_t scaleFactorA_ = 0, scaleFactorB_ = 0;
    uint64_t scaleKL1_ = 0;
    uint64_t mBlockCnt_ = 0, nBlockCnt_ = 0;
    uint64_t tailBlockCnt_ = 0, tailM_ = 0, tailN_ = 0;
    uint32_t dbL0c_ = 1;

    static constexpr uint64_t CB    = 16;
    static constexpr uint64_t MG    = 32;
    static constexpr uint64_t SA    = 64;
    static constexpr uint64_t SM    = 2;
    static constexpr uint64_t DB    = 2;
    static constexpr uint64_t B256  = 256;
    static constexpr uint64_t B128  = 128;
    static constexpr uint64_t B512  = 512;
    static constexpr uint64_t SKMAX = 4;
    static constexpr uint64_t SFMAX = 127;
    static constexpr uint64_t MTE2M = 32768;
    static constexpr uint64_t MTE2C = 128;
    static constexpr uint64_t DATA_C = 4;

    static uint64_t CD(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
    static uint64_t AL(uint64_t a, uint64_t b) { return ((a + b - 1) / b) * b; }
    static uint64_t FA(uint64_t a, uint64_t b) { return (a / b) * b; }

    void CalcBasicBlock()
    {
        baseM_ = AL(std::min(m_, B256), CB);
        baseN_ = AL(std::min(n_, B256), CB);
        baseK_ = B128;

        mBlockCnt_ = CD(m_, baseM_);
        nBlockCnt_ = CD(n_, baseN_);

        uint64_t blockNum = mBlockCnt_ * nBlockCnt_;
        if (blockNum < aicNum_) AdjustBasicBlock();
        if (baseM_ == 0 || baseN_ == 0 || baseK_ == 0) return;

        mBlockCnt_ = CD(m_, baseM_);
        nBlockCnt_ = CD(n_, baseN_);

        // 参考 CalcBasicBlock: dbL0c = (baseM*baseN*8 <= L0C) ? 2 : 1
        dbL0c_ = (baseM_ * baseN_ * DATA_C * DB <= l0c_) ? 2 : 1;

        // tail statistics
        uint64_t totalBlock = mBlockCnt_ * nBlockCnt_;
        tailBlockCnt_ = totalBlock % aicNum_;
        tailM_ = m_ - (mBlockCnt_ - 1) * baseM_;
        tailN_ = n_ - (nBlockCnt_ - 1) * baseN_;
    }

    void AdjustBasicBlock()
    {
        uint64_t mM = CD(m_, CB), nM = CD(n_, CB);
        uint64_t tbM = baseM_, tbN = baseN_, mC = CD(m_, baseM_), nC = CD(n_, baseN_);
        if (mM > nM) {
            tbN = AL(CD(n_, nC), CB); nC = CD(n_, tbN);
            mC = aicNum_ / nC; tbM = AL(CD(m_, mC), CB);
        } else {
            tbM = AL(CD(m_, mC), CB); mC = CD(m_, tbM);
            nC = aicNum_ / mC; tbN = AL(CD(n_, nC), CB);
        }
        baseM_ = tbM; baseN_ = tbN;
    }

    void CalcPathSpecificL1()
    {
        uint64_t aSize = baseM_ * baseK_ * aB_;
        uint64_t bSize = baseN_ * baseK_ * bB_;
        uint64_t saSize = AL(CD(baseK_, MG), SM) * baseM_;
        uint64_t sbSize = AL(CD(baseK_, MG), SM) * baseN_;
        uint64_t baseL1 = aSize + bSize + saSize + sbSize;
        uint64_t dInit = GetDepth(l1_, baseL1, 1);
        uint64_t left = l1_ - dInit * baseL1;
        uint64_t dASec = GetDepth(left, (aSize + saSize) * dInit, dInit);
        uint64_t dBSec = GetDepth(left, (bSize + sbSize) * dInit, dInit);
        depthA1_ = std::max(dASec, dBSec);
        depthB1_ = depthA1_;
        if (depthA1_ * baseL1 > l1_) {
            depthA1_ = dASec >= dBSec ? dASec : dInit;
            depthB1_ = dASec < dBSec ? dBSec : dInit;
        }
        CalStepKs();
        CalScaleFactors(aSize, bSize, saSize, sbSize);
    }

    uint64_t GetDepth(uint64_t leftSize, uint64_t perDepth, uint64_t dInit)
    {
        if (dInit > 1 && perDepth > DB * MTE2M) return dInit;
        uint64_t ds = leftSize / perDepth;
        if (dInit > 1) {
            uint64_t bkSize = baseM_ * baseK_ * aB_;
            while ((ds * bkSize) % B512 != 0 && ds * bkSize > B512) ds--;
            if ((ds * bkSize) % B512 != 0 && ds * bkSize >= B256) ds = B256 / bkSize;
            ds = std::max(ds, 1UL);
        } else {
            ds = 1;
            while (ds * perDepth < leftSize) ds *= 2;
            ds = (ds == 1) ? ds : ds / 2;
        }
        return dInit * ds;
    }

    void CalStepKs()
    {
        stepKa_ = depthA1_ / DB; stepKb_ = depthB1_ / DB;
        if (stepKa_ * baseK_ > k_) stepKa_ = CD(k_, baseK_);
        if (stepKb_ * baseK_ > k_) stepKb_ = CD(k_, baseK_);
        if (stepKa_ > stepKb_) stepKa_ = stepKa_ / stepKb_ * stepKb_;
        if (stepKb_ > stepKa_) stepKb_ = stepKb_ / stepKa_ * stepKa_;
        stepKa_ = std::min(stepKa_, SKMAX); stepKb_ = std::min(stepKb_, SKMAX);
        depthA1_ = stepKa_ * DB; depthB1_ = stepKb_ * DB;
    }

    void CalScaleFactors(uint64_t aSize, uint64_t bSize, uint64_t saSize, uint64_t sbSize)
    {
        uint64_t fAM = std::min(MTE2M / saSize, SFMAX);
        uint64_t fBM = std::min(MTE2M / sbSize, SFMAX);
        uint64_t fA = k_ / (stepKa_ * baseK_);
        uint64_t fB = k_ / (stepKb_ * baseK_);
        scaleFactorA_ = std::min(std::max(1UL, fA), fAM);
        scaleFactorB_ = std::min(std::max(1UL, fB), fBM);

        uint64_t left = l1_ - (depthA1_ * aSize + depthB1_ * bSize);
        uint64_t sInit = left / (depthA1_ * saSize + depthB1_ * sbSize);
        if (scaleFactorA_ <= sInit && scaleFactorB_ > sInit) {
            left -= scaleFactorA_ * depthA1_ * saSize;
            scaleFactorB_ = std::min(left / (depthB1_ * sbSize), scaleFactorB_);
        } else if (scaleFactorB_ <= sInit && scaleFactorA_ > sInit) {
            left -= scaleFactorB_ * depthB1_ * sbSize;
            scaleFactorA_ = std::min(left / (depthA1_ * saSize), scaleFactorA_);
        } else if (scaleFactorA_ > sInit && scaleFactorB_ > sInit) {
            left -= sInit * (depthB1_ * sbSize + depthA1_ * saSize);
            uint64_t saSec = std::min(left / (depthA1_ * saSize), scaleFactorA_ - sInit);
            scaleFactorA_ = saSec + sInit;
            scaleFactorB_ = sInit;
        }
    }

    uint64_t CalcScaleKL1()
    {
        return std::min(scaleFactorA_ * stepKa_ * baseK_, scaleFactorB_ * stepKb_ * baseK_);
    }
};
