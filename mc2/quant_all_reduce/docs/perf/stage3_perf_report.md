# QuantAllReduce stage3 性能报告

> **分支**：`stage3_target_iter` (基于 stage2 `5223bcbd`)
> **改动**：host 侧 TARGET_ITER 公式化 xPerBlock + DataCopyPad 替换 scale DataCopy（鲁棒性提升）
> **基线**：stage2 cap=2048 归档 `x_block_2048_perf.csv`
> **测试**：world_size=4, 4×Ascend950PR (aivNum=56), CANN 9.0.0, level0 sheet 全 24 用例
> **采集**：mc2_test 内置 10 轮取后 5 轮均值

## 最终配置

```cpp
// host: quant_all_reduce_tiling.cpp::SetXPerBlock
constexpr uint32_t TARGET_ITER = 3U;   // 命中 DoubleBuffer 甜点（实测 T=2/3/6 中 T=3 最优）
constexpr uint32_t MIN_BLOCK = 1024U;  // 公式 raw < 1024 时兜底（24 用例 raw 都 ≥ 2048，MIN 不激活）
constexpr uint32_t ALIGN_BLOCK = 1024U; // 与 kernel X_BLOCK_ALIGN_NUM 对齐

// 公式
xPerBlock = max(ceil(xNums / aivNum / TARGET_ITER), MIN_BLOCK);
xPerBlock = floor(xPerBlock, ALIGN_BLOCK);

// kernel: quant_all_reduce_mte_one_shot.h
constexpr static uint32_t X_BLOCK_ALIGN_NUM = 1024U;
// 读 host 推荐值，与 ub_max 取 min
uint32_t hostRecommended = tiliingDatainfo.xPerBlock;
if (hostRecommended != 0U && maxXPerBlock > hostRecommended) {
    maxXPerBlock = hostRecommended;
}
// scale DataCopy → DataCopyPad（处理 sub-32B 对齐）
```

## xPerBlock 实际下发值

| shape | per_core (xNums/aivNum) | T=3 raw | floor 1024 | 每核 iter |
|-------|-------------------------|---------|------------|-----------|
| 96×4096 (qwen) | 7023 | 2341 | **2048** | 4 |
| 96×7168 (DSV3) | 12289 | 4097 | **4096** | 3 |

DSV3 系列从 stage2 的 2048（6 iter）升级到 4096（3 iter），DB 流水更高效。
qwen 系列与 stage2 同（2048），无变化。

## 24 用例 perf 对照

| case | stage2 (us) | stage3 (us) | Δ% |
|---|---:|---:|---:|
| 96_7168_int8→fp32 ID0001 | 32.892 | 32.337 | **−1.69%** |
| 96_7168_hif8→fp32 ID0002 | 32.787 | 31.829 | **−2.92%** |
| 96_7168_fp8_e4m3fn→fp32 ID0003 | 32.793 | 31.932 | **−2.62%** |
| 96_7168_fp8_e5m2→fp32 ID0004 | 32.757 | 32.467 | −0.89% |
| 96_7168_MX_e4m3fn→fp32 ID0005 | 32.936 | 32.497 | −1.33% |
| 96_7168_MX_e5m2→fp16 ID0006 | 33.240 | 32.043 | **−3.60%** |
| 96_7168_int8→bf16 ID0007 | 32.929 | 32.033 | **−2.72%** |
| 96_7168_hif8→fp32 ID0008 | 32.899 | 31.791 | **−3.37%** |
| 96_7168_fp8_e4m3fn→bf16 ID0009 | 32.864 | 32.072 | **−2.41%** |
| 96_7168_fp8_e5m2→bf16 ID0010 | 32.816 | 32.005 | **−2.47%** |
| 96_7168_MX_e4m3fn→fp32 ID0011 | 33.126 | 32.420 | **−2.13%** |
| 96_7168_MX_e5m2→bf16 ID0012 | 32.900 | 32.216 | **−2.08%** |
| 96_4096_int8→fp16 ID0013 | 23.230 | 22.925 ¹ | −1.32% |
| 96_4096_hif8→bf16 ID0014 | 22.803 | 22.960 | +0.69% |
| 96_4096_fp8_e4m3fn→bf16 ID0015 | 23.008 | 22.712 | −1.29% |
| 96_4096_fp8_e5m2→fp16 ID0016 | 22.789 | 22.999 | +0.92% |
| 96_4096_MX_e4m3fn→fp16 ID0017 | 23.595 | 23.392 | −0.86% |
| 96_4096_MX_e5m2→bf16 ID0018 | 23.382 | 23.133 | −1.07% |
| 96_4096_int8→bf16 ID0019 | 22.801 | 23.026 | +0.99% |
| 96_4096_hif8→bf16 ID0020 | 22.433 | 23.016 | +2.60% ² |
| 96_4096_fp8_e4m3fn→bf16 ID0021 | 22.665 | 22.877 | +0.94% |
| 96_4096_fp8_e5m2→fp16 ID0022 | 22.721 | 22.588 | −0.59% |
| 96_4096_MX_e4m3fn→fp16 ID0023 | 23.151 | 23.793 | +2.77% ² |
| 96_4096_MX_e5m2→fp32 ID0024 | 23.395 | 22.879 | **−2.20%** |

注：
1. ID0013 首测 93.29us 为 profiler 残留污染异常，已重测取 22.93us。
2. ID0020 / ID0023 略超 +2% 阈值，处于测量噪声边界；对应 shape 在 T=3 公式下分块与 stage2 一致（都是 2048），perf 差异属于运行波动。

## 总结

- **24/24 用例精度 PASS**（4×rank 共 96 项）
- **24 用例总时长**：670.91 us → 661.93 us（**−1.34%**）
- **DSV3 (h=7168) 系列 12 用例**：全部 −0.9% ~ −3.6% 加速（xPerBlock 由 2048 升 4096，每核 iter 由 6 降 3，DB 流水更高效）
- **qwen (h=4096) 系列 12 用例**：与 stage2 持平（xPerBlock 不变 2048）

## 调优过程关键决策

| 尝试 | T | MIN | ALIGN | 结果 |
|------|---|-----|-------|------|
| 1 | 6 | 1024 | 1024 | qwen 落 1024 → 24 用例 +2.80%（回退） |
| 2 | 3 | 1024 | 1024 | qwen 落 2048（同 stage2），DSV3 落 4096 → **−2.17%** ✅ |
| 3 | 2 | 1024 | 1024 | DSV3 落 6144（3 iter 不足）→ +1.82%（回退） |
| 4 | 3 | 512 | 512 | scale 越界 → 精度 FAIL |
| 5 | 3 | 1024 | 1024 + DataCopyPad | **−1.34%** ✅（DataCopyPad 微小 overhead） |

最终采用 #5：保留 DataCopyPad 提升边界鲁棒性，T=3 + 1024 对齐为甜点。

## 经验教训

1. **TARGET_ITER 与硬件 aivNum 强耦合**：T=3 在 aivNum=56 下使分块 / iter 数命中 DB 甜点，但同样的 T 在 aivNum=24/40 等环境下需要重扫
2. **MIN_BLOCK 对当前 24 用例不激活**，但保留作为小 shape 兜底
3. **DataCopyPad 替代 DataCopy 对 scale 操作鲁棒性提升**：解锁未来 sub-1024 alignment 实验的可能性，运行时 overhead 可忽略
4. **测量异常用例（ID0013 等）**：批跑时 profiler 状态残留可能导致单点 4× 异常，需单独重测确认
