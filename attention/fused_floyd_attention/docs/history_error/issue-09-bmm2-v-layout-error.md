# Bmm2 的 V 矩阵 Nd2Nz 布局错误

## 现象

qCoreOffset 修复后，Output 2（attention_out = Vec2(Bmm2(...)))cosine ≈ 0，
数值随机但位置正确。

## 根因（子 agent 结论）

adapter 把两件独立的事**耦合到同一个 `transB_` 标志**：
1. B 矩阵的 Nd2Nz 布局方向；
2. LoadData 的 ifTranspose。

对 Bmm2（`SetTensorB(value)`，默认 transB_=false）：
- LoadData `ifTranspose = !false = true` ✓ 正确（匹配 GQA 的 V load）
- 但 Nd2Nz 走 transB_=false 分支：`nValue=N(=d), dValue=K(=s2), srcDValue=singleN_, dstNzC0Stride=alignN`
  → 把 V 当成物理 `[d,s2]`，C0-stride 放 d 轴。
- **V 物理上是 `[s2,d]`**（和 K 相同布局），GQA 把 C0-stride 放 s2 轴（`dstNzC0Stride=alignK`）。
- 结果：V 以错误方向搬进 L1，再被正确的转置 LoadData 转置了垃圾 → cosine≈0。

**关键**：不能简单翻转 transB_——单标志无法表达 “V 物理布局同 K(`[s2,d]`) 但需要相反转置”。

## 修复方向

解耦 Nd2Nz 布局 与 ifTranspose，不再都从 transB_ 推导。Bmm2 的 B(V) 应该：
- **Nd2Nz**：`nValue=s2, dValue=d, srcDValue=singleN_(=dSize 的 GM d-stride), dstNzC0Stride=alignK, dstNzNStride=1`
- **LoadData**：`ifTranspose=true, srcStride=alignK/16, mStep=alignK/16 (K fractals), kStep=nFrac (N fractals)`
- 即 gqa_kernel.asc 562-571 的模式。

## 实测结果：修复后无改善，方向被纠正

实施 Bmm2 布局修复后：`Output2 cosine 0.000549→0.000517`，几乎无变化，Output0/1 完全不变。

**纠正判断**：根因不在 Bmm2，而在 **Bmm1**（见 [issue-10](issue-10-bmm1-systematic-error.md)）。
- Bmm1 错 → mm1Res 错 → softmax(P) 错 → Bmm2 无论对错，Output2 都错。
- Bmm2 的这个布局修改本身是对的，先保留，只是当前被 Bmm1 的错误掩盖、显现不出来。

## 待二次验证（大 D case）

1. **L0B 大小 / N 分块**：adapter 单次 LoadData 全部 N=dSizeAlign16；GQA 对 N 分 128（8 fractal）块，因为 K_BASE×d(=512) 会溢出 L0B。case 61 d=64 单次 OK，大 D 需 N 分段循环。
2. **A(P) Nd2Nz stride**：adapter 用 `srcDValue=singleK_(=s2BaseSize)`，但 stage1Res 由 Vec1 写入的行 stride 是 s2AlignedSize。需确认 s2BaseSize == s2AlignedSize，不等则要用真实值。
