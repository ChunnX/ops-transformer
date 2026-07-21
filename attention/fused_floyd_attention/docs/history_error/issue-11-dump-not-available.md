# A5 MIX 模式下 DumpTensor / printf 无输出

## 现象

为定位 [issue-10](issue-10-bmm1-systematic-error.md) 的 Bmm1 数值误差，需 dump kernel 内部的
Q/K/mm1Res 手算对比，但两种手段都拿不到输出。

## DumpTensor

在 `IterateBmm1` 里插入：

```cpp
AscendC::DumpTensor(queryGm[...], 100, 32);   // desc=自定义ID, dumpSize=元素个数
```

**实际结果**：ATK 框架下所有日志（debug.log、atk.log、task output）里均**无 DumpTensor 输出**。

推测：ATK 拦截/重定向了 dump，或 DumpTensor 需要 Kernel 直调场景才输出（文档约束）。

## printf

之前尝试 `printf("[Bmm1Debug]...")` 也无任何输出 → A5 MIX 模式下标准输出机制不可用。

## 影响

无法拿到 kernel 内部实测数据（Q/K/mm1Res），难以手算对比验证传参是否正确，
成为定位 Bmm1 系统性误差的主要卡点。

## 规避方案

- **msopst 单算子直调**：脱离 ATK 框架，DumpTensor 应能正常输出（准备成本高）。
- **A3(910B)对照**：在能正常 dump 的平台上验证原版精度与中间结果。

> 备份：`/home/j00946653/830/backup/s1s2_before_dump_110144.h`（DumpTensor 插入前）。
