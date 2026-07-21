# FusedFloydAttention A3→A5 改造问题索引

按遇到顺序梳理，前段是测试框架/环境问题（已解决），后段是 kernel 迁移问题（部分未决）。

## 测试框架与环境

| # | 问题 | 状态 |
|---|------|------|
| [01](issue-01-atk-register-name-mismatch.md) | ATK executor 注册名不匹配（`KeyError: ascend_function_ffa`） | ✅ 已解决 |
| [02](issue-02-headdim-constraint.md) | headDim 不符合约束（仅支持 32/64/128） | ✅ 已解决 |
| [03](issue-03-accuracy-registry-missing.md) | 精度比对方法 `cv_fused_double_benchmark` 未注册 | ✅ 已解决 |
| [04](issue-04-output2-dtype-mismatch.md) | output[2] dtype 不匹配（golden fp32 / npu bf16） | ✅ 已解决 |
| [05](issue-05-soc-verification-failed.md) | SoC 版本验证失败（自定义 .so 未加载） | ✅ 已解决 |

## Kernel A3→A5 迁移

| # | 问题 | 状态 |
|---|------|------|
| [06](issue-06-aicore-error-cube-on-aiv.md) | AI Core Error：Cube 指令在 AIV 核执行（MIX 模式架构） | ✅ 已定位，架构改造中 |
| [07](issue-07-unitflag-deadlock.md) | unitFlag 导致 Cube matmul 死锁/timeout | ✅ 已解决 |
| [08](issue-08-qcoreoffset-uninitialized.md) | qCoreOffset 未初始化 → Output2 几乎全零 | ✅ 已解决 |
| [09](issue-09-bmm2-v-layout-error.md) | Bmm2 的 V 矩阵 Nd2Nz 布局错误 | ⚠️ 已修，被上游 Bmm1 掩盖 |
| [10](issue-10-bmm1-systematic-error.md) | Bmm1 系统性数值误差（softmaxMax cosine 0.965） | ⏳ **当前主攻，根因未定** |
| [11](issue-11-dump-not-available.md) | A5 MIX 下 DumpTensor/printf 无输出 | ⏳ 卡点，待规避 |

## 当前主线

精度问题收敛到 **[issue-10 Bmm1 系统性误差](issue-10-bmm1-systematic-error.md)**：手写 adapter 与库 matmul 误差形态一致，根因在共同上游（输入数据 / host tiling 参数 / A5 适配语义）。因 [issue-11](issue-11-dump-not-available.md) 拿不到 kernel 内部数据，下一步倾向先在 A3(910B) 上跑 case 61 对照，判定是否为 A5 特有问题。
