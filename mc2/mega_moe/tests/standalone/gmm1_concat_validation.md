# arch35 GMM1 内部拼接

输入 weight1 仍是原来的 gate/up 分离布局，scale1 也保持原顺序。设备入口固定启用片内拼接，
不读取 `weight1_interleave`，也不再使用 `MEGA_MOE_WEIGHT1_INTERLEAVED` 编译宏。

设完整权重的逻辑 N 为 `2H`，GMM1 的计算 tile 宽度为 256，第 t 个 tile 搬运：

```text
gate: weight1[128*t : 128*t+128, :]
up:   weight1[H+128*t : H+128*t+128, :]
片内: [gate128, up128]
输出: 一次 BlockMmad 产生 [gate128, up128]
激活: SiLU(gate128) * up128，写回原来的连续 H 维输出
```

这里的一次调用指每个配对的 GMM1 tile；BlockMmad 内部仍按 K 分块迭代。
现有 tiling 保证完整 N 是 256 的倍数，故每个 gate/up 半 tile 都有 128 列。

| 路径 | 权重搬运 | scale 搬运 |
| --- | --- | --- |
| A8W8 DN | 复用 ops-tensor `CopyConcatGM2L1`，在 GM→L1 时拼接 | 同一接口拼接 ScaleBDN |
| A8W8 NZ | GM→L1 按 K/C0 行交替读取 gate/up | 同上 |
| A8W4 | GM→UB 读取 C0=32 的 packed W4 并拼接，再走已有 W4→W8、UB→L1 | 同上 |
| A4W4 GMM1 | 原生 FP4、C0=64，GM→L1 时拼接 | 同上 |

A4W4 的 GMM1 没有 W4→W8 prologue；它的 GMM2 仍走原来的转换路径。
GMM2 的权重布局不变，其输入就绪计数按新的 GMM1 tile 数计算。
wave、shared expert、TopK weight prefetch 与 layered 调用都传递相同的片内布局信息。

通用 BlockMmad 保留 ops-tensor `block_mmad_qbmm_mx.h`（de3bf6f）的 L1/L0/Fixpipe 调度，
仅替换 GMM1 的右矩阵及 scale 搬运。DN/scale 使用该依赖已有的 concat 接口；
其语义也与核对过的 ops-tensor master ddda90f66a 一致。NZ 使用本地 burst-copy 扩展。

## 验证方法与边界

- standalone CPU 测试：108 组 NZ 地址、packed nibble、K 尾块、缓冲区边界及 SwiGLU 配对检查。
  运行方法见同目录 README。
- Ascend950、CANN 9.2.0-beta.2 的 MTE 设备编译：三条纯模板、A8W8 DN/NZ、
  routed/shared 六种混合搭配、三条 Combine E4M3 分支，以及 A8W8 E5M2 NZ/Combine E5M2；
  每种签名均编译 prefetch 开/关，共 30 个最终配置内核。
- A8W8、A8W4 的独立中间提交另做设备编译，检查尚未启用的 W4 shared 路径仍可实例化。
- 本地 Hcomm 缺少上游 layered 路径使用的 `AscendC::BatchHandle`、`BatchCommit` 等接口，
  完整 URMA 编译未通过。MTE 编译使用仅位于 build 外部验证脚本中的版本 shim 排除 URMA；
  产品代码未添加依赖绕过。
- 本机没有 NPU；设备上的端到端精度、同步和性能尚未验证。上板应比较普通分离权重输入
  的 A8W8/A8W4/A4W4 输出，并覆盖多专家、多个 wave、M 尾块、prefetch 和 shared 混合配置。
