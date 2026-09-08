# TurboQuantSparseFlashAttention 实现说明

输入格式与调用约束见[算子说明](../README.md)。

## 反量化与缩放

令码本解码后的归一化 latent 为 $\hat{K}_t$，token 的归一化系数为 $s_t$。真实 latent 为 $s_t\hat{K}_t$，存储的 rope 为 $R_t/s_t$。QK 计算后按 KV 列乘以 $s_t$，得到：

$$
s_t\left(Q_{latent}\hat{K}_t^T + Q_{rope}(R_t/s_t)^T\right)
= Q_{latent}(s_t\hat{K}_t)^T + Q_{rope}R_t^T
$$

当前 DSA 路径采用 MLA-absorb 结构，K/V 共用 latent。softmax 后将概率的对应列乘以 $s_t$，再与归一化 latent 相乘，等价于使用反量化后的 V 计算输出。

## 分页寻址与搬运

页地址按实际 `stride(0)` 计算，使用 64 位字节偏移，以支持页间 padding 和首轴非连续。合并搬运的源间距超出 DMA 参数范围时，退回单条搬运。
