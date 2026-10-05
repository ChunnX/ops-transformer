# 与 CUDA 默认接口的差异

Ascend 950 实现采用 **`per_token` Q/K INT8 量化、每 channel 每 32 个 token 的 V MXFP8 量化，以及 `fp32+fp32` 两级累加**。相对于 CUDA 通用 FP8 接口默认的 `fp32+fp16` 路径，本实现的 PV 局部累加精度更高；Q/K 和 V 的 scale 分组也更细。以下区分接口参数、实际计算类型和硬件路径。

## 对照版本与默认接口

CUDA 源码来自 [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)，核对日期为 2026-09-14，固定提交为 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`。本说明针对该提交的 SageAttention2 INT8 QK / FP8 PV 路径。

CUDA 的 `sageattn` 会按 GPU 架构选择实现，不能把某一个分支的默认参数推广到全部 GPU：

| 入口或分支 | 默认 Q/K 量化参数 | 默认 PV 累加参数 |
|---|---|---|
| `sageattn_qk_int8_pv_fp8_cuda`；`sageattn` 的 SM89 分支 | `per_thread` | `fp32+fp16` |
| `sageattn_qk_int8_pv_fp8_cuda_sm90`；`sageattn` 的 SM90 分支 | `per_thread` | `fp32+fp32` |
| `sageattn` 的 SM120 / SM121 分支 | `per_warp` | `fp32+fp16` |
| 本实现 `sageattn_qk_int8_pv_fp8_asc`；NPU `sageattn` | `per_token` | `fp32+fp32` |

上游通用 FP8 函数的 docstring 仍写默认 `fp32+fp32`，但函数签名和 SM89 调度实际传入 `fp32+fp16`；此处以执行代码为准。SM80 / SM86 默认选择 FP16 PV，属于另一条计算路径。

## PV 累加类型：为什么使用 fp32+fp32

本实现分两级保留累加状态：

1. Cube 执行 MXFP8 PV 矩阵乘，局部结果保留在 FP32 L0C 中，并以 FP32 搬到 Vector UB。
2. Vector 用 FP32 维护跨块输出，执行 online softmax 对应的 `O = alpha × O + PV`，分母及最终归一化也使用 FP32。最终写回时才将 O 转为输入对应的 FP16 / BF16。

因此公开参数默认值使用 **`pv_accum_dtype="fp32+fp32"`**，表明 PV 局部累加和跨块输出累加均为 FP32。旧值 `"fp32"` 是无告警的兼容别名，调用相同实现；它不会切换到另一种单级累加 kernel。其他值沿用兼容回退行为：Python 发出 `UserWarning`，独立 ACLNN 发出 `OP_LOGW`，实际仍使用 `fp32+fp32`。

CUDA 参数字符串也需要结合 kernel 理解：该版本 `fp32+fp16` 分支调用 `compute_fp8_sv_inst_buf_fp16_accu`，其 PV MMA 使用 FP16 局部累加，将结果解包为 float 后加入 FP32 输出状态；不是“先用 FP32 MMA，再把输出总和降为 FP16”。`fp32+fp32` 分支则使用 FP32 PV MMA 局部缓冲，再加入 FP32 输出状态。

**相对于 CUDA 的 `fp32+fp16` 默认分支，NPU 在 PV 局部累加阶段使用更高精度，避免该阶段的 FP16 舍入和动态范围限制。** CUDA 的 SM90 默认已经采用 `fp32+fp32`，两者在软件累加类型上都是两级 FP32，不能仅凭参数名宣称 NPU 比该分支精度更高。上游还记录了部分 FP8 MMA / WGMMA 路径的有效累加位数限制；具体硬件有效位数不能只由源码里的 `float` 类型推导。

两级 FP32 描述的是累加状态，Q/K 仍量化为 INT8，P/V 仍量化为 FP8，score/max 仍有 FP16 处理。它不等于完整 FP32 Attention，也不保证任意输入的最终误差都小于 CUDA。

## 量化粒度：Q/K 与 V 的 scale 分组更细

### Q/K：每个 token 独立 scale

NPU 为每个 batch、head、token 沿整个 head dimension 求绝对值最大值，生成一个 INT8 scale；不同 token 不共享该 scale。

CUDA `per_thread` 中的 thread 指与 warp / MMA 映射对应的交错 token 分组，不是“每个元素一个 scale”。按该版本实际调用参数，满分组的覆盖范围如下，尾组只统计有效 token：

| Q/K 分组 | 每个 Q scale 覆盖的 token 数 | 每个 K scale 覆盖的 token 数 |
|---|---:|---:|
| CUDA 通用 FP8 `per_thread`：WARPQ=32、WARPK=64 | 4 | 16 |
| CUDA SM90 FP8 `per_thread`：WARPQ=16、WARPK=128 | 2 | 32 |
| CUDA 通用 FP8 `per_warp`：WARPQ=32、BLKK=64 | 32 | 64 |
| CUDA SM90 FP8 `per_warp`：WARPQ=16、BLKK=128 | 16 | 128 |
| NPU `per_token` | 1 | 1 |

`per_thread` 的 Q 分组按 8 个 token 的步距取行，单个 scale 覆盖 `WARPQ/8` 行；K 分组按相邻两行、步距 8 取行，覆盖 `WARPK/4` 行。各组均覆盖完整 head dimension。因此这里的“更细”具体指 **scale 沿 token 维的共享范围更小**，减少某个 token 的大值对其他 token 量化步长的影响。

本实现接受 `per_thread` / `per_warp` 参数以兼容调用，但告警后实际执行 `per_token`。Ascend 的 AIV / Cube 分工和矩阵布局与 CUDA lane / warp / MMA 映射不同；使用这些字符串不会复现 CUDA 的分组、量化结果和舍入顺序。

### V：每 channel 每 32 个 token 独立 scale

CUDA `per_channel_fp8` 为每个 batch、KV head、channel 生成一个 FP32 scale，在该 channel 的整条序列上共享。NPU 则将序列按 32 个 token 分组，每个 channel、每组分别生成 E8M0 scale，V 数据为 E4M3；尾组补零。

例如 Sk=8192 时，CUDA 每个 channel 使用 1 个 scale，NPU 每个 channel 使用 256 个 scale。局部组可适应各自的数值范围，降低整条序列中离群值对其他组的影响。E8M0 scale 只能表示 2 的整数次幂，与 CUDA 的 FP32 scale 表示精度不同；更细的分组并不意味着每个输入都获得更小误差。

P 使用 E4M3 数据与固定单位 MX scale，概率幅度通过 softmax 的 448 平移方案处理。上述“更细粒度”针对 Q/K 和 V，不表示 P 也采用自适应的每 32-token scale。

## 其他接口差异与使用方式

| 项目 | CUDA FP8 路径 | 当前 NPU 实现 |
|---|---|---|
| `smooth_k` | 默认 `True`，对 K 减均值 | 默认 `True`，保留 K 平滑与 LSE 补偿 |
| 最终输出 | FP16 / BF16，与输入类型对应 | FP16 / BF16，与输入类型对应；中间两级 FP32 不改变输出 dtype |
| 计算组织 | CUDA 预处理量化后调用 attention kernel | 完整融合算子包含 Stats、Quant、Attention，内部按 Cube / Vector 流水调度 |

V 的局部分组量化是当前省去 V 均值统计、减均值及输出补偿的设计依据，可减少计算与访存；它不表示 V 平滑对任何分布都无益。

```python
from sageattention2 import sageattn_qk_int8_pv_fp8_asc

out = sageattn_qk_int8_pv_fp8_asc(
    q, k, v,
    qk_quant_gran="per_token",
    pv_accum_dtype="fp32+fp32",
    smooth_k=True,
)
```

默认调用已使用上述配置。已有调用中的 `pv_accum_dtype="fp32"` 无需修改即可继续运行，结果语义相同。更多形状、布局与功能限制见 [算子 README](../README.md)。由于 Q/K、V 的量化分组及数值处理不同，CUDA 同名参数不能直接视为 NPU 的同配置逐元素 golden。

## 源码依据

上游链接固定到本次核对提交：

- [core.py：架构调度、默认参数及实际调用分支](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py)
- [attn_utils.cuh：FP32 / FP16 局部 PV 累加与 FP32 输出缓冲](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/csrc/qattn/attn_utils.cuh)
- [quant_per_thread.py：Q/K 交错 token 分组](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/triton/quant_per_thread.py)
- [quant.py：per-warp Q/K 与 per-channel V](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/quant.py)

NPU 实现对应位置：

- [Python 默认接口](../python/sageattention2/__init__.py)、[Dispatcher 与参数兼容](../python/sageattention2/_ops.py)、[C++/torch_npu 接入](../python/sageattention2/csrc/attention.cpp)、[ACLNN 参数兼容](../op_api/aclnn_sage_attention2.cpp)
- [Cube PV 累加与 FP32 搬运](../op_kernel/attention_block_cube.h)、[Vector 输出累加与归一化](../op_kernel/output_update.h)
- [Q/K per-token 量化](../op_kernel/quant.h)、[V 每 32-token MXFP8 量化](../op_kernel/quant_v_mx.h)
