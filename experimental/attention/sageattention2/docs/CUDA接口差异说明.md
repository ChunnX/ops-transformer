# 与 CUDA 默认接口的差异

本文逐项声明 NPU 侧 `sageattn` / `sageattn_qk_int8_pv_fp8_asc` / 两段式 `aclnnSageAttention2` 相对 CUDA 上游 SageAttention2 INT8 QK / FP8 PV 路径的真实语义。所有 NPU 行为均给出本仓库可核验的源码位置；CUDA 侧行为仅在上游源码可核验的范围内陈述。

## 对照基准与范围

- CUDA 上游基准：thu-ml/SageAttention，核对提交固定为 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`（2026-09-14 核对），对应 `sageattention/core.py` 的 `sageattn_qk_int8_pv_fp8_cuda` 分支族
- NPU 侧声明范围：本目录 `./quant/`、`./sageattn_fp8/ascendc/`、`./aclnn/`、`./python/sageattention2/` 的当前实现
- 数值结论类断言（逐位对齐、误差界）一律以仓库根 golden 数据集（`golden/manifest.json`，随交付置备：`bash docker/init.sh --golden <数据集>`）的 TC 系列数据比对为准，本文不给出无实测出处的误差数字

## 量化粒度：per_thread 与 per_warp 均为真实现

NPU 侧两种粒度是两套独立存在的量化内核，不是兼容别名；这与"仅实现单一分组、其余字符串告警回退"的适配方式有本质区别。

- `qk_quant_gran="per_thread"` 真实分组（`./quant/quant_int8.cpp` per-thread 路径，x86 可单测）

- Q：每 128 token 块切 4 个 32-token 子块，每子块按步距 8 取 4 组，单 scale 覆盖 8 个交错 token 的完整 head dim，每块共 32 个 scale（`QScaleCount`，`./quant/quant_common.h`）
- K：每 64 token 块切 4 组，单 scale 覆盖 16 个交错 token（8 对相邻 token 对，`n mod 8 == 2g`）的完整 head dim，每块共 4 个 scale（`KScaleCount`，`./quant/quant_common.h`）
- 舍入为 half-away-from-zero（见下节）

- `qk_quant_gran="per_warp"` 真实分组（`./quant/quant_int8.cpp` per-warp 路径，RNE 舍入）

- Q：每 128 token 块 4 个 scale，单 scale 覆盖 32 个 token
- K：每 64 token 块 1 个 scale，单 scale 覆盖 64 个 token

- 与 CUDA 分组数对照（上游 `sageattention/triton/quant_per_thread.py`、`sageattention/quant.py`，WARPQ/WARPK/BLKK 为上游源码常量）

- CUDA 通用 FP8 `per_thread`（WARPQ=32、WARPK=64）：Q 单 scale 覆盖 4 token，K 覆盖 16 token
- CUDA SM90 FP8 `per_thread`（WARPQ=16、WARPK=128）：Q 单 scale 覆盖 2 token，K 覆盖 32 token
- CUDA 通用 FP8 `per_warp`（WARPQ=32、BLKK=64）：Q 单 scale 覆盖 32 token，K 覆盖 64 token
- NPU `per_warp` 的覆盖数与 CUDA 通用 per_warp 相同（32/64）；NPU `per_thread` 为 8/16，介于 CUDA 两种 per_thread 配置之间
- 结论：同名参数不构成逐位 golden。CUDA 的 thread/warp 分组与 lane、MMA fragment 映射绑定，NPU 的 AIV/Cube 布局不同；仓库根 golden 数据集的 TC-02 系列给出 NPU 侧两种粒度的逐位期望值

## S_FP8_OFFSET：P 矩阵按 log2(448) 精确放大

- NPU 语义：Softmax 全程在 exp2 域计算，running max 记为 $m$；P 矩阵按 $P = 2^{\text{score} - m + 8.807355}$ 生成，其中偏置常数 $8.807355 = \log_2 448$（`S_FP8_OFFSET`，唯一定义位于 `./sageattn_fp8/ascendc/tiling.h`；设备侧 P 量化注入点为 `./op_kernel/sage_device_vf.h` 的 `SoftmaxProbRowsVF` 与宿主基线 `./op_kernel/sageattention2.cpp` 的 `ComputeSoftmaxProbability`——448· 放大经乘法恒启用，LSE 出口恒以该常量补扣）
- 效果：P 落在 $[0, 448]$，直接按 e4m3 饱和量化，P 的量化 scale 固定为单位 1，省去逐元素乘 scale 的 Vector 开销
- 分母一致性：softmax 分母按同一放大尺度在线累计，归一化经 `OnlineSoftmaxTransformer::GetFinalInvDenominator`（O 与分母同处 448· 放大域，除法处放大精确相消，见 `./sageattn_fp8/ascendc/online_softmax.h`）；偏置抵消由 `./tests/test_kernel_twin.py` 的 golden 交叉验证覆盖
- 与 CUDA 的关系：上游 FP8 路径同样通过 448 平移/缩放方案把概率幅度送入 e4m3 表示域（见上游 `csrc/qattn/attn_utils.cuh`），数值设计目标一致；但上游在 FP16 域做 max 平移后再换算，本实现在 exp2 域以单次加法注入偏置，舍入轨迹不同，不承诺逐位一致

## PV 两级累加：NPU 三档，CUDA 两档

- NPU 档位（`PvAccumDtype`，`./sageattn_fp8/ascendc/tiling.h`；Host 侧实现 `./aclnn/sage_attn_host_tiling.h`，设备侧 PV 交接为 `./sageattn_fp8/catlass/sage_pv_mmad.hpp` 的 FixPipe 裸搬 + `MergePvPartialRowsVF` 档位合并，宿主真值 `./sageattn_fp8/ascendc/twin_reference.h` 的 `AccumulatePvRow`）

- `0`（`"fp32"`）：单级全程 FP32 累加，无第二级低精度 buffer；`smooth_v` 正路径仅在该档可用
- `1`（`"fp32+fp32"`）：Cube FP8 MMA 硬件累加，每 `k_step` 周期归约至 FP32 输出 buffer
- `2`（`"fp32+fp16"`）：每 `k_step` 周期归约至 FP16 buffer，融合 FixPipe 随路量化；默认验收档
- CUDA 上游仅有 `fp32+fp16` 与 `fp32+fp32` 两档；NPU 的 `"fp32"` 单级档为新增，三者互不别名，Python 侧非法档位直接 `ValueError`（`./python/sageattention2/error_check.py`），无静默回退

- 档位与 V 量化饱和上限联动（`GetVScaleMax`，`./quant/quant_host.h`）

- `"fp32+fp16"` 档：`v_scale_max = 2.25`（保护 FP16 归约 buffer 动态范围，常数 `SCALE_MAX_FP16_ACCUM`）
- 其余档位：`v_scale_max = 448.0`（e4m3 满幅，常数 `SCALE_MAX_FP32_ACCUM`）
- 该取值影响 V 量化 scale 与饱和截断半径，是跨档位结果不可互比的原因之一

- 附带的 NPU 侧机制：softmax 采用恒逐块强锚定重缩放——每到来一个 KV 分块必对 O 累加器执行重缩放（`OnlineSoftmaxTransformer::Step`，锚点 `m_scale` 恒等于 `m_run`），对齐 golden 的逐块锚定 e4m3 量化网格。此为 NPU 微架构设计，CUDA 无对应物，也不改变对外数值契约

## 舍入模式：half-away-from-zero 与 RNE 并存

- NPU `per_thread` INT8 量化：half-away-from-zero，对量化前数值的绝对值加 `0.5` 后向下取整、再乘回符号，随后饱和至 `[-127, 127]`（舍入原语 `RoundHalfAwayFromZero`/`RoundNearestEven` 唯一定义于 `./quant/quant_common.h`，注释明确区分两种模式）
- NPU `per_warp` INT8 量化：RNE（round to nearest even，`RoundNearestEven` 显式整数运算实现，不依赖 `std::nearbyint`——设备侧 <cmath> 无该符号，见 `./quant/quant_common.h`）
- 两种舍入在 x86 可单测内核与 AscendC 设备内核中逐路径保持一致；仓库根 golden 数据集（`golden/manifest.json`，随交付置备）的 TC-02 系列按粒度分别固化期望值
- 含义：即使分组数一致，不同舍入模式也会产生不同 int8 码本；任何 CUDA 逐位对齐声明必须先核对舍入轨迹，不能只看分组参数

## smooth_v：fp32 档真实正路径，其余档告警忽略

- `pv_accum_dtype="fp32"` 且 `smooth_v=True`：真实执行 V 平滑正路径

- Host/量化侧：`QuantizeVFp8Ascend` 沿序列维统计 V 均值 `vm` 并输出（`./quant/quant_fp8.cpp`），workspace 预留 `vm` 空间（TilingData `workspace_vm_offset`，`./sageattn_fp8/ascendc/tiling.h`）
- 计算侧：V 减均值后量化，输出端加回 `vm` 补偿
- 这是 NPU 特有路径：CUDA 上游 FP8 两级累加分支固定不做 V 平滑，`smooth_v` 参数在上游通用接口中会告警并被忽略

- `pv_accum_dtype` 为 `"fp32+fp32"` 或 `"fp32+fp16"` 时传 `smooth_v=True`：Python 层发 `UserWarning` 后置 False（`./python/sageattention2/error_check.py` 的 `handle_smooth_v`），独立 ACLNN 层等价拒绝，不静默改语义

## smooth_k 与 LSE 补偿

- NPU 语义：`smooth_k=True`（默认）时 K 沿序列维减均值 `km`（量化核 `QuantizeKInt8Ascend` 的 `km_in` 入参，`./quant/quant_int8.cpp`），并预计算 LSE 修正项 `lse_correction = Q * km^T`（融合在设备核内，`./op_kernel/sageattention2.cpp` 的 `lse_corr`）
- 对外 LSE 换算：`lse_out = (lse_exp2 - S_FP8_OFFSET) * ln(2) + lse_correction * sm_scale`（`ConvertLseToHost`，`./sageattn_fp8/ascendc/online_softmax.h`；即 exp2 域 LSE 先补扣 `S_FP8_OFFSET` 常量再除以 $\log_2 e$ 换回自然对数域），减均值引入的行常数在返回前精确补偿
- 意义：开启 smooth_k 后对外 LSE 与"不减均值"的参考实现可证等价；CUDA 上游 `return_lse` 语义仅在部分入口提供，且补偿路径实现不同，跨实现比较 LSE 时以本公式为准

## 差异速查（条目式对照）

- Q/K INT8 量化分组

- NPU `per_thread`：Q 每 scale 8 token / K 每 scale 16 token，两套真实内核
- NPU `per_warp`：Q 每 scale 32 token / K 每 scale 64 token，真实内核
- CUDA `per_thread`：Q 4（通用）或 2（SM90），K 16（通用）或 32（SM90）
- CUDA `per_warp`：Q 32 / K 64（通用配置）

- INT8 舍入

- NPU：per_thread 用 half-away-from-zero，per_warp 用 RNE
- CUDA：分组与 lane/warp 映射绑定，舍入轨迹依赖具体 kernel 分支，逐位不可移植

- P（概率）FP8 表示

- NPU：exp2 域注入 `S_FP8_OFFSET = 8.807355`，P 放大 448 倍后单位 scale 量化 e4m3，分母同尺度累计并在 Finalize 抵消
- CUDA：448 平移/缩放方案送入 e4m3，P 的 MX scale 固定单位

- PV 累加档位

- NPU：`fp32`、`fp32+fp32`、`fp32+fp16` 三档，无别名无回退
- CUDA：`fp32+fp16`、`fp32+fp32` 两档，不同 SM 默认值不同

- V FP8 量化

- NPU：沿 head dim 逐 channel，饱和上限按档位取 `2.25`（fp32+fp16）或 `448.0`（其余）
- CUDA：`per_channel_fp8` 逐 channel 单 scale，饱和域为 e4m3 满幅

- smooth_v

- NPU：`fp32` 档真实正路径（vm 统计 + 减均值 + 输出补偿），其余档告警忽略
- CUDA：FP8 两级累加分支不支持，参数告警后忽略

## 源码依据

- NPU 量化内核与舍入：`./quant/quant_int8.cpp`（half-away 与 RNE 两条路径）、`./quant/quant_fp8.cpp`（V e4m3 与 vm 统计）、`./sageattn_fp8/ascendc/tiling.h`（`S_FP8_OFFSET`、`SCALE_MAX_*` 等常量唯一定义）、`./quant/quant_common.h`（`QScaleCount`/`KScaleCount` 分组数公式）、`./quant/quant_host.h`（using 引入常量并提供 `GetVScaleMax`）
- 设备侧累加与偏置：`./sageattn_fp8/ascendc/online_softmax.h`（`GetFinalInvDenominator` 归一化）、`./op_kernel/sageattention2.cpp`（`ComputeSoftmaxProbability` 注入偏置）、`./op_kernel/sage_device_vf.h`（设备态 `SoftmaxProbRowVF` / `FinalizeRowVF`）
- Tiling 契约：`./sageattn_fp8/ascendc/tiling.h`（`PvAccumDtype`、`QkQuantGran`、`workspace_vm_offset`）
- 接口与告警语义：`./python/sageattention2/error_check.py`（`handle_smooth_v`、档位白名单）、`./python/sageattention2/dispatch.py`（`per_thread`/`per_warp` 双实现参考实现）
- 数值期望值：仓库根 golden 数据集的 TC-02（量化粒度）、TC-03（smooth_v/no_smooth_k）、TC-09（长序列）系列
- CUDA 上游分组常量（WARPQ/WARPK/BLKK）：thu-ml/SageAttention 提交 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` 的 `sageattention/triton/quant_per_thread.py`、`sageattention/quant.py`、`csrc/qattn/attn_utils.cuh`
