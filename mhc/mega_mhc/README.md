# mega_mhc on Ascend 950

950 专用 m-k 基线：N=4、2560≤D≤7168 且 D 为256的倍数，FP16/BF16 输入，FP32/HF32 投影。
扩展范围目前为实验实现，数值验收例外见[通用 D 分块实验](docs/generic_d_tiling_experiment.md)。
`gamma_0` 必须非空；gamma_2=None 为四算子，提供 [4D] gamma_2 为五算子。
保留四个输出及阶段舍入点，无 Sinkhorn、3BF16、bs/nd 调度。

## 目录与调用链

- op_host/op_api：aclnnAiInfraMegaMhc 参数校验、Contiguous、L0 launch、ViewCopy。
- op_host/op_tiling：GE总注册与arch35 host分块；memory_plan逐buffer计数，TilingData负责下发。
- op_kernel/arch35/compute：五阶段RegBase数学函数，公式与舍入点写在函数注释中。
- op_kernel/arch35/schedule_mk.h、epilogue.h：四/五调度、行归属、同步和最终归约。
- op_kernel/arch35/compute/projection_cube.h：Cube Basic API微流水，读取host给定L1/L0/Mmad/LoadData参数。
- op_kernel/arch35/transfer.h：GM、UB→L1双槽传输及READY/credit协议。
- torch_extension：仓内自动收集的mega_mhc Python、Meta、PrivateUse1/JIT C++封装。
- tests：host分块、Python metadata合约、NPU数值及确定性测试。

FP16/BF16 共用同一套 RegBase 计算与归约顺序，仅加载/写回的类型转换不同。
RMS0/RMS1 平方和使用双链累加；Post 内融合的 RMSMid 平方和按 head 0/1、2/3
先用 FMA 求局部平方和再累计，复用已消费的 Post 寄存器，仅保留一个累加寄存器。
该顺序缩短长累加链，无 BF16 专用补偿分支。

## 当前调度

四算子由 Host 按1024元素partial的整除性选择2或4路K分核，Mtile=32*kGrid、L1K上限=512/kGrid。
仅当完整批、行归属与 UB 容量允许时启用，不匹配特定 D。
M=3072、32 AIC/64 AIV 时，D2560采用M64/K2/L1K256，其余三个目标D保持M128/K4/L1K128，
均为每批1024行。每个AIV的projection基本块都是8192元素。
每个 AIV 每批生成16个完整 Post 行，首批预先生成，后续批次与前一批
投影穿插执行。Post 原位复用已消费的 residual UB 槽；两份输入缓冲隐藏下一行搬运。
Post 整行异步写回，在槽复用前等待 MTE3 完成，完成计数按 MTE3 顺序发布。
每个 M 分组通过完成计数等待全部 Post 行就绪，再读取对应的 x2。
Host 按最短 K 分片算每次推进行数和间隔，保证下一批读取前全部行已产生。
D7168 的 producer+projection 峰值为247.75 KiB；不满足条件时采用阶段屏障。

五算子先按完整行执行 RMS0/Post/Mid，输入双缓冲，Post 与 residual 原位复用，
Mid 保留阶段舍入，完成后写出 x2。gamma2 在容量允许时整行常驻，否则按 Host 规划分块加载；
平方和仍覆盖完整4D，只计算一次 invRms。D5120 的 producer UB 为211.25 KiB。
之后按 m-k 投影；目标配置为 M192/K2。四、五算子均在投影完成后统一执行 Pre/RMS1，
当前默认不启用跨批 epilogue 重叠。

所有 Cube 分块、循环次数和尾块参数由 Host 下发；partialK 上限保持1024。
GM staging 提交写回后，延迟到复用 UB 前等待完成；READY 在 MTE3 写回后发布，
Cube 消费后归还 credit。空行子核参与协议，退出时回收剩余 credit 与写回事件。

Epilogue 合并 RMS 分片 DMA，与 MM partial 共用完成等待，按 Host 规划的 reduceRows
处理系数。容量允许时，归并后的 partial UB 复用为第二份 x2 输入，输出也交替计算与写回。
wave的epilogue按每个AIV的token数和自身UB容量选择批量，不受projection行块大小限制。
RMS0/RMS1 的双链累加和 Mid 平方和逻辑保持不变，FP16/BF16 共用数学函数。
所有阶段的输入类型舍入点均保留。

## 范围与限制

GM 是当前验证过的默认传输路径。UB→L1 传输实现作为独立策略保留，
切换路径需要同步核对调度模板、Host 预算并重新验收。
UB 按同时存活的分配逐项256B对齐检查，但尚未引入完整的 Host offset/lifetime 布局。
Producer 仍使用完整行，尚无 D 流式分块；Cube 接口仍接收完整 TilingData。

见[Torch接口](docs/torchapi_mega_mhc.md)和[测试说明](tests/README.md)。
