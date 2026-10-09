# mega_mhc on Ascend 950

950 专用 M-K/ND/BS 实现：N=4、2560≤D≤7168 且 D 为256的倍数，FP16/BF16 输入，FP32/HF32 投影。
扩展范围目前为实验实现，仍存在未通过精度阈值的案例，不代表全范围已通过数值验收。
`gamma_0` 必须非空；gamma_2=None 为四算子，提供 [4D] gamma_2 为五算子。
`pre_mix` 位于 `gamma_0` 前，`h_pre` 位于 `x_2_out` 后；两者均为可选预留字段，
本版要求 pre_mix=None、h_pre 输出指针为空。Torch 返回 `(h_in_prime,h_post_new,h_res_new,x_2_out,None)`。
未来二者约定 FP32 ND `[T,4]` 或 `[B,S,4]`，pre_mix 不支持非连续；本版不启用相关计算或写出。
原四个输出及阶段舍入点保持不变，无 Sinkhorn、3BF16。

## 目录与调用链

- op_host/op_api：aclnnMegaMhc 参数校验、Contiguous、L0 launch、ViewCopy。
- op_host/op_tiling：GE总注册与arch35 host分块；memory_plan逐buffer计数，TilingData负责下发。
- op_kernel/arch35/compute：五阶段RegBase数学函数，公式与舍入点写在函数注释中。
- op_kernel/arch35/schedule_mk.h、schedule_bs.h、schedule_nd.h、epilogue.h：四/五调度、行归属、同步和最终归约。
- op_kernel/arch35/compute/projection_cube.h：Cube Basic API微流水，读取host给定L1/L0/Mmad/LoadData参数。
- op_kernel/arch35/transfer.h：GM、UB→L1双槽传输及READY/credit协议。
- torch_extension：仓内自动收集的mega_mhc Python、Meta、PrivateUse1/JIT C++封装。
- tests：host分块、Python metadata合约、NPU数值及确定性测试。

FP16/BF16 共用同一套 RegBase 计算与归约顺序，仅加载/写回的类型转换不同。
RMS0/RMS1 平方和使用双链累加；Post 内融合的 RMSMid 平方和按 head 0/1、2/3
先用 FMA 求局部平方和再累计，复用已消费的 Post 寄存器，仅保留一个累加寄存器。
该顺序缩短长累加链，无 BF16 专用补偿分支。

## 模板选择

调度模板由 Host 内部决定，不属于公开属性。本版保持默认 M-K；ND/BS 实现保留，
自动选择策略尚未启用。三种模板通过 tiling key 区分，公共计算继续复用。
公开属性严格依次为 `norm_eps`、`mhc_pre_norm_eps`、`hc_eps`、`compute_mode`。
`mhc_pre_norm_eps` 仅用于 Pre 投影归一化；其余 RMS 阶段使用 `norm_eps`。
更正后的 ACLNN/Torch 接口需要同步重新编译，不再提供调度参数的 V2 接口。

- ND：`mGrid=1`，沿 `K=N*D` 分核；所有 K 分片共同计算同一批 token，最后按 K 顺序归约。
- BS：`kGrid=1`，沿 token 分核；每核遍历完整 K，按既有规则输出 1024 长度有序 partial。
- M-K：保留已有二维分核及四算子跨批流水。

ND/BS 的 `Mcap=alignDown16(min(256, L0C/(2*32*4), L1A/(2*128*4)))`。
ND 取 `Mtile=min(Mcap,alignUp16(T))`；BS 取 `Mtile=min(Mcap,alignUp16(ceil(T/C)))`。
BS 的 `mGrid=min(C,ceil(T/Mtile))`；ND 的 K 路数先取 `min(C,ceil(K/256))`，
再依照共同的 K 对齐/partial 规则去掉空分片。其余 L1/L0 块与尾块由同一 CubePlan 计算。
这里 C 为可用 AIC 数，L1A 为 A 操作数双槽总预算；无 D 特化。

ND/BS 首版在完整 RMS0/Post/可选 Mid 后同步，投影内部继续 CV 双缓冲，
投影完成后同步并复用 Pre/RMS1 尾处理。与 mhc_pre 对齐的是分核方向，
尚未迁移其 ND 特有跨核流水或 BS 每个 token 块独立完成 epilogue 的同步方式。
不声称跨模板逐位一致，也不因 BS 分核而新增跨 batch 一致性承诺。

## 当前调度

外层调度按模板分为 `schedule_mk.h`、`schedule_bs.h`、`schedule_nd.h`，
各自负责阶段顺序、TPipe 生命周期和阶段屏障。公共投影循环及 producer 辅助函数位于
`schedule_common.h`；RegBase 数学逻辑继续由 `compute/` 复用。
M-K 的跨批 Post producer 和完成计数协议归属 `schedule_mk.h`。

`mega_mhc_tiling_key.h` 使用仓内 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 形式，
Host 通过 `GET_TPL_TILING_KEY(TILING_MODE, WITH_MID, USE_HF32)` 选择实例。
三个维度分别为 M-K/ND/BS、是否包含 RMSNorm Mid、是否使用 HF32，共12种组合。
Kernel 入口和 Cube 精度选择使用编译期分支；TilingData 只传递分块、容量和运行参数，
不再重复存储这些模式字段。输入 FP16/BF16 仍由算子 dtype 实例区分。

四算子由 Host 按 T 动态选择每个 AIV 每批的行数 R：在不超过16的2次幂中，
选择满足每批 `2*C*R` 行、尽量保留至少两批的最大值。
若最小批能整除T，再逐级缩小R使整批覆盖；若整批候选不合法，则尝试容量上限批量并保留尾批。
Host下发尾批有效行数。
C取可用AIC数；不为开启流水而缩核，小T条件不满足时保留阶段调度。
固定projection块预算8192元素；`W=8192/R`，K分核候选为 `gMax=W/128` 和 `gMax/2`。
比较最忙核工作量 `g*ceil(P/g)`，其中 `P=ceil(K/1024)`；取较小者，相同保留较大M块。
首选候选因K对齐缩减核数、无法保留完整行分工时，再检查另一个候选；
顺序partial按商余数分配，保留可用K路，不先按最大分片压缩核数。
Mtile=2*R*g，L1K上限=W/g；常规核组使用M64/128。
核数受可用核数约束，与输入 n 无关；详见通用分块实验记录。
仅当行归属、流水迭代数与 UB 容量允许时启用，不匹配特定 D。
M=3072、32 AIC/64 AIV 时，D2560采用M64/K2/L1K256，其余三个目标D保持M128/K4/L1K128，
均为每批1024行。每个AIV的projection基本块都是8192元素。
T=1024时R=8，每批512行；T≥2048时R上限为16，每批至多1024行（32 AIC），整除规则可进一步减小。
每个 AIV 每批生成R个完整 Post 行，首批预先生成，后续批次与前一批
投影穿插执行。Post 原位复用已消费的 residual UB 槽；两份输入缓冲隐藏下一行搬运。
Post 整行异步写回，在槽复用前等待 MTE3 完成，完成计数按 MTE3 顺序发布。
每个 M 分组通过完成计数等待全部 Post 行就绪，再读取对应的 x2。
Host提供尾批完整producer数与最后producer的有效行数；无有效行成员跳过数据访问，
仍发布完成计数并参与全部同步。T<=16保留非wave路径，无法形成两个最小M块。
Host 按最短 K 分片算每次推进行数和间隔，保证下一批读取前全部行已产生。
D7168 的 producer+projection 峰值为247.75 KiB；不满足条件时采用阶段屏障。
完成标志复用RMS0 workspace，Host保证该区域至少容纳每个AIV的128字节槽。

五算子先按完整行执行 RMS0/Post/Mid，输入双缓冲，Post 与 residual 原位复用，
Mid 保留阶段舍入，完成后写出 x2。gamma2 在容量允许时整行常驻，否则按 Host 规划分块加载；
平方和仍覆盖完整4D，只计算一次 invRms。D5120 的 producer UB 为211.25 KiB。
之后按 m-k 投影；目标配置为 M192/K2。四、五算子均在投影完成后统一执行 Pre/RMS1，
当前默认不启用跨批 epilogue 重叠。

所有 Cube 分块、循环次数和尾块参数由 Host 下发；partialK 判断与 mhc_pre 一致：初始 singleCoreK 按256对齐，达到2048时按1024分段累加，否则按完整单核 K 分片累加。
GM staging 提交写回后，延迟到复用 UB 前等待完成；READY 在 MTE3 写回后发布，
Cube 消费后归还 credit。空行子核参与协议，退出时回收剩余 credit 与写回事件。

Epilogue 按每核目标行数 `ceil(T/activeAiv)` 限制批大小，非wave路径同时保留projection行块上限。
先按 UB 容量求上限 F，再将目标行数 R 均分到 `ceil(R/F)` 批，
每批按8行向上对齐且不超过F。该规则不判断具体 T/D，避免固定批大小造成核间负载不均。

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
