**QBSA common / VF 与 master 对齐及专用实现说明**

本次改动以 `master` 提交 `b39ebe3a4a6952813d569e53e4516d7c705a8d01` 为直接基线。通用实现来自该提交的 `attention/common/op_kernel`；专用函数来自同一提交已有的 QBSA 实现。目录保持基线的 `op_kernel/arch35/common` 和 `op_kernel/arch35/vf` 布局。

通用代码保留原文件名；QBSA 扩展使用原文件名加 `_QBSA.h`。本轮涉及 4 个专用头文件、36 个函数定义（包括一个转调主线的普通 FlashUpdate wrapper）。基础 VF 的常量和 CastTrait 可直接复用，因此无需新增 `vf_basic_block_utils_QBSA.h`。

**独立头文件是维护边界，不是算法的硬性要求。** 应分别判断是否存在必须保留的行为差异，以及主线接口能否表达这些差异。本轮优先保留基线已有的函数体，故专用文件中也存在可进一步收敛的重复 helper 和薄封装。

| 专用文件 | 实现内容 | 是否需要专用逻辑 |
|---|---|---|
| [CopyInL1_QBSA.h](/D:/wk_code/QBSA/ops-transformer_plan/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/common/CopyInL1_QBSA.h) | MXFP8 scale 的 GM→L1 搬运、PA 查表寻址 | 底层搬运可复用；PA scale 地址和步长策略目前需要保留 |
| [matmul_QBSA.h](/D:/wk_code/QBSA/ops-transformer_plan/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/common/matmul_QBSA.h) | 参数构造、MX 调用封装、跨 subLoop 复用 L0B | 前两项可收敛；维持当前 L0B 复用调度时需要第三项行为 |
| [vf_flashupdate_new_QBSA.h](/D:/wk_code/QBSA/ops-transformer_plan/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/vf/vf_flashupdate_new_QBSA.h) | 普通 FlashUpdate 适配、空行归一化和 LSE | 普通更新已复用；空行结果约定需要保留 |
| [vf_mul_sel_softmaxflashv2_cast_nz_dn_QBSA.h](/D:/wk_code/QBSA/ops-transformer_plan/experimental/attention/quant_block_sparse_attn/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_QBSA.h) | FP8/MXFP8 softmax、scale/mask 布局、尾块处理、P/PScale 写出 | 布局和状态更新策略需要保留，不要求永久维护整份重复循环 |

在 **CopyInL1_QBSA.h** 中，4 个函数的必要性不同。

| 函数 | 实现了什么 | 与 master 的关系及独立必要性 |
|---|---|---|
| `MxCopyScaleToL1Nd2Nz` | 将两个 E8M0 元素作为一个 BF16 搬运单元；保持行数，列数和源行步长除以 2；允许指定 L1 C0 stride | 不能直接改名为 `CopyScaleToL1Nd2Nz`，后者将 nValue 除以 2。可以通过参数适配复用 `DataCopyGmScaleNDToL1`，不必保留独立的 DataCopy 参数构造算法 |
| `MxCopyScaleToL1Dn2Nz` | Q/K scale 的 DN→NZ 搬运；将 nValue 和源步长除以 2，按 BF16 粒度拷贝 | 与 `CopyScaleToL1Dn2Nz` 相同，差异为名字、注释和无符号整数 `>> 1` / `/ 2` 写法。**不必独立，可以直接使用主线函数** |
| `MxGmFlatScaleCopyInToL1PAForDN` | 根据 block table 查询物理页，处理跨页和尾块，计算 K scale 的 GM 地址，再按 DN→NZ 写入 L1 | 主线 `GmScaleCopyInToL1PAForDN` 的地址公式和 shape 单位不完全相同，不能只更名替换。需要保留寻址策略，底层 DN 搬运可复用 |
| `MxGmFlatScaleCopyInToL1PAForND` | 对 V scale 执行同类 PA 遍历；每个 scale 行对应 64 个 value token，按指定对齐行数写入 L1 | 主线 `GmScaleCopyInToL1PAForND` 的 ND 分支采用 `gmTensor[offset * 2]`，并以 copyRowCnt 作为对齐行数；QBSA 采用显式 scale 步长和 copyRowNumAlign。需要保留布局适配，底层 ND 搬运可复用 |

`MxCopyScaleToL1Nd2Nz` 可以映射到 `DataCopyGmScaleNDToL1` 的参数为：`rowAct=nValue`、`rowAlign=dstNzC0Stride`、`col=dValue >> 1`、`colStride=srcDValue >> 1`。主线接口接收非 const 张量引用，封装还需适配引用形式；两者最终均将 scale 张量重解释成 BF16 后执行 DataCopy。

两个 PA helper 保留的关键地址公式为：`physicalBlock * pageStride + head * blockSize * rowStride + rowInBlock * rowStride + dIdx`。page、row 和 head 的单位必须按 scale 张量解释。未来可将 PA 遍历与 scale 地址策略分开，使 QBSA 只提供布局参数。

在 **matmul_QBSA.h** 中，前两个函数没有独立的矩阵乘计算逻辑。

| 函数 | 实现了什么 | 必须独立吗、如何复用 |
|---|---|---|
| `MxMakeMMParam` | 填写 M/N/K、转置、累加初值、unitFlag 和 realM | **不必独立。** 主线已有 `MakeMMParam`，额外的 loadK/loadN 默认均为 0，与当前 QBSA 构造结果一致 |
| `MxMatmulFull` | 将模板类型、数据张量、scale 张量和 MMParam 原样转交给 `MatmulFull` | **不必独立，计算已经复用。** 可保留 MX 命名入口，也可让调用者直接使用主线函数 |
| `MxMatmulFullReuseB` | 由 LOAD_B / RELEASE_B 控制 L0B 的取得、加载和释放；后一轮通过 GetPre() 使用保留的 B；每轮更新 A 并执行 Mmad | **维持现有复用调度时需要保留该策略。** 主线 MatmulFull 每次取得、加载并释放 A/B，没有表达跨调用保留 B 的接口。可扩展主线策略，或取消该优化并重新安排调用和同步 |

QBSA 的 BMM1 使用 `K_subLoop × Qᵀ`，此处复用的 B 实际是 Q。第一轮加载并保留 Q，第二轮复用后释放，可减少重复的 L1→L0B 搬运。数学计算不要求必须复用 L0B，但当前调用协议和同步事件按此生命周期配套，不能仅替换为无条件释放 B 的函数。

在 **vf_flashupdate_new_QBSA.h** 中，普通更新与末块归一化的复用情况不同。

| 函数或函数组 | 实现了什么 | 必须独立吗、如何复用 |
|---|---|---|
| `FlashUpdateNew` | 保留 QBSA 旧接口，调用主线 `FlashUpdateNew<..., isMlaFullQuant=false>` | **不必独立，已经复用。** 为未使用的 rowMaxTensor 参数传入有效的 expMaxTensor，因为主线 wrapper 会取其地址；也可在调用点直接适配 |
| `FlashUpdateLastBasicVF`、`FlashUpdateLastGeneralVF` | 重缩放前一块结果，加上当前块，再除以最终 sum；分别处理按 64 对齐和有 D 尾部的情况；通过 zeroWhenSumZero 控制空行输出为 0 | 需要保留空行输出策略，当前主线相应接口不能直接表达该差异。可增加策略参数后共享主体 |
| `FlashUpdateLastBasic`、`FlashUpdateLastGeneral`、`FlashUpdateLastNew` | 获取 UB 地址、准备尾部参数，选择 Basic/General VF | wrapper 没有必须独立的计算逻辑，随底层空行策略保留或收敛 |
| `LastDivNewVF`、`LastDivNew` | 单个 S2 tile 的最终归一化，sum 为 0 时选择零输出 | 需要保留空行行为，可在主线增加可选策略；wrapper 不要求独立 |
| `ComputeLseOutputVF` 的两个重载 | 计算 `log(sum) + max`；识别 max 哨兵或 sum 为 0 的空行，一个重载实现寄存器操作，一个提供 LocalTensor 接口 | **空行结果约定需要保留。** 主线在 max 命中哨兵时返回 +Inf；QBSA 当前调用使用 `<float, true>` 返回 -Inf，并额外检查 sum 为 0。可合并为可配置空行结果的通用接口 |

当前普通 FP8、MXFP8 两处 LSE 调用均选择 `negativeInfWhenEmpty=true`。模板的 false 分支仍保留 -FLT_MAX 返回约定。内部 max 的有限哨兵和最终 LSE 的 -Inf 分别承担不同作用，合并时需要分别保留。

这里迁移的是基线已有的算术顺序：Div/Log 后执行对应的结果选择。本轮没有增加新的算术保护，也没有改变正常行的操作顺序。不能把源码中的空行输出处理当作所有硬件执行情况均已验证的结论。

在 **vf_mul_sel_softmaxflashv2_cast_nz_dn_QBSA.h** 中，19 个函数按路径整理如下。通用 exp、归约和 cast 指令可复用，但 scale、mask、历史状态和输出布局共同决定当前接口。

| 函数或函数组 | 实现了什么 | 必须独立吗、如何复用 |
|---|---|---|
| `ProcessVec1DnNoUpdateVF`、`ProcessVec1DnUpdateVF` | 普通 FP8 DN：逐行 Q scale、逐列 K scale 反量化，应用 mask，计算 softmax/在线 max/sum，量化并打包 P；Update 合并前序块状态 | 主线普通 DN 接收标量 deScaleQK，不能直接表达 QBSA 的 Q/K scale 张量布局。需要保留 scale/mask 策略，可扩展通用接口后合并 |
| `ProcessVec1DnNoUpdateAttenTailVF`、`ProcessVec1DnUpdateAttenTailVF` | 对 originN < ubN 的尾块补列；缩放前使用有限安全值，缩放后恢复 minValue | 需要保留尾块与缩放的处理顺序；可提取为通用尾块策略，不要求两份完整 VF |
| `ProcessVec1DnNoUpdate`、`ProcessVec1DnUpdate`、`ProcessVec1VfDn` | 获取 UB 指针及 Q scale gather 索引，按是否更新、是否需要 mask、是否有尾块选择 VF | 属于接口和分发适配；底层策略统一后可收敛，同名主线入口目前参数不同 |
| `ProcessVec1DnNoUpdatePerTokenHeadVF`、`ProcessVec1DnUpdatePerTokenHeadVF` | NTD 调用路径的 per-token/head scale 版本，使用连续 K scale 和外部 mask，按 64 行物理布局处理 | 主线 PerTokenHead 使用 NUM_264 分段 K scale 读取及 maskLine 生成的 mask，当前不能直接替换。可将 scale 读取和 mask 来源参数化 |
| `ProcessVec1DnNoUpdatePerTokenHeadAttenTailVF`、`ProcessVec1DnUpdatePerTokenHeadAttenTailVF` | 在 PerTokenHead 布局中补充尾列安全填充和缩放后 minValue 恢复 | 尾块策略需要保留，可与普通 DN 共用 helper |
| `ProcessVec1DnNoUpdatePerTokenHead`、`ProcessVec1DnUpdatePerTokenHead`、`ProcessVec1VfDnPerTokenHead` | 提供 PerTokenHead 的 LocalTensor 接口及完整块/尾块、mask、更新分发 | 属于适配封装，底层接口收敛后可一起收敛 |
| `ProcessVec1DnNoUpdateMxfp8VF`、`ProcessVec1DnNoUpdateMxfp8` | 首个逻辑 S2 tile 的 MXFP8 softmax，按编译期 SUB_LOOP=0/1 处理两个部分，输出 FP8 P 与 E8M0 PScale，合并 max/sum | 需要保留外部稀疏 mask、物理布局、PScale 打包及编译期分发；主线支持这些策略后可共享主体 |
| `ProcessVec1DnUpdateMxfp8VF`、`ProcessVec1DnUpdateMxfp8` | 在已有逻辑 S2 历史上更新 max/sum 和 P/PScale，保存进入逻辑 tile 前的 max/sum 及第一个 subLoop 的 sum | 需要保留所选历史状态协议，合并时同步核对调用者的状态 buffer；不要求完整循环永久独立 |
| `ProcessVec1VfDnMxfp8` | MXFP8 入口，获取索引张量，传入 preMax/preSum/firstSum 并选择首轮或更新版本 | 属于调用协议适配，可保持薄入口并共享通用主体 |

几个容易误判的差异需要明确。

1. **Q/K scale 布局。** 普通 DN 中有 Q scale gather，索引 buffer 为 QBSA 的第 4 号槽位；PerTokenHead 直接加载。QBSA 的 K scale 地址是连续的 `4*i + {0,1,2,3}`，主线 PerTokenHead 对后三组还分别增加 264、528、792 的偏移。没有同时转换 scale buffer 布局时，同名 VF 不能互换。
2. **mask 来源。** 主线普通 DN 已支持外部 mask。主要差异在于 QBSA 各路径使用外部生成的稀疏/部分可见 mask，而主线 PerTokenHead/MXFP8 通过 maskLine 生成规则边界。单个 maskLine 无法一般性表示稀疏块拼接后的全部可见性关系。
3. **尾列和空行。** `VF_QBSA_SAFE_PAD_VALUE=-25690112` 用于 FP8 AttenTail 的缩放前填充，缩放后恢复 minValue；空行 max/sum 与 LSE 输出还有配套约定。这些是保留处理顺序的理由，不代表必须复制完整 softmax。
4. **MXFP8 布局。** QBSA 使用 mConst=64，当前将 512 列逻辑 tile 分成两个 256 列 subLoop；主线 MX 入口保留运行时 m/subLoop 参数。QBSA 的 P 第二路写指针为 `x_exp + (ubN >> 3)`，当前 ubN=256 时就是主线 FP8 路径的 +32，因此这个表达式差异本身不构成当前实例必须独立的理由。
5. **PScale 生成。** subLoop 0 写单位 scale 编码 0x7f；subLoop 1 根据 max 变化生成缩放系数，经 BF16 舍入、E8M0 Cast、Or/DeInterleave 后写出。基础 CastTrait 直接复用主线；打包方式需要与 C2 读取的布局一致。
6. **sum 的含义。** 当前先将 FP32 exp 结果累加到 sum，再把 P cast 为 FP8 写出；没有将 FP8 P 反量化后重新计算分母，不能用“分母与量化后的 P 完全一致”解释独立必要性。

后续减少专用代码可按以下顺序推进。本说明仅记录建议，没有执行这些进一步改动。

| 顺序 | 可收敛内容 | 前提 |
|---|---|---|
| 1 | MxCopyScaleToL1Dn2Nz → CopyScaleToL1Dn2Nz；MxMakeMMParam → MakeMMParam | 参数和默认值核对后，更新调用并编译检查 |
| 2 | MxMatmulFull、普通 FlashUpdateNew 薄封装 | 可保留命名入口，也可由调用点直接传主线参数；删除封装名不会减少新的计算逻辑 |
| 3 | ND scale 搬运调用 DataCopyGmScaleNDToL1 | 适配搬运单位和张量引用，核对 L1 实际结果 |
| 4 | 空行归一化/LSE 策略下沉通用接口 | 保持主线默认行为，覆盖 sum=0、max 哨兵、D 尾部和 LSE 行尾 |
| 5 | PA 地址、scale/mask/尾块及 L0B 生命周期策略 | 同时核对生产者、消费者和同步协议，完成编译、精度及性能回归 |

独立文件解决维护边界，接口适配仍需完成。例如当前保留的 PA helper 访问 `shape.rowStride`，而复制来的主线 PAShape 不含该字段；`MxMatmulFullReuseB` 保留的 `LoadDataToL0AMx<A, L0ADType>` 调用还需适配主线要求的 baseK 模板参数。这些属于主线替换后的整合工作，不能以文件拆分代替编译和设备验证。

本说明依据所列 master 基线的源码和调用点。函数迁移、参数及通用文件一致性已做静态核对，尚未进行 CANN 编译或 NPU 精度/性能回归。
