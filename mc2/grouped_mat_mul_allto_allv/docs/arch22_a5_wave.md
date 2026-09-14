# GMMA2AV arch22 Wave 调度

参考 `ops-transformer/mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h` 的有限 M-group Wave、独立 AIV 通信角色，以及 `stage/mega_moe_gmm2_combine.h` 的 `NotifyWaveGmm2Ready` GM 完成标记机制。

GMMA2AV 是 GMM down projection + AllToAllV，只有一段矩阵乘法，因此适配为下列流水；没有引入 MegaMoe 的 GMM1/激活/GMM2 算子语义。

```text
AIC   : GMM wave 0 ─ FIX_S/GM ready ─ GMM wave 1 ─ ... ─ optional shared MM
AIV0  :     wait wave 0 ─ stage rows ─ wait wave 1 ─ ... ─ publish expert progress
AIV1  :     wait local expert progress ─ publish peer ready ─ wait/pull peers ─ ...
```

## 分层

- `grouped_mat_mul_allto_allv_mte.h`：组合入口，保持调用方的通信类型名。
- `grouped_mat_mul_allto_allv_mte_wave.h`：AIC、AIV0、AIV1 的独立循环和 Wave 发布回调。
- `grouped_mat_mul_allto_allv_mte_catlass.h`：GMM 计算；跨专家保持一个 CATLASS resource 生命周期。
- `common/gmma2av_mte_context.h`：运行期状态和方法声明；UB、窗口、进度和游标只有一份。
- `common/gmma2av_mte_constants.h`、`workspace.h`、`copy.h`、`sync.h`：布局约束、窗口/UB访问、MTE 搬运和同步。
- `stage/gmma2av_mte_prepare.h`、`stage.h`、`combine.h`：准备/结束、workspace staging、专家发布和 peer pull。

## Wave 与任务归属

一个 M-group 对应每个 AIC 最多一个完整 M tile（包括该 tile 的全部 N 切片）。`M_GROUPS_PER_WAVE=1`，每个 AIC 每完成一个拥有的 M tile 就发布一次 Wave 进度。专家尾部不足一个 group 时提前结束；Wave 不跨专家合并。

M tile 的归属延续当前跨专家轮转：`startM=(core+coreCount-startCore)%coreCount`，步长为 `coreCount`。专家结束后 `startCore` 增加该专家的 M tile 数。计算与 staging 使用同样的归属和 Wave 分界，空专家、空闲核不产生也不等待 Wave；Stage 的序列号跨专家连续。

AIV1 不再消费 AIC 的 mode=0x2 通知，不再归还通知信用。其循环只依赖 Stage 发布的专家完成进度。Stage 不等待 peer，AIC 不等待通信角色消费，因此 peer 延迟不会通过反向信用阻塞计算。

各 rank 的专家 token 数可以不同，跨卡通知仍以完整专家为边界，不能把本地 Wave 序列当成跨卡共同的 Wave 编号。AIV1 在本地所有实际生产者完成 staging 后才发布 ExpertReady。

## arch22 接口与内存顺序

数据路径保持 `L0C → GM workspace → UB → 本 rank CCL window → peer UB → 输出 GM`。不使用 A5 的 L0C→UB、MX 量化、arch35 matmul 或其 UB 尺寸配置。

AIC 在 Wave 边界执行 `SetFlag/WaitFlag<HardEvent::FIX_S>`，随后标量 `SetValue` 写本核独占的 64-byte cache line 中的进度值，再执行 `DataCacheCleanAndInvalid<…, SINGLE_CACHE_LINE, CACHELINE_OUT>`（dcci）和 DDR barrier。这是计算输出可见性的发布边界。AIV0 用普通 MTE2 读取 AIC 发布的进度，观察到 `progress >= expected` 后才读 workspace。当前调用内 workspace 行不会复用，所以观察到更晚的进度也有效。

Stage 完成专家的全部本核 M tile 搬运后，经 MTE3 完成与 DDR barrier 发布专家进度；Comm 等待所有拥有该专家行的 Stage 核，然后执行既有跨卡协议。

Workspace 的 count scratch 后放置两段按 512-byte 对齐的进度区：Stage→Comm、AIC→Stage，每段每 AIC 一个 cache line。Host 分配和 device 地址都使用 `WaveWorkspaceBytes` / `ComputeProgressOffset`。相对原实现新增一段进度区，20 AIC 时增加 1536 bytes。AIV0 在初始化清零两类本核进度，全体 AIC/AIV 在计算前会合一次，防止清零与新 Wave 发布竞争；最后的全核会合保证本调用工作结束后才复用 workspace。

`FIX_S` 会排空该 AIC 先前的 FIX 工作，这是将 A5 GM ready 机制迁移到当前 GM staging 数据路径的成本。分核解耦并不自动保证所有 shape 更快，应使用独立的性能对比验证。

## 验证

编译目标为 `ascend910_93`，使用 py310 和当前 `env_0828` CANN。精度回归脚本 `ep_test/cases/gmma2av_v2/single_script/test_gmma2av_wave_correctness.py` 使用行、列均可区分的精确置换矩阵 oracle，包含空专家、129-row 轮转和 2561-row Wave 尾部，并覆盖 BF16/FP16、GMM 两种转置、可选 shared MM 两种转置、连续多次变化输入。

实际执行结果和日志写入工作区 `a5_refactor_20260915`；以测试汇总为准，不以编译成功替代硬件精度验证。

## 跨调用同步

参考 `trans_GMMA2AV/mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode.h` 和 `matmul_reduce_scatter_aiv_mode_util.h` 的 rotating buffer、入口清 flag、出口清 flag0 并等待各卡清零的顺序。

每次 kernel 入口固定选择 buffer 0，选择逻辑直接写在 device 的 `GetBufferIndex()`，不再依赖 tiling 或 host 调用计数。入口清理当前 buffer 的全部 sync 槽，发布 count table，再 COUNT_READY=1 并等待各卡。通信完成后 RECV_DONE=2 并等待各卡，随后 COUNT_READY=0 并等待各卡归零，最后 RECV_DONE=0 并等待各卡确认已观察到入口清零。

最后的清零确认是同一 buffer 连续复用的必要条件：旧调用还在等待退出确认时，新调用可以保持 RECV_DONE=0，但不能越过下一次 COUNT_READY 会合发布 RECV_DONE=2。这样旧调用等待的退出零值不会被提前覆盖。仅入口清零加单次出口清零不足以保证这一点。
这是对生命周期的参考，未照搬 reduce-scatter 的 atomic 累加 flag：本算子使用每卡单写者的阶段槽和按专家 peer pull，跨卡专家数一致但本地 Wave 数可以不同。

每个 payload buffer 容量为 `floor((windowBytes-21MiB)/2/512)*512`，由所有 rank 共用的 window 大小决定；count 区和 sync 区使用固定 stride，与本地 shape 无关。Host 要求至少 `2*alignUp(sendBytes,512)+21MiB`，且保留原 200MiB 下限。已删除host轮次计数及tiling的 `syncRound` 字段；完整重建host与device以更新序列化布局。保留两套区域的布局和容量检查，本版本实际只使用buffer 0；未同时调整内存容量策略。固定入口消除了对“每次调用重新tiling”的依赖，但不据此宣称图回放等其他执行模式已经验证。

最后的 RECV_DONE 清零确认使同一 buffer 立即复用时仍有完整握手：下一轮 RECV_DONE 不能在所有卡进入下一轮 COUNT_READY 前发布，尚未退出的卡保持 COUNT_READY=0，阻止其确认零值被下一轮有效值覆盖。

CPU 协议模型覆盖逐 peer 轮询、rank 进度偏差、双 buffer 再次复用和单 buffer 立即复用；其结果不能替代真实 NPU 缓存可见性测试。
