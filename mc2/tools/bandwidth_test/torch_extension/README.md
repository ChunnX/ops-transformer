# Torch Extension

## URMA 共享 jetty

在 `config.py` 中设置 `shared_jetty=True`、`shared_jetty_num=N`，保留 `comm_alg="urma"`。
`N` 为实际建立的共享队列数量，也是按 BS 分片的发送核数。tiling 仍沿用原有 `aiv_num`
规则选择并启动全部 AIV（设备当前最多 72 个）；只允许 `1 <= N <= min(所选 AIV 数, 56)`。
`aiv_num=-1` 时使用设备 AIV 数。该数量关系在 tiling 中校验。
关闭共享 jetty 时沿用现有 MTE / 普通 URMA 路径。MTE 开启此开关会报参数错误。

```python
sym_buffer = BandwidthTestBuffer(group, comm_alg="urma", shared_jetty=True, shared_jetty_num=8)
y, receive_cnt = bandwidth_test(
    sym_buffer, x, dstrank_id, sym_buffer.world_size, x.size(0),
    mode=0, comm_alg="urma", shared_jetty=True, shared_jetty_num=8, batch_tokens=128)
```

依赖 [asc-comm 0eff184](https://gitcode.com/cann/asc-comm/commit/0eff184c367e6d67263745e57abc485237d34019)
提供的公开 `MakeMultiChannelHandle` / `GetHandleRef` 接口及对应 HCCL 共享队列支持。
Host 使用 `torch_extension/shared_jetty_context.py` 和同目录 `csrc` 中的独立实现，
逐对端查询实际网络层，优先选择域内 `UBC_CTP`（即 `UB_CTP`），否则选择跨超 `UB_RTP`（UBG）。
两者均按 URMA 建链，保留各对端实际的 `channelProtocol`。UBG 共享 jetty 使用用户提供的
支持该能力的联调包；不按公开仓库旧版文档拦截，也不将 UBG 归类为 MTE。
每个共享 jetty 使用独立 tag，包含按 rank 排序、跳过本卡的所有对端 descriptor；
context 保留共享 jetty 的环境和兼容性检查：仅支持 Ascend950，
从各对端候选链中选择共同的本地 endpoint，无交集时明确报错。这是本算子参考官方示例
采用的共享队列选择约束；不会因为链路协议是 `UB_RTP` 而拒绝它。
共享队列 API 不可用或 HCCL 建链失败时，同样在 context 创建阶段报错，报告 jetty 编号及返回码。
底层 C++ context 不查询设备核数，也不承担共享 jetty 数量上限校验；数量限制由 tiling 统一处理。
不会隐式创建额外 jetty 或回退普通 channel；单 rank 无远端，跳过建链。

Win 通过 HCCL engine context 分配、清零并注册；构造阶段通过组内 collective 交换 Win 地址，
同时核对共享 jetty 数、窗口大小和构造顺序。句柄、注册和 Win 均由 communicator 管理，
Python owner 保留 ProcessGroup。多配置和图重放应复用 owner；所有 kernel 完成后销毁通信组
才释放这些资源。构造中途失败的 HCCL 资源同样保留到通信组销毁，不能重试同一个失败 owner。

tiling key 先选择通信算法，再在 URMA 下选择 `SHARED_JETTY`；kernel 独立文件为
`op_kernel/bandwidth_test_urma_shared_jetty.h`，保留 mode 0/1/3：

- mode 0/1：将输入 BS 的完整 token 连续分给 N 核，先在 workspace 写各核各对端计数，
  同步后按核号前缀计算紧凑发送位置。每核独占中转 token 区间，全部数据完成后再汇总并发送
  count / ready。其余已启动 AIV 继续参与全核同步、状态等待和接收搬运，mode 0 沿用原来的
  接收和输出排列。
- mode 3：将每个对端的 `floor(BS/world_size)` 个完整 token 分给 N 核；每核遍历全部远端。
  总流量口径不变，各 `(源 rank, 核)` 使用独立 epoch flag，空区间仍通知完成。
- 每个发送核只操作自己的共享 jetty。数据、count、ready 和 epoch 的远端写都使用
  `GetHandleRef → WriteNbi(batch) → BatchCommit(outer batch) → Drain(outer batch)`。
  `batch_tokens` 是每核向一个目标卡提交的 payload token 数，尾批可以不足；
  统计表根据实际 BS 分核后的路由计算最大数据批量，状态通知不计入 payload。
  `WriteNbi`、`BatchCommit`、`Drain` 返回错误时触发 `Trap()`，中止 kernel，避免继续发布成功标志。

共享模式的新增 workspace 为 `N * align_up(world_size, 8) * 4` 字节计数表；
发送 UB 还包含一份按 world_size 对齐的本核计数。tiling 按 248 KiB 可用 UB 校验容量。
以下原有普通 URMA / MTE 布局说明中的按目标卡分核和 channel 锁仅适用于非共享路径。

当前交付基于静态分析；仍需在支持上述接口的 CANN 上编译，并验证多 rank 的
mode 0 精度、mode 1/3 重复调用、图重放、非整除 BS、空区间、56 jetty，以及联调包的
跨超 UBG 和混合 CTP/UBG 拓扑。这里的静态检查不代表目标包编译或实机通信已验证。

## 安装和原有调用方式

实现复用 MegaMoe 的 `OpBuilder`、`CommContextManager` 和 `ACLNN_CMD`。
现有安装脚本将三级目录收集到 `cann_ops_transformer.ops.tools.bandwidth_test`，
C++ 源码放入 `csrc/tools/`，无需修改公共打包逻辑。

先安装包含 BandwidthTest 的算子包并加载其环境，再从仓库根目录安装扩展：

```bash
bash build.sh --pkg --soc=ascend950 --ops=bandwidth_test
# 执行 build_out 中本次生成的 .run 包，并 source 其安装目录下的 set_env.bash。
python -m pip install ./torch_extension --no-build-isolation
```

`bash build.sh --torch_extension` 只生成 Torch wheel，随后直接退出，不编译 ACLNN、tiling 或 kernel。
更新本算子时需要重新安装算子包和 Torch 扩展，并重启测试进程。

完整安装同时包含 `mc2/common` 的 context 支持。按算子打包时，
`TORCH_EXTENSION_OPS` 必须包含 `bandwidth_test,common`，并从生成包的
`ops.tools.bandwidth_test` 模块导入下面两个接口。

```python
import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from cann_ops_transformer.ops import BandwidthTestBuffer, bandwidth_test

# 先设置当前 NPU、HCCL_BUFFSIZE，再初始化 distributed 并创建 group。
# 所有 rank 都要调用；context 创建和 C++ JIT 编译在图捕获前完成。
sym_buffer = BandwidthTestBuffer(group)

class Model(torch.nn.Module):
    def __init__(self, sym_buffer, max_bs):
        super().__init__()
        self.sym_buffer = sym_buffer
        self.max_bs = max_bs

    def forward(self, x, dstrank_id):
        return bandwidth_test(self.sym_buffer, x, dstrank_id,
                              self.sym_buffer.world_size, self.max_bs, mode=0)

model = Model(sym_buffer, max_bs=32)
config = CompilerConfig()
config.mode = "reduce-overhead"
config.debug.aclgraph.clone_input = False
config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./static_kernel"
compiled = torch.compile(model, backend=torchair.get_npu_backend(compiler_config=config),
                         dynamic=False, fullgraph=True)
y, receive_cnt = compiled(x, dstrank_id)
torch.npu.synchronize()
# 所有重放完成后才能释放 model/compiled/sym_buffer 或销毁进程组。
```

原测试脚本 `graph_type=0` 为 eager，`graph_type=3` 为 ACLNN 静态 kernel / ACL graph。
Meta 输出为 `y[max_bs * world_size, hidden]`（FP16）和
`receive_cnt[world_size]`（INT32）；GE Converter 与主线 OpDef 对齐。
Eager 路径经 `aclnnInnerBandwidthTest` 显式传递 Torch 创建的 context，
不再由公开 ACLNN 包装层重复创建 context。
可选参数在 schema、Meta、PrivateUse1、GE converter 和 OpDef 中统一为
`comm_alg=""`（MTE）、`aiv_num=-1`（平台默认核数）、`batch_tokens=0`（自动批量）、
`shared_jetty=False`、`shared_jetty_num=1`；
Python 始终将这些值显式传给 C++/ACLNN。新增属性改变了生成的 inner ACLNN 签名，
更新后需同步重装算子包和 Torch 扩展；公开 MTE ACLNN 包装接口保持原签名，
内部为新增参数传入 `batch_tokens=0, shared_jetty=false, shared_jetty_num=1`。

Ascend950 支持 MTE（同一 UB 内存域，`comm_alg="mte"` 或空字符串）和
URMA（`comm_alg="urma"`，由已有 Torch context manager 建立 UBC_CTP channel）。
URMA 要求安装支持 `MakeBatchHandle`、`BatchCommit`、`Drain(batchHandle)` 和
`Lock/Unlock` 的 Hcomm 版本。创建和调用必须选择相同算法：

```python
sym_buffer = BandwidthTestBuffer(group, comm_alg="urma")
y, receive_cnt = bandwidth_test(sym_buffer, x, dstrank_id,
                                sym_buffer.world_size, x.size(0), mode=0, comm_alg="urma",
                                batch_tokens=128)
```

URMA 通过 Torch 的 inner ACLNN 路径使用现有 context；公开的
`aclnnBandwidthTestGetWorkspaceSize` 包装层不创建 URMA context，会拒绝 `urma`。
`x` 必须为二维 FP16，`dstrank_id` 为长度等于 bs 的 INT32；
所有输入位于同一 NPU。主线 kernel 按 max_bs 读取输入，要求 `bs == max_bs`、
hidden 是 16 的倍数且小于 32768，
各 rank 使用相同的 max_bs、hidden、world_size、mode、aiv_num、batch_tokens。
目标 rank ID 应在 `[0, world_size)`；扩展不会在捕获期间读取 NPU tensor 的值。
`aiv_num=-1` 使用平台默认值，正值不应超过设备 AIV 核数。
测试脚本在 `config.py` 的 `ModelConfig.batch_tokens` 设置每次提交的 token/WQE 数：
0 自动选择容量；正数由 tiling 校验，通过后按原值提交；负数非法。
`batch_tokens` 仅适用于 URMA，MTE 保持 0。每张目标卡的最后一批可以不足指定数，
状态通知仍分别提交 count 和 ready flag，不受数据批量凑整影响。

当前共用的 `CommContextManager` 创建并注册自定义通信窗口；扩展显式传入
`2 * HCCL_BUFFSIZE * MiB` 字节，使实际分配与主线 A5 tiling 的窗口推导一致。
未设置该变量时沿用 tiling 的 200 MiB 默认配置；不能在初始化后更改它。
该窗口是 HCCL 自身分配以外的额外显存开销。每个 owner 使用独立的 context tag，
避免共用 manager 的缓存恢复分支；同组各 rank 必须以相同顺序创建 owner，
多配置和图重放应复用同一个 owner。
MTE 和 URMA 共用 Win 布局：1 MiB 状态区和两个数据半区，每个半区需要
`(world_size + 1) * max_bs * align_up(hidden * 2, 512)` 字节。
URMA 将 MTE 已预留的首个 max_bs 数据块作为中转区；接收区仍按源 rank 分块。
状态仍是 MTE 的 `[1.0f ready flag, count, ...]`；两条路径只翻转原位置的双缓冲位，
不再执行通用 `InitWinState` 的诊断字段和调用次数记账。
中转数据和通知源地址全部位于 context 注册的 Win；workspace 仅保留框架系统空间。
URMA 通过 MTE 的 `SplitToCore(world_size, aiv_num, ...)` 将连续的目标卡范围分配给各核，
前面的核处理余数卡，本卡仍使用 `ProcessToken` 本地搬运。
kernel 通过已有的通信算法 tiling key 和 `if constexpr` 选择 MTE/URMA；mode 3 不新增 tiling key。
URMA 是独立类，声明、类外实现和 `Process()` 分支排列与 MTE 一致。
`mode=0` 顺序执行 `SendToMoeExpert → SetStatus → WaitDispatch → LocalWindowCopy`；
`mode=1` 顺序执行前三个阶段。
通道句柄在初始化时按本核负责的目标卡读取一次并缓存。
数据和状态通知的所有远端 Write 都经 batch 接口提交，包括只有一条请求的尾批。
Batch 接口使用 handle 内的缓冲区，不再分配普通接口的 512 字节 scratch 或调用其 `Init`。
每个阶段处理一张远端卡时创建一次 batch handle，全部 token 和 commit 复用它。
URMA 分块读取 rank ID，发送时累计各目标卡的 token 数，不再分配全量 ID 和三个计数临时区。
token 搬运只使用一行 UB，每行 MTE3 完成后复用；ID 和 WQE buffer 则逐批复用。
host 按完整的 248 KiB 可用 UB 计算分块容量，ID buffer 按 32 字节对齐：

```text
N = ceil(world_size / aiv_num)
fixed_bytes = 32 + align_up(N, 8) * 32 + hidden * 2
available_bytes = 248 * 1024 - fixed_bytes
ub_maximum = 最大满足 64 * B + align_up(4 * B, 32) <= available_bytes 的正整数 B
maximum = min(ub_maximum, 2048)
capacity = maximum if batch_tokens == 0 else batch_tokens
rank_id_buffer_bytes = align_up(capacity * 4, 32)
batch_buffer_bytes = capacity * 64
```

Ascend950 的 256 KiB UB 扣除系统预留的 8 KiB 后，算子使用完整的 248 KiB，
不再对平台查询值额外扣减 8 KiB；不按 UB 批容量给 BS 设置上限，也不按 BS 截断 maximum。
总 token 数可以超过单批容量，`batch_tokens` 也可以超过 BS（实际数据不足时形成尾批）。
现有 INT32 计数表示检查和实际 tensor/通信窗口容量检查仍然有效。
若传入正数超过 maximum，tiling 返回失败并报告 `batch_tokens=... exceeds maximum=...`；
合法值会打印 `requested / selected / maximum`。UB 最大值与 BS 无关。
例如 `world_size=8、aiv_num=64` 时，`max_bs=4096、hidden=16384` 和
`max_bs=8192、hidden=8192` 的 tiling 最大提交数均为 **2048 个 token/WQE**。
设备端分块扫描 ID，为当前目标卡跨 ID 块累计 WQE；凑够指定数量就 `BatchCommit`。
每批经 `BatchCommit` 提交、`Drain` 完成后，再准备下一批。
所有 URMA 模式直接调用 `hcomm_.WriteNbi(handle, dst, src, len)`，不传 WQE 配置或模板实参。
WQE 行为完全由 SDK 默认值决定，算子不覆盖 `odr`、`fence` 或 `cqe`。
下述已核对 SDK 版本的默认值为 `odr=5、fence=1、cqe=1`，直接调用并不表示把这些字段置零。
当前 SDK 默认每条请求生成 CQE，`Drain` 收齐该批完成通知；不会因为一个 ID 块结束就提交未满的数据批次。
空目标卡不提交数据批次。
发送中的单行 UB 不必保留到 URMA 完成：URMA 源数据已位于注册的 GM 中转区。
中转区仍使用原 token 下标，避免不同目标卡所属核并发覆盖同一槽位。
所有数据和状态通知完成后，`TPipe::Reset` 将发送阶段 UB 复用于等待和接收阶段。
这是带宽测试：调用方必须保证 channel、目标 rank、Win 容量和各 rank 配置正确，
并保证 tiling 选出的 `batch_tokens` 小于每条使用的 SQ 和 CQ 深度；使用 0/最大值时也须自行确认。
本算子 kernel 直接使用 tiling 的有效批量，不额外做参数校验、打印、Trap 或自动缩小批量；SDK 内部检查沿用原实现。
kernel 只通过公开 Hcomm 接口提交、完成和锁定通信，不读取 channel entity、队列深度、游标或 batch handle 内部字段。
当前 Batch 接口未提供队列余量查询，也未承诺自动处理多批在途的溢出；因此每批提交后调用 Drain，再复用下一批。
多批数据的计时包含这些完成等待，不能假定更小的 batch 会降低耗时。
已核对 [Hcomm 公共接口](https://gitcode.com/cann/asc-comm/blob/master/include/aicore/hcomm/hcomm.h)，asc-comm 提交为
`1529377b9e908d78ba5ba5d579edf1669fb99b01`，部署时须使用兼容版本。
该 SDK 的 `PollBatchCq` 在目标 CQ 游标等于 0 时直接返回，即使还有未完成请求。
长期打流中，32 位 `cqHead` 恰好在 Drain 边界回绕为 0 时可能漏等完成；
部署 SDK 应修复为仅在没有待回收 CQE 时跳过轮询，不能仅判断目标游标为 0。
本算子不为此增加 BS 上限、设备侧补丁或强序设置；目标 SDK 未修复前，不能保证触及该边界时的完成语义。
数据阶段的通道锁覆盖 batch handle 创建至最终 Drain，避免共享 SQ 的其他核并发更新游标。
`SetStatus` 复用缓存的通道句柄，重新持锁并创建该阶段的 batch handle；
计数和 ready flag 均经 batch 接口提交并分别等待完成，保证先计数后就绪。
阶段间重新绑定避免复用已解锁、可能被共享 SQ 的其他核推进过的游标；不会在 token 或 commit 循环里重新获取 handle。
`mode=1` 与现有 MTE 一致，包含发送、状态通知和等待，仅省略接收区到输出的搬运。
`mode=0` 在相同流程后调用与 MTE 一致的 `LocalWindowCopy`，按源卡和原 token 顺序拼接有效输出。
同一 context 的调用与图重放应串行下发，避免并发覆盖通信窗口。
切换 shape、mode 或测试配置前，所有 rank 必须完成此前调用并同步；各 rank 调用次数和顺序须相同。
普通 URMA owner 的 `aiv_num` 必须固定；该路径按核保存双缓冲位，改变核数需要同步后重建 owner。
共享 jetty owner 使用公共双缓冲位，同步完成后允许改变 `aiv_num`，但不得小于 `shared_jetty_num`。
`mode=0` 仅 `y[:receive_cnt.sum()]` 有效；`mode=1/3` 不搬运输出，两个输出均不保证有效，
测试脚本的图内循环因此继续使用原始输入。schema 将 context 声明为可变输入，
沿用仓库通信算子的副作用约定；目标环境仍须确认优化后的图保留全部通信次数和顺序。

`mode=3` 仅支持此 Torch context 的 URMA 路径，直接从 Win 发送 `BS // world_size` 个 token 到每个远端卡。
它在同一个 `BandwidthTestUrma::Process()` 的 `mode_ == 3` 分支中执行，沿用原有 URMA tiling key。
`Init()` 读取通信参数后提前返回，不做 mode 0/1 的业务初始化、ID 扫描和数据搬运；
`StreamFromWindow()` 只为有远端任务的核分配 `batch_tokens * 64` 字节 UB。
Win 使用紧凑 H 行，最小容量为 `1 MiB + 2 * BS * H * 2`；该模式不适用上文 mode 0/1 的双缓冲及 ID 预算公式。
flag 位于 Win 的 128 KiB 处（每个来源卡 32B），本核序号位于 160 KiB 处（每核 32B）。
有远端任务的核每次调用递增本核序号；尾批未满时，将 flag 并入尾批 Commit，WQE 总数不超过配置批量。
满批先 Commit/Drain，随后单独 Commit/Drain flag；合并尾批时只需一次 Commit/Drain。
flag 的先后关系依赖当前 SDK 默认的 `fence=1`，算子不显式传配置；部署 SDK 的默认值必须保留该完成顺序及 CQE。
等待时允许对方序号超前一轮，支持 uint32 回绕。
不读取 x/dstrank_id 内容，仍保留原 tensor shape/type 接口约定；完整使用和计时口径见主目录 README。

需在装有 CANN、PyTorch、torch_npu 和兼容 TorchAir 的 Ascend950 多 rank 环境
分别验证 eager 精度、图捕获及重复重放。本地静态检查不能替代上述验证。
两份接收实现均包含状态 buffer 字节数、DataCopy 块长及 UB 记账修正；host 按实际缓冲区检查可用 UB。
URMA kernel 不检查通信参数及 Hcomm 失败返回值；调用方负责保证上述通信前置条件。
这些修改仍需在目标 SDK 下验证编译和多卡可见性。
