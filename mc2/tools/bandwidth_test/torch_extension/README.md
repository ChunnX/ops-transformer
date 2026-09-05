# Torch Extension

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
`comm_alg=""`（MTE）、`aiv_num=-1`（平台默认核数）；Python 始终将这两个值显式传给 C++/ACLNN。

Ascend950 支持 MTE（同一 UB 内存域，`comm_alg="mte"` 或空字符串）和
URMA（`comm_alg="urma"`，由已有 Torch context manager 建立 UBC_CTP channel）。
URMA 要求安装支持 `MakeBatchHandle`、`BatchCommit`、`Drain(batchHandle)` 和
`Lock/Unlock` 的 Hcomm 版本。创建和调用必须选择相同算法：

```python
sym_buffer = BandwidthTestBuffer(group, comm_alg="urma")
y, receive_cnt = bandwidth_test(sym_buffer, x, dstrank_id,
                                sym_buffer.world_size, x.size(0), mode=0, comm_alg="urma")
```

URMA 通过 Torch 的 inner ACLNN 路径使用现有 context；公开的
`aclnnBandwidthTestGetWorkspaceSize` 包装层不创建 URMA context，会拒绝 `urma`。
`x` 必须为二维 FP16，`dstrank_id` 为长度等于 bs 的 INT32；
所有输入位于同一 NPU。主线 kernel 按 max_bs 读取输入，要求 `bs == max_bs`、
hidden 是 16 的倍数且小于 32768，
各 rank 使用相同的 max_bs、hidden、world_size、mode、aiv_num。
目标 rank ID 应在 `[0, world_size)`；扩展不会在捕获期间读取 NPU tensor 的值。
`aiv_num=-1` 使用平台默认值，正值不应超过设备 AIV 核数。

当前共用的 `CommContextManager` 创建并注册自定义通信窗口；扩展显式传入
`2 * HCCL_BUFFSIZE * MiB` 字节，使实际分配与主线 A5 tiling 的窗口推导一致。
未设置该变量时沿用 tiling 的 200 MiB 默认配置；不能在初始化后更改它。
该窗口是 HCCL 自身分配以外的额外显存开销。每个 owner 使用独立的 context tag，
避免共用 manager 的缓存恢复分支；同组各 rank 必须以相同顺序创建 owner，
多配置和图重放应复用同一个 owner。
MTE 和 URMA 共用 Win 布局：1 MiB 状态区和两个数据半区，每个半区需要
`(world_size + 1) * max_bs * align_up(hidden * 2, 512)` 字节。
URMA 将 MTE 已预留的首个 max_bs 数据块作为中转区；接收区仍按源 rank 分块。
状态仍是 MTE 的 `[1.0f ready flag, count, ...]`，双缓冲由同一个 `InitWinState` 推进。
中转数据和通知源地址全部位于 context 注册的 Win；workspace 仅保留框架系统空间。
URMA 通过 MTE 的 `SplitToCore(world_size, aiv_num, ...)` 将连续的目标卡范围分配给各核，
前面的核处理余数卡，本卡仍使用 `ProcessToken` 本地搬运。
kernel 新增通信模式 tiling key，通过 `if constexpr` 选择 MTE/URMA，不在设备运行时判断算法字段。
URMA 是独立类，声明、类外实现和 `Process()` 分支排列与 MTE 一致。
`mode=0` 顺序执行 `SendToMoeExpert → SetStatus → WaitDispatch → LocalWindowCopy`；
`mode=1` 顺序执行前三个阶段。
通道句柄在初始化时按本核负责的目标卡读取一次并缓存。
数据和状态通知的所有远端 Write 都经 batch 接口提交，包括只有一条请求的尾批。
每个阶段处理一张远端卡时创建一次 batch handle，全部 token 和 commit 复用它。
保留现有工作 buffer 和 token 队列配置，WQE 缓冲区取剩余可用 UB 的最大 64 字节整数倍：

```text
N = ceil(world_size / aiv_num)
R = align_up(max_bs * 4, 32)
I = 32 + align_up(N, 8) * 32
    + max(R, align_up(max_bs * 2, 256)) + 3 * max(hidden * 4, R)
Q = 1 if I + hidden * 4 > 170 * 1024 else 2
batch_buffer_bytes = align_down(248 * 1024 - I - Q * hidden * 2 - 512, 64)
```

平台实际可用 UB 更小时按更小值计算；至少要能容纳一条 64 字节 WQE。
例如 `world_size=8、aiv_num=64、max_bs=2、hidden=8192` 时，其他 buffer 占 132128 字节，
batch 可用 **121792 字节（1903 条 WQE）**，仅余 32 字节不足一条 WQE。
卡数使用实际 `world_size` 动态计算，`max_bs=2` 表示每卡 token 数，不是卡数。
保持上述 `max_bs` 和 `hidden`，128 卡、64 核时每核处理 2 卡；状态 buffer 按 8 卡对齐，
因此 batch 仍为 121792 字节。128 卡、8 核时每核处理 16 卡，状态 buffer 增加 256 字节，
batch 相应减为 **121536 字节（1899 条 WQE）**。非整除时前面的核多处理一张卡。
实际单批上限为 `min(batch_buffer_bytes / 64, SQ深度-1)`。
UB 满时仅 `BatchCommit`，清空已准备计数后继续填充；在途 WQE 达到 SQ 上限或该卡尾批结束时，
末条请求 CQE、提交并 `Drain`，防止 SQ 覆盖及源数据提前复用。
SQ 深度在持锁时从 URMA channel context 读取一次，要求深度至少为 2；不修改 batch handle 内部字段。
该读取依赖 [Hcomm URMA 实现](https://gitcode.com/cann/asc-comm/blob/master/src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h)
的 `ChannelEntity` / `sqContextAddr[0]` 布局；静态核对的 asc-comm 提交为
`1529377b9e908d78ba5ba5d579edf1669fb99b01`，部署时须使用兼容版本。
数据阶段的通道锁覆盖 batch handle 创建至最终 Drain，避免共享 SQ 的其他核并发更新游标。
`SetStatus` 复用缓存的通道句柄，重新持锁并创建该阶段的 batch handle；
计数和 ready flag 均经 batch 接口提交并分别等待完成，保证先计数后就绪。
阶段间重新绑定避免复用已解锁、可能被共享 SQ 的其他核推进过的游标；不会在 token 或 commit 循环里重新获取 handle。
`mode=1` 与现有 MTE 一致，包含发送、状态通知和等待，仅省略接收区到输出的搬运。
`mode=0` 在相同流程后调用与 MTE 一致的 `LocalWindowCopy`，按源卡和原 token 顺序拼接有效输出。
同一 context 的调用与图重放应串行下发，避免并发覆盖通信窗口。
切换 shape、mode 或测试配置前，所有 rank 必须完成此前调用并同步；各 rank 调用次数和顺序须相同。
同一 owner 的 `aiv_num` 必须固定；`InitWinState` 按核保存双缓冲位，改变核数需要同步后重建 owner。
`mode=0` 仅 `y[:receive_cnt.sum()]` 有效；`mode=1` 不搬运输出，两个输出均不保证有效，
测试脚本的图内循环因此继续使用原始输入。schema 将 context 声明为可变输入，
沿用仓库通信算子的副作用约定；目标环境仍须确认优化后的图保留全部通信次数和顺序。

需在装有 CANN、PyTorch、torch_npu 和兼容 TorchAir 的 Ascend950 多 rank 环境
分别验证 eager 精度、图捕获及重复重放。本地静态检查不能替代上述验证。
两份接收实现均包含状态 buffer 字节数、DataCopy 块长及 UB 记账修正；host 按实际缓冲区检查可用 UB。
URMA kernel 不使用 ASSERT，也不检查通信接口返回值；参数和通信正确性由测试调用方保证。
这些修改仍需在目标 SDK 下验证编译和多卡可见性。
