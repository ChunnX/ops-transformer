# Torch Extension

## 多机启动（双机，每机 16P）

两台机器都修改 `mc2/tools/bandwidth_test/config.py`，并分别启动一次脚本：

| 配置 | 机器 0 | 机器 1 |
| --- | --- | --- |
| `parallel.server_num` | `2` | `2` |
| `parallel.rank_per_dev` | `16` | `16` |
| `parallel.server_index` | `0` | `1` |
| `parallel.is_cloud` | `False` | `False` |
| `parallel.master_ip` | 机器 0 可达的 IP 字符串 | 相同 IP |
| `parallel.master_port` | `50001` | `50001` |
| `model.comm_alg` | `"roce"` | `"roce"` |
| `model.mode` | `0` | `0` |
| `model.shared_jetty` | `False` | `False` |
| `model.batch_tokens` | `0` | `0` |

两台机器使用相同的 shape、测试模式、随机种子、图模式和其他通信配置，且各自可见 16 个 NPU。
脚本在每台机器自行启动 16 个进程：机器 0 负责全局 rank 0–15，机器 1 负责 16–31，
本地设备均为 0–15，通信组总大小为 32。脚本没有 SSH 远程拉起功能，不要再用 torchrun 重复启动它。

```bash
cd mc2/tools/bandwidth_test
python bandwidth_test.py
```

精度、单配置带宽、多配置带宽三个入口均支持按本机 rank 启动。精度测试在每台机器生成相同的
全局输入和 golden，只比较本机输出；两台机器均通过才表示完整测试通过。带宽报告只统计本机
profiling 样本，并记录服务器编号及本机全局 rank 范围，不自动汇总跨机结果。
profiling、静态图构建和缓存目录应使用各机独立目录，避免共享文件系统上的并发清理或覆盖。
云环境仍通过 `VC_TASK_INDEX`、`VC_WORKER_HOSTS` 获取节点编号和主节点地址。

ROCE 会检查全部 rank 对（包括同机卡之间）的 ROCE link，仅跨机链路支持 ROCE 不满足此模式。
当前算子仍面向 Ascend950；每机 16P 只描述进程数量，不扩展到其他芯片架构。

## ROCE mode 0

在 `config.py` 中设置 `comm_alg="roce"`、`mode=0`、`shared_jetty=False`，脚本会选择
ROCE context 和独立的 ROCE tiling 分支。`mode=None` 时 ROCE 默认使用 mode 0；显式设置
其他 mode 会报错。支持 `aiv_num`；`batch_tokens` 必须为 0，ROCE 不支持 Batch 接口。

```python
sym_buffer = BandwidthTestBuffer(group, comm_alg="roce")
y, receive_cnt = bandwidth_test(
    sym_buffer, x, dst_rank_id, sym_buffer.world_size, x.size(0),
    mode=0, comm_alg="roce", batch_tokens=0)
```

`HcclChannelContextBuilder` 逐 peer 遍历网络层并确认
`COMM_PROTOCOL_ROCE` link；缺少任一远端 peer 的 ROCE link 时直接报错，不回退 URMA/UBG。
ROCE 与 URMA 共用 `aclrtMalloc → HcclCommMemReg` 的窗口申请和注册流程。
Channel 的 `memHandles=customMemHandle_`、`memHandleNum=1`，远端 Win 通过
`HcclChannelGetRemoteMems` 按注册 tag 查找。本地 Win 容量为 `2 * HCCL_BUFFSIZE` MiB。
所有 rank 的窗口容量必须一致，ROCE 建链时校验远端注册地址非空且容量与本地相同。
需在创建通信组前设置足够大的 `HCCL_BUFFSIZE`，mode 0 每卡至少需要
`1 MiB + 2 * (world_size + 1) * max_bs * align_up(hidden * 2, 512)` 字节。

`bandwidth_test_roce.h` 是独立的 mode 0 模板，使用 `Hcomm<COMM_PROTOCOL_ROCE>` 的
`Init`、普通 `WriteNbi(ChannelHandle, dst, src, len)` 和 `Drain(ChannelHandle)`，不使用
`BatchHandle`、`MakeBatchHandle`、`BatchCommit` 或锁接口。每个目标 rank 由一个 AIV 负责，
逐 token 写入（默认立即提交），对该 peer 的数据全部发送后 Drain，再依次发布 count、Drain、
ready flag、Drain。SDK 负责普通写入的队列空间管理；通信错误会 Trap，不静默继续等待。
网络写入的数据和通知的源地址、目的地址全部位于 HCCL 注册 Win；输入先经 UB 拷贝到本地 Win，
接收完成后再从本地 Win 拷贝到输出 tensor。

ROCE 使用 512 字节 Hcomm 初始化 UB 和 256 个 rank ID 的分块扫描缓冲，BS 超过 256 时继续
分块扫描，不限制总 token 数。该分块仅用于读取路由，不是通信 Batch；tiling 的 Batch 字段为 0，
性能表的 Batch 列显示 N/A。URMA 保留原来的 Batch 接口和批量设置。

窗口由 context manager 分配并持有，复用原有注册、建链和释放流程，context tensor 使用原有布局。
每个 buffer 使用独立的注册 tag；窗口在首次申请时清零，再注册和建链。所有 rank 都要在图捕获前
创建 buffer，并持有 buffer 和通信组直到执行结束。同一 buffer 的算子调用和图重放需串行。
网络收发不使用 tiling 分配的普通 workspace。
公开 `aclnnBandwidthTestGetWorkspaceSize` 不创建 ROCE context，ROCE 使用 Torch inner ACLNN 路径。
新增 tiling mode 扩展了 key 的位宽，需同步重新构建安装 host tiling、kernel 和 Torch 扩展。

## URMA 共享 jetty

在 `config.py` 中设置 `shared_jetty=True`、`shared_jetty_num=N`，保留 `comm_alg="urma"`。
`shared_jetty_comm_level` 只用于共享 context 建链：0 表示 UB 域内使用 UB_CTP、域间使用
UB_RTP（UBG）；1 表示全部远端使用 UB_CTP；2 表示全部远端使用 UB_RTP。
`N` 通常为实际建立的共享队列数量，也是发送核数；mode 4 忽略配置值并要求调用方建立
`world_size - 1` 个 Jetty。tiling 仍沿用原有 `aiv_num`
规则选择并启动全部 AIV（设备当前最多 72 个）；只允许 `1 <= N <= min(所选 AIV 数, 56)`。
`aiv_num=-1` 时使用设备 AIV 数。该数量关系在 tiling 中校验。
关闭共享 jetty 时沿用现有 MTE / 普通 URMA 路径。MTE 开启此开关会报参数错误。

```python
sym_buffer = BandwidthTestBuffer(
    group, comm_alg="urma", shared_jetty=True, shared_jetty_num=8, shared_jetty_comm_level=0)
y, receive_cnt = bandwidth_test(
    sym_buffer, x, dstrank_id, sym_buffer.world_size, x.size(0),
    mode=0, comm_alg="urma", shared_jetty=True, shared_jetty_num=8, batch_tokens=128)
```

Host 控制面建联使用 [hcomm 的 HCCL shared-channel 接口](https://gitcode.com/cann/hcomm/blob/7c391e58a83fbd64a999f3892498596e009034b3/include/hccl/hccl_channel.h)。已有 `HcclComm` 时，每个逻辑 Jetty
按以下顺序构造：`HcclChannelConfigCreate → SetInt(IS_SHARED_QUEUE) → SetStr(SHARED_QUEUE_TAG)
→ HcclChannelAcquireWithConfig → HcclChannelConfigDestroy`。这条 HCCL 接口内部完成底层
`HcommChannelCreateWithConfig`、READY 等待及 device ChannelEntity 填充。已有 HCCL 句柄继续复用公共
`hccl_common.h` 的 loader；新增的 shared-channel 配置接口也统一放在该公共 loader 中从 `libhcomm.so` 加载。

Host 使用 `torch_extension/shared_jetty_context.py` 和同目录 `csrc` 中的独立实现，按
`shared_jetty_comm_level` 选择协议和网络层。自动模式先按原 `HcclChannelContextBuilder` 的拓扑方式确定连续
UB_CTP 域，用它划分域内、域外 peer，再联合寻找本地 endpoint 完全相同的 UB_CTP 层和 UB_RTP 层。
全 UB_CTP/全 UB_RTP 模式要求一个 layer 覆盖所有远端并支持对应协议。一个 Jetty 的所有 peer descriptor
使用同一个 queue tag；不同 Jetty 使用不同 tag，因而得到相互独立的底层队列。本实现按 rank 排列 descriptor
并跳过本卡，标准接口按 descriptor 对应位置回填普通 `ChannelHandle`。Host 用第一组标准 Channel 查询远端 Win；随后仅为适配 asc-comm 的
shared-Jetty device batch ABI，为每个 Jetty 生成一个 `MultiChannelHandle`。当前 asc-comm 没有公开的
from-Channels 构造接口，所以公开 `MakeMultiChannelHandle` helper 会以相同 tag 再次 acquire；hcomm
shared-channel pool 复用前一步的 Channel/Jetty，不创建第二套物理 Jetty。最终 context 只保存
`MultiChannelHandle[jetty]`，不把普通 Channel 矩阵暴露给 kernel，也不读取 `ChannelEntity` 私有字段。
context 继续复用原有 `sharedJettyHandle_[1024]`，前 `shared_jetty_num` 项保存有效句柄。
HCCL shared queue 要求 V2 communicator、AIV engine、UB_CTP/UB_RTP 协议及相同的 local endpoint；不满足时
在 context 创建阶段明确报错，不隐式回退普通 channel。同一通信域的所有 rank 必须使用相同的 Jetty 数量、
通信层级、窗口大小和构造顺序；这是标准建链接口的调用前提。单 rank 无远端时跳过 Channel 建联。

业务 Win 由 manager 通过 `aclrtMalloc` 独立申请、清零，再由 `HcclCommMemReg` 注册；每个 descriptor 携带其
mem handle。Channel 建立后，Host 对每个 peer 调用 `HcclChannelGetRemoteMems` 并按 Win tag 精确匹配远端地址，
因此不再通过 Python collective 交换裸地址，也不再执行 `std::copy(remoteBuffers, ...)`。参照公共
`HcclChannelContextBuilder`，`HcclEngineCtxCreate/Copy` 仅保存算子自己的 `SharedJettyContext` blob；它既不申请
业务 Win，也不负责 Channel 建联。context 内的句柄表与远端 Win 表随后直接复制到 Torch NPU tensor。

Channel 和共享 Jetty 由 communicator 管理，Win 由 manager 持有。Python owner 保留 ProcessGroup，并应在所有
kernel 和图重放完成前保持存活；销毁时必须先停止使用并同步，且应先销毁 ProcessGroup/communicator，再释放
owner 持有的 Win。C++ 内部不查询或切换 NPU，调用方须保持正确的 current device。Win 初始化或注册失败会回滚
尚未交给 communicator 的内存；注册成功后若后续步骤失败，公开 HCCL API 没有可依赖的 AIV Win 反注册/
共享 Channel 逐项销毁接口，因此 Python wrapper 会先销毁已污染的 ProcessGroup，再释放 Win，并将原异常继续抛出；
调用方必须重建通信组。若 ProcessGroup 销毁本身也失败，则为避免释放仍被 HCCL 引用的地址，Win 会被有意保留到
进程退出，并在异常中明确报告。Host hcomm 当前支持 UB_RTP shared channel；但
Host 建联使用 hcomm 的 `COMM_PROTOCOL_UB_CTP`；CANN 9.2 的 Device `adv_api/hcomm` 只提供兼容名
`COMM_PROTOCOL_UBC_CTP`，因此两个 kernel 的 `Hcomm<>` 必须使用该 SDK 枚举。UB_RTP device batch 路径仍须用目标联调包验证。

tiling key 先选择通信算法，再在 URMA 下选择 `SHARED_JETTY`；kernel 独立文件为
`op_kernel/bandwidth_test_urma_shared_jetty.h`，支持 mode 0/1/2/3/4：

- mode 0/1：将输入 BS 的完整 token 连续分给 N 核，先在 workspace 写各核各对端计数，
  同步后按核号前缀计算紧凑发送位置。每核独占中转 token 区间，全部数据完成后再汇总并发送
  count / ready。其余已启动 AIV 继续参与全核同步、状态等待和接收搬运，mode 0 沿用原来的
  接收和输出排列。
- mode 2：将每个对端的 `floor(BS/world_size)` 个完整 token 分给 N 核；每核遍历全部远端。
  总流量口径不变，各 `(源 rank, 核)` 使用独立 epoch flag，空区间仍通知完成。
- mode 3：N 个 Jetty 都遍历全部远端，每个 Jetty 向每张卡发送 `floor(BS/world_size)` 个 token。
  `batch_tokens` 不生效，实际 batch 取 `min(floor(BS/world_size), tiling最大值)`；接收区按
  `(源 rank, Jetty)` 分槽，避免并发覆盖。
- mode 4：实际 Jetty 数固定为 `world_size - 1`，每个 Jetty 绑定一张远端卡并发送完整 BS。
  有效批量为 `min(batch_tokens or BS, BS, tiling最大值)`；接收区按源 rank 分槽。
- 每个发送核只操作自己的共享 jetty，并在切换 peer 前完成上一批 Drain；同一 Jetty 上的 Channel 不并发使用。
  数据、count、ready 和 epoch 的远端写都使用
  `MakeBatchHandle(MultiChannelHandle) → GetHandleRef(peerIndex) → WriteNbi(batch) → BatchCommit → Drain`。
  `batch_tokens` 是每核向一个目标卡提交的 payload token 数，mode 3 除外，尾批可以不足；
  统计表根据实际 BS 分核后的路由计算最大数据批量，状态通知不计入 payload。
  `WriteNbi`、`BatchCommit`、`Drain` 返回错误时触发 `Trap()`，中止 kernel，避免继续发布成功标志。

共享模式的新增 workspace 为 `N * align_up(world_size, 8) * 4` 字节计数表；
发送 UB 还包含一份按 world_size 对齐的本核计数。tiling 按 248 KiB 可用 UB 校验容量。
以下原有普通 URMA / MTE 布局说明中的按目标卡分核和 channel 锁仅适用于非共享路径。

当前交付基于静态分析；仍需在支持上述接口的 CANN 上编译，并验证多 rank 的
mode 0 精度、mode 1/2/3/4 重复调用、图重放、非整除 BS、空区间、56 jetty，以及联调包的
跨超 UB_RTP 和混合 UB_CTP/UB_RTP 拓扑。这里的静态检查不代表目标包编译或实机通信已验证。

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
更新后需同步重装算子包和 Torch 扩展；公开 ACLNN 第一段接口也声明并透传
`sharedJetty` / `sharedJettyNum`，`batch_tokens` 仍使用自动批量值 0。该包装层仍只创建 MTE context，
URMA 共享 jetty 继续由 Torch 接口传入独立 context。

Ascend950 支持 MTE（同一 UB 内存域，`comm_alg="mte"` 或空字符串）和
URMA（`comm_alg="urma"`，由已有 Torch context manager 建立 UB_CTP channel）。
URMA 要求安装支持 `MakeBatchHandle`、`BatchCommit` 和 `Drain(batchHandle)` 的 Hcomm 版本；
普通 URMA 路径还要求 `Lock/Unlock`。创建和调用必须选择相同算法：

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
测试脚本在 `config.py` 的 `ModelConfig.batch_tokens` 设置每次提交的 token/WQE 数。
mode 0/1/2 中 0 自动选择容量，正数由 tiling 校验后按原值提交；mode 3 忽略该值并取
`min(BS // world_size, maximum)`；mode 4 取 `min(配置或BS, BS, maximum)`。负数始终非法。
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
kernel 通过已有的通信算法 tiling key 和 `if constexpr` 选择 MTE/URMA；mode 2/3/4 不新增 tiling key。
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
mode 0/1/2 的总 token 数可以超过单批容量，`batch_tokens` 也可以超过 BS（实际数据不足时形成尾批）。
现有 INT32 计数表示检查和实际 tensor/通信窗口容量检查仍然有效。
mode 0/1/2 若传入正数超过 maximum，tiling 返回失败；mode 3/4 则把各自规则算出的批量截断到
maximum。合法值会打印 `requested / selected / maximum`。UB 最大值与 BS 无关。
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
本算子 kernel 直接使用 tiling 的有效批量，不额外校验批量范围或自动缩小批量；共享路径只断言
context 与 tiling 的 Jetty 数量一致，通信接口报错时仍按上文说明 Trap。SDK 内部检查沿用原实现。
kernel 只通过公开 Hcomm 接口提交和完成通信；普通 URMA 路径还通过公开 `Lock/Unlock` 串行化通道使用。
两条路径都不读取 channel entity、队列深度、游标或 batch handle 内部字段。
当前 Batch 接口未提供队列余量查询，也未承诺自动处理多批在途的溢出；因此每批提交后调用 Drain，再复用下一批。
多批数据的计时包含这些完成等待，不能假定更小的 batch 会降低耗时。
已核对 [Hcomm 公共接口](https://gitcode.com/cann/asc-comm/blob/1e575b02fb8bb3cf3e8f1e8f617972eac0d2a66e/include/aicore/hcomm/hcomm.h)，asc-comm 提交为
`1e575b02fb8bb3cf3e8f1e8f617972eac0d2a66e`，部署时须使用兼容版本。
该 SDK 的 `PollBatchCq` 在目标 CQ 游标等于 0 时直接返回，即使还有未完成请求。
长期打流中，32 位 `cqHead` 恰好在 Drain 边界回绕为 0 时可能漏等完成；
部署 SDK 应修复为仅在没有待回收 CQE 时跳过轮询，不能仅判断目标游标为 0。
本算子不为此增加 BS 上限、设备侧补丁或强序设置；目标 SDK 未修复前，不能保证触及该边界时的完成语义。
普通 URMA 数据阶段的通道锁覆盖 batch handle 创建至最终 Drain，避免其他核并发更新共享 SQ 游标。
普通 URMA 的 `SetStatus` 复用缓存的通道句柄，重新持锁并创建该阶段的 batch handle；
计数和 ready flag 均经 batch 接口提交并分别等待完成，保证先计数后就绪。
普通 URMA 在阶段间重新绑定，避免复用已解锁、可能被其他核推进过共享 SQ 游标的 handle；
不会在 token 或 commit 循环里重新获取 handle。共享 Jetty 路径则按上文所述由每个发送核独占一个 Jetty，
在切换 peer 前完成 Drain，不调用 `Lock/Unlock`。
`mode=1` 与现有 MTE 一致，包含发送、状态通知和等待，仅省略接收区到输出的搬运。
`mode=0` 在相同流程后调用与 MTE 一致的 `LocalWindowCopy`，按源卡和原 token 顺序拼接有效输出。
同一 context 的调用与图重放应串行下发，避免并发覆盖通信窗口。
切换 shape、mode 或测试配置前，所有 rank 必须完成此前调用并同步；各 rank 调用次数和顺序须相同。
普通 URMA owner 的 `aiv_num` 必须固定；该路径按核保存双缓冲位，改变核数需要同步后重建 owner。
共享 jetty owner 使用公共双缓冲位，同步完成后允许改变 `aiv_num`，但不得小于 `shared_jetty_num`。
`mode=0` 仅 `y[:receive_cnt.sum()]` 有效；`mode=1/2/3/4` 不搬运输出，两个输出均不保证有效，
测试脚本的图内循环因此继续使用原始输入。schema 将 context 声明为可变输入，
沿用仓库通信算子的副作用约定；目标环境仍须确认优化后的图保留全部通信次数和顺序。

`mode=2` 仅支持此 Torch context 的 URMA 路径，直接从 Win 发送 `BS // world_size` 个 token 到每个远端卡。
它在同一个 `BandwidthTestUrma::Process()` 的 `mode_ == 2` 分支中执行，沿用原有 URMA tiling key。
`Init()` 读取通信参数后提前返回，不做 mode 0/1 的业务初始化、ID 扫描和数据搬运；
`StreamFromWindow()` 只为有远端任务的核分配 `batch_tokens * 64` 字节 UB。
Win 使用紧凑 H 行，最小容量为 `1 MiB + 2 * BS * H * 2`；该模式不适用上文 mode 0/1 的双缓冲及 ID 预算公式。
flag 位于 Win 的 128 KiB 处（每个来源卡 32B），本核序号位于 160 KiB 处（每核 32B）。
普通 URMA mode 2 的尾批未满时将 flag 并入尾批 Commit；满批则先 Commit/Drain 数据，再单独提交 flag。
合并尾批的先后关系依赖当前 SDK 默认的 `fence=1` 和 CQE 语义。

`mode=3/4` 仅支持共享 Jetty。mode 3 的每个 Jetty 都向全部远端发送 `BS // world_size` 个 token，
按 tiling 最大容量决定一次提交或分批；mode 4 使用 `world_size - 1` 个 Jetty，一对一向远端发送完整 BS。
mode 3/4 与 mode 2 共用 Win 状态区和公开 Hcomm batch 接口，但采用独立的接收槽布局；脚本、Torch 入参检查
和 host tiling 使用同一套有效 Jetty 数、批量和 Win 容量公式。
共享 mode 2/3/4 始终先 Commit/Drain 完整数据批，再单独 Commit/Drain flag，不将 flag 合入 payload 尾批。
有远端任务的核每次调用递增本核序号；等待时允许对方序号超前一轮，支持 uint32 回绕。
不读取 x/dstrank_id 内容，仍保留原 tensor shape/type 接口约定；完整使用和计时口径见主目录 README。

需在装有 CANN、PyTorch、torch_npu 和兼容 TorchAir 的 Ascend950 多 rank 环境
分别验证 eager 精度、图捕获及重复重放。本地静态检查不能替代上述验证。
两份接收实现均包含状态 buffer 字节数、DataCopy 块长及 UB 记账修正；host 按实际缓冲区检查可用 UB。
普通 URMA kernel 沿用原有通信返回值处理；共享路径在 WriteNbi、BatchCommit 或 Drain 失败时 Trap。
调用方仍负责保证上述通信前置条件。
这些修改仍需在目标 SDK 下验证编译和多卡可见性。
