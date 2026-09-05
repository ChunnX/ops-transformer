# Torch Extension

实现复用 MegaMoe 的 `OpBuilder`、`CommContextManager` 和 `ACLNN_CMD`。
现有安装脚本将三级目录收集到 `cann_ops_transformer.ops.tools.bandwidth_test`，
C++ 源码放入 `csrc/tools/`，无需修改公共打包逻辑。

先安装包含 BandwidthTest 的算子包并加载其环境，再从仓库根目录安装扩展：

```bash
python -m pip install ./torch_extension --no-build-isolation
```

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

当前 kernel 只实现 Ascend950、同一 UB 内存域内的 MTE 通信，`comm_alg` 为
`mte` 或空字符串。`x` 必须为二维 FP16，`dstrank_id` 为长度等于 bs 的 INT32；
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
窗口至少包含 1 MiB 状态区和两个数据半区，每个半区需要
`(world_size + 1) * max_bs * align_up(hidden * 2, 512)` 字节。
同一 context 的调用与图重放应串行下发，避免并发覆盖通信窗口。
`mode=0` 仅 `y[:receive_cnt.sum()]` 有效；`mode=1` 只发数据，两个输出均不保证有效，
测试脚本的图内循环因此继续使用原始输入。schema 将 context 声明为可变输入，
沿用仓库通信算子的副作用约定；目标环境仍须确认优化后的图保留全部通信次数和顺序。

需在装有 CANN、PyTorch、torch_npu 和兼容 TorchAir 的 Ascend950 多 rank 环境
分别验证 eager 精度、图捕获及重复重放。本地静态检查不能替代上述验证。
主线 kernel 的极端尺寸 UB 占用及接收状态 buffer 分配还存在静态疑点，
本次仅适配 Torch 接口，未修改 kernel，也不据此保证全部尺寸的运行安全性。
