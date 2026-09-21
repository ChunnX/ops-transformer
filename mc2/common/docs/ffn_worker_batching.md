# ffn\_worker\_batching

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35） | 支持 |
| Atlas A2 / A3 | 本 Torch 接口不支持 |

按专家 ID 排序并重排 Attention 发送的 token，调用 `aclnnFfnWorkerBatching`。实现位于 `ffn/ffn_worker_batching/torch_extension`，通过仓库 `OpBuilder` 首次 JIT 编译 C++ wrapper；kernel 使用已安装的自定义 OPP 二进制。

## 函数原型

```python
cann_ops_transformer.ffn_worker_batching(
    schedule_context: torch.Tensor,
    expert_num: int,
    max_out_shape: List[int],
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
    layer_num: int = 0,
    sync_flag: int = 0,
    mx_output_uint8: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

同时注册 `torch.ops.cann_ops_transformer.ffn_worker_batching`，参数及返回值一致。六种 dtype 均支持前向 eager、Meta/FakeTensor 和固定形状的 `torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False)` 推理图。图模式复用 ACLNN 下发，无需 GE Converter；MX（token_dtype=3/4/5）图模式需设置 `mx_output_uint8=True`；当前不支持 TorchAir GE 后端和反向。图模式的二级指针与状态约束见下文。

## 参数

| 参数 | 说明 |
| --- | --- |
| schedule_context | 必选；连续一维 NPU `torch.int8` 张量，至少 1024 字节，ND 格式；保存 ScheduleContext 及真实设备指针。不支持非连续输入；eager wrapper 不自动复制上下文，图框架可为函数化建立副本并回写。 |
| expert_num | 必选；专家数，范围 1..8192。 |
| max_out_shape | 必选；`[A,BS,K,H]`，A=1..1024，BS>0，K=1..64，H>0；K 是含共享专家后的选择数。上下文中的 A/BS/K 必须与属性一致。 |
| token_dtype | 可选；0 FP16、1 BF16、2 INT8、3 FP8 E5M2、4 FP8 E4M3FN、5 FP4 E2M1，默认 0；FP4 要求 H 为偶数。 |
| need_schedule | 可选；0 直接 batching；1 轮询接收数据后 batching，默认 0。 |
| layer_num | 可选；0..expert_num，默认 0，与 ACLNN 属性一致。 |
| sync_flag | 可选；默认 0；仅 need_schedule=1 生效。0 等当前 micro batch 全部 A 个 session ready；1 循环扫描，处理首个非空 ready 快照。无数据时持续等待。 |

| mx_output_uint8 | 可选；默认 False 保留原生 dtype。True 时 MX scale 和 FP4 y 返回 uint8 原始编码，FP8 y 保持原生 dtype；对 token_dtype=0/1/2 无影响。仅为 Torch 输出表示选项，不增加 ACLNN 属性。 |

## 返回值

`Y=A*BS*K`，下表为 `mx_output_uint8=False` 时的返回顺序与类型。全部张量位于输入所在 NPU，连续存储。

| 输出 | PyTorch shape | dtype |
| --- | --- | --- |
| y | `[Y,H]`；FP4 为 `[Y,H/2]` | 随 token_dtype：float16 / bfloat16 / int8 / float8_e5m2 / float8_e4m3fn / float4_e2m1fn_x2 |
| group_list | `[expert_num,2]` | int64，每行 `[expert_id, token_count]`，末尾空行 `[0,0]` |
| session_ids | `[Y]` | int32 |
| micro_batch_ids | `[Y]` | int32 |
| token_ids | `[Y]` | int32 |
| expert_offsets | `[Y]` | int32 |
| dynamic_scale | INT8 为 `[Y]`；MX 为 `[Y,ceil(H/32)]`；FP16/BF16 为 `[Y]` 但无有效值 | INT8/FP16/BF16 为 float32；MX 为 float8_e8m0fnu |
| actual_token_num | `[1]` | int64 |

FP4 的 `float4_e2m1fn_x2` 每个 PyTorch 元素存两个逻辑 FP4 值，低半字节在前。单字节 `[1,1]` 输出使用 `(2,1)` stride 明确最后一维为打包轴；单元素张量仍是连续的，Meta 返回相同 stride。公共 ACLNN 转换层把最后一维及 stride 转换为逻辑元素单位，底层 ACLNN 输出仍是 `[Y,H]`。可用 `y.view(torch.uint8)` 查看原始编码，不将 FP4 当作 INT8 数值计算。

`mx_output_uint8=True` 时，仅 MX `dynamic_scale` 和 FP4 `y` 的 dtype 改为 uint8，shape、stride 和编码不变。C++ 在原生类型 ACLNN 下发后创建 dtype view，Meta 同步返回对应 uint8 元数据。

仅 `y`、四个索引及有效 scale 的前 `actual_token_num` 行有意义。接口不主动读取该计数或裁剪输出，避免引入 host/device 同步；需要在 Python 读取时，由调用方显式 `.item()`。

## 上下文与流约定

- RECV 会原地推进上下文的 `polling_index`；schema 使用 `Tensor(a!)` 标记这一副作用。
- 原 `token_info_buf` 的选中行在 prepare 读入后立即清 ids/flag；未选中行保持不变。后续 gather 仍读取原 `token_data_buf`，遵循已确认的生产者不覆盖本次消费数据的约定。
- 指针指向的 token、描述符及 NORM 索引缓冲必须位于同一 NPU。Python 调用方或通信内存管理器负责保持这些缓冲存活，直至当前流执行完成；wrapper 无法从裸指针恢复 Tensor 所有权。
- 输入数据准备、flag 发布和消费之间的流依赖由调用方维护。同一上下文的消费者调用应串行；生产者在数据写入完成后发布 flag。跨流共享 Tensor 时按 PyTorch 的流生命周期约定使用 `wait_stream` / `record_stream` 或同步。
- wrapper 的 DeviceGuard 覆盖输出申请、workspace 申请和 ACLNN 下发，使用输入设备的当前流，不插入全局设备同步。
- MX scale 紧跟有效 token 字节；上下文 `attn_to_ffn_token_size` 是整行字节步长，可以包含行尾 padding。

## 构建与安装

需要支持上述 dtype 的 PyTorch/torch_npu（本次使用 2.9.0 / 2.9.0.post8）、CANN Ascend950 环境、C++ 编译器和 ninja。自定义 OPP 必须包含 `syncFlag` 参数的新 ACLNN 签名及 arch35 二进制。

```bash
source /path/to/cann/set_env.sh
# 先构建/安装 OPP；也可复用已验证且源码匹配的安装包。
cd /path/to/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching -j8
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/path/to/custom_opp
source /path/to/custom_opp/vendors/custom_transformer/bin/set_env.bash
cd torch_extension
python -m pip install ninja
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps dist/*.whl
```

首次真实 NPU 调用编译 wrapper；仅 import 或 Meta/FakeTensor 调用不触发本算子的 JIT。

## 调用示例

下面使用真实 NPU 地址构造最小 FP16 RECV 上下文；token 和描述符变量保留至执行完成。生产系统通常由调度器或通信接口管理这些缓冲。

```python
import struct
import torch
import torch_npu
import cann_ops_transformer as ops

h = 32
tokens = torch.arange(h, dtype=torch.float16, device='npu').reshape(1, 1, 1, 1, h)
# [A,M,flag + layer_id + BS*K ids]；只发送 expert 0 的一个 token。
desc = torch.tensor([[[1, 0, 0]]], dtype=torch.int32, device='npu')
context_cpu = torch.zeros(1024, dtype=torch.int8)
raw = context_cpu.numpy()
for offset, value in [(0, 1), (4, 1), (8, 1), (12, 1), (16, 1), (20, h * 2)]:
    struct.pack_into('<I', raw, offset, value)
for offset, tensor in [(384, desc), (400, tokens)]:
    struct.pack_into('<Q', raw, offset, tensor.data_ptr())
    struct.pack_into('<Q', raw, offset + 8, tensor.numel() * tensor.element_size())
context = context_cpu.npu()
outputs = ops.ffn_worker_batching(context, 1, [1, 1, 1, h], need_schedule=1, sync_flag=1)
y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, scales, count = outputs
torch.npu.synchronize()
assert count.item() == 1
assert torch.equal(y.cpu()[0], tokens.cpu().reshape(h))
assert desc.cpu().tolist() == [[[0, 0, 2147483647]]]
```

本例完成后 flag 已清；再次 RECV 前需要生产者重新发布数据，不能直接重复调用并期待返回。

## 验证

用例：`ffn/ffn_worker_batching/tests/st/arch35/verify_torch_interface.py`。使用真实设备地址检查六种 dtype、NORM/同步/异步 RECV、FP4 打包、非对齐 scale、行尾 padding、空有效输出、上下文连续调用、非默认流及延迟 flag 发布；另检查 Meta/FakeTensor、参数拒绝与 schema 可变性。执行时已安装扩展可直接运行：

```bash
python ffn/ffn_worker_batching/tests/st/arch35/verify_torch_interface.py
```

最新 eager/Meta 回归 **98/98**、图模式 **46/46** 通过。三个 verify 命令的分工、环境与运行顺序见 [验证命令指南](../../../../ffn/ffn_worker_batching/docs/arch35_verify_guide.md)。

## npugraph_ex 图模式

六种 token_dtype 均可用于固定形状图；MX 类型需设置 `mx_output_uint8=True`。`torch_npu` 导入时注册 `npugraph_ex` 后端；无需额外安装 TorchAir 或编写 GE Converter。先按上面的示例建立真实设备指针上下文，然后将消费函数编译为图：

```python
import torch
import torch_npu
import cann_ops_transformer as ops

# expert_num、max_out_shape 和 dtype/调度属性在本图中固定。
# schedule_context 及其指向的缓冲由外部管理，并已按通信协议准备好。
def consume(schedule_context):
    return ops.ffn_worker_batching(
        schedule_context, expert_num, max_out_shape,
        token_dtype=0, need_schedule=1, sync_flag=1, mx_output_uint8=True)

consume_graph = torch.compile(
    consume, backend="npugraph_ex", fullgraph=True, dynamic=False)
outputs = consume_graph(schedule_context)
# 后续调用重放图，每次仍执行设备端轮询并消费一次就绪快照。
# actual_token_num.item() 如有需要应放在编译函数之外。
```

### 执行流程与状态传递

1. Dynamo 追踪公开接口，保留 `cann_ops_transformer::ffn_worker_batching` 节点。`fullgraph=True` 使断图显式报错，不能静默退回 eager。
2. Meta 根据属性生成八个输出的 shape/dtype/stride；不读取上下文或 ready 标志。实际 token 数在设备运行时写入固定形状的 `actual_token_num`。
3. `Tensor(a!) schedule_context` 告诉函数化处理输入会被修改。框架负责保持同一上下文前后调用的依赖及更新回写；不返回 context，符合原地自定义算子的 schema 约定。
4. `npugraph_ex` 捕获 C++ wrapper 下发的 ACLNN 任务，后续 replay 执行相同的设备任务。内核每次读取当前指针缓冲、轮询 ready、清选中描述符并推进游标，不能把首次就绪快照当作常量。
5. 返回输出后，调用方可在图外读取计数。输出存储可能被后续 replay 复用，需要长期保存的结果应在下次 replay 前复制并完成必要的流同步。

### 支持边界

- 图模式支持 token_dtype=0..5；MX（3/4/5）设置 `mx_output_uint8=True`，避免原生 FP4/E8M0 输出阻塞框架函数化分解。默认 False 保持原生 eager 返回类型。
- 开启该选项时，MX `dynamic_scale` 返回 uint8；FP4 `y` 返回 `[Y,H/2]` uint8；FP8 `y` 仍为原生 FP8。这里只重解释存储，不做数值转换，ACLNN 使用原生类型描述符。图外需要原生类型时，使用 `scales.view(torch.float8_e8m0fnu)` 和 `y.view(torch.float4_e2m1fn_x2)`（仅 FP4），不要用 `.to(dtype)`。图内下游算子需支持此字节表示，不能据此推断任意 MX 算子链均可入图。

- 固定 shape、dtype 和标量属性的单卡推理；本次未验证动态 shape、反向、多卡、跨进程图缓存和性能收益。
- 保持同一个上下文对象及全部二级指针缓冲存活、地址稳定；不使用 TEST_MAGIC 相对地址。图框架不能管理裸指针背后的 Tensor 所有权。
- 同一上下文的消费者串行调用。外部生产者必须先写数据再发布 flag，并保证消费期间数据不被覆盖。已验证同图两次消费和丢弃第一次输出仍保留副作用。
- 二级指针没有作为独立 Tensor 参数进入图。不要把这些缓冲的普通 Tensor 读写或通信生产者任意放进同一编译图，编译器无法推导这类隐藏依赖；当前验证范围是外部维护缓冲、图内串行消费。
- 不额外设置会执行算子的 warmup/eager 预跑选项：每次真实执行都会消费状态。`need_schedule=1` 无数据时会等待，首次调用也必须遵循数据发布协议。
- 当前不注册 GE Converter。GE 探针选用了 CANN 内置的旧封装，缺少 `sync_flag`，导致 tiling 属性错位；本报告不把 GE 图列为支持项。

完整测试、复现命令和结果见 [图模式验证报告](../../../../ffn/ffn_worker_batching/docs/arch35_torch_graph_validation.md)。实现依据为 TorchAir 的 [npugraph_ex 快速上手](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md) 与 [In-place 算子入图](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/custom_op_graph/in_place_op_cases.md)，行为以本机版本实测为准。
