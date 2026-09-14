# ffn_worker_batching

## 产品支持情况

本页的异步混层示例支持 Ascend 950PR/Ascend 950DT。

## 功能说明

根据调度上下文中的专家号排序，重排 token、scale 和来源索引，并统计各专家的 token 数。算子只搬运数据，不执行 FFN 矩阵计算或量化计算。

异步接收（`need_schedule=1、sync_flag=1`）允许同一 micro batch 中已就绪的不同 session 属于不同层，使用跨层专家号：

```text
experts_per_layer = expert_num / layer_num
new_expert_id = layer_id * experts_per_layer + expert_id
```

`expert_num`为本卡参与层的专家总数；`expert_id`为层内本卡专家号。`layer_id`位于各 session 的`FfnDataDesc`描述符中，不是独立的 Torch 入参，也不是 NORM 使用的`layer_ids_buf`。

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

| 参数名 | 类型 | 必选/可选 | 描述 |
| --- | --- | --- | --- |
| schedule_context | Tensor | 必选 | 连续一维 NPU int8 张量，至少1024字节，按`ScheduleContext`布局保存维度及有效Device地址。 |
| expert_num | int | 必选 | 专家数，范围[1,8192]。异步混层模式表示本卡所有参与层的专家总数。 |
| max_out_shape | List[int] | 必选 | `[A, BS, K, H]`，分别为session容量、micro batch大小、每token的专家选择数、逻辑hidden维度。 |
| token_dtype | int | 可选 | 0：FP16；1：BF16；2：INT8；3：FP8 E5M2；4：FP8 E4M3FN；5：FP4 E2M1。示例使用0。 |
| need_schedule | int | 可选 | 0：NORM；1：RECV。示例使用1，直接消费接收描述符。 |
| layer_num | int | 可选 | 参与层数。异步RECV必须大于0且整除`expert_num`；示例使用2。 |
| sync_flag | int | 可选 | 0：同步接收，等待全部session；1：异步接收，处理ready快照选中的session。仅RECV解释该属性。 |
| mx_output_uint8 | bool | 可选 | 默认False。True时MX scale以uint8返回，FP4 y也以uint8返回，保持原始编码；FP16示例无需设置。 |

## 返回值说明

令`Y=A*BS*K`，返回值顺序如下。示例采用FP16，输出shape如下表。

| 返回值 | dtype | shape | 描述 |
| --- | --- | --- | --- |
| y | float16 | `[Y,H]` | 按跨层专家号排序后的token。 |
| group_list | int64 | `[expert_num,2]` | 每个非空专家占一行，内容为`[跨层专家号, token数]`，其余行补零。 |
| session_ids | int32 | `[Y]` | 原始session编号。 |
| micro_batch_ids | int32 | `[Y]` | 原始micro batch编号。 |
| token_ids | int32 | `[Y]` | 原始micro batch内的token编号。 |
| expert_offsets | int32 | `[Y]` | token的专家选择位置，范围[0,K)。 |
| dynamic_scale | float32 | `[Y]` | FP16/BF16下无有效含义。量化数据的scale布局见[V2接口文档](aclnnFfnWorkerBatchingV2.md)。 |
| actual_token_num | int64 | `[1]` | 本次有效输出行数。 |

`y`、来源索引及有意义的scale只读取前`actual_token_num`行，不读取未使用的输出容量。

## 约束说明

- 各层在本卡的专家数相同。`layer_id`为[0,layer_num)内的连续编号，输入专家号必须为层内局部编号，不能预先加层偏移。
- 非法层号使该已选中描述符的token全部被mask；专家号不在[0,experts_per_layer)内时被mask，包含原有mask值。
- 上下文中的地址必须来自Device分配，不能直接填CPU指针。上下文以及它引用的token、描述符缓冲区必须保持存活，直到执行完成。
- 接收过程会清理已选中描述符的flag及expert_ids，并推进轮询游标。重复调用或图replay前必须重新发布数据；没有ready数据时算子会继续等待。
- 示例在同一stream上先准备数据再调用消费者，以演示异步选择语义。实际跨stream或跨设备生产者应遵守“先完成数据写入，再发布ready”的接收协议。
- 同步RECV和NORM保留原专家编号规则，不使用该跨层公式。

## 调用示例

完整脚本：[torch_ffn_worker_batching_async.py](../examples/torch_ffn_worker_batching_async.py)。不依赖ops-test-kit或测试辅助模块。

按[Torch扩展构建说明](../../../torch_extension/README.md)安装whl，并加载配套的自定义OPP环境后，从仓库根目录运行：

```bash
# 单算子直调。
python ffn/ffn_worker_batching/examples/torch_ffn_worker_batching_async.py --mode eager

# npugraph_ex完整图及第二次replay；每次调用前原地重新发布描述符。
python ffn/ffn_worker_batching/examples/torch_ffn_worker_batching_async.py --mode graph

# TorchAir GE完整图及第二次执行，同样校验异步混层结果和描述符状态。
python ffn/ffn_worker_batching/examples/torch_ffn_worker_batching_async.py --mode ge
```

默认导入`cann_ops_transformer`。选择性打包时可通过`--package <实际包名>`或环境变量`FFN_TORCH_PACKAGE`指定，设备通过`--device 0`指定。`--mode graph`使用`npugraph_ex`；`--mode ge`使用TorchAir GE，需安装配套TorchAir及包含`sync_flag`属性的自定义OPP。两种图模式均设置`fullgraph=True、dynamic=False`，失败直接报错，不回退到eager或记为SKIP。

示例设置`expert_num=8、layer_num=2`，每层4个专家。session 0属于layer 1，session 1属于layer 0，两者使用相同的层内专家号`[[1,0],[3,1]]`。micro batch 0未就绪，micro batch 1仅这两个session就绪，session 2不应被消费。

核心调用为：

```python
y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num = (
    ffn_worker_batching(
        schedule_context, 8, [3, 2, 2, 256],
        token_dtype=0, need_schedule=1, layer_num=2, sync_flag=1,
    )
)
```

预期有效token数为8，排序后的跨层专家号为`[0,1,1,3,4,5,5,7]`；`group_list`的非空行为：

```text
[[0,1], [1,2], [3,1], [4,1], [5,2], [7,1]]
```

脚本同时校验token搬运、来源索引、group_list、已消费描述符清理以及未选中描述符保持不变；失败时抛出异常，不跳过。

### TorchAir GE调用方式

脚本中的GE分支采用以下编译方式，调用参数和异步混层输入与直调相同：

```python
import torchair
from torchair.configs.compiler_config import CompilerConfig

backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
compiled_call = torch.compile(call, backend=backend, fullgraph=True, dynamic=False)

# 原地重新发布描述符，保持context引用的Device地址不变。
case.publish()
outputs = compiled_call(case.context)
torch.npu.synchronize()
case.check(outputs)
```

`call`、`case`见完整脚本。图中仅包含算子调用，发布输入、读取有效输出行数及结果校验均在图外执行。第二次执行前同样重新发布描述符，避免消费已清空的输入。

本示例仅覆盖FP16异步混层。当前GE converter不支持FP4及MX uint8输出视图；不能将本示例的结论扩展到这些类型。

验证记录：2026-09-14在Ascend950DT_9574、CANN 9.2.0环境使用配套OPP/whl运行本脚本，GE首次编译执行和第二次执行均通过；每次有效token数为8，跨层专家号、token来源、group_list和描述符消费状态均符合预期。
