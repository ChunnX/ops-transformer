# cann_ops_transformer.attention_worker_combine

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 函数原型

```python
cann_ops_transformer.attention_worker_combine(schedule_context, expert_scales, layer_id, hidden_size, *, token_dtype=0, need_schedule=0) -> (Tensor, Tensor)
```

## 功能说明

在Attention与FFN分离部署场景下，对FFN侧返回的专家token数据进行加权融合，并输出下一层ID。

Ascend 950PR/Ascend 950DT同时支持MXFP8（E5M2、E4M3FN）和MXFP4（E2M1）反量化融合。

## 参数说明

设R为token数，K为路由专家数，H为hidden_size，其中R > 0，1 ≤ K ≤ 64，H > 0。

- **schedule_context**（`Tensor`）：调度上下文，保存有效的ScheduleContext结构体。数据类型为`torch.int8`，shape为`(1024,)`。
- **expert_scales**（`Tensor`）：非量化场景为专家权重，数据类型为`torch.float32`，shape为`(R, K)`；MXFP反量化场景为E8M0缩放因子，使用`torch.uint8`承载E8M0编码，也支持`float8_e8m0fnu`类型，shape为`(R, K+1, S)`，其中S为Ceil(H/32)向上取整到偶数，K+1包含共享专家。
- **layer_id**（`Tensor`）：当前层ID，数据类型为`torch.int32`，shape为`(1,)`。
- **hidden_size**（`int`）：每行token的元素数H。
- **token_dtype**（`int`）：可选关键字参数，默认值为0。0表示FP16，1表示BF16，2表示MXFP8_E5M2，3表示MXFP8_E4M3FN，4表示MXFP4_E2M1。Atlas A2/A3训练推理系列产品仅支持0和1；Ascend 950PR/Ascend 950DT支持0～4。
- **need_schedule**（`int`）：可选关键字参数，默认值为0。取值为0或1，0表示不等待token数据就绪，1表示等待token数据就绪。

## 返回值

- **y**（`Tensor`）：融合结果，shape为`(R, H)`。`token_dtype=0`时数据类型为`torch.float16`，其余为`torch.bfloat16`。
- **next_layer_id**（`Tensor`）：下一层ID，值为`layer_id+1`，数据类型为`torch.int32`，shape为`(1,)`。

## 约束说明

所有输入须为同一NPU设备上的连续Tensor，数据格式为ND，不支持空Tensor。`schedule_context`及其关联数据须由上游调度流程准备。

## 安装

在本仓库的`torch_extension`目录执行`python -m pip install .`，首次调用时会JIT编译C++适配层。运行环境需安装支持该算子的CANN/自定义算子包及匹配的PyTorch、torch_npu。

## 调用示例

以下为BF16场景的完整调用示例，使用`torch_npu._afd.create_schedule_context_holder`构造调度上下文，需运行环境提供该接口。窗口中的token数据初始化为0，`need_schedule=0`时直接计算。

```python
import torch
import torch_npu
import cann_ops_transformer


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


torch.npu.set_device(0)
R, K, H = 256, 6, 4096

# Attention窗口包含就绪标志和K+1行专家数据，BF16每个元素占2字节。
info_bytes = align_up(4 * R * (K + 1), 512)
data_bytes = 2 * R * (K + 1) * H
window_bytes = align_up(info_bytes + data_bytes, 2 * 1024 * 1024)
window = torch.zeros(window_bytes, dtype=torch.uint8, device="npu")

holder = torch_npu._afd.create_schedule_context_holder(
    schedule_mode=1,
    session_num=4,
    micro_batch_num=1,
    micro_batch_size=R,
    selected_expert_num=K + 1,
    expert_num=257,
    attn_to_ffn_token_size=2 * H,
    ffn_to_attn_token_size=2 * H,
    attention_window=window.data_ptr(),
    attention_window_size=window.numel(),
)

try:
    expert_scales = torch.ones((R, K), dtype=torch.float32, device="npu")
    layer_id = torch.zeros((1,), dtype=torch.int32, device="npu")
    y, next_layer_id = cann_ops_transformer.attention_worker_combine(
        holder.get_schedule_context_tensor(),
        expert_scales,
        layer_id,
        H,
        token_dtype=1,
        need_schedule=0,
    )
    torch.npu.synchronize()
    # 本例窗口中的token数据全为0，因此融合结果为0，下一层ID为1。
    print(y.shape, y.dtype)  # torch.Size([256, 4096]) torch.bfloat16
    print(torch.count_nonzero(y).item(), next_layer_id.item())  # 0 1
finally:
    holder.stop_schedule()
```

## 验证

安装扩展包并配置CANN及自定义OPP环境后，在代码仓根目录执行：

```bash
# 五种token类型，分别验证调度关闭和开启，并与CPU标杆比较。
python attention/attention_worker_combine/tests/st/test_torch_extension.py
# 使用原生E8M0 Tensor传入反量化scale。
python attention/attention_worker_combine/tests/st/test_torch_extension.py --native-scales
# TorchAir静态图、动态图；分别验证need_schedule=0和1。
python attention/attention_worker_combine/tests/st/test_torch_extension.py --backend ge
python attention/attention_worker_combine/tests/st/test_torch_extension.py --backend ge --dynamic
# 扩展覆盖BS/K/H边界、微批次回绕、特殊编码和storage offset；可输出CSV结果。
python attention/attention_worker_combine/tests/st/test_torch_extension.py --suite --output eager_results.csv
# 同一个动态图连续改变R，并重复执行检查context/flags消费。
python attention/attention_worker_combine/tests/st/test_torch_extension.py --suite --backend ge --dynamic --output dynamic_results.csv
```

验证脚本额外依赖NumPy和ml_dtypes。图模式需要匹配的TorchAir及包含AttentionWorkerCombine原型的算子包。开启调度时会消费flags并更新context，重复调用前须由生产者准备下一批数据；外部缓冲区须在执行期间保持有效。
