# npu\_get\_mega\_moe\_ccl\_buffer\_size

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575834_section14441124184111"></a>

-   API功能：
    
    需与[mega\_moe](mega_moe.md)配套使用，用于计算mega_moe算子所需的HCCL通信buffer\_size大小（单位：MB）。
-   计算公式：

    -   计算dispatch阶段存储所有专家接收到整个EP域中的Token数量空间大小：

        $$dispatch\_token\_per\_expert = ep\_world\_size * Align128(ep\_world\_size * local\_moe\_expert\_num) * sizeof(int32)$$

        其中ep\_world\_size为专家并行通信域大小

    -   计算dispatch量化后输出的token及scales存放需要的空间大小：

        $$quant\_token\_scales = Align512(num\_max\_token\_per\_rank * topK * (hidden + hidden / 32))$$

        其中num\_max\_token\_per\_rank为每张卡上的token数量
    -   计算combine阶段处理发回数据所需空间大小：

        $$combine\_out = Align512(num\_max\_token\_per\_rank * topK * y\_out\_dtype\_size)$$
    
        其中y\_out\_dtype\_size为输出的token类型转为字节数
    -   计算ccl\_buffer\_size大小：

        $$ccl\_buffer\_size = Align2(Align1MB(peermem\_data\_offset + dispatch\_token\_per\_expert + quant\_token\_scales + combine\_out) / 1MB) / 2$$
        其中peermem\_data\_offset为全卡软同步所需空间大小`60 * 1024B`

## 函数原型

```
npu_get_mega_moe_ccl_buffer_size(ep_world_size: int, moe_expert_num: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, dispatch_quant_mode: int = 0, dispatch_quant_out_type: int = 28, combine_quant_mode: int = 0, comm_alg: str = "") -> int:
```

## 参数说明

-   **ep\_world\_size** (`int`)：必选参数，表示通信域的大小。取值范围[2, 768]。
-   **moe\_expert\_num** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]。
-   **num\_max\_dispatch\_tokens\_per\_rank** (`int`)：必选参数，表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小。
-   **num\_topk** (`int`)：必选参数，表示选取topK个专家，目前仅支持 6 或 8 。
-   **hidden** (`int`)：必选参数，表示hidden size隐藏层大小。目前仅支持 4096、5120、7168。
-   **dispatch\_quant\_mode** (`int`)：可选参数，dispatch通信时量化模式，目前仅支持4（MX模式）。
-   **dispatch\_quant\_out\_type** (`int`)：可选参数，dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。
-   **comm\_alg** (`string`)：暂不支持该参数，使用默认值即可。

## 输出说明

`int`

表示计算得到的ccl\_buffer\_size大小，单位为MB。

## 约束说明

-   npu\_get\_mega\_moe\_ccl\_buffer\_size、mega\_moe必须配套使用。
-   本文公式中的“/”表示整除。

## 调用示例

```python
import os
import torch
import torch_npu
from npu_ops_transformer.ops import npu_get_mega_moe_ccl_buffer_size

server_num = 1
rank_per_dev = 2
world_size = server_num * rank_per_dev
num_experts = 8  # moe专家数
BS = 256  # token数量
H = 4096  # 每个token的长度
topK = 6

buffer_size = npu_get_mega_moe_ccl_buffer_size(world_size, num_experts, BS, topK, H)
os.environ['HCCL_BUFFSIZE'] = f'buffer_size'
print(f"buffer_size is {buffer_size}")
print("run ccl_buffer_size success.")
```
