# ElasticBuffer.get\_moe\_ep\_ccl\_buffer\_size<a name="ZH-CN_TOPIC_0000002343094194"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575834_section14441124184111"></a>

-   API功能：

    需与[ElasticBuffer.dispatch](ElasticBuffer_dispatch.md)和[ElasticBuffer.combine](ElasticBuffer_combine.md)配套使用，用于计算dispatch和combine算子所需的HCCL通信buffer\_size大小（单位：MB）。该接口为静态方法，可在初始化`ElasticBuffer`前调用。
-   计算公式：

    -   计算每个token所需空间：

        $$per\_token\_bytes = Align512(2 \times H)$$

    -   计算最小buffer大小：

        $$minimum\_buffer\_size = 2 \times ep\_world\_size \times num\_max\_tokens\_per\_rank \times per\_token\_bytes$$

    -   计算ccl\_buffer\_size大小（单位MB）：

        $$ccl\_buffer\_size = Align2(Align1MB(minimum\_buffer\_size) / 1MB) / 2$$

        其中 Align512(x) = ((x+512-1)/512)\*512, Align1MB(x) = ((x+1048576-1)/1048576)\*1048576, Align2(x) = ((x+2-1)/2)\*2

## 函数原型<a name="zh-cn_topic_0000002203575834_section45077510412"></a>

```
ElasticBuffer.get_moe_ep_ccl_buffer_size(world_size, num_max_tokens_per_rank, hidden, num_experts, topk) -> int
```

## 参数说明<a name="zh-cn_topic_0000002203575834_section112637109430"></a>

-   **world\_size** (`int`)：必选参数，表示EP通信域的大小（即参与EP通信的卡数）。取值范围\[2, 768\]。
-   **num\_max\_tokens\_per\_rank** (`int`)：必选参数，表示每张卡上的最大token数量上限。
-   **hidden** (`int`)：必选参数，表示hidden size隐藏层大小。取值范围\[1024, 8192\]。
-   **num\_experts** (`int`)：必选参数，MoE专家总数量，取值范围\[1, 2048\]，且满足num\_experts % ep\_world\_size = 0。
-   **topk** (`int`)：必选参数，表示选取topK个专家，取值范围\[1, 32\]。

## 输出说明<a name="zh-cn_topic_0000002203575834_section22231435518"></a>

`int`

表示计算得到的ccl\_buffer\_size大小，单位为MB。将该值设置为HCCL\_BUFFSIZE环境变量即可满足通信域的内存需求。

## 约束说明<a name="zh-cn_topic_0000002203575834_section12345537164215"></a>

-   `ElasticBuffer.get_moe_ep_ccl_buffer_size`、[ElasticBuffer.dispatch](ElasticBuffer_dispatch.md)和[ElasticBuffer.combine](ElasticBuffer_combine.md)必须配套使用。
-   计算得到的ccl\_buffer\_size需通过环境变量HCCL\_BUFFSIZE设置，每个通信域独占一组"2 \* HCCL\_BUFFSIZE"大小的内存。
-   本文公式中的"/"表示整除。

## 调用示例<a name="zh-cn_topic_0000002203575834_section14459801436"></a>

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from npu_ops_transformer.ops.elastic_buffer import ElasticBuffer

world_size = 2
num_experts = 8
num_max_tokens_per_rank = 128
hidden = 4096
topk = 4

buffer_size = ElasticBuffer.get_moe_ep_ccl_buffer_size(
    world_size, num_max_tokens_per_rank, hidden, num_experts, topk)
os.environ['HCCL_BUFFSIZE'] = str(buffer_size)
print(f"buffer_size is {buffer_size} MB")
print("run ccl_buffer_size success.")
```
