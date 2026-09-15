# FlashMlaMetadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：FlashMlaMetadata是[FlashMla](../flash_mla/README.md)的前置算子，根据Query和KV的实际序列长度及输入配置，生成各计算核的任务划分信息，用于FlashMla的负载均衡。
- 使用流程：先调用FlashMlaMetadata生成metadata，再将metadata传入FlashMla完成注意力计算。支持FLOAT16、BFLOAT16非量化及INT8全量化场景，KV支持连续存储和分页注意力（Paged Attention，PA）存储，实际序列长度以设备Tensor传入。

## 参数说明

<table style="table-layout: fixed; width: 1390px">
  <colgroup>
    <col style="width: 190px">
    <col style="width: 150px">
    <col style="width: 750px">
    <col style="width: 200px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>actual_seq_lengths_q</td>
      <td>可选输入</td>
      <td>一维连续NPU Query实长张量，默认不传入。TND下使用不含起始0的累计长度，非空batch时必选。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv</td>
      <td>可选输入</td>
      <td>一维连续NPU KV实长张量，默认不传入。连续TND KV使用累计长度，PA KV使用各batch的实际长度；PA或TND且batch非空时必选。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads</td>
      <td>属性</td>
      <td>Query头数，取值范围为[1, 128]；KV头数固定为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>逻辑batch数，取值范围为[0, 65536]。PyTorch接口默认值为None，TND下可从actual_seq_lengths_q的元素个数推导，其他布局必须传入。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>q_seq_size</td>
      <td>可选属性</td>
      <td>Query的存储序列长度，非TND为S，TND为总长度T，取值范围为[0, 4294967295]。PyTorch接口默认值为None，TND且batch非空时可从Query累计长度末项推导，其他情况必须传入。INT8场景另有长度约束。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_seq_size</td>
      <td>可选属性</td>
      <td>连续KV的存储序列长度S/T，或PA场景下每个batch的存储容量（分页表列数乘block_size），取值范围为[0, 2147483647]。PyTorch接口默认值为None，连续TND KV且batch非空时可从KV累计长度末项推导，其他情况必须传入。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layer</td>
      <td>可选属性</td>
      <td>Query输入的数据排布格式，对应FlashMla的input_layout。支持BSH、BSND、BNSD、TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD，默认值为BSH。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>可选属性</td>
      <td>连续KV存储时为0；PA时为不大于1024的正16倍数，INT8时必须为128。默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>可选属性</td>
      <td>非量化支持0、3、4、9，INT8支持0、3、9，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>可选属性</td>
      <td>与FlashMla一致的左窗口参数，默认值为2147483647。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>可选属性</td>
      <td>与FlashMla一致的右窗口参数，默认值为2147483647。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_atten_mask</td>
      <td>可选属性</td>
      <td>FlashMla是否使用非空atten_mask，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>可选属性</td>
      <td>整数0表示非量化，整数1表示INT8全量化，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>FlashMla的负载均衡结果，shape为(480,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- FlashMlaMetadata需与FlashMla配套使用，仅支持单算子模式（eager），不支持反向传播和TorchAir图模式。
- 所有实际序列长度Tensor和metadata必须位于同一NPU，实际序列长度Tensor必须一维连续。
- TND布局下，实际序列长度Tensor的元素个数须等于batch_size。Query和连续TND KV使用不含起始0的累计长度，累计值非负且单调不减，末项等于对应的存储总长度。PA的KV使用各batch的实际长度。
- 非TND布局下，非量化Query和连续KV的实际序列长度支持缺省、空Tensor、单项广播或不少于batch_size项的Tensor；PA场景在batch非空时必须提供非空KV实际序列长度Tensor。
- batch_size、q_seq_size和kv_seq_size通过PyTorch接口省略时须满足[尺寸参数推导](docs/torchapi_flash_mla_metadata.md#约束说明)的条件；显式传入时应与FlashMla输入的shape及分页表容量一致。
- INT8场景要求num_heads为2的幂，block_size为128，Query布局支持BSH、BSND、TND及对应输出布局后缀。非TND的Query存储长度不超过32；TND的Query总长度不超过1048576，各batch的Query实际长度不超过32。
- input_layer、block_size、sparse_mode、pre_tokens和next_tokens应与FlashMla的对应配置一致，has_atten_mask应与其是否使用非空掩码一致，quant_mode应与其输入数据类型一致。
- 在同一设备和stream上依次更新实际序列长度、生成metadata、调用FlashMla，两个算子的有效核数须一致。在FlashMla执行完成前，应保持实际序列长度Tensor和metadata有效且不被修改；实际序列长度或调度参数变化后，应重新生成metadata。
- 完整参数组合约束参见[PyTorch接口约束说明](docs/torchapi_flash_mla_metadata.md#约束说明)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | [test_torch_flash_mla](../flash_mla/examples/test_torch_flash_mla.py) | 通过[flash_mla_metadata](docs/torchapi_flash_mla_metadata.md)接口调用FlashMlaMetadata算子，输出作为[flash_mla](../flash_mla/docs/torchapi_flash_mla.md)接口的metadata输入。 |
