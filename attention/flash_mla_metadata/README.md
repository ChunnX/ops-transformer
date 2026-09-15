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

- 算子功能：为[FlashMla](../flash_mla/README.md)生成调度元数据（metadata）。AI CPU读取设备上的Query和KV序列实长，结合输入配置与当前有效核数，计算后续注意力任务的分核方案。
- 输出为固定大小的NPU INT32张量，包含FlashMla执行需要的调度信息。该接口不计算注意力结果；调用方将输出原样传给FlashMla。
- 支持非量化和INT8全量化两种调度模式。FLOAT16与BFLOAT16共用非量化调度模式；连续KV存储和分页注意力（Paged Attention，PA）采用不同的序列长度语义。

## 参数说明

下表采用torch接口的参数名，`device`用于选择输出设备。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| num_heads | 属性 | Query头数，取值范围为[1, 128]；KV头数固定为1。 | INT | - |
| batch_size | 可选属性 | 默认None，TND下从Query实长张量的项数推导；其他布局必须显式传入。逻辑batch数，取值范围为[0, 65536]。 | INT | - |
| q_seq_size | 可选属性 | 默认None，TND且batch非空时从Query累计实长末项推导；其他情况必须显式传入。非TND下为Query存储S；TND下为存储总T。取值范围为[0, 4294967295]，INT8另有约束。 | INT | - |
| kv_seq_size | 可选属性 | 默认None，连续TND KV且batch非空时从KV累计实长末项推导；其他情况必须显式传入。连续KV为存储S/T，PA为block_table.shape[1]乘block_size。取值范围为[0, 2147483647]。 | INT | - |
| actual_seq_lengths_q | 可选输入 | 一维连续NPU Query实长张量，默认不传入。TND下使用不含起始0的累计长度，非空batch时必选。 | INT64 | ND |
| actual_seq_lengths_kv | 可选输入 | 一维连续NPU KV实长张量，默认不传入。连续TND KV使用累计长度；PA KV使用逐batch绝对长度，非空batch时必选。 | INT64 | ND |
| input_layer | 可选属性 | 对应FlashMla的input_layout，默认值为BSH；名称为input_layer。输出布局后缀不影响调度。 | STRING | - |
| block_size | 可选属性 | 连续KV存储时为0；PA时为不大于1024的正16倍数，INT8时必须为128。默认值为0。 | INT | - |
| sparse_mode | 可选属性 | 非量化支持0、3、4、9，INT8支持0、3、9，默认值为0。 | INT | - |
| pre_tokens | 可选属性 | 与FlashMla一致的左窗口参数，默认值为2147483647。 | INT | - |
| next_tokens | 可选属性 | 与FlashMla一致的右窗口参数，默认值为2147483647。 | INT | - |
| has_atten_mask | 可选属性 | FlashMla是否使用非空atten_mask，默认值为false。 | BOOL | - |
| quant_mode | 可选属性 | 整数0表示非量化，整数1表示INT8全量化，默认值为0。 | INT | - |
| device | 可选接口参数 | torch输出设备，默认值为None。优先采用Query实长张量的设备，其次采用KV实长张量的设备，两者均未传入时使用当前NPU。 | torch.device、str | - |
| metadata | 输出 | 新分配的连续NPU张量，shape为(480,)，存储偏移为0，首地址按64字节对齐。 | INT32 | ND |

## 约束说明

- 尺寸参数显式传入时使用传入值并保留校验；省略时按上述条件推导，信息不足时报错。0不表示省略。累计末项由AI CPU读取，Host不读取设备实长值；完整规则见[尺寸参数推导](docs/torchapi_flash_mla_metadata.md#尺寸参数推导)。

- 支持BSH、BSND、BNSD、TND，以及BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD。连续KV与Query采用相同输入布局；PA KV的存储布局由FlashMla的输入张量描述确定，本接口无需接收KV布局。
- 非TND实长通常支持缺省、空张量、单项广播或不少于batch_size项；多余项不参与计算。INT8非TND的Query直接使用存储长度，忽略Query实长的值与项数，但传入张量仍需满足设备、类型、维度和连续性要求。
- TND下两个实长张量均须恰好包含batch_size项。Query和连续TND KV的累计长度应非负、单调不减，末项等于对应存储总长度；PA KV始终使用逐batch绝对长度，不累加。
- PA的非空batch必须传入非空KV实长，其值不得超过kv_seq_size。完整序列规则见[torch接口约束说明](docs/torchapi_flash_mla_metadata.md#约束说明)。
- INT8要求Query头数为2的幂、block_size为128；Query仅支持BSH、BSND、TND及对应输出后缀。非TND的Query存储长度不超过32；TND的总长度不超过1048576，且每个batch的Query实长不超过32。配套FlashMla还要求NZ KV和三个FLOAT反量化系数张量。
- sparse_mode非0时必须设置has_atten_mask为true，并由FlashMla提供相应掩码。INT8的单token及多token掩码组合见torch接口文档。
- 所有实长张量与输出必须位于同一NPU。调用顺序为：更新序列实长、生成metadata、调用FlashMla；三步应在同一stream执行，且两算子的有效核数一致。
- 序列值、序列描述或调度配置变化后必须重新生成metadata；在FlashMla执行完成前，不能修改序列实长或metadata。保持配置不变时，非量化metadata可用于FLOAT16和BFLOAT16。
- 当前torch接口只支持单算子模式（eager），暂不支持反向传播和torchair图模式。

## 调用说明

对外仅提供 PyTorch API；aclnn 为 Torch 扩展的内部调用入口，不提供独立接口头文件。

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | [test_torch_flash_mla.py](../flash_mla/examples/test_torch_flash_mla.py) | 通过[flash_mla_metadata](docs/torchapi_flash_mla_metadata.md)生成metadata，再传给[flash_mla](../flash_mla/docs/torchapi_flash_mla.md)。 |
