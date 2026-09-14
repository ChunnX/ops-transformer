# IndexerQuantCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在Indexer注意力机制的Epilog阶段对KV Cache进行原地压缩更新。将`float16`/`bfloat16`激活值`x`量化为FP8（E4M3/E5M2）、HiFloat8（以uint8存储编码）或MX-FP4格式，并按`slot_mapping`将量化结果散写到`cache`；动态量化模式同时更新对应的`cache_scale`。`slot_mapping`中值为-1的token跳过不处理。支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。

- 计算公式：

  对`x`的最后一维（d轴）进行量化，记第g组为$x_g$：

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - `quant_mode=0`（MX-FP8）：scale存储为`float8_e8m0`，`round_scale=true`时对scale进行舍入。
  - `quant_mode=1`（Normal）：动态逐行量化，scale存储为`float32`；`round_scale=true`时向上取整到2的幂。
  - `quant_mode=2`（HiFloat8）：输出为$x \times x\_scale$后的hifloat8；不产生逐行scale，`cache_scale`保持输入值。
  - `quant_mode=3`（MX-FP4）：按标准MX块（每32元素）量化为FP4（uint8每字节打包2个fp4值），scale存储为`float8_e8m0`。

## 参数说明

| 参数名       | 输入/输出/属性 | 描述                                                                          | 数据类型                                                      | 数据格式 |
| ------------ | -------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ | -------- |
| cache        | 输入/输出      | 当前层的KV Cache向量缓存，原地更新。MX-FP4使用FP4数据类型，每字节打包2个FP4值；UINT8表示HiFloat8编码。 | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输入/输出      | 每块量化的scale因子，原地更新，即公式中`scale`。                            | FLOAT、FLOAT8_E8M0                                            | ND       |
| x            | 输入           | 待量化的激活输入，即公式中`x`。shape为[bs, d]。                              | FLOAT16、BFLOAT16                                             | ND       |
| slot_mapping | 输入           | token到cache slot的索引映射，值为 -1表示跳过该token。shape为[bs]。       | INT32                                                        | ND       |
| cache        | 输出           | 原地更新后的KV Cache，与输入`cache`为同一tensor。                            | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输出           | 原地更新后的scale，与输入`cache_scale`为同一tensor。                         | FLOAT、FLOAT8_E8M0                                            | ND       |
| quant_mode   | 可选属性       | 量化模式：0=MX-FP8，1=Normal，2=HiFloat8，3=MX-FP4。默认值为1。                | INT                                                          | -        |
| round_scale  | 可选属性       | 动态量化模式下控制scale舍入；Normal模式为向上取整到2的幂。HiFloat8模式不使用该属性。默认值为true。 | BOOL                                                         | -        |
| x_scale      | 可选属性       | HiFloat8模式（quant_mode=2）下的全局scale乘数。默认值为1.0。                 | FLOAT                                                        | -        |

## 约束说明

- `x`的最后一维`d`须能被32整除且 `d ≤ 8192`。
- `slot_mapping`的shape应等于`x`去掉最后一维后的shape，前导各维逐维对应；内部将这些维度展平为token行。
- 输入不支持空Tensor。`cache`和`cache_scale`仅支持四维shape `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1，`num_slots = blockNum × blockSize`。
- `cache.headDim ≥ d`；MX-FP4的headDim以FP4元素计，必须为偶数，分页block跨度也须为偶数。`cache_scale.headDim ≥ scaleCol`，其中MX-FP8/MX-FP4的`scaleCol = d/32`，Normal/HiFloat8为1。
- `cache`和`cache_scale`仅支持blockNum维的分页非连续布局，block内须连续；`x`和`slot_mapping`须连续。
- 动态shape在执行时具体化后，同样需要满足上述rank、维度、布局及数据类型约束；未知维不能作为实际张量尺寸传入ACLNN接口。
- Normal模式未对极小scale增加固定下限或次正规数补偿；极小输入不保证与理想浮点量化结果一致。
- 数据类型组合需匹配量化模式：Normal模式cache为FP8或UINT8（HiFloat8编码），HiFloat8模式cache为UINT8，两者cache_scale均为FLOAT；MX-FP8模式cache为FP8且cache_scale为FLOAT8_E8M0；MX-FP4模式cache为FP4且cache_scale为FLOAT8_E8M0。
- `slot_mapping`中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用说明

| 调用方式  | 样例代码                                                                          | 说明                                                                          |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_indexer_quant_cache](./examples/test_aclnn_indexer_quant_cache.cpp)    | 通过[aclnnIndexerQuantCache](./docs/aclnnIndexerQuantCache.md)调用IndexerQuantCache算子。 |
| 图模式    | -                                                                                 | 通过[算子IR](./op_graph/indexer_quant_cache_proto.h)接入GE图模式调用IndexerQuantCache算子。 |
| PyTorch API | - | 通过[indexer_quant_cache](../../attention/indexer_quant_cache/docs/torchapi_indexer_quant_cache.md)接口调用indexer_quant_cache算子。 |

### ACLNN样例运行

先安装支持该算子的CANN环境，按[编译与运行样例](../../docs/zh/context/compile_and_run_sample.md)配置编译器和运行环境。
下面命令在仓库根目录执行，`IQC_VENDOR_PATH`替换为实际安装的私有算子vendor目录；使用内置包时按上述通用文档选择对应库。

```bash
source /path/to/cann/set_env.sh
export IQC_VENDOR_PATH=/path/to/private-install/vendors/indexer_mode2_fix_transformer
export ASCEND_CUSTOM_OPP_PATH="$IQC_VENDOR_PATH"
export LD_LIBRARY_PATH="$IQC_VENDOR_PATH/op_api/lib:$LD_LIBRARY_PATH"
g++ -std=c++17 attention/indexer_quant_cache/examples/test_aclnn_indexer_quant_cache.cpp \
    -I "$ASCEND_HOME_PATH/include" -I "$IQC_VENDOR_PATH/op_api/include/aclnnop" \
    -L "$IQC_VENDOR_PATH/op_api/lib" -L "$ASCEND_HOME_PATH/lib64" \
    -lcust_opapi -lascendcl -lnnopbase -o /tmp/test_aclnn_indexer_quant_cache
/tmp/test_aclnn_indexer_quant_cache
```

样例使用逻辑设备0，自行构造FP16全1输入及`slot_mapping=[1,-1]`，执行Normal E4M3量化。
它回读并检查全部cache和scale，包括跳过token后未更新的slot；成功时输出`IndexerQuantCache PASS`，失败返回非零退出码。
