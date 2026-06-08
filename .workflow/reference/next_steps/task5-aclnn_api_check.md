# op_api Aclnn接口参数检查

> 你应当在必要时候使用subagent来完成任务，提供一个独立、聚焦的上下文来完成具体的任务，并且为其提供必要的知识或信息。

请对attention/qkv_rms_norm_rope_cache_with_kscale算子aclnn接口的inplace参数情况和多输出情况进行检查，查看是否和已有代码的处理方法一致，如果不一致请进行修改。

## 任务说明

inplace参数情况：该算子设计文档的aclnn接口说明中提到kCache、vCache、kScaleCache三个参数是输入输出参数，也就是算子会进行原地操作(Inplace)请检查对于这种同时是输入输出的参数，aclnn的l0接口和l2接口是否进行了正确处理。通常来说你可以查看op_host中的xxx_def.cpp文件了解一个算子的输入输出和类型，如果有参数同时是Input和Output那么就是这种情况，请检查qkv_rms_norm_rope_cache_with_kscale算子和已有代码对l0接口和aclnn的两段式接口和实现是否按照同样的方法进行了处理，如果不一致请进行修改。你可以参考gmm/quant_grouped_matmul_inplace_add是如何处理inplace参数的。

多输出情况：当前算子存在多个输出参数的情况，请检查l0接口和aclnn两段式接口是否正确处理了多返回值的情况，你可以参考attention/scatter_pa_kv_cache算子的是如何处理keyCacheRef和valueCacheRef两个返回值的方法来修改当前算子的实现。
尤其需要注意l0接口，scatter_pa_kv_cache算子的返回值有两个，因此l0接口使用tuple返回两个参数，请你查看调研其他算子是否都是这种写法，如果是请帮我修改，并检查aclnn的两段式接口是否正确处理多返回值的情况。

## 任务计划

- [x] 检查inplace参数(kCache/vCache/kScaleCache)在L0/L2的处理方式
- [x] 对比quant_grouped_matmul_inplace_add的inplace处理模式
- [x] 对比scatter_pa_kv_cache的多输出处理模式
- [x] 确认当前实现是否符合规范

## 任务完成情况

### 初次检查结果：无需修改

**inplace参数处理**：当前实现已符合规范
- L2签名：kCache/vCache/kScaleCache为`aclTensor*`（非const），正确标记为inplace
- L0实现：同一tensor同时出现在`OP_INPUT`和`OP_OUTPUT`中
- L2 Contiguous()：不对cache参数调用Contiguous()，保留原始stride
- def.cpp：Input("k_cache")/Output("k_cache")同名注册，正确标记inplace

**多输出处理**：当前实现已符合规范
- L0返回`qOut`（第一个输出），所有5个输出都在`OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache)`中
- 与scatter_pa_kv_cache的tuple返回模式不同，但本算子有独立的qOut/qScale输出+3个inplace cache输出，使用单返回值是合理的

### 二次验证结果（RECHECK）：确认无需修改

**对比算子**：quant_grouped_matmul_inplace_add（inplace）、scatter_pa_kv_cache/gather_pa_kv_cache（多输出）、recurrent_gated_delta_rule（多OP_OUTPUT单返回）、nsa_compress_with_cache（单输出）

**Inplace参数验证**：
- def.cpp：Input("y")/Output("y")同名注册（quant_grouped_matmul_inplace_add）→ 本算子Input("k_cache")/Output("k_cache")同名 → 一致
- L0：yRef在OP_INPUT和OP_OUTPUT中，非const aclTensor* → 本算子kCache/vCache/kScaleCache同理 → 一致
- L2：yRef非const，不调用Contiguous() → 本算子cache参数同理 → 一致
- 无需NnopbaseSetRef — 整个ops-transformer codebase均未使用此机制，inplace语义通过同名Input/Output自动推断

**多输出验证**：
- scatter_pa_kv_cache/gather_pa_kv_cache使用tuple返回 → 因其输出由L0内部AllocTensor创建，需通过tuple返回给L2做ViewCopy
- recurrent_gated_delta_rule有2个OP_OUTPUT但单返回out → 因stateRef为caller-provided mutable param，inplace更新
- 本算子5个OP_OUTPUT全部为caller-provided mutable pointers（qOut/qScale/kCache/vCache/kScaleCache），单返回qOut → 与recurrent_gated_delta_rule模式一致 → 正确

**结论**：无需ViewCopy — 本算子所有输出均为caller-provided mutable pointers，L0直接写入原始tensor，无需中间contiguous副本

### 三次验证结果（RECHECK + FIX）：L0多输出返回模式修正

**问题发现**：用户指出多输出算子应使用tuple返回所有输出指针，当前实现仅返回qOut，qScale等其他输出完全丢失。

**深度调研**：对比8+个算子的L0返回模式，发现两种主流模式：
1. **全部输出返回**（scatter_pa_kv_cache, gather_pa_kv_cache, attention_update, chunk_gated_delta_rule）：tuple包含OP_OUTPUT中所有输出
2. **仅返回新分配输出**（qkv_rms_norm_rope_cache, recurrent_gated_delta_rule）：tuple仅含AllocTensor创建的输出，inplace排除

**修复决策**：采用模式1（全部输出返回），与scatter_pa_kv_cache/gather_pa_kv_cache一致，因为：
- 主流模式，更多算子采用
- 返回所有输出（包括inplace）更安全，L2可逐一验证
- 用户明确要求"至少要包括qOut和qScale，其他inplace参数也需要返回"

**修复内容**：
- L0 .h：返回类型从`const aclTensor*`改为`std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>`（5个输出）
- L0 .cpp：返回`{qOut, qScale, kCache, vCache, kScaleCache}`，错误路径返回`{nullptr, nullptr, nullptr, nullptr, nullptr}`
- L2 .cpp：解包tuple，使用`std::get<0>`到`std::get<4>`逐一检查所有5个返回值

**验证结果**：编译通过 + 57/57测试通过