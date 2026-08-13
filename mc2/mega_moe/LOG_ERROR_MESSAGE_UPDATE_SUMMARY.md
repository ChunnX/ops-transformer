# MegaMoe Arch22 错误日志规范化修改总结

## 1. 修改背景与目的

本次修改针对 `mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp` 中的错误日志进行规范化，参考以下内容：

- `ops-transformer` PR 7234 的日志改造方式；
- `mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp` 的校验语义；
- Err_Msg_Skill 对结构化错误日志的要求；
- 当前仓库 `log.h` 和 `mc2_log_compat.h` 中已有的日志宏定义。

修改目的如下：

1. 将可归类的普通 `OP_LOGE` 转换为结构化错误日志，输出稳定的错误码和错误字段；
2. 准确标识错误所属的输入、属性、shape、dtype、format 或 TensorList；
3. 保留真实错误值、期望值以及必要的上下文，提升问题定位效率；
4. 避免底层结构化错误已经上报后，上层包装日志再次上报通用错误；
5. 不改变原有校验条件、返回值和 tiling 控制流程。

## 2. 修改原则

### 2.1 按错误对象选择宏

- 缺少必选输入：使用 `OP_LOGE_WITH_INVALID_INPUT`；
- 属性值错误：使用 `OP_LOGE_WITH_INVALID_ATTR`；
- 属性列表长度错误：使用 `OP_LOGE_WITH_INVALID_ATTR_SIZE`；
- shape、shape 维数、dtype、format、TensorList 数量错误：使用对应的 `OP_LOGE_FOR_INVALID_*` 宏；
- 框架内部对象获取失败、helper 失败等不适合用户参数错误码的场景：使用 `OP_LOGE_WITHOUT_REPORT` 保留内部诊断信息。

### 2.2 区分算子名和调试标签

- `K_OP_NAME = "MegaMoe"`：仅用于结构化错误宏，保证错误信息中的 `OP[...]` 是正式算子名；
- `K_INNER_DEBUG = "MegaMoeA2A3 Tiling Debug"`：继续用于普通调试日志、内部错误日志和 `OP_LOGE_WITHOUT_REPORT`。

### 2.3 使用对外参数名

日志中的属性名与算子定义保持一致，例如：

- `moe_expert_num`
- `ep_world_size`
- `dispatch_quant_mode`
- `dispatch_quant_out_dtype`
- `activation_params`
- `transpose_weight1`
- `weight1_interleave`

输入、输出名称同样采用算子定义中的名称，如 `topk_ids`、`topk_weights`、`x_active_mask` 和 `expert_token_nums`。

### 2.4 保证错误信息真实完整

- dtype 和 format 使用 `Ops::Base::ToString` 输出可读名称；
- shape 校验输出完整 shape，而不是只输出某个维度的数值；
- TensorList 中的错误携带元素索引，如 `weight1[2]`、`bias1[0]`；
- 依赖其他参数计算期望值时，同时输出实际值和计算后的期望值；
- HCCL buffer 校验保留实际大小、最低要求、推荐大小和相关计算参数。

## 3. 实际修改内容

### 3.1 引入类型字符串转换接口

新增：

```cpp
#include "graph/utils/type_utils.h"
```

用于显式引用 `Ops::Base::ToString`，将 `ge::DataType`、`ge::Format` 和 shape 转换为可读字符串，避免依赖其他头文件的间接包含。

### 3.2 属性校验日志

对以下属性的空指针、取值范围、关联约束或列表长度错误进行了结构化改造：

- `moe_expert_num`
- `ep_world_size`
- `ccl_buffer_size`
- `max_recv_token_num`
- `dispatch_quant_mode`
- `dispatch_quant_out_dtype`
- `combine_quant_mode`
- `comm_alg`
- `num_max_tokens_per_rank`
- `activation`
- `activation_params`
- `activation_out_dtype`
- `transpose_weight1`
- `transpose_weight2`
- `weight1_interleave`

其中 `activation_params` 的长度和元素错误会带上激活类型或具体元素索引；量化模式相关错误会输出由 `weight1` dtype 推导出的期望值。

### 3.3 输入、输出及 TensorList 校验日志

完成以下类型的结构化改造：

- 必选输入缺失；
- shape 维数、shape 范围及 shape 关联关系错误；
- dtype 和 format 错误；
- `weight1`、`weight2`、权重 scale、bias 的 TensorList 数量错误；
- TensorList 内单个 Tensor 的维数或 shape 错误；
- 输出 `y` 和 `expert_token_nums` 的 shape、dtype 错误。

动态输入逐项校验时构造带索引的名称，避免日志只能定位到整个 TensorList。

### 3.4 内部日志和重复上报处理

以下日志保留为内部诊断，不归类为用户输入错误：

- 获取 attributes、tiling data、workspace、输出描述等内部对象失败；
- 下层校验函数失败后的上层包装日志；
- tiling 各阶段的汇总失败日志。

这些场景使用 `OP_LOGE_WITHOUT_REPORT`，从而保留本地日志，同时避免覆盖或重复上报下层已经生成的结构化错误。

### 3.5 HCCL buffer 与量化输出日志

- `ccl_buffer_size` 错误按用户设置的总 buffer 大小进行报告，并给出最低总大小、推荐 MB 数以及计算所依赖的关键参数；
- `dispatch_quant_out_dtype` 报告用户实际配置值，不再使用内部派生的 tiling key 枚举值代替用户值。

## 4. 未修改范围

- `arch35/mega_moe_tiling.cpp` 作为参考文件，本次未修改；
- `op_api/aclnn_mega_moe.cpp` 中携带 `ACLNN_ERR_*` 返回码的 `OP_LOGE` 属于 ACLNN API 错误处理形式，本次未机械替换；
- 未修改任何原有校验条件、返回分支、tiling 计算或算子接口定义。

## 5. 验证结果

执行以下检查：

```bash
git diff --check
bash build.sh -u --ophost --ops=mega_moe -j4
```

结果：

- `git diff --check` 通过；
- MegaMoe ophost 编译通过；
- 8 个单元测试全部通过，其中包含 2 个 Arch22 tiling 测试和 6 个 infer shape/dtype 测试。

现有单元测试主要验证返回状态和 tiling/infer 行为，尚未逐项断言错误码及错误文本字段。
