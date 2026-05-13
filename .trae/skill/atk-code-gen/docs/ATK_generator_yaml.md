## Generator 与 CaseConfig Inputs 参考

Generator的核心职责是在 `after_case_config` 中对ATK框架随机生成的用例参数施加约束，使参数间满足算子文档规定的依赖关系。理解 `case_config.inputs` 的数据结构是编写generator的关键。
Yaml负责对入参进行一一声明，需要与aclnn接口的入参顺序严格一致且一一对应（除了aclnn中输出的参数以及executor和workspacesize），如若需要额外信息进行场景枚举，需要在AclnnBaseApi的子类中的init_by_input_data方法中进行pop处理。

### CaseConfig Inputs 与 YAML 声明的对应关系

`case_config.inputs` 是一个列表，**索引与yaml中 `inputs` 的声明顺序一一对应**。例如：

| yaml声明顺序 | 索引 | name | type |
|-------------|------|------|------|
| 第1个 | 0 | x1 | tensor |
| 第2个 | 1 | x2 | tensor |
| 第3个 | 2 | bias | tensor |
| 第4个 | 3 | alltoAllAxesOptional | attrs |
| 第5个 | 7 | transposeX1 | attr |
| 第6个 | 8 | transposeX2 | attr |

> **重要**：索引必须与yaml声明顺序严格一致，编写generator时先确认yaml中inputs的声明顺序。

### Tensor 类型 Input 的数据结构

对应yaml中 `type: tensor` 的参数，ATK框架生成的JSON结构如下：

```json
{
    "name": "x1",
    "type": "tensor",
    "required": true,
    "dtype": "fp16",
    "shape": [1174, 8],
    "range_values": [-5, 5],
    "backward": true,
    "align_32B": null,
    "outlier_values": null
}
```

在generator中通过 `case_config.inputs[i]` 访问，可操作属性：

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `name` | str | 参数名，只读 | `"x1"` |
| `dtype` | str | 数据类型，**可写** | `"fp16"`, `"bf16"`, `"fp32"`, `"int8"` |
| `shape` | List[int] | 张量维度，**可写** | `[1174, 8]` |
| `range_values` | List | 数值范围 | `[-5, 5]` |
| `required` | bool | 是否必选 | `true` |
| `backward` | bool | 是否参与反向 | `true` |

### Attr 类型 Input 的数据结构

对应yaml中 `type: attr` 的参数，ATK框架生成的JSON结构如下：

```json
{
    "name": "hasBias",
    "type": "attr",
    "required": true,
    "dtype": "bool",
    "shape": null,
    "range_values": true,
    "backward": false
}
```

```json
{
    "name": "group",
    "type": "attr",
    "required": true,
    "dtype": "string",
    "shape": null,
    "range_values": ["-inf", "inf"],
    "backward": false
}
```

在generator中可操作属性：

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `name` | str | 参数名，只读 | `"hasBias"` |
| `dtype` | str | 数据类型 | `"bool"`, `"string"`, `"int64_t"` |
| `range_values` | Any | yaml中定义的取值范围 | `true`, `[0, 1, 2]` |

### 5.4 Attrs 类型 Input 的数据结构

对应yaml中 `type: attrs` 的参数（如 `alltoAllAxesOptional`），ATK框架生成的JSON结构被**额外包裹一层数组**：

```json
[
    {
        "name": "alltoAllAxesOptional",
        "type": "attrs",
        "required": true,
        "dtype": "int64_t",
        "shape": null,
        "range_values": ["-inf", "inf"],
        "backward": false
    }
]
```

> **注意**：attrs类型在JSON中是 `[{...}]` 形式（外层多一个数组），但在 `case_config.inputs` 中与tensor/attr一样占一个索引位。generator中通常不需要修改attrs类型的参数（它们在executor中统一处理为AclIntArray或空指针）。

### 5.5 dtype 字符串对照表

在 `case_config.inputs[i].dtype` 中，dtype可能以以下形式出现，generator中需要兼容处理：

| ATK内部表示 | torch表示 | 说明 |
|------------|----------|------|
| `"fp16"` | `"torch.float16"` | 半精度浮点 |
| `"bf16"` | `"torch.bfloat16"` | BFloat16 |
| `"fp32"` | `"torch.float32"` | 单精度浮点 |
| `"int8"` | `"torch.int8"` | 8位有符号整数 |
| `"int32"` | `"torch.int32"` | 32位有符号整数 |
| `"bool"` | `"torch.bool"` | 布尔类型 |

### 5.6 常见约束场景速查

| 约束场景 | 代码模式 |
|---------|---------|
| 维度整除 | `x1.shape[0] = max(int(bs / rank_size) * rank_size, rank_size)` |
| 维度联动 | `x2.shape[0] = x1.shape[1] * rank_size` |
| dtype联动 | `x2.dtype = x1.dtype` |
| 条件dtype | `bias.dtype = 'fp16' if x1.dtype in ['fp16', ...] else 'fp32'` |
| 固定attr值 | `case_config.inputs[i].range_values = False` |
| 维度范围限制 | `if h * rank_size > 65535: x1.shape[1] = 65535 // rank_size` |
| N最小值约束 | `if x2.shape[1] < 1: x2.shape[1] = 1` |