# aclnn接口场景拦截梳理指南

本文面向所有 aclnn 两段式接口，不绑定 GroupedMatmul。工具会从任意接口 `.h` 文件中解析
`aclnnStatus XxxGetWorkspaceSize(...)` 这类函数原型，根据参数类型和参数名给出一组可编辑的场景轴，
再引导录入哪些组合允许、哪些组合需要在入口拦截。

## 目标

入口层只回答三类问题：

1. 当前接口版本是否支持目标平台和基础参数组合。
2. 当前场景组合是否应该在入口直接拦截。
3. 进入完整 shape/dtype/format 校验前，是否需要做输入表示归一化。

不要把完整 shape 推导、tiling 选择或 kernel 分支混在入口 guard 里。入口 guard 的产物应该是：

- 稳定的校验顺序。
- 稳定的首个报错。
- 一张能被评审的场景组合表。

## 通用校验顺序

1. 必选指针与 tensor-list 非空检查：只检查访问安全，不做业务判断。
2. DFX phase-1：必选输入可安全读取后立即记录。
3. 平台与接口版本支持检查。
4. 标量和枚举类属性检查。
5. 可选输入的存在性、rank、dtype、format、array size 检查。
6. 决策矩阵中记录的接口专属场景拦截。
7. 输入表示归一化：仅当该接口确实存在 packed dtype、format alias 等转换时执行。
8. 深层公共校验：dtype、format、shape、optional 参数关系和输出关系。

## 使用程序

交互式梳理任意 aclnn 头文件：

```bash
python scripts/tools/api_scenario_guard_wizard.py attention/flash_attention_score/op_api/aclnn_flash_attention_score.h
```

指定函数和输出目录：

```bash
python scripts/tools/api_scenario_guard_wizard.py \
  moe/moe_fused_topk/op_host/op_api/aclnn_moe_fused_topk.h \
  --function aclnnMoeFusedTopkGetWorkspaceSize \
  --out-dir docs/generated_scenario_guards
```

程序会依次询问：

1. 使用哪个接口函数。
2. 哪些场景轴参与组合。
3. 每个场景轴有哪些取值。
4. 哪些组合需要拦截，使用 `key=value,key2=value` 形式录入规则。
5. 未命中规则的组合默认 `allow` 还是 `block`。

通用场景轴来自参数类型：

- `aclTensor*`：建议 `xxx_present`、`xxx_rank`、`xxx_dtype`、`xxx_format`。
- `aclTensorList*`：建议 `xxx_list_size`、`xxx_dtype`、`xxx_format`。
- `aclIntArray*` / `aclFloatArray*` / `aclBoolArray*`：建议 `xxx_size`。
- `int64_t` / `bool` 等标量：建议参数名本身作为场景轴。
- 多个 tensor 输入：额外建议 `dtype_pattern` 作为跨 tensor 的 dtype 组合轴。

部分常见参数会带默认取值提示，例如 `splitItem`、`groupType`、`groupListType`、`actType`、`inputLayout`。
这些只是建议值，交互时可以删改或新增任何自定义场景轴。

示例拦截规则：

```text
platform=Ascend310P,input_dtype=int8
action: block
reason: int8 input is not supported on Ascend310P.
```

可用 `|` 表示多个取值：

```text
platform=Ascend310P,groupType=-1|2
```

非交互模式用于先生成模板：

```bash
python scripts/tools/api_scenario_guard_wizard.py \
  attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v5.h \
  --non-interactive
```

## 输出文件

默认输出到 `docs/generated_scenario_guards`：

- `*_scenario_guard.json`：机器可读的接口参数、场景轴、规则和决策矩阵。
- `*_scenario_guard.md`：评审用文档，包含推荐校验顺序和场景表。
- `*_scenario_guard_skeleton.cpp`：根据规则 selector 生成的 C++ guard 骨架。

如果组合数过大，程序会按 `--max-combinations` 截断 Markdown/JSON 中的明细矩阵，但保留总组合数与拦截规则。

## 落代码建议

入口代码建议按三层拆分：

1. `CheckXxxEntryParam`：只做平台、标量参数、可选输入形态和接口专属拦截。
2. `NormalizeXxxInputs`：只做 packed dtype、format alias 等输入表示归一化。
3. `CheckParam` 或公共 checker：做完整 dtype/shape/format/optional 参数关系校验。

新增规则时优先更新场景表，再改 guard。评审时先看“新增组合是否有明确 allow/block”，再看 C++ 报错顺序是否与表一致。
