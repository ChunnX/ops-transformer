# 算子 aclnn 与 tiling 构造指南

本文面向新算子或重构算子的端到端代码骨架：从 aclnn 一阶段入口校验，到 op_host tiling 分流，再到 op_kernel 模板实例选择。
配套工具为 `scripts/tools/aclnn_tiling_framework_wizard.py`。

## 基本分层

### 1. aclnn 入口层

aclnn 负责把用户输入拦在算子边界外。建议校验顺序固定为：

1. 芯片平台校验：根据 `*_def.cpp` 中 `AICore().AddConfig(...)` 和需求确认 SOC 是否支持。
2. 数据类型与格式校验：按参数类别分别处理 `aclTensorList`、`aclTensor`、`aclIntArray`、`aclFloatArray`、`aclBoolArray`、`int/uint`、`float/double`、`bool`、`char/string`。
3. 必选入参校验：空指针、空 tensor-list、输出指针、`workspaceSize`、`executor`。
4. 可选入参校验：可选输入是否允许为空，是否受某个 bool/attr 控制，存在时校验 dtype/rank/shape。
5. 组合场景校验：跨参数关系，例如 dtype 组合、layout 组合、flag 与 optional 输入的对应关系。
6. 空 tensor 快速返回与输入归一化：contiguous、cast、format alias、packed dtype 等只在校验通过后执行。
7. 调用 l0op 并 `ViewCopy` 到用户输出。

### 2. tiling 层

aclnn 通过后，tiling 不再承担用户输入合法性兜底，主要做：

- 读取 shape、attr、compileInfo。
- 根据场景组合生成 `tilingKey`。
- 根据芯片资源计算 blockDim、workspace、UB 切分和 tiling data。
- 调用公共 tiling 库时检查返回值和硬件资源上限。

### 3. op_kernel 层

kernel 根据编译期 dtype 宏和运行期 `tilingKey` 选择模板实例：

- `ORIG_DTYPE_*` 决定编译期类型分支。
- `TILING_KEY_IS(n)` 决定同 dtype 下的场景模板。
- 模板参数必须和 tilingKey 的语义一一对应。

## 工具输入

最少提供 aclnn 头文件：

```bash
python scripts/tools/aclnn_tiling_framework_wizard.py \
  --aclnn-h path/to/op_host/op_api/aclnn_xxx.h
```

推荐提供三类文件：

```bash
python scripts/tools/aclnn_tiling_framework_wizard.py \
  --aclnn-h moe/moe_fused_topk/op_host/op_api/aclnn_moe_fused_topk.h \
  --def-cpp moe/moe_fused_topk/op_host/moe_fused_topk_def.cpp \
  --kernel-cpp moe/moe_fused_topk/op_kernel/moe_fused_topk.cpp \
  --out-dir docs/generated_aclnn_tiling_frameworks
```

如果 kernel cpp 还没有写好，也可以手动给模板类别：

```bash
python scripts/tools/aclnn_tiling_framework_wizard.py \
  --aclnn-h path/to/aclnn_xxx.h \
  --def-cpp path/to/xxx_def.cpp \
  --template-category "key=0;dtype=ORIG_DTYPE_X==DT_FLOAT;invoke=INVOKE_OP_IMPL;args=Xxx,float,0" \
  --template-category "key=1;dtype=ORIG_DTYPE_X==DT_FLOAT;invoke=INVOKE_OP_IMPL;args=Xxx,float,1"
```

## 工具输出

默认输出到 `docs/generated_aclnn_tiling_frameworks`：

- `*_framework.json`：解析出的接口、OpDef 和模板类别。
- `*_framework.md`：评审用构造说明。
- `*_aclnn_skeleton.cpp`：入口校验和 executor 框架。
- `*_tiling_skeleton.cpp`：tiling parse/init/key/data 框架。
- `*_kernel_dispatch_skeleton.cpp`：kernel dtype + tilingKey 分发框架。

这些文件是代码框架，不是可直接编译的最终实现。需要把 TODO 替换为算子的真实 shape 规则、资源切分、l0op 调用和模板参数。

## MoeFusedTopk 观察点

`moe/moe_fused_topk` 是一个比较清晰的例子：

- aclnn 层先检查空指针，再检查 dtype，然后检查 shape 和 `enableExpertMapping` 控制的可选输入。
- `*_def.cpp` 中 `mapping_num`、`mapping_table` 是 OPTIONAL，但 aclnn 中当 `enableExpertMapping=true` 时把它们提升为必选。
- tiling 中 `enableExpertMapping` 影响 `tilingKey`：0 表示普通场景，1 表示 expert mapping 场景。
- kernel 中同一组 tilingKey 在 `DT_FLOAT`、`DT_FLOAT16`、`DT_BF16` 三个 dtype 宏下分别实例化模板。

这类关系应该同时写进 aclnn 组合场景表、tiling key 表和 kernel 分发表，三者保持同一个命名和语义。

## 评审清单

1. aclnn 支持的 dtype/format 是否与 `*_def.cpp` 一致。
2. optional 输入在 aclnn、def、tiling 中的语义是否一致。
3. 每个 `tilingKey` 是否有清晰场景定义。
4. 每个 `tilingKey + dtype` 是否能在 kernel 找到模板实例。
5. tiling 是否只做资源与分流，不重新定义用户输入合法性。
6. 空 tensor、workspace、contiguous、ViewCopy 是否按固定顺序处理。
