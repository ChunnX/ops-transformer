# Common Pitfalls

## 不要套用 ops-math 的当前机器事实

`ops-math` 的 CANN 9.0.0、`arch35`、ACL device id、ST coverage、adapter 支持范围都是它当前工作树的事实。`ops-transformer` 当前默认是用户声明的 910B B3、`aarch64`，仓库静态映射是 `ascend910b -> arch22`。

## 不要引用不存在或未覆盖的脚本/能力

当前 `ops-transformer` 没有：

- `scripts/agent_context/validate_agent_context.py`

当前已有：

- `.agents/skills/` repo-local skills。
- `agent_ops/lightning_indexer/` 和 `agent_ops/recurrent_gated_delta_rule/` 两个轻量账本。

这些账本只覆盖已完成闭环的两个 attention 算子，不能外推为全仓库单算子账本都已建立。

例外：当前已迁移裁剪 `scripts/st_accuracy/run_st_accuracy.py`，但它只覆盖 `attention/lightning_indexer` 和 `attention/recurrent_gated_delta_rule` 两个重点算子，不能外推为全仓库统一 ST runner。

## 不要把 example 当完整精度

`build.sh --run_example` 只保证样例编译并执行。只有样例源码明确比较 golden，才能把它写成该范围内的精度验证。

## 不要把 opkernel 当部署

`--opkernel` 不会自动更新运行时 vendor 目录中的 op_api、op_host、op info 和全部 config。profiling 前要确认运行加载的是新产物。

## 不要把缺少 custom_transformer 误判为算子不可用

两个重点 attention 算子的 ST runner 需要本仓库自定义 package：`/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer`。只安装了 CANN 内置 `libopapi_transformer.so` 或其他仓库的 `custom_math` 时，runner 会缺少 `custom_transformer/op_api/lib/libcust_opapi.so`。正确处理是 `--pkg --vendor_name=custom` 构建并安装本仓库 package，再运行 ST。

## 注意 build 清理

`build.sh` 会清理 `build/` 和 `output/`。不要把唯一实验记录放在这些目录。

## 注意 MC2 环境

MC2 算子常涉及 HCCL、多卡、rank 和通信拓扑。单卡 smoke 不能代表通信融合性能，simulator 也不能替代真实多卡 profiling。

## 注意公共目录影响面

`attention/common/`、`gmm/common/`、`mc2/common/`、`mc2/3rd/`、`moe/3rd/` 可能影响多个算子。改这些目录前先做依赖和回归范围分析。
