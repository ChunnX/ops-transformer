# FlashKDA

本目录将原 FlashKDA 发布为可随 ops-transformer 的 custom / builtin run 安装的二进制算子。
原 Python 接口、设备计算、Channel 和同步逻辑保持不变；新增 `@entry(name="FlashKda")`
负责描述输入、输出和工作空间，并调用原 `FlashKDA.run`。

## 使用 Python 算子

原来的调用方式继续可用，返回 `(output, final_state)`：

```python
from flash_kda import flash_kda

output, final_state = flash_kda(
    q, k, v, g, beta, scale, initial_state, A_log, dt_bias,
    lower_bound=-1.0, layout_qkv="BNSD",
)
```

支持 BNSD / BSND，输出布局跟随输入；原 `cu_seqlens` 仅接受 `None`。
Python 调用仍按原方式 JIT 编译。发布后的二进制通过生成的 ACLNN 接口调用，
运行时无需 Python 源码或 JIT 编译。两种入口复用同一份设备计算实现。

## 发布二进制

先加载 CANN 环境，并将 **framework run 的安装目录**加入 Python 和 CMake 搜索路径。
在算子目录执行以下命令即可查看参数和构建 bundle：

```bash
python flash_kda.py --help
python flash_kda.py /absolute/output/bundle --aic-count 28 --head-pairs 1:1,1:2
```

`--aic-count` 是目标设备可用的 AIC 核数，请按实际设备设置；`--head-pairs` 中
`1:2` 表示 Q/K 有 1 个头、V 有 2 个头。发布过程同时支持两种布局。
默认发布 B=1..4、S=64..8192、head_dim=128；S 必须为 64 的倍数。
这些范围描述当前包能接收的输入，调整后需要重新构建，不是原算子的功能上限。

| 参数 | 含义 |
| --- | --- |
| `--head-pairs 1:1,1:2` | 需要发布的 Q/K 头数与 V 头数组合；V 头数须为 Q/K 头数的整数倍 |
| `--batch-min` / `--batch-max` | 支持的 batch 范围，默认 1..4 |
| `--sequence-min` / `--sequence-max` | 支持的序列长度范围，默认 64..8192 |
| `--npu-arch` | 目标架构，默认 dav-3510（Ascend 950） |
| `--group` / `--dv-base` | 高级调优覆盖；通常保持自动策略即可 |

也可以在 Python 中构建，不必调用命令行：

```python
import cannbotdsl as cb
from flash_kda import build

bundle = build(
    cb.Target("dav-3510", aic_count=28, aiv_count=56),
    head_pairs=((1, 1), (1, 2)),
    batch_max=4,
    sequence_max=8192,
)
bundle.save("/absolute/output/bundle")
```

将 bundle 转成 ops-transformer 可接入的 stage：

```bash
python -m cannbot_framework.cli stage-bundle \
  --bundle /absolute/output/bundle --operator FlashKda \
  --vendor custom_transformer --parent-opp /absolute/ops-transformer \
  --output /absolute/output/stage
```

从 ops-transformer 根目录配置构建时，设置
`ENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON`、`FLASH_KDA_FROZEN_STAGE=/absolute/output/stage`，
即可由本目录 CMake 接入生成的 API 与 Provider。沿用仓库的正式 run 打包流程，
`ENABLE_BUILT_IN=ON` 选择 builtin 布局。仅构建此算子时使用 `ASCEND_OP_NAME=flash_kda`；
Provider-only 构建不需要占位 C++ 或空的传统算子库。

## 二进制接口约定

生成的 ACLNN 接口用 `is_bsnd` 表示布局：false=BNSD，true=BSND。
这是由于当前框架的 ACLNN 属性支持布尔值而不支持布局字符串；原 Python 接口保持字符串。
二进制接口没有仅允许 `None` 的 `cu_seqlens` 参数。

`group_override` / `dv_base_override` 是发布时固定的高级属性，0 表示原自动策略。
命令行未显式指定时，沿用原 `KDA_GROUP` / `KDA_DV_BASE` 环境配置；
二进制调用必须传入该包声明的值，运行时不会再读取这些环境变量。
普通发布与调用均使用 0 即可。

发布签名保留 `lower_bound` 的 [-5, 0] 范围，不人为排除 0；`scale` 为 float32，
不限制在 [0, 1]。连续张量中大小为 1 的维度允许独立的非负 stride。
输出按物理 shape 分配，单例维的 stride 可能规范化，有效元素和布局保持不变。

## 适配层保留了什么

- **Entry 与发布范围**：声明合法输入、分配输出和工作空间，让框架生成二进制调用计划。
- **静态调度接入**：从显式 Target 获取核数，用仅保存静态字段的接收对象调用原 `FlashKDA.run`，避免冻结过程访问设备查询和 Python 缓存。
- **自动 group 的可捕获表达**：按原函数文档中的 L2 规则选择 group；当前框架不接受原辅助函数里的 `while`，因此保留一个有界循环。原辅助函数本身不改动。
- **单位矩阵初始化**：用小型 Provider 写入原计算所需的 16×16 FP16 单位矩阵。原 run 的 mask 参数保留工作空间，但原 stage1 没有读取它，新增适配无需初始化 mask。

没有第二份 kernel 实现，没有独立的 `kernels.py`、`package_flash_kda.py` 或
`provider_only_stub.cpp`。测试集中在 `tests/test_flash_kda.py`。

## 验证

测试先核对原文件完整字节前缀、AST 和原符号未被覆盖，再用相同输入比较原入口、
native bundle 和安装后的 ACLNN 调用。检查 output / final_state 逐位一致、重复调用一致、
initial_state 不变，并单独记录 CPU recurrence 的精度结果。

原基线为 `cannbot-arena/samples/flash_kda/flash_kda.py`，共 2183 行，SHA-256：
`6dc4c436098784b314962c37a94edeb4cd16dbe9e61dbb2d02a03e8e659b996e`。
已知主测试矩阵的三个 `lower_bound=0` 用例在原版自身也超过 CPU 阈值；
测试必须如实报告，不能通过修改数学实现、放宽阈值或排除输入来掩盖。
