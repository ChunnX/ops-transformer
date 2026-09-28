# FlashMLA 参数校验维护说明

主算子入口使用 `FlashMlaWithKvcacheChecker` 组合各参数组的校验。
metadata 算子的 Host 参数校验位于
`../../../flash_mla_with_kvcache_metadata/op_host/flash_mla_with_kvcache_metadata_check.h`。

## 校验顺序

1. 空 Tensor：parser 先检查 Q、k_cache、attn_out 的零轴，组合 checker 再检查空 Tensor 标志。
2. `CheckParaExistence`：建立必需 Tensor、desc、shape 的存在性，以及可选输入的必传/禁传关系。
3. `CheckSinglePara`：检查类型、格式、维数和属性范围。
4. `CheckFeature`：检查当前特性是否适用。
5. `CheckMultiPara`：检查多个输入之间的布局、头数和 shape 一致性。

检查失败立即返回错误。必须先判空再解引用、先确认维数再读取轴，先确认除数大于0再做取模或除法。
不在 Host checker 中解引用设备数据指针。

## 每个参数组的职责

| Checker | 职责 |
| --- | --- |
| Common | Q/KV/out 的布局、类型、轴大小与 MLA 维度 |
| ActualSeqLen | 序列长度参数的存在性、类型、shape 和 max length 属性 |
| PagedAttention | block_size 范围/对齐、block_table 和 KV stride 限制 |
| Mask | mode、窗口属性以及 mask 的存在性、类型、shape |
| Sinks | 可选 sink 的类型及头数匹配 |
| Metadata | 主算子 metadata 输入的存在性、类型与非空一维 shape |
| SoftmaxLSE | 开关、输出存在性、类型与完整输出 shape |

`CommonChecker` 中的 `Q_OUTPUT_LAYOUTS` 是 Q/输出布局规则的唯一表，
`PAGED_KV_LAYOUTS` 是 KV 布局白名单。单参数检查从同一张表导出支持集合，多参数检查验证对应关系。
增加 layout 时还需同步 parser、shape 解析和实际计算路径，不能仅放开 checker。

## 可选输入语义

- mode=0 禁传 mask；mode=3/4 必传 mask。
- 非空 sinks 必须是 FP32、shape 为 `[Q头数]`；None 或一维空 Tensor 保留关闭 sink 的语义。
- 关闭 LSE 时不检查 LSE 输出；开启时 desc/shape 必须存在，TND 对应 `[Nq,T]`，BSND/BNSD 对应 `[B,Nq,S]`。
- 主算子和 metadata 的 Q 头数范围均为1–128；修改范围时同步 Host、metadata AICPU 的属性校验。

## 扩展与回归

扩展已有参数优先修改所属 checker；新增独立参数组时继承 `FlashMlaWithKvcacheBaseChecker`，
只覆盖需要的阶段，并在 `RegisterCheckers()` 注册。错误信息应包含参数名、实际值和期望范围。

每次调整应覆盖正常输入、边界值、越界值、缺失输入和相关组合。
Host 回归入口：`tests/assets/run_checker_tests.py`，用法见 `tests/assets/README.md`。

序列元素值、分页索引、mask 内容和 metadata 调度内容不属于当前 Host checker 的已执行检查。
校验放行不能替代内核或 NPU 端到端验证。
