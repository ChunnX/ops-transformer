# 算子转测准入扫描

- 代码根目录：`/home/z50058944/moe_transfer_scan_root.C9Kjly`
- Golden 根目录：`/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/assets`
- 第 4 项为静态证据收集，必须人工定版。
- 第 5 项只有通过显式绑定的 Ascend950 静态与动态 GEIR 证据校验后才能为 PASS。
- 第 9～13 项只有通过算子约束清单与 Ascend950 TTK 异常实跑证据校验后才能为 PASS/N/A。
- 第 9～13 项的 FAIL 不等于算子已拦截；逐项结论会区分有效拦截、应拦截但未拦截和不可判定。

## 汇总

| 算子 | 1 原型 | 2 一致性 | 3 950 ini | 4 Golden | 5 GEIR | 6 旧 Tiling | 7 def dtype | 8 Infer | 9 dtype 异常 | 10 格式异常 | 11 9D | 12 属性值域 | 13 Shape/维度 | 14 属性可选性 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MoeTokenUnpermuteWithEp | FAIL | N/A | PASS | FAIL | REMEDIATE | PASS | PASS | FAIL | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |

## 逐算子证据

### MoeTokenUnpermuteWithEp

- 开源算子目录：`/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep`
- `1-prototype-uniqueness`：**FAIL** — 三个指定位置均未找到原型
- `2-prototype-consistency`：**N/A** — 没有可比较的重复原型
- `3-ascend950-ini`：**PASS** — Ascend 950 TBE ini 中无该 section
- `4-golden`：**FAIL** — 未找到 Golden Python 文件
- `5-geir`：**REMEDIATE** — 动态 example 缺少关键结构，需自动修复: RunGraph, Shape/dtype/values verification
  - `/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_def.cpp:68` — DynamicShapeSupportFlag(true)
  - `/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_def.cpp:67` — DynamicRankSupportFlag(true)
  - `/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/examples/arch35/test_geir_moe_token_unpermute_with_ep.cpp:1` — 静态 GEIR example
  - `/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/examples/arch35/test_geir_moe_token_unpermute_with_ep_dynamic.cpp:1` — 动态 GEIR example
- `6-legacy-tiling`：**PASS** — 未发现指定的旧 Tiling 注册宏
- `7-def-driven-dtype`：**PASS** — 未发现 dtype TilingKey 字段，kernel 使用 def 生成的 DTYPE 宏
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:20` — DTYPE_PERMUTED_TOKENS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:21` — DTYPE_PERMUTED_TOKENS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:23` — DTYPE_PROBS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:24` — DTYPE_PROBS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:24` — DTYPE_PERMUTED_TOKENS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:46` — DTYPE_PERMUTED_TOKENS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:46` — DTYPE_PROBS
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep_apt.cpp:48` — DTYPE_PERMUTED_TOKENS
- `8-infer`：**FAIL** — runtime 未注册，缺少规定位置的注册: InferDataType: /home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_graph/moe_token_unpermute_with_ep_graph_infer.cpp
  - `/home/z50058944/moe_transfer_scan_root.C9Kjly/ops-transformer_moe/moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_infershape.cpp:1` — 开源仓 infer 文件
- `9-exception-dtype`：**UNAVAILABLE** — 未提供 TTK 异常实跑证据目录: 未绑定
- `10-exception-format`：**UNAVAILABLE** — 未提供 TTK 异常实跑证据目录: 未绑定
- `11-exception-rank9`：**UNAVAILABLE** — 未提供 TTK 异常实跑证据目录: 未绑定
- `12-exception-attribute`：**UNAVAILABLE** — 未提供 TTK 异常实跑证据目录: 未绑定
- `13-exception-shape-dimension`：**UNAVAILABLE** — 未提供 TTK 异常实跑证据目录: 未绑定
- `14-attribute-optionality`：**UNAVAILABLE** — 未找到算子原型，无法比较属性可选/必选定义
