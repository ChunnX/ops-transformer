# GroupedMatmulSwigluQuantWeightNzV3 Tensor API 门槛用例

以下用例从 Ascend 950 的随机精度验证中筛选，用于后续修改
`swiglu_mode=2` MXFP8 weight-NZ 路径后的回归。每条用例均应同时完成
ACLNN GetWorkspace、kernel 执行和 CPU golden 对比；精度门限为：

- `rel_err > 2e-3` 的点数为 0；
- `rel_err > 1e-3` 的点数不超过总点数的千分之一；
- 不允许出现非有限值。

`N` 是 SwiGLU 前的完整输出宽度，输出宽度为 `N / 2`。所有用例均使用
FP8 E4M3FN 的 x/weight/output、E8M0 的 xScale/weightScale/outputScale，
`group_list_type=0`、`round_mode="rint"`、`dst_type_max=0.0`。

| 名称 | Layout | M | N | K | E | 分组模式 | scale_alg | 预期路径 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| nz_tensor_boundary_e1 | NZ | 513 | 128 | 128 | 1 | 单组 | 0 | Tensor API, key 16 |
| nz_tensor_boundary_e2 | NZ | 1026 | 128 | 128 | 2 | 均分 | 1 | Tensor API, key 16 |
| nz_tensor_uneven_e3 | NZ | 1539 | 256 | 256 | 3 | 极不均分 | 0 | Tensor API, key 16 |
| nz_tensor_random_e4 | NZ | 2053 | 512 | 384 | 4 | 随机不均分 | 1 | Tensor API, key 16 |
| nz_tensor_large | NZ | 3000 | 4096 | 4096 | 3 | 随机不均分 | 0 | Tensor API, key 16 |
| zn_cgmct_boundary_e1 | ZN | 128 | 128 | 128 | 1 | 单组 | 1 | CGMCT, key 1 |
| zn_tensor_boundary_e1 | ZN | 129 | 128 | 128 | 1 | 单组 | 1 | Tensor API, key 17 |
| zn_tensor_boundary_e2 | ZN | 258 | 128 | 128 | 2 | 均分 | 0 | Tensor API, key 17 |
| zn_tensor_uneven_e3 | ZN | 390 | 256 | 256 | 3 | 极不均分 | 1 | Tensor API, key 17 |
| zn_tensor_random_e4 | ZN | 517 | 512 | 384 | 4 | 随机不均分 | 0 | Tensor API, key 17 |
| zn_tensor_medium | ZN | 1026 | 1024 | 1024 | 2 | 随机不均分 | 1 | Tensor API, key 17 |

这十一条覆盖 NZ/ZN、平均 M 准入临界点、E=1~4、均匀与不均匀分组、
`scale_alg=0/1` 及一组大 shape。`zn_cgmct_boundary_e1` 验证转置场景平均 M
等于 128 时回退 CGMCT，`zn_tensor_boundary_e1` 验证平均 M 为 129 时进入
Tensor API；其余九条覆盖当前迭代的 V3 Tensor API 主路径。该门槛保证
Tensor API 使用对称的 256×256 临时 L1 预算，避免 A/B scale 窗口不一致。
K 均为 128 对齐。

V3 与 V2 复用 CGMCT 的 key 0/1 和 Tensor API 的 key 16/17；各自保持旧
tiling 前缀布局，并在尾部追加 SwiGLU 参数。`swiglu_mode` 与 `scale_alg`
通过 tiling 参数传递，不再为 V3 或两种量化算法增加编译 key。
