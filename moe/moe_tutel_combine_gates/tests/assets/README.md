# MoeTutelCombineGates 算子 TTK 测试资产

## 功能说明

基于 ops-test-kit（TTK）测试框架，完成 MoeTutelCombineGates 算子的 **Kernel（AscendC）精度验证**：

- 用例文件：`tests/st/arch35/ttk_kernel_moe_tutel_combine_gates.csv`（13 条，仅 Ascend 950 / arch35）
- TestSpec：`tests/assets/moe_tutel_combine_gates.py`，同时提供 golden 与输入定制
- 每条用例都会走 **动态 shape（dyn）** 与 **静态 shape（cst）** 两条编译/执行通路，逐元素与 CPU golden 比对

> 本算子只注册了 `ascend950` 一份 AICore 配置，因此 ST 用例只放在 `arch35` 目录下。

## 精度标准

golden 是算子数学定义的独立 CPU 实现，不调用算子自身的任何产物，因此直接使用生态默认标准 `mix_tolerance`（按输出 dtype 取内置阈值）：

```python
tolerance = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}
```

数值口径与 kernel 的累加契约对齐：hidden 维乘积以 float64 精确计算（float32/bfloat16/float16 操作数的 float64 乘积即精确乘积，对应 kernel fp32 路径的 TwoProduct 误差通道与 16b 路径的精确 fp32 乘积），再以 float64 求和、一次性舍入到输出 dtype（等价于 kernel 补偿归约的正确舍入结果）。实测各用例与 golden 的最大绝对误差均在输出 dtype 的 1 个 ulp 量级，`matched_ratio` 恒为 `1.0`。

## golden 与输入定制

`tests/assets/moe_tutel_combine_gates.py` 通过 `__spec__ = {"moe_tutel_combine_gates": "MoeTutelCombineGatesTestSpec"}` 注册（Kernel 路径的注册键就是算子名），包含：

| 成员 | 作用 |
| --- | --- |
| `golden` | 输入按算子定义顺序 `x / y_grad / indices / locations` 传入 `numpy.ndarray`，返回 `[gates_grad]`。展平 task 下标 `t = b * S + s`，`token = t % S`，`slot = e * C + p`；非法路由（`e < 0` / `e >= B` / `p < 0` / `p >= C`）结果为精确 `0` |
| `customize_inputs` | TTK 默认会独立随机生成 `indices` / `locations`，那不是一组合法路由。该钩子保持 `x` / `y_grad` 不变，按用例名与 shape 派生的固定种子重写两个下标张量，保证每个 task 都指向真实的 `y_grad` 行 |
| `tolerance` | 见上 |

`illegal` 用例（用例名含 `illegal`）只在 `e < 0` 与 `p >= C` 两类取值上注入非法路由：这两类在任何通路上都不会被校验层拒收，因此用例仍然合法，同时钉住「越界路由输出精确 0 且不读 `y_grad`」的语义。

> 注意：`x` / `y_grad` 仍由 TTK 随机生成（CI 命令未指定 `--seed`），只有下标张量是确定性的。因此每次运行的数值与最大绝对误差会有小幅波动；判定标准是逐元素的 `mix_tolerance` 与 `precision_status`，不要以某一次的 `max_abs_error` 具体数值为准。

## 用例清单

| 用例 | dtype | S × H × B × C | 覆盖点 |
| --- | --- | --- | --- |
| `mtcg_fp32_tc01` | float32 | 500 × 64 × 2 × 500 | fp32 分支（tilingKey 1）；taskNum=1000 非 16 对齐，核尾切分 |
| `mtcg_fp32_h1` | float32 | 4 × 1 × 2 × 3 | 规约轴退化 H=1 |
| `mtcg_fp32_h0` | float32 | 4 × 0 × 2 × 3 | H=0 且 S,B>0（非空张量），整表精确 0 |
| `mtcg_fp32_illegal_zero` | float32 | 8 × 32 × 2 × 4 | `e < 0` / `p >= C` 精确 0，且不发起越界访存 |
| `mtcg_bf16_tc03` | bfloat16 | 1024 × 1024 × 8 × 256 | bf16 分支（tilingKey 0）；大 shape，taskNum=8192 |
| `mtcg_bf16_core_tail` | bfloat16 | 16 × 128 × 40 × 4 | taskNum=640 恰为 AIV 核数整数倍 |
| `mtcg_bf16_unaligned` | bfloat16 | 128 × 513 × 2 × 64 | H=513 未按 UB 路由粒度对齐 |
| `mtcg_bf16_c0` | bfloat16 | 4 × 8 × 2 × 0 | C=0 使所有 task 非法，整表精确 0 |
| `mtcg_fp16_unaligned` | float16 | 96 × 130 × 3 × 17 | fp16 分支（tilingKey 2）；H 与 C 均非对齐 |
| `mtcg_fp16_s0` | float16 | 0 × 8 × 2 × 3 | S=0 空张量，输出元素数为 0 |
| `mtcg_fp32_b0` | float32 | 4 × 8 × 0 × 3 | B=0 空张量，输出元素数为 0 |
| `mtcg_fp16_s0h0` | float16 | 0 × 0 × 2 × 3 | S=0 与 H=0 同时成立：taskNum==0 优先于 H 规则，输出为空张量 |
| `mtcg_bf16_b0c0` | bfloat16 | 4 × 8 × 0 × 0 | B=0 与 C=0 同时成立：taskNum==0 优先于 C 规则，输出为空张量 |

## 执行方式

前置：已经编译并安装本算子的 run 包（`bash build.sh --pkg --soc=ascend950 --ops=moe_tutel_combine_gates`，随后安装 `build_out` 下的包），并 `source <vendor>/bin/set_env.bash` 使 `ASCEND_CUSTOM_OPP_PATH` 指向该 vendor 包。

```bash
# 仓库自带入口（首次会自动通过 cmake 下载 ops-test-kit 到 build/third_party/ops-test-kit）
bash scripts/ci/ops_st_test.sh --soc_version=ascend950 --ops=moe_tutel_combine_gates --test_type=kernel
```

也可以直接调用 TTK：

```bash
cd build/third_party/ops-test-kit
python3 -m ttk kernel \
    -i <repo>/moe/moe_tutel_combine_gates/tests/st/arch35/ttk_kernel_moe_tutel_combine_gates.csv \
    -o /tmp/ttk_kernel_moe_tutel_combine_gates_result.csv \
    --plugin <repo>/moe/moe_tutel_combine_gates/tests \
    -c --pc=8 --warmup=false
```

结果 CSV 中 `precision_status` 为 `PASS` 即通过；`dyn_tiling_key` 字段可用来核对 dtype → tilingKey 的路由（bf16 → 0、fp32 → 1、fp16 → 2）。
