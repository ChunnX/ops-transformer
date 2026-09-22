# MoeTutelCombineGates算子TTK测试资产

## 功能说明

基于ops-test-kit（TTK）测试框架，完成MoeTutelCombineGates算子的**Kernel（AscendC）精度验证**：

- 用例文件：`tests/st/arch35/ttk_kernel_moe_tutel_combine_gates.csv`（13条，仅Ascend 950/arch35）
- TestSpec：`tests/assets/moe_tutel_combine_gates.py`，同时提供golden与输入定制
- 每条用例都会走**动态shape（dyn）**与**静态shape（cst）**两条编译/执行通路，逐元素与CPU golden比对

> [!NOTE]
> 本算子只注册了`ascend950`一份AICore配置，因此ST用例只放在`arch35`目录下。

## 精度标准

golden是算子数学定义的独立CPU实现，不调用算子自身的任何产物，因此直接使用生态默认标准`mix_tolerance`（按输出dtype取内置阈值）：

```python
tolerance = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}
```

数值口径与kernel的累加契约对齐：hidden维乘积以float64精确计算（float32/bfloat16/float16操作数的float64乘积即精确乘积，对应kernel fp32路径的TwoProduct误差通道与16b路径的精确fp32乘积），再以float64求和、一次性舍入到输出dtype（等价于kernel补偿归约的正确舍入结果）。实测各用例与golden的最大绝对误差均在输出dtype的1个ulp量级，`matched_ratio`恒为`1.0`。

## golden与输入定制

`tests/assets/moe_tutel_combine_gates.py`通过`__spec__ = {"moe_tutel_combine_gates": "MoeTutelCombineGatesTestSpec"}`注册（Kernel路径的注册键就是算子名），包含：

| 成员 | 作用 |
| --- | --- |
| `golden` | 输入按算子定义顺序`x / y_grad / indices / locations`传入`numpy.ndarray`，返回`[gates_grad]`。展平task下标`t = b * S + s`，`token = t % S`，`slot = e * C + p`；非法路由（`e < 0`/`e >= B`/`p < 0`/`p >= C`）结果为精确`0` |
| `customize_inputs` | TTK默认会独立随机生成`indices`/`locations`，那不是一组合法路由。该钩子保持`x`/`y_grad`不变，按用例名与shape派生的固定种子重写两个下标张量，保证每个task都指向真实的`y_grad`行 |
| `tolerance` | 见上 |

`illegal`用例（用例名含`illegal`）只在`e < 0`与`p >= C`两类取值上注入非法路由：这两类在任何通路上都不会被校验层拒收，因此用例仍然合法，同时钉住「越界路由输出精确0且不读`y_grad`」的语义。

> [!CAUTION]
> `x`/`y_grad`仍由TTK随机生成（CI命令未指定`--seed`），只有下标张量是确定性的。因此每次运行的数值与最大绝对误差会有小幅波动；判定标准是逐元素的`mix_tolerance`与`precision_status`，不要以某一次的`max_abs_error`具体数值为准。

## 用例清单

| 用例 | dtype | S×H×B×C | 覆盖点 |
| --- | --- | --- | --- |
| `mtcg_fp32_tc01` | float32 | 500×64×2×500 | fp32分支（tilingKey 1）；taskNum=1000非16对齐，核尾切分 |
| `mtcg_fp32_h1` | float32 | 4×1×2×3 | 规约轴退化H=1 |
| `mtcg_fp32_h0` | float32 | 4×0×2×3 | H=0且S,B>0（非空张量），整表精确0 |
| `mtcg_fp32_illegal_zero` | float32 | 8×32×2×4 | `e < 0`/`p >= C`精确0，且不发起越界访存 |
| `mtcg_bf16_tc03` | bfloat16 | 1024×1024×8×256 | bf16分支（tilingKey 0）；大shape，taskNum=8192 |
| `mtcg_bf16_core_tail` | bfloat16 | 16×128×40×4 | taskNum=640恰为AIV核数整数倍 |
| `mtcg_bf16_unaligned` | bfloat16 | 128×513×2×64 | H=513未按UB路由粒度对齐 |
| `mtcg_bf16_c0` | bfloat16 | 4×8×2×0 | C=0使所有task非法，整表精确0 |
| `mtcg_fp16_unaligned` | float16 | 96×130×3×17 | fp16分支（tilingKey 2）；H与C均非对齐 |
| `mtcg_fp16_s0` | float16 | 0×8×2×3 | S=0空张量，输出元素数为0 |
| `mtcg_fp32_b0` | float32 | 4×8×0×3 | B=0空张量，输出元素数为0 |
| `mtcg_fp16_s0h0` | float16 | 0×0×2×3 | S=0与H=0同时成立：taskNum==0优先于H规则，输出为空张量 |
| `mtcg_bf16_b0c0` | bfloat16 | 4×8×0×0 | B=0与C=0同时成立：taskNum==0优先于C规则，输出为空张量 |

## 执行方式

前置：已经编译并安装本算子的run包（`bash build.sh --pkg --soc=ascend950 --ops=moe_tutel_combine_gates`，随后安装`build_out`下的包），并`source <vendor>/bin/set_env.bash`使`ASCEND_CUSTOM_OPP_PATH`指向该vendor包。

```bash
# 仓库自带入口（首次会自动通过 cmake 下载 ops-test-kit 到 build/third_party/ops-test-kit）
bash scripts/ci/ops_st_test.sh --soc_version=ascend950 --ops=moe_tutel_combine_gates --test_type=kernel
```

也可以直接调用TTK：

```bash
cd build/third_party/ops-test-kit
python3 -m ttk kernel \
    -i <repo>/moe/moe_tutel_combine_gates/tests/st/arch35/ttk_kernel_moe_tutel_combine_gates.csv \
    -o /tmp/ttk_kernel_moe_tutel_combine_gates_result.csv \
    --plugin <repo>/moe/moe_tutel_combine_gates/tests \
    -c --pc=8 --warmup=false
```

结果CSV中`precision_status`为`PASS`即通过；`dyn_tiling_key`字段可用来核对dtype → tilingKey的路由（bf16 → 0、fp32 → 1、fp16 → 2）。
