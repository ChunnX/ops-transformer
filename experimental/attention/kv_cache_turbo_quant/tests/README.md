# KvCacheTurboQuant 自测说明

验收人按下面步骤可以复现精度和性能结果。环境为 CANN 9.1.0，Atlas A2（ascend910b）或 Atlas A3（ascend910_93）。本机实测卡为 Ascend910_93，Vector/Cube 指令集为 `dav-c220`。

## 用例

精度和性能使用同一组 shape，与任务书一致。`rows = num_tokens * num_kv_heads`，`head_dim = qjl_dim = 128`。

| case | num_tokens | num_kv_heads | mse_bits | eager 基线 (us) | 10× 上限 (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gqa_decode_b1 | 1 | 8 | 3 | 1491.676 | 149.168 |
| gqa_decode_b64 | 64 | 8 | 3 | 1552.784 | 155.278 |
| gqa_prefill_t2048 | 2048 | 8 | 3 | 3855.422 | 385.542 |
| gqa_mse2_t2048 | 2048 | 8 | 2 | 3607.963 | 360.796 |
| gqa_mse4_t2048 | 2048 | 8 | 4 | 5286.502 | 528.650 |

另外两组只做精度、不计入性能表：`gqa_heads4_t32`（32×4 head，3-bit）和 `gqa_heads32_t8`（8×32 head，3-bit），覆盖任务书 `num_kv_heads` 的下界和上界。

`recon`：256 行、3-bit、Sylvester Hadamard 与高斯投影，检查重构相对 MSE 和内积相对误差 95 分位。旋转必须正交，重构公式里的 `H^T` 才是逆。`case.json` 的旋转取值范围是 `[-1, 1]`，那是编码一致性用的任意矩阵，不拿来算重构误差。

`tests/case.json`、`tests/op.json`、`tests/golden.py` 是任务附带的 msopst 描述。`golden.py` 用 torch_npu 的 FP32 matmul 计算旋转和投影，再用 numpy bfloat16（round-to-nearest-even）写出 norm 和 gamma。验收对比的就是这份 golden，阈值来自 `case.json`：idx/qjl 的 atol 约为 1e-5（uint8 差 1 即失败），norm/gamma 为 `[0, 0]`（bf16 逐位一致）。

## 精度判定

- `build.sh` 跑完直调样例后，会调用 `tests/compare_golden.py`，把设备输出和 `golden.calc_expect_func` 逐元素比较。
- `quant_idx`、`quant_qjl`、`quant_norm`、`quant_gamma` 都必须是 0 失配。验收只看 `golden.py`。主机上逐通道标量点积和 Cube GEMM 不是同一条舍入路径，不参与判定。
- 重构：`MSE / ||x||^2` 的均值和最大值都小于 0.05。
- 内积：`|<q, x_hat> - <q, x>| / |<q, x>|` 在 `|cos| > 0.2` 的样本上，95 分位小于 0.1。查询取 `q = x + 0.35 * gauss`。

## 性能判定

五个性能 case 各预热 5 次，再用 `aclrt` event 统计 20 次 encode 的平均耗时。一次 encode 含 BF16 转换、两次 `aclnn` 行范数与除法、`gamma` 乘法、两次 Cube GEMM、Lloyd-Max、BF16 写回和 QJL 符号打包，全部在同一次计时里。平均耗时必须低于上表 10× 上限。头数边界 case 只做精度。

## 复现命令

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd kv_cache_turbo_quant
bash examples/build.sh /tmp/kv_cache_turbo_quant_test
```

进程退出码 0，样例打印 `PASSED`，随后 `compare_golden.py` 打印七行 `PASS` 和 `GOLDEN PASSED`，才算精度和性能都通过。五行性能 case 带微秒耗时，两行头数边界 case 标 `accuracy_only`。golden 日志给出四路输出的失配个数。

只编译不运行：

```bash
KVTQ_BUILD_ONLY=1 bash examples/build.sh /tmp/kv_cache_turbo_quant_test
```

设备选择：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash examples/build.sh
```
