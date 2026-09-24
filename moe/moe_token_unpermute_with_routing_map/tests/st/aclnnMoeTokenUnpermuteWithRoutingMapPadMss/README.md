# MoeTokenUnpermuteWithRoutingMap pad 路径内存/功能回归用例

针对 issue [#4278](https://gitcode.com/cann/ops-transformer/issues/4278)（pad 路径
0 字节 UB 队列）与多循环偏移缺陷，补充的定向回归矩阵，覆盖：

- issue 小 shape（`num_tokens_each_front_core < maxNPerLoopForUb`，修复前 each_loop=0）
- 常规中 shape、rowVal≈coreNum、front/tail 分核、每 expert 1 token
- 多循环大 shape（fp32 2.4M / bf16 4.8M token）
- fp32 / fp16 / bf16 / 混精（bf16 token + fp32 probs）
- 极值（E=1、C=2、rowVal=9）

## 文件

- `executor_moe_token_unpermute_with_routing_map.py`：ATK executor，
  解析 `case_spec` 生成输入，`run_npu` 调 `torch.ops.npu._npu_moe_token_unpermute_with_routing_map`，
  `run_cpu` 提供参考。
- `atk_moe_unpermute_mss_matrix.json`：15 个 case 的矩阵（`case_spec` 内嵌 shape/dtype）。
- `verify_matrix.py`：不依赖 ATK 精度标准的 NPU vs CPU 功能校验：
  确定性输出（permute_probs / out_index / permute_token_id）逐位一致，
  unpermuted_tokens 按 dtype 容差（fp32 ≤1e-3，half ≤0.05）。

## 运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
conda activate <atk 环境>

# 1) 功能校验（NPU vs CPU 参考）
python verify_matrix.py

# 2) 内存检测（mssanitizer memcheck）
mssanitizer --tool=memcheck --log-file=$PWD/mssanitizer_matrix.log -- \
  atk node --name npu_dut --backend npu --devices 0 task \
    -c ./atk_moe_unpermute_mss_matrix.json \
    -p ./executor_moe_token_unpermute_with_routing_map.py \
    --task run --mssanitizer \
    -msl $PWD/mssanitizer_matrix.log -to 3600
```

## 说明

- 非混精 half：NPU 按 token dtype 量化乘积后 fp32 累加，`run_cpu` 参考需对齐该行为，
  否则 bf16 累加会出现数 ULP 级位差（默认精度标准下会误报失败）。
- `mssanitizer` 的 `--log-file` 必须与 `-msl` 指向同一绝对路径。
