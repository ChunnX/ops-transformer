# BlockAttentionResiduals 正向 Torch 精度测试

测试对象为主仓目录：

```text
mhc/block_attention_residuals
```

Torch 接口：

```python
block_attention_residuals(
    partial_block, block_res, proj_weight, norm_weight,
    valid_block_num=-1, norm_eps=1.0e-6,
)
```

输入 shape：

```text
partial_block: [T, H]
block_res:    [T, N, H]，由连续存储 [T, k, H] 沿 block 维切片得到（k >= N）
proj_weight:  [H] 或 [1, H]
norm_weight:  [H]
```

主输入支持 `float16`、`bfloat16` 和 `float32`；`1 <= N <= 100`，`H >= 1`，`valid_block_num` 为 `-1` 或 `N`。

测试默认将 `block_res` 初始化为 `[T, k, H]`（`k = 2N`，可用 `--storage-blocks` 指定），再切片为 `[T, N, H]`。该 view 在 NPU 上保持非连续（token 维 stride 为 `k*H`），算子按 stride0 直接读取，不先转 contiguous。

## 参考计算

脚本使用 PyTorch float32 计算参考结果：

```text
values = cat(block_res, partial_block)
inv_norm = rsqrt(mean(values^2, dim=-1) + norm_eps)
score_weight = proj_weight * norm_weight
probs = softmax(sum(values * inv_norm * score_weight, dim=-1))
hidden_states = sum(probs * values)
```

参考结果最后转换回主输入 dtype，与 NPU 正向输出比较。

## 234 环境

```bash
cd /data/kdy/ops-transformer/tmp/ops-transformer-cann-master-20260912
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
```

如需指定设备，先确认物理卡空闲：

```bash
export ASCEND_RT_VISIBLE_DEVICES=<空闲物理卡号>
```

## 运行

CPU 参考自检：

```bash
python3 mhc/block_attention_residuals/tests/test_torch_block_attention_residuals_forward.py --device cpu
```

NPU 精度测试：

```bash
python3 mhc/block_attention_residuals/tests/test_torch_block_attention_residuals_forward.py \
  --device npu --device-id 0
```

指定测试规模和 dtype：

```bash
python3 mhc/block_attention_residuals/tests/test_torch_block_attention_residuals_forward.py \
  --tokens 4 --blocks 8 --hidden-size 128 --dtype fp16 --seed 1
```

默认最大绝对误差阈值为 `2.0e-2`，通过后输出 `PASS`。该阈值用于冒烟测试，正式验收应按具体 dtype 和测试规范调整。
