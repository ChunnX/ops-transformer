# IndexerQuantCache 验证

## 主机参数校验

`tests/ut/op_host/test_indexer_quant_cache_tiling.cpp` 覆盖四维 cache/scale 约束、量化模式与 dtype 配对、空 batch、batch 乘积溢出、分页维度不匹配、UB 不足及 MX-FP4 UB 预算。构建并执行：

```bash
bash build.sh --ophost_test --ops=indexer_quant_cache --soc=ascend950
```

## 设备 TTK

安装待测算子包后，设置 `ASCEND_CUSTOM_OPP_PATH` 指向对应 vendor，并将 `TTK_OP_API_SO_PATH` 和 `TTK_OP_API_HEADER_PATH` 指向该包的 ACLNN 动态库和头文件目录，避免误用其他已安装版本。

```bash
ttk aclnn -i attention/indexer_quant_cache/tests/st/arch35/ttk_aclnn_mxfp4_ub.csv \
  --plugin attention/indexer_quant_cache/tests/assets/golden.py \
  --seed 42 --pc 1 --task-prof false --run 1 --warmup false
```

- `ttk_aclnn_mxfp4_ub.csv`：小 batch 对照及 FP16/BF16 × FP4 E2M1/E1M2 的大 batch UB 回归，共 5 条；cache 和 scale 按编码逐字节比较。
- `ttk_aclnn_scale_preservation.csv`：四模式、单/多 token 以及未命中 slot 保留，共 16 条。
- 对照验证可通过 `--no-prof --dump in,golden --manual-data-dirs <目录>` 准备一次数据，再去掉 `--no-prof --dump in,golden`，由两个版本分别回放同一目录。

slot 仅使用唯一合法索引和 `-1`。越界 slot 属于不支持的输入，不将其纳入正常输出比较，也不保证设备安全跳过。Torch MX-FP8 E5M2 的当前限制见 [Torch 接口文档](../docs/torchapi_indexer_quant_cache.md)。
