# IndexerQuantCache 回归验证

## CPU Golden

安装 numpy、ml_dtypes、en_dtypes 后，从仓库根目录执行：

```bash
python attention/indexer_quant_cache/tests/st/arch35/test_quant_golden.py
```

覆盖 Normal 全零/非有限值、非法 slot 保留缓存、Torch MX 载体打包转换和 E5M2。

## 主机和 kernel UT

加载对应 CANN 环境后执行：

```bash
bash build.sh --ophost_test --ops=indexer_quant_cache --soc=ascend950
bash build.sh --opkernel_test --ops=indexer_quant_cache --soc=ascend950
```

主机测试覆盖空 batch、UB 无法容纳一行及非法模式/dtype 配对。kernel 测试覆盖 single-row 连续跳过、越界 slot 后继续处理合法行，以及 round_scale 两值下全零行的 cache/scale。

本次新增 numSlots 到 TilingData，主机与 kernel 必须从同一版本重新构建，不能混用旧的预编译二进制。

## TTK Torch eager

先安装由当前源码构建的算子包与 Torch 扩展，并配置 ASCEND_CUSTOM_OPP_PATH。
如系统同时安装其他版本的扩展，应确认加载的是当前版本。

```bash
export PYTHONPATH="$PWD/attention/indexer_quant_cache/tests/assets:$PYTHONPATH"
python -m ttk e2e \
  -i attention/indexer_quant_cache/tests/st/arch35/ttk_torch_static_regressions.csv \
  --plugin attention/indexer_quant_cache/tests/assets/golden.py
```

测试适配器调用公开 Python API，再返回两个原地更新张量供 TTK 比较；不改变公开接口的 None 返回值。CSV 不把 cache 标记为 out 参数。8 个用例覆盖四种模式的单 token/多 token，以及 MX-FP8 E5M2。Golden 为 Torch 的 uint8 MX 载体执行位重解释，MX-FP4 按 E2M1 解包/打包，不把 uint8 当作数值量化。

TensorFlow 适配、Torch FP4-E1M2 及图模式非连续原地视图未列入当前交付范围，见 [算子 README](../README.md)。
