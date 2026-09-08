# TurboQuantSparseFlashAttention 测试说明

## 执行方式

先按[构建与安装](../docs/install.md)准备算子包和标准 PyTorch 包，再在仓库根目录执行：

```bash
# Host tiling/infer-shape UT
bash build.sh --ops=turbo_quant_sparse_flash_attention --ophost_test -j8
# 安装兼容包，验证旧接口转发
python3 -m pip install --no-deps attention/turbo_quant_sparse_flash_attention/torch_ops_extension
# 按实际空闲设备设置物理设备编号；pytest 在进程内使用逻辑设备 0
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest attention/turbo_quant_sparse_flash_attention/tests/pytest -q
```

## 覆盖范围

`pytest` 覆盖 FP16/BF16 query、PA_BSND/PA_BNBD、不同 query head 数、空序列、页间 padding、首轴 stride、storage offset、可选参数、非法元数据，以及图回放时更新输入的场景。rope 主导且归一化系数远离 1 的用例用于检查 `rope/scale` 语义。

[spec.py](assets/spec.py) 提供 TestSpec 入口。[独立参考实现](assets/tq_sfa_reference.py)从实际打包 cache 字节解码，使用 CPU float64 计算 golden；输入生成器原地填充张量，保留 cache 的 stride 和 storage offset。

[资产测试](pytest/test_turbo_quant_sparse_flash_attention_assets.py)覆盖稀疏块大小 1/2/4/8/16、页大小 16/1024、8K/32K KV，并通过故意扰动输出检查比较器能否检出错误。直接运行这些 pytest 不等同于执行完整 TTK 流程。

## 直接 ACLNN 验证

[C++ 验证程序](../examples/test_aclnn_turbo_quant_sparse_flash_attention.cpp)覆盖 FP16/BF16、PA_BSND/PA_BNBD、空 query、零 KV 长度及带 storage offset 的非连续 cache，共 16 个用例，并检查结果与输入 cache 未被修改。

安装自定义验证包后，在仓库根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
export CUSTOM_OP=/your/operator/install/vendors/custom_transformer
source "$CUSTOM_OP/bin/set_env.bash"
g++ -std=c++17 -O2 \
  attention/turbo_quant_sparse_flash_attention/examples/test_aclnn_turbo_quant_sparse_flash_attention.cpp \
  -I"$ASCEND_HOME_PATH/include" -I"$CUSTOM_OP/op_api/include" \
  -L"$ASCEND_HOME_PATH/lib64" -L"$CUSTOM_OP/op_api/lib" \
  -lcust_opapi -lnnopbase -lascendcl -o test_turbo_quant_sfa
ASCEND_RT_VISIBLE_DEVICES=0 ./test_turbo_quant_sfa
```

使用标准内置包时，按[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)调整头文件、库路径及链接库。

## 精度判据

当前回归用例中，FP16/BF16 均使用 `atol=rtol=2^-9`，同时满足以下条件才判定通过：

- 满足 $|actual-golden| \le atol+rtol\times|golden|$ 的元素占比至少为 `0.99`。
- 最大绝对误差不超过 `0.1`。

上述判据用于与历史回归结果比较。发布验收需按对应版本的正式精度门槛和验证流程执行。
