
MambaV2系列算子（Ascend CANN / cann-ops-transformer）

本目录包含针对Nemotron-h系列模型的Mamba v2定制算子，基于cann-ops-transformer框架，在华为Ascend AI（昇腾NPU）910B平台上实现高性能加速。

Mamba v2提出基于状态空间模型（State-Space Model, SSM）的序列建模方法，在长序列建模方面相较Transformer具有更优的计算效率与内存效率。本目录中提供Mamba v2推理prefill阶段所需的关键算子。

**Nemotron-H网络中的Mambav2整层计算流：**

<img src="https://raw.gitcode.com/user-images/assets/7673863/5163260c-fe1e-4a06-90c6-5355ba23ea90/image.png" height="900">

**Mambav2 chunk scan combined计算流和融合算子设计：**

![image.png](https://raw.gitcode.com/user-images/assets/7673863/b98b23ee-bb4b-42b5-8270-92828d17c2d2/image.png 'image.png')

目录结构

```
experimental/
|
├──attention/
    |
    ├──mambav2/
        |
        ├── mamba2_causal_conv1d/       # 基于状态空间模型的因果卷积（Causal SSM Convolution）
        ├── mamba2_chunk_cumsum/        # 按chunk进行cumulative sum，用于streaming状态累积
        ├── mamba2_chunk_state/         # chunk状态更新
        ├── mamba2_chunk_state_passing/ # chunk间状态传递
        ├── mamba2_chunk_scan/          # selective scan扫描机制
        ├── mamba2_rmsnormgated/        # RMSNorm + Gate融合算子
        └── utils/                      # 公共工具包括精度比对和性能profiling

```

其中mamba2_chunk_xxx四个算子为Prefill过程中Chunk计算的核心实现模块。

每个算子子目录包含：
    - 算子实现(op/op_kernel/)
    - torch封装(torch_interface.cpp)
    - 精度和性能测试脚本(op/test/)
    - 算子介绍说明文档

编译与使用

1. 编译目录

```
path="path/to/experimental"
cd $path$/npu_ops_transformer_ext
```

2. 编译命令

```
python3 -m build --wheel -n
```

3. 安装

```
cd dist
pip3 install *.whl --force-reinstall -no-deps
```

4. 测试，以mamba2_causal_conv1d为例

```
export PYTHONPATH=$PYTHONPATH:/path/to/experimental/attention/mambav2
cd $path$/attention/mambav2/mamba2_causal_conv1d/tests
python test_causal_conv1d.py
```

特性说明：

1. 当前版本算子已支持FP32 / FP16输入输出精度；
2. 所有mamba2_chunk_xxx系列算子均支持BSND数据布局，其中S维度需在调用前pad至chunk_size的整数倍；
3. 当前版本仅支持固定chunk_size = 256；
4. 已通过PyTorch参考实现的精度比对验证（测试脚本见各算子的test目录）。

**单融合算子性能加速比(每个算子tests下测试脚本在910B3的profile结果)：**

<img src="https://raw.gitcode.com/user-images/assets/7673863/4d4b226f-cce2-4c93-bb01-fdaade6fff7e/image.png" height="200">
