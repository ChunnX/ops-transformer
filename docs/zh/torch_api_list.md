# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  `torch_extension`接口定义于`cann_ops_transformer`包中。更多接口说明及使用示例正在完善中，敬请期待。如有疑问或建议，欢迎通过Issue反馈。

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：因CANN或NPU型号不同等原因，可能无法保证同一个API运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分接口可通过PyTorch中控制算法确定性的全局开关[torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py)开启确定性算法，使多次运行结果一致。

|    接口名   |   说明     |  确定性说明（A2/A3）  | 确定性说明（Ascend 950） |
| ----------- | ------------------- | ------------------- | ------------------- |
|[flash_attn](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md)|FlashAttention非量化注意力计算。|-|默认支持确定性计算|
|[lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)|基于一系列操作得到每一个token对应的Top-k个位置。支持KV压缩场景。|默认确定性实现|-|
|[lightning_indexer_metadata](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)|lightning_indexer接口的前置接口，用于计算lightning_indexer的负载均衡。|默认确定性实现|默认确定性实现|
|[mhc_post](../../torch_extension/cann_ops_transformer/docs/zh/mhc_post.md)|实现MHC Post组件的前向计算，用于Transformer模型中多层残差连接的后处理阶段。该算子将残差矩阵变换与输出状态投影融合为单次计算，避免多次独立算子调用带来的额外开销。|默认确定性实现|-|
|[mhc_pre_sinkhorn](../../torch_extension/cann_ops_transformer/docs/zh/mhc_pre_sinkhorn.md)|基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。|默认确定性实现|-|
|[mixed_quant_sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla.md)|量化场景下基于共享KV完成MixedQuantSparseFlashMla稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[mixed_quant_sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla.md)|mixed_quant_sparse_flash_mla接口的前置接口，用于计算mixed_quant_sparse_flash_mla的负载均衡。|-|默认确定性实现|
|[quant_lightning_indexer_metadata](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)|quant_lightning_indexer接口的前置接口，用于计算quant_lightning_indexer的负载均衡。|默认确定性实现|默认确定性实现|
|[sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)|基于共享KV完成SparseFlashMla稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)|生成SparseFlashMla主算子使用的任务切分metadata。|默认支持确定性计算；默认支持batch invariance。|
|[inplace_partial_rotary_mul](../../torch_extension/cann_ops_transformer/docs/zh/inplace_partial_rotary_mul.md)|执行单路旋转位置编码的Inplace计算，直接修改输入张量，不产生新的输出张量。|默认确定性实现|默认确定性实现|
|[compressor](../../torch_extension/cann_ops_transformer/docs/zh/compressor.md)|将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。|默认支持确定性计算。|