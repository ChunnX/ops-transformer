# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  `torch_extension`接口定义于`cann_ops_transformer`包中。更多接口说明及使用示例正在完善中，敬请期待。如有疑问或建议，欢迎通过Issue反馈。

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：因CANN或NPU型号不同等原因，可能无法保证同一个API运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分接口可通过PyTorch中控制算法确定性的全局开关[torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py)开启确定性算法，使多次运行结果一致。

|    接口名   |   说明     | 确定性说明 |
| ----------- | ------------------- | ------------------- |
|[flash_attn](../../torch_extension/npu_ops_transformer/doc/npu_flash_attn.md)|完成xx计算。|xx|
