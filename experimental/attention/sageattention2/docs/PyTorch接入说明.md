# PyTorch 接入说明

本算子以独立 Python 包 `sageattention2-npu` 提供 `sageattention2` 模块，默认调用链为：

```text
sageattn / sageattn_qk_int8_pv_fp8_asc
  → torch.ops.sageattention2.forward
  → 本算子 C++ 扩展
  → torch_npu EXEC_NPU_CMD_EXT
  → aclnnSageAttention2GetWorkspaceSize / aclnnSageAttention2
```

C++ 扩展只编译本包 `python/sageattention2/csrc/attention.cpp`，通过 torch_npu 随附的 `op_plugin/utils/op_api_common_base.h` 复用 Tensor 转换、workspace 管理、任务队列和 ACLNN 提交机制。C++ 侧在输出分配前设置 DeviceGuard，并检查形状、dtype、布局和 causal 约束。输入须为基础 ND/NCHW 存储；950 当前运行时不支持四维 FRACTAL_NZ 转换，接口对私有格式明确报错。

## 安装与使用

在算子目录执行，使用匹配的 PyTorch / torch_npu，已验证版本见 [README](../README.md)。

```bash
source /usr/local/Ascend/cann/set_env.sh
export CATLASS_ROOT=/path/to/catlass
bash install_opp.sh
source build/opp/vendors/custom_transformer/bin/set_env.bash
python -m pip install --no-deps .
```

首次 NPU 调用使用 `torch.utils.cpp_extension.load` 编译 bridge，需要 C++17 编译器、Ninja，以及 CANN / torch_npu 头文件和动态库。编译产物放在 PyTorch 扩展缓存；可用 `TORCH_EXTENSIONS_DIR` 指定位置。`SAGEATTN_BUILD_VERBOSE=1` 可输出编译日志。kernel 仍由 `install_opp.sh` 完整构建安装，bridge 编译不替代 OPP 构建。

```python
from sageattention2 import sageattn

out, lse = sageattn(q, k, v, return_lse=True)
```

原有 `sageattn_qk_int8_pv_fp8_asc` 和 `sageattention2.aclnn.run` 入口继续可用，均调用相同的已注册算子。`pv_accum_dtype` 默认 `fp32+fp32`，`fp32` 为兼容别名；量化、平滑和参数回退语义见 [CUDA 接口差异说明](CUDA接口差异说明.md)。

## Dispatcher 与图模式

底层 `torch.ops.sageattention2.forward` 返回固定结构 `(out, optional_lse)`；Python 公开接口在 `return_lse=False` 时只返回 out。注册了 PrivateUse1 实现和 Meta 实现，Meta/FakeTensor 推导不执行 NPU kernel，也不触发 C++ 编译。

`torch.compile(..., backend="eager", fullgraph=True)` 测试用于验证 Dynamo 捕获该算子的能力；不代表已经提供 torchair/GE Converter 或验证所有编译后端。当前为前向算子，不支持 autograd。

## 性能计时

使用 `benchmarks/profile_performance.py` / `run_validation_performance.py` 从公开 PyTorch 接口比较 Sage 和 FIA。先预热并同步以排除 JIT/初始化，再用一对 NPU Event 包住批量调用，结果为总耗时除以调用次数；统计包含完整 Stats、Quant、Attention，不缓存最终输出或预处理结果。

迁移主要改变 host 提交路径。kernel 算法和 AIV 指令没有因接入迁移而改变，不能将 Event 耗时变化直接解释为 AIV time 下降。
