# FlashKDA

FlashKDA 接收 BF16 的 `q/k/v/g/beta`，返回 BF16 输出和 FP32 末态；支持
`BNSD`、`BSND`、`TND` 和 GQA，当前 head dimension 为 128，目标为 Ascend 950。
`TND` 的序列边界由 `flash_kda_metadata` 的 `cu_seqlens` 输入指定。

## 构建和安装 run 包

先加载 CANN 9.2 环境，并设置 `PYTHONPATH` 指向 **DSL run 安装**的
`cann/python/site-packages`。无需安装编译器或算子 wheel。
在 transformer 仓根目录执行标准 custom 构建命令：

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_kda,flash_kda_metadata --vendor_name=custom
bash build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path="$PWD/install-custom"
source "$PWD/install-custom/vendors/custom_transformer/op_impl/cannbotdsl/set_env.bash"
```

标准 `bash build.sh --pkg --soc=ascend950` 构建全仓 builtin 包，也包含所选芯片支持的
Native 算子。只验证这两个算子的 builtin 交付时，使用已有 CMake 参数选择算子：

```bash
cmake -S . -B build/builtin -DENABLE_BUILT_IN=ON -DENABLE_BUILD_PKG=ON \
  -DPACKAGE_TYPE=run -DASCEND_COMPUTE_UNIT=ascend950 \
  '-DASCEND_OP_NAME=flash_kda;flash_kda_metadata'
cmake --build build/builtin --target package -j16
bash build_out/cann-950-ops-transformer_9.1.0_linux-x86_64.run \
  --full --quiet --install-path="$PWD/install-builtin"
source "$PWD/install-builtin/cann/opp/built-in/op_impl/cannbotdsl/set_env.bash"
```

builtin 文件名中的组件版本来自仓库 `version.cmake`。custom 和 builtin 都直接安装
Python 入口、Native Provider 和 AICPU `.so`，无需安装时再编译或解包 wheel。
资源分别位于 vendor 或 builtin 的 `op_impl/cannbotdsl/python` 下。
custom 导入 `cann_ops_transformer_custom`；builtin 导入 `cann_ops_transformer_builtin`。
这两个入口通过 Torch dispatcher 调用现有算子，不新增 ACLNN API。

## Torch wheel（可选）

需要独立 Python 扩展包时，仍可在 `torch_extension` 目录执行：

```bash
TORCH_EXTENSION_OPS=flash_kda,flash_kda_metadata \
CANNBOTDSL_NPU_ARCH=dav-3510 python setup.py bdist_wheel
```

加载对应 run 的 `set_env.bash` 或安装可选 wheel 后：

```python
import cann_ops_transformer_custom as ops

metadata = ops.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens)
out, final_state = ops.flash_kda(
    q, k, v, g, beta, scale, initial_state,
    A_log, dt_bias, lower_bound, layout_qkv, metadata,
)
```

`initial_state`、`A_log` 和 `dt_bias` 使用 FP32；所有输入放在同一 NPU 上且连续。
`initial_state` 为 `[B, Nv, 128, 128]`，`A_log` 为 `[Nv]`，`dt_bias` 为
`[Nv, 128]`；`lower_bound` 取 `[-5, 0]`。非 TND 布局可传 `cu_seqlens=None`。
公开参数和计算逻辑沿用 `op_kernel/flash_kda.py`。

## 二进制接入

- `op_kernel/flash_kda.py` 保留计算实现，在文件末尾用框架示例中的
  `@register("flash_kda")` 收集原有动态编译入口，生成 Native Provider。
  不另设 shape 清单，不拆分或复制算子实现。
- `torch_extension` 负责注册 Torch schema、NPU 调用和 Meta 推导，同时注册
  包内 `_native` 资源。它不生成 C++ 占位文件，也不经过 ACLNN 包装。
- 框架 Native 加载器按编译契约匹配二进制。设置
  `CANNBOTDSL_NATIVE_BINARY_MODE=require` 可要求匹配安装的二进制，缺失时直接报错。
  Python 保留原有参数检查、工作区分配和调度入口；设备计算来自预编译 `.so`。
- `flash_kda_metadata` 的 AICPU 编译链独立于 Native Provider，详见相邻目录说明。

## 验证

只维护 `test/flash_kda_test.py`，其中同时验证 metadata 和 FlashKDA 的联动：

```bash
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 \
CANNBOTDSL_NPU_ARCH=dav-3510 \
python -m pytest attention/flash_kda/test/flash_kda_test.py -v -s
```

选择空闲的设备编号后，在仓根目录执行。测试使用 run 安装出的模块；builtin 测试需
增加 `KDA_PACKAGE=cann_ops_transformer_builtin`。测试禁用编译缓存并拦截 AIC/AIV 和 AICPU 编译调用，确认
二进制命中；精度通过独立 FP32 token 递推校验，保留原测试的
`abs(actual-reference)/(1+abs(reference))` 的 P99.9 不超过 `5e-3` 标准。
覆盖三种布局、非整块序列、2:1/3:1 GQA、多 batch、跨轮工作区复用和 Meta。
