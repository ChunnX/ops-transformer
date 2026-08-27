# CANNBotDSL Provider 混合算子包

本目录把 CANNBotDSL 原生 Provider 算子并入 ops-transformer 的自定义 OPP。开启
`ENABLE_CANNBOTDSL_PROVIDER_ACLNN` 后，传统 AscendC 算子和 Provider 算子共用
`libcust_opapi.so` 与 vendor 安装目录；这表示同一个包中可混合两类 L0，而不是为
同一个算子同时生成两种实现。

当前首个接入算子是 FlashKDA：

- Python 源码及有限变体声明保留在
  `attention/flash_kda/flash_kda.py` 和 `package.toml`；
- 构建阶段调用 CANNBotDSL AOT，生成 6 个 `ascend950pr` Provider DSO；
- 生成的 Provider registry 与 `flash_kda_l0.cpp` 一同编入
  `libcust_opapi.so`；
- `l0op::FlashKda` 可被其他 L2 ACLNN 实现直接组装；
- `aclnnFlashKdaGetWorkspaceSize` / `aclnnFlashKda` 提供独立的两阶段 ACLNN
  调用入口；
- L0 executor 内同时包含传统 `Contiguous`/`Reshape` 节点和 Provider 节点，
  因而验证了混合执行拓扑，而非仅把两类文件放在同一个安装目录。

## 依赖与 ABI

Provider 路径要求同步升级 CANNBotDSL、ops-transformer 和 opbase。opbase 必须包含
Provider kernel ABI v1，并在构建及运行时提供下列符号：

- `op::MakeProviderKernelSpec`
- `op::AllocProviderWorkspaceTensor`
- `op::SetProviderWorkspaceTensors`
- `op::AddProviderKernelLauncher`

仅将新 opbase 头文件传给 ops-transformer、但运行时仍加载旧 `libnnopbase.so`，会
导致动态链接失败。开发验证时应先从对应 opbase 分支构建并安装 `nnopbase`；正式
发布时应把该 opbase 变更纳入匹配的 CANN 基础包。`flash_kda_l0.cpp` 对 Provider
kernel/registry ABI v1 做了编译期断言，但共享库符号是否存在仍由匹配的 opbase
运行库保证。

当前方案不生成传统 OpDef、InferShape、TilingData 或 TilingKey。FlashKDA 是供
L2 组装的 ACLNN L0，输入输出契约和 workspace 规划由本目录的手写 L0 adapter
负责，Provider 变体选择由生成 registry 完成。

## 构建

先构建 CANNBotDSL Python runtime，然后在 ops-transformer 根目录配置：

```bash
cmake -S . -B build-provider \
  -DBUILD_OPEN_PROJECT=ON \
  -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  -DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON \
  -DCANNBOTDSL_SOURCE_DIR=/path/to/cannbot-dsl \
  -DCANNBOTDSL_RUNTIME_ROOT=/path/to/cannbot-dsl/build/python_packages \
  -DCANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR=/path/to/opbase \
  -DOPBASE_SOURCE_PATH=/path/to/opbase \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/path/to/cann

cmake --build build-provider --target cust_opapi -j
cmake --install build-provider --prefix /path/to/stage
```

未开启 `ENABLE_CANNBOTDSL_PROVIDER_ACLNN` 时，既有传统算子构建流程不变。

安装后的关键布局如下：

```text
packages/vendors/<vendor>_transformer/
├── op_api/
│   ├── include/aclnnop/aclnn_flash_kda.h
│   └── lib/libcust_opapi.so
└── op_impl/cannbotdsl/
    ├── index.json
    └── packages/ops-transformer-flash-kda/0.1.0/
        ├── manifest.json
        └── artifacts/ascend950pr/<variant-id>/provider.so
```

## 当前 FlashKDA 契约

- 布局固定为 BNSD，`K/V/Q/G/O` 为 BF16；`A_log/dt_bias/initial_state/
  final_state` 为 FP32；
- head dimension 固定为 128，chunk size 固定为 64；
- 当前仅发布 package provider 声明中的 6 组有限 `(B, N, S)` 变体；
- 7 个内部临时张量由 executor workspace 分配并在 launch 前回填地址；
- Provider 物理 ABI 为 16 个 tensor inputs、2 个 tensor outputs、4 个 scalar
  attributes；顺序必须与生成 registry 完全一致；
- 当前完成 AOT、registry、host 编译/链接、混合符号及安装布局验证；真实 NPU 数值
  与性能回归仍需在 Ascend 950 设备环境执行。
