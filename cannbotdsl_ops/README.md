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
- L0 executor 内同时包含 `CopyToNpu`/`CreateView` 常量准备和 Provider
  节点，因而验证了混合执行拓扑，而非仅把两类文件放在同一个安装目录。

## 依赖与 ABI

Provider 路径要求同步升级 CANNBotDSL、ops-transformer 和 opbase。opbase 必须包含
Provider kernel ABI v1，并在构建及运行时提供下列符号：

- `op::MakeProviderKernelSpec`
- `op::AllocProviderWorkspaceTensor`
- `op::SetProviderWorkspaceTensors`
- `op::AddProviderKernelLauncher`
- `op::RequireProviderAclnnRuntimeAbiV1`

仅将新 opbase 头文件传给 ops-transformer、但运行时仍加载旧 `libnnopbase.so`，会
导致动态链接失败。开发验证时应先从对应 opbase 分支构建并安装 `nnopbase`；正式
发布时应把该 opbase 变更纳入匹配的 CANN 基础包。`flash_kda_l0.cpp` 对 Provider
kernel/registry ABI v1 做了编译期断言；CMake 还会链接
`RequireProviderAclnnRuntimeAbiV1`，在配置阶段拒绝旧运行库。

当前方案不生成传统 OpDef、InferShape、TilingData 或 TilingKey。FlashKDA 是供
L2 组装的 ACLNN L0，输入输出契约和 workspace 规划由本目录的手写 L0 adapter
负责，Provider 变体选择由生成 registry 完成。

FlashKDA 的 `@entry` 公共参数严格对应 `l0op::FlashKda`：8 个输入 tensor、
`scale/lowerBound` 两个属性、2 个输出 tensor；`group/dv_base/block_num` 是
构建期 variant 选择值，只记录在 schema v2 的 `internal_parameters` 中，不进入 L0
公共签名。registry 生成阶段会把每个 variant 的 entry binding plan 与 Provider
sidecar/HostSchema 逐项校验；runtime binding、constexpr、workspace dtype/shape 或
六个 variant 间的公共 entry 身份任一漂移都会使混合包构建失败。

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
  -DCANNBOTDSL_PROVIDER_OPBASE_LIBRARY=/path/to/new/libnnopbase.so \
  -DOPBASE_SOURCE_PATH=/path/to/opbase \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/path/to/cann

cmake --build build-provider --target cust_opapi -j
cmake --install build-provider --prefix /path/to/stage
cmake --build build-provider --target package -j
```

需要编译设备 smoke 时，在配置命令中增加
`-DCANNBOTDSL_BUILD_FLASH_KDA_ACLNN_SMOKE=ON`，然后执行：

```bash
cmake --build build-provider --target cannbotdsl_flash_kda_aclnn_smoke -j
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke

# 同进程双 stream × 两轮，并验证 lower_bound 越界 fail-closed
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke --stress --negative

# 指定任一已发布有限 variant 的 shape 做设备执行/有限值检查
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke \
  --batch 2 --heads 96 --sequence 10240

# 导出一组 ACLNN 输入输出并执行 ACLNN/native/CPU golden 三方对拍
mkdir -p /tmp/flash-kda-dump
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke \
  --dump-dir /tmp/flash-kda-dump
PYTHONPATH=/path/to/cannbot-dsl/build/python_packages \
python cannbotdsl_ops/tests/verify_flash_kda_aclnn_dump.py \
  --dump-dir /tmp/flash-kda-dump \
  --cannbotdsl-source /path/to/cannbot-dsl --verify-native
```

未开启 `ENABLE_CANNBOTDSL_PROVIDER_ACLNN` 时，既有传统算子构建流程不变。

安装后的关键布局如下：

```text
packages/vendors/<vendor>_transformer/
├── op_api/
│   ├── include/aclnnop/aclnn_flash_kda.h
│   └── lib/libcust_opapi.so
└── op_impl/cannbotdsl/
    ├── registry.json
    └── packages/ops-transformer-flash-kda/0.1.0/
        ├── manifest.json
        └── artifacts/ascend950pr/<variant-id>/provider.so
```

### 混合 `.run` 离线验收

`package` 目标生成的同一个自解压包必须同时包含传统 AscendC 算子和 Provider
算子。建议在安装前使用只解包模式完成归档、内容和 ABI 三层检查：

```bash
run_file=/path/to/cann-ops-transformer-<vendor>_linux-<arch>.run
python cannbotdsl_ops/tests/verify_mixed_run.py --run "${run_file}"

# 下列命令用于按层排查验收失败
extract_dir=$(mktemp -d /tmp/ops-transformer-run.XXXXXX)

sha256sum "${run_file}"
"${run_file}" --check --noexec --extract="${extract_dir}"

provider_pkg="${extract_dir}/packages/vendors/<vendor>_transformer/op_impl/cannbotdsl/packages/ops-transformer-flash-kda/0.1.0"
(cd "${provider_pkg}" && sha256sum -c checksums.sha256)

nm -D -C "${extract_dir}/packages/vendors/<vendor>_transformer/op_api/lib/libcust_opapi.so" \
  | grep -E 'aclnn(ChunkKdaFwd|FlashKda)|l0op::FlashKda'
```

该 makeself 封装的单独 `--check` 会继续执行内嵌 `install.sh`；内嵌脚本又会忽略
未知的 `--check` 参数，因此它不是只读检查。必须同时传入 `--noexec` 和
`--extract`。动态库依赖应对解包后的 `libcust_opapi.so` 使用与目标 CANN、驱动和
新版 Provider ABI `libnnopbase.so` 匹配的 `LD_LIBRARY_PATH` 做 `ldd`/`dlopen`
检查。离线构建还必须预置项目声明的第三方依赖，否则 CPack preinstall 可能进入
依赖下载流程。`verify_mixed_run.py` 不安装归档，并会自动检查 registry/manifest
语义摘要、entry schema v2、12 个公共参数、3 个内部 specialization、artifact
digest、包内 checksum、传统 kernel 数量、Provider DSO 数量和混合导出符号。

## 实机验证

Ascend950PR 上已完成以下验证：

- `aclnnFlashKdaGetWorkspaceSize` 成功并返回 29,430,272 字节；
- `aclnnFlashKda` 成功入队并完成 stream 同步；
- 确定性有效输入下 output 与 final state 均为有限值；
- 同输入 ACLNN 与 native Provider 的 output/final state 逐字节一致；
- 相对 CPU chunk golden，output/state 最大绝对误差分别为
  `2.350081e-4` 和 `6.781593e-4`，通过 `atol=rtol=5e-3`；
- 同进程 2 streams × 2 waves 通过，无非有限值且第二轮结果与第一轮逐字节一致；
- 六个有限 variant 均已用最小合法 shape 完成 GetWorkspace、执行与有限值检查；
  覆盖 B1/N3/S8192、B1/N3/S32768、B4/N3/S8192、B1/N96/S1024、
  B2/N96/S10240 和 B4/N96/S1024；
- Provider launch 固定为 28 blocks，满足该目标 28 个 Cube core 的整网格
  barrier 单波驻留约束；
- `lower_bound=-5.01` 在 GetWorkspace 阶段返回 `161002`，调用前设置的非零
  workspace/executor 哨兵被主动清为 0/null；
- 已生成并只读解包实际混合 `.run`：归档内 SHA256 校验通过，Provider 包内全部
  文件 checksum 通过；包中同时存在 2 个传统 ChunkKda `.o`、6 个 FlashKDA
  `provider.so`，且 `libcust_opapi.so` 同时导出 ChunkKda ACLNN、FlashKDA ACLNN
  和 `l0op::FlashKda`。本次验证样本大小为 5,792,597 字节，整包 SHA256 为
  `f03be6c3132873f29dc5a4994b2795568d45d0bcb6f7f087cc5c11540deffded`；

## 当前 FlashKDA 契约

- 布局固定为 BNSD，`K/V/Q/G/O` 为 BF16；`A_log/dt_bias/initial_state/
  final_state` 为 FP32；
- head dimension 固定为 128，chunk size 固定为 64；
- `lower_bound` 必须为有限值且位于闭区间 `[-5, 0]`；
- 当前仅发布 package provider 声明中的 6 组有限 `(B, N, S)` 变体；
- 7 个内部临时张量由 executor workspace 分配并在 launch 前回填地址；
- Provider 物理 ABI 为 16 个 tensor inputs、2 个 tensor outputs、4 个 scalar
  attributes；顺序必须与生成 registry 完全一致；
- 当前完成 AOT、registry、host 编译/链接、混合符号、安装布局、B1/N3/S8192
  三方精度、双 stream/repeat 和六变体最小 shape 实机执行。其余 5 个变体的
  三方精度、全变体性能、更多错误注入和长稳压力仍是量产验收项。
