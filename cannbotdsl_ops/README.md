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
- L0 executor 内同时包含生成的 workspace 分配、`CopyToNpu`/`CreateView`
  常量与 view 准备以及 Provider 节点，因而验证了混合执行拓扑，而非仅把
  两类文件放在同一个安装目录。

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
L2 组装的 ACLNN L0；本目录只手写公共契约校验以及 `group/dv_base/block_num`
调度策略，`@entry` 的 workspace/constant/view、checked shape 表达式、物理
Provider 参数结构、`OP_INPUT/OP_OUTPUT/OP_ATTR` 顺序和 workspace 登记均由
schema v3 adapter 生成，Provider 变体选择由生成 registry 完成。

FlashKDA 的 `@entry` 公共参数严格对应 `l0op::FlashKda`：8 个输入 tensor、
`scale/lowerBound` 两个属性、2 个输出 tensor；`group/dv_base/block_num` 是
构建期 variant 选择值，只记录在 schema v3 的 `internal_parameters` 中，不进入 L0
公共签名。registry 生成阶段会把每个 variant 的 entry binding plan 与 Provider
sidecar/HostSchema 逐项校验；runtime binding、constexpr、workspace dtype/shape 或
六个 variant 间的公共 entry 身份任一漂移都会使混合包构建失败。
schema v3 还显式声明 `output_parameters=["o", "st_out"]`；当前生成器已据此生成
并接入 `FlashKdaEntryArguments`、`PrepareFlashKdaProviderArguments`、
`FlashKdaProviderArguments`、`MakeFlashKdaProviderOpArgContext` 和
`SetFlashKdaProviderWorkspaceTensors`。因此七段 scratch、identity 常量、view、
checked runtime shape 求值、物理参数顺序和底层 workspace 登记均无需手写。

六个交付域及其 `group/dv_base/block_num` 由
`attention/flash_kda/flash_kda_variant_policy.py` 单点声明。Python AOT variant
直接消费该表；构建在生成 `libcust_opapi.so` 前会穷举每个 64-token 交付点，拒绝
覆盖空洞/重叠，并逐 artifact 校验真实 sidecar 的 B/N/S 约束与 constexpr，随后
生成同源 C++ L0/smoke matcher。任何 manifest、specialization 或 L0 支持域漂移
都会在编译阶段 fail-closed。

## 构建

先构建 CANNBotDSL Python runtime，然后在 ops-transformer 根目录配置：

```bash
cmake -S . -B build-provider \
  -DBUILD_OPEN_PROJECT=ON \
  -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  -DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON \
  -DENABLE_BUILD_PKG=ON \
  -DCANN_3RD_LIB_PATH=/path/to/preinstalled/third_party \
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

离线打包时，`CANN_3RD_LIB_PATH` 下可预置 `makeself/makeself.sh`；CPack 会优先
复用该目录，不要求在 ops-transformer 源码树内另建 `third_party/makeself`。

需要编译设备 smoke 时，在配置命令中增加
`-DCANNBOTDSL_BUILD_FLASH_KDA_ACLNN_SMOKE=ON`，然后执行：

```bash
cmake --build build-provider --target cannbotdsl_flash_kda_aclnn_smoke -j

# Provider 路径按最终 op_api/lib -> ../../op_impl/cannbotdsl 布局解析 DSO。
# 先生成安装树，并确保运行时优先加载安装树中的 libcust_opapi.so。
stage=/tmp/ops-transformer-provider-stage
cmake --install build-provider --prefix "${stage}"
vendor_dir=custom_transformer
export LD_LIBRARY_PATH="${stage}/packages/vendors/${vendor_dir}/op_api/lib:${LD_LIBRARY_PATH}"
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke

# 同进程双 stream × 20 轮，执行错误矩阵、确定性长稳和 launch+sync 统计
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke \
  --stress --negative --waves 20

# 指定任一已发布有限 variant 的 shape 做设备执行/有限值检查
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke \
  --batch 2 --heads 96 --sequence 10240

# 导出任一已发布 shape 的完整 ACLNN 输入输出；metadata.txt 自描述 B/N/S 和标量
mkdir -p /tmp/flash-kda-dump
./build-provider/cannbotdsl_ops/cannbotdsl_flash_kda_aclnn_smoke \
  --batch 2 --heads 96 --sequence 10240 \
  --dump-dir /tmp/flash-kda-dump

# 完整输出先做流式 finite 检查，再对指定 batch/head 切片执行
# ACLNN/native/CPU golden 三方对拍。--golden-sample 可重复，CPU 峰值内存为 O(S*D)。
PYTHONPATH=/path/to/cannbot-dsl/build/python_packages \
python cannbotdsl_ops/tests/verify_flash_kda_aclnn_dump.py \
  --dump-dir /tmp/flash-kda-dump \
  --cannbotdsl-source /path/to/cannbot-dsl \
  --golden-sample 0:0 --golden-sample 1:95 --verify-native

# 发布验收可不启用 native，顺序遍历所有 batch/head，完成全输出逐元素 CPU golden：
PYTHONPATH=/path/to/cannbot-dsl/build/python_packages \
python cannbotdsl_ops/tests/verify_flash_kda_aclnn_dump.py \
  --dump-dir /tmp/flash-kda-dump \
  --cannbotdsl-source /path/to/cannbot-dsl \
  --all-golden-samples --summary-only
```

不要把构建根目录中的 `libcust_opapi.so` 放在上述安装树之前；其相对 Provider
根目录不满足交付布局，会使 registry 在 `dlopen` 阶段 fail-closed。运行时还必须
优先加载与构建头文件匹配、含 Provider ABI v1 的 `libnnopbase.so`。

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

# 发布配置需要锁定传统 kernel 变体数时，可增加精确数量门禁
python cannbotdsl_ops/tests/verify_mixed_run.py --run "${run_file}" \
  --expected-traditional-kernel-count 8

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
语义摘要、entry schema v3、12 个公共参数、2 个输出参数、3 个内部 specialization、artifact
digest、包内 checksum、传统 kernel 数量下限、Provider DSO 数量和混合导出符号；
发布配置还可用命令行参数锁定传统 kernel 的精确数量。

## 实机验证

Ascend950PR 上已完成以下验证：

- `aclnnFlashKdaGetWorkspaceSize` 成功并返回 29,430,272 字节；
- `aclnnFlashKda` 成功入队并完成 stream 同步；
- 确定性有效输入下 output 与 final state 均为有限值；
- 六个有限 variant 的首尾代表性 `(B,H)` 切片中，同输入 ACLNN 与 native
  Provider 的 output/final state 均逐字节一致；
- 六个 variant 的最小合法 shape 均已按 `(B,H)` 顺序遍历，完成全部 output
  `S×D` 和 final state `128×128` 元素的 CPU chunk golden 校验；累计验证
  690 个切片，矩阵最坏 output/state 最大绝对误差分别为 `2.527554e-4` 和
  `1.048915e-3`，通过 `atol=rtol=5e-3`；
- 同进程 2 streams × 20 waves 通过，40 次完整调用均无非有限值，且每个 stream
  的后续结果均与首轮逐字节一致；双流 launch+sync 为 `0.698–1.067 ms`，平均
  `0.744 ms`；
- 六个有限 variant 均已用最小合法 shape 完成 GetWorkspace、执行、完整 CPU
  golden 与代表性 native 对拍；
  覆盖 B1/N3/S8192、B1/N3/S32768、B4/N3/S8192、B1/N96/S1024、
  B2/N96/S10240 和 B4/N96/S1024；
- Provider launch 固定为 28 blocks，满足该目标 28 个 Cube core 的整网格
  barrier 单波驻留约束；
- GetWorkspace 错误矩阵覆盖 lower_bound 上下越界/NaN、scale NaN/float 溢出、
  null K、错误 K 契约、未发布 variant、null workspaceSize/executor 输出指针；
  10 类均返回 `161002`，非空输出哨兵被主动清为 0/null，随后合法调用仍成功；
- 已生成并只读解包实际混合 `.run`：归档内 SHA256 校验通过，Provider 包内全部
  文件 checksum 通过；包中同时存在 8 个传统 ChunkKda `.o`、6 个 FlashKDA
  `provider.so`，且 `libcust_opapi.so` 同时导出 ChunkKda ACLNN、FlashKDA ACLNN
  和 `l0op::FlashKda`。本次官方 CPack 完整构建样本大小为 11,650,113 字节，
  整包 SHA256 为
  `eafe8f89e89f5ee2c05e87dce8c60f427615fb52d77ffdc6f8fc2aaec6ebd566`；

单流基础时延使用相同 smoke，在每轮重新构造 executor 后，仅测量
`aclnnFlashKda` 入队至 stream 同步完成的区间；不含输入构造、GetWorkspace、
workspace 分配和 D2H finite/确定性检查。它是当前 Ascend950PR 工程基线，不是
尚未定义 SLA 的产品性能准入值：

| 最小合法 shape | waves | min / avg / max (ms) |
| --- | ---: | ---: |
| B1/N3/S8192 | 5 | 0.573 / 0.617 / 0.651 |
| B1/N3/S32768 | 3 | 1.123 / 1.167 / 1.223 |
| B4/N3/S8192 | 3 | 0.863 / 1.591 / 3.026 |
| B1/N96/S1024 | 3 | 0.815 / 0.929 / 1.089 |
| B2/N96/S10240 | 3 | 9.436 / 9.464 / 9.516 |
| B4/N96/S1024 | 3 | 2.579 / 2.637 / 2.729 |

## 当前 FlashKDA 契约

- 布局固定为 BNSD，`K/V/Q/G/O` 为 BF16；`A_log/dt_bias/initial_state/
  final_state` 为 FP32；
- head dimension 固定为 128，chunk size 固定为 64；
- `lower_bound` 必须为有限值且位于闭区间 `[-5, 0]`；
- 当前仅发布单一策略表声明并经 manifest/sidecar 构建门禁证明唯一匹配的
  6 组有限 `(B, N, S)` 变体；
- 7 个内部临时张量由 executor workspace 分配并在 launch 前回填地址；
- Provider 物理 ABI 为 16 个 tensor inputs、2 个 tensor outputs、4 个 scalar
  attributes；顺序必须与生成 registry 完全一致；
- 当前完成 AOT、registry、host 编译/链接、混合符号、安装布局、六变体最小
  shape 的完整 CPU golden/代表性 native 对拍、六 shape 基础时延、10 类错误
  注入，以及双 stream × 20 waves。产品目标输入域确认与扩展、SLA/基线对比、
  更长 soak 和目标发布包安装验收仍是量产项。
