# mla_prolog_v5 精简设计方案（快速迁移）

> 快速迁移模式：本文档仅记录迁移映射与关键工程决策，不做完整白盒设计。

## 1. 工程结构（目标）

```
operators/mla_prolog_v5/
├── CMakeLists.txt              # 独立算子工程根（npu_op_package，ascendc-registry-invoke-template 模式）
├── build.sh                    # 构建入口（--soc=ascend910b）
├── README.md                   # 从 v3 README 迁移（A2 only）
├── docs/
│   ├── REQUIREMENTS.md / DESIGN.md / LOG.md / aclnnMlaPrologV5WeightNz.md
├── op_kernel/
│   ├── mla_prolog_v5.cpp       # v3 kernel 改名；去掉 __CCE_AICORE__==310(arch35) 分支
│   ├── mla_prolog_tiling_data.h        # vendored from attention/mla_prolog/op_kernel/
│   ├── mla_prolog_template_tiling_key.h# vendored from attention/mla_prolog/op_kernel/
│   └── arch22/                 # vendored from attention/mla_prolog/op_kernel/arch22/（12 个头文件）
├── op_host/
│   ├── CMakeLists.txt
│   ├── mla_prolog_v5_def.cpp           # v3 def 改名 MlaPrologV5；删 950 config，保留 910b/910_93
│   ├── mla_prolog_v5_infershape.{h,cpp}# v3 infershape 改名（依赖 vendored v1 infershape）
│   ├── mla_prolog_infershape.{h,cpp}   # vendored v1（路径本地化）
│   ├── mla_prolog_v5_tiling.h          # v3 tiling.h 改名
│   ├── mla_prolog_tiling.{h,cpp}       # vendored v1（路径本地化）
│   ├── mla_prolog_tiling_check.{h,cpp} # vendored v1
│   └── mla_prolog_v5_tiling_register.cpp # IMPL_OP_OPTILING(MlaPrologV5)
├── op_api/
│   ├── aclnn_mla_prolog_v5_weight_nz.{h,cpp}  # 外层：v4 cpp 改名（含 doRope 参数 + 校验）
│   └── mla_prolog_v5.cpp               # 手写 L0/Inner：OP_TYPE_REGISTER + ADD_TO_LAUNCHER_LIST_AICORE
├── torch_extension/            # pybind
│   ├── csrc/mla_prolog_v5.cpp  # v3 torch_extension/csrc/mla_prolog.cpp 改调 aclnnMlaPrologV5WeightNz
│   ├── CMakeLists.txt          # 模板 torch_adapter 模式（find Torch/torch_npu/cust_opapi）
│   └── mla_prolog_v5.py        # Python 接口（独立，不依赖仓级 cann_ops_transformer 包）
├── tests/
│   └── pybind/test.py          # Python 接口测试（torch_npu + CPU golden 比对）
├── examples/                   # aclnn 直调 C++ 示例（从 v3 examples 迁移一个 bf16 用例）
└── issues/
```

## 2. 关键工程决策

| # | 决策 | 理由 |
|---|------|------|
| D1 | aclnn 外层取 **V4 语义**（doRope 参数），算子注册名 `MlaPrologV5` | v3 目录内现成最全外层；torch_extension 现成代码即调 V4；def.cpp 本就含 do_rope attr，零逻辑改动 |
| D2 | **Inner/L0 层手写**（`op_api/mla_prolog_v5.cpp`） | 独立工程无 ops-transformer 仓的 aclnnInner 生成框架；参照仓内先例（nsa_compress_with_cache 等：OP_TYPE_REGISTER + ADD_TO_LAUNCHER_LIST_AICORE，输出由外部传入，无需 InferShape/AllocTensor） |
| D3 | 构建系统用 **npu_op_package**（ascendc-registry-invoke-template 模式），仅 ascend910b | CANN 9.1 自定义算子包标准路径；A2-only 要求 |
| D4 | v1 依赖 **vendor 进本工程**（arch22 头、tiling/infershape），include 路径本地化 | 自包含，不依赖源仓其他目录；源仓零改动 |
| D5 | kernel 编译选项沿用源仓：`--cce-auto-sync=off -Wno-deprecated-declarations -mllvm -cce-aicore-hoist-movemask=false` | 保证行为一致 |
| D6 | pybind 用 **CMake 直编 .so**（链接 cust_opapi + torch_npu），Python 侧独立薄封装 | 用户要求"独立算子工程 + python 接口测试"；不引入仓级 JIT op_builder |
| D7 | `MlaPrologTilingData` 等 v1 共享结构体名**不改**（仅 tiling 注册绑定到 MlaPrologV5） | tiling 数据结构由 hash 注册、跨 op 复用安全；改名侵入面大、违背"只迁移"原则 |

## 3. 重命名规则（机械替换）

- `MlaPrologV3` → `MlaPrologV5`（类名、OP_ADD、IMPL_OP_OPTILING、日志 op 名）
- `mla_prolog_v3` / `aclnn_mla_prolog_v3_weight_nz` → `mla_prolog_v5` / `aclnn_mla_prolog_v5_weight_nz`（文件名、include、kernel 入口）
- `aclnnMlaPrologV3WeightNz` / `aclnnMlaPrologV4WeightNz` → `aclnnMlaPrologV5WeightNz`（API 函数名，V4 文件为外层蓝本）
- `aclnnInnerMlaPrologV3*` → 由手写 L0 替代（`l0op::MlaPrologV5` + 两段式）
- vendored v1 文件内部符号（TilingMlaProlog、MlaPrologTilingData 等）保持不变

## 4. A2 裁剪清单

1. `op_kernel/mla_prolog_v5.cpp`：删除两段 `#if __CCE_AICORE__ == 310` 块（arch35 include 与 mxfp8/fp8/hif8 kernel 分支——这些分支仅 arch35 可达）；`#if __CCE_AICORE__ == 310` 的 GetCtrlSpr 块同步删除（`__NPU_ARCH__ == 3510` 仅 950）。
2. `op_host/mla_prolog_v5_def.cpp`：删除 `aicore_config_95` 整段与 `AddConfig("ascend950")`；保留 ascend910b / ascend910_93（同为 A2 系 arch22）。
3. `op_api` 外层校验中 `NpuArch::DAV_3510` 分支保留原样（运行时走 else 分支，逻辑不变）。

## 5. 编译与运行链路

```
bash build.sh --soc=ascend910b
  → cmake(npu_op_package) → cust_optiling.so / cust_opapi.so / ascendc kernel binary(.o)
  → custom_opp_ubuntu_aarch64.run
  → 安装 .run（opp/vendors/mla_prolog_v5_custom）
torch_extension: cmake → libmla_prolog_v5.so（rpath 指向 vendors op_api/lib + CANN lib64）
Python: import mla_prolog_v5 → 调 .so 的 mla_prolog()（内部 aclnnMlaPrologV5WeightNz 两段式）
```

## 6. 测试方案（基本功能验证）

- `tests/pybind/test.py`：
  - 用例组 1（L0 必测）：非量化 bf16，BS 合轴 (T,He) 与非合轴 (B,S,He) × cache_mode ∈ {PA_BSND, TND}，与 CPU golden 比对（query / query_rope / kv_cache / kr_cache / query_norm）。
  - 用例组 2（加测）：部分量化（weight_quant_mode=1, kv perchannel）int8 场景。
  - 比对标准：bf16 用相对误差（社区标准放宽阈值），量化输出按 int8 语义比对。
- CPU golden：torch CPU 实现 RmsNorm/matmul/ROPE/scatter（与 v3 examples 的数据构造对齐）。
- examples/：迁移一个 aclnn 直调 C++ 示例作编译级验证（可选运行）。

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| npu_op_code_gen 对复杂 def（optional 输入/输出同名 kv_cache）解析问题 | 编译阶段验证；必要时回退为模板同款简化 def 描述（保持 dtype/attr 不变） |
| OP_INPUT 宏对 nullptr（optional）tensor 的支持 | 参照仓内先例（nsa_compress_with_cache 等）传 nullptr 的写法；必要时空 tensor 用 shape{0} 占位 |
| vendored v1 tiling 对 arch22 特有路径依赖（如 GetPlatformInfo） | v1 tiling.cpp 本就服务 910b/950 双路径，保留原样编译验证 |
| torch_npu 头文件 API（ACLNN_CMD/TensorWrapper）编译环境差异 | torch_extension csrc 已在源仓验证过该写法，CMake include 路径对齐 torch_npu/include |
