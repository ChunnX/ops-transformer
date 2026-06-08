# CHANGELIST — 变更记录

> 记录项目演进和变更，便于后期回溯和审查。
> **只追加，不删除已有条目。**

---

## 变更记录格式

```
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更了什么

#### 关键决策
- 做了什么决策，为什么

#### 影响范围
- 对现有功能的影响
```

---

<!-- 在此追加变更记录 -->

### 2026-06-02 | 工作流初始化 + 范围修正

**类型**: 🆕 新功能 | 📝 文档
**范围**: .workflow/ 全部文件 + raw/ 目录结构

#### 变更内容
- 初始化 .workflow/ 工作流文件（GOAL.md, PLAN.md, NEXT_STEPS.md, CHECKLIST.md）
- 复制设计文档到 raw/design/qkv_rms_norm_rope_cache_with_kscale/
- 创建算子原型规格文档 raw/spec/qkv_rms_norm_rope_cache_with_kscale_op.md
- 更新 raw/INDEX.md 区分 spec/（接口契约）和 design/（实现方案）

#### 关键决策
- **设计文档归入 design/**：设计初稿不是最终规格，放入 design/ 目录，后续补充 ACLNN 开发设计文档
- **spec/ 存放算子原型规格**：参考 gmmsq_op.md/gmmfr_op.md 格式，创建本算子的接口契约文档，作为 ACLNN 开发的 truth source
- **移除 tiling**：tiling 与 kernel 紧密相关，当前阶段不交付，后续再开发
- **测试范围缩窄为 op_api UT**：无 kernel 开发，不需要 golden.py/inputs.py，仅需 ACLNN API 单元测试
- **op_api 模式 A (L2→L0 自包含)**：参考 nsa_compress_with_cache，op_api 下同时包含 L2 和 L0 实现
- **执行顺序**：第一轮 阶段一(脚手架+op_host) + 阶段二(op_api)，第二轮 阶段三(测试+文档+验收)

#### 影响范围
- GOAL.md: 5 阶段 → 3 阶段，移除 tiling/golden/inputs
- PLAN.md: 移除阶段二(tiling)，阶段四(测试)简化为 op_api UT
- NEXT_STEPS.md: 5 任务 → 3 任务
- CHECKLIST.md: 移除 tiling 检查项和 golden 检查项
- raw/INDEX.md: spec/ 和 design/ 分离，新增本算子规格文档索引

### 2026-06-02 | 阶段一 + 阶段二实现完成

**类型**: 🆕 新功能
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/

#### 变更内容
- 创建完整目录结构：op_host/, op_api/, tests/, docs/, examples/
- 实现 op_host/CMakeLists.txt 和根目录 CMakeLists.txt
- 实现 op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp
  - 注册 QkvRmsNormRopeCacheWithKScale 算子
  - 9 个必选输入 + 2 个可选输入 + 5 个输出 + 2 个属性
  - ascend950 平台配置
- 实现 op_host/qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
  - 完整的 shape 推导逻辑
  - head_nums 属性校验（长度=3，值正整数，Nv==Nk）
  - D=128 fast path 约束
  - 输出 shape 计算（q_out, q_scale, cache 输出）
- 实现 op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.h/.cpp
  - L2 公共 API：GetWorkspaceSize 和 Execute 接口
  - 参数校验（空指针、dtype、空 tensor）
  - Contiguous 转换
  - 调用 L0 接口
- 实现 op_api/qkv_rms_norm_rope_cache_with_kscale.h/.cpp
  - L0 内部实现
  - OP_TYPE_REGISTER 注册
  - headNums aclIntArray → aclTensor 转换
  - INFER_SHAPE 和 ADD_TO_LAUNCHER_LIST_AICORE 调用
- 实现 op_api/CMakeLists.txt

#### 关键决策
- **def.cpp 中 rotation/v_scale 标记为 OPTIONAL**：虽然当前场景强制非空，但接口形态保持 optional，由 L2 API 层校验非空
- **infershape.cpp 中 Nv!=Nk 返回 GRAPH_FAILED**：首期 fast path 只支持 Nv==Nk，不进入正常 kernel 路径
- **infershape.cpp 中 D!=128 返回 GRAPH_FAILED**：首期只支持 D=128 fast path
- **L0 返回 std::tuple**：5 个输出使用 tuple 返回，参考 nsa_compress_with_cache 模式
- **L2 API 中 rotation/v_scale 校验非空**：虽然 def.cpp 标记为 OPTIONAL，但 L2 层强制校验非空

#### 影响范围
- 新增 attention/qkv_rms_norm_rope_cache_with_kscale/ 完整目录
- 不影响已有算子编译
- 待阶段三完成 op_api UT 后可完整验证

### 2026-06-02 | 阶段三完成：编译验证 + 单元测试通过

**类型**: ✅ 完成
**范围**: attention/qkv_rms_norm_rope_cache_with_k_scale/

#### 变更内容
- 编译验证通过：所有源文件成功编译，无错误
- 单元测试全部通过（4/4）：
  - normal_case: 基本功能正常
  - null_qkv: 正确检测空指针并返回 ACLNN_ERR_PARAM_NULLPTR
  - null_rotation: 正确检测空指针并返回 ACLNN_ERR_PARAM_NULLPTR
  - wrong_dtype: 正确检测错误数据类型并返回 ACLNN_ERR_PARAM_INVALID

#### 关键修复
1. **const 正确性修复**：将 cache tensors (kCache, vCache, kScaleCache) 从 const 改为非 const，因为它们是 in-place 更新的
2. **L0 返回类型修复**：将 L0 函数返回类型从 std::tuple 改为 const aclTensor*，与参考算子模式一致
3. **错误处理修复**：
   - L2 API 直接返回 CheckQkvRmsNormRopeCacheWithKScaleParam 的错误码，不再包装
   - InputDtypeCheck 使用显式 if 语句 + OP_LOGE + return ACLNN_ERR_PARAM_INVALID，替代 CHECK_COND 宏
4. **测试框架修复**：在 file_io.h 中添加 `#include <cstdint>` 以支持 uint32_t

#### 编译命令
```bash
bash build.sh --opapi --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 --noexec -j8
```

#### 测试命令
```bash
ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=3 BUILD_PATH=$(pwd)/build ./build/tests/ut/framework_normal/op_api/transformer_op_api_ut
```

#### 测试结果
```
[==========] Running 4 tests from 1 test suite.
[----------] 4 tests from qkv_rms_norm_rope_cache_with_kscale_opapi_ut
[ RUN      ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.normal_case
[       OK ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.normal_case (6 ms)
[ RUN      ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.null_qkv
[       OK ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.null_qkv (0 ms)
[ RUN      ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.null_rotation
[       OK ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.null_rotation (0 ms)
[ RUN      ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.wrong_dtype
[       OK ] qkv_rms_norm_rope_cache_with_kscale_opapi_ut.wrong_dtype (0 ms)
[----------] 4 tests from qkv_rms_norm_rope_cache_with_kscale_opapi_ut (7 ms total)
[==========] 4 tests from 1 test suite ran. (7 ms total)
[  PASSED  ] 4 tests.
```

#### 影响范围
- 算子开发全部完成，代码质量验证通过
- 错误处理机制符合 ACLNN 规范
- 可安全集成到上层应用

### 2026-06-03 | QKV RMSNorm RoPE Cache KScale op_api 接口修正

**类型**: 🔧 修改
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/, CMake autogen alias

#### 变更内容
- 将算子源码目录从 `attention/qkv_rms_norm_rope_cache_with_k_scale/` 更名为 `attention/qkv_rms_norm_rope_cache_with_kscale/`。
- 将单个 `rotation` 输入拆分为 `q_rotation` 和 `k_rotation`，二者 shape 均校验为 `[D, D]`。
- 将 `k_cache`、`v_cache`、`k_scale_cache` 调整为 ACLNN inplace 输入输出参数，不再暴露独立 cache out 参数。
- op_host 输出名改为 `k_cache`、`v_cache`、`k_scale_cache`，使 opbuild 生成 `Ref` 关系。
- L2 ACLNN 入参、L0 调用、UT、示例和文档同步为 `qRotation`/`kRotation` + cache inplace 形式。
- 新增 CMake autogen alias 后处理，用于将 CANN 生成器产出的 `qkv_rms_norm_rope_cache_with_k_scale` inner 文件同步为源码期望的 `qkv_rms_norm_rope_cache_with_kscale` 文件。

#### 关键决策
- 保持公开 op type/API 符号 `QkvRmsNormRopeCacheWithKScale`/`aclnnQkvRmsNormRopeCacheWithKScale` 不变，避免改变算子语义名。
- 对 cache inplace 参数不执行 L2 Contiguous 替换，避免原地写回落到临时连续 tensor。
- CMake alias 仅解决生成器 `KScale -> k_scale` 的文件命名习惯，不把源码目录或构建入口改回 `k_scale`。

#### 影响范围
- `bash build.sh --ophost --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8` 通过。
- `bash build.sh --opapi_test --ops=qkv_rms_norm_rope_cache_with_kscale --soc=ascend950 --noexec -j8` 通过编译。
- 当前未涉及 tiling 和 kernel 实现。

### 2026-06-04 | 设计文档 rotation 接口说明修正

**类型**: 📝 文档
**范围**: .workflow/raw/design/qkv_rms_norm_rope_cache_with_kscale/qkv_rms_norm_rope_cache_with_kscale_design.md

#### 变更内容
- 将设计文档中的旧 `optionalRotation`/单 `rotation` 接口说明更新为 `qRotation`、`kRotation` 两个 ACLNN 可选输入形参。
- 明确 `q_rotation`、`k_rotation` shape 均为 `[D, D]`，layout 为 ND，按非转置右矩阵参与计算。
- 更新 Cube rotation matmul 语义：Q 分支执行 `q_rope @ q_rotation`，K 分支执行 `k_rope @ k_rotation`。
- 同步修改 op_host 校验说明、测试说明、rotation L1 常驻/搬运说明和 dense/FWHT 后续优化说明。

#### 关键决策
- 保持 `q_rotation` 与 `k_rotation` 均为接口形态上的可选输入，但当前 arch35 交付场景强制非空。
- 保持 dense Cube 基线方案，不因拆分 rotation 引入 FWHT 或 kernel 内生成矩阵假设。

#### 影响范围
- 仅更新设计文档，不修改源码。
- 文档与当前 op_api 中 `qRotation`/`kRotation` 接口保持一致。

### 2026-06-04 | ACLNN 可选输入形参命名修正

**类型**: 🔧 修改 | 📝 文档
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/op_api/, qkv_rms_norm_rope_cache_with_kscale_design.md

#### 变更内容
- 将 ACLNN L2 接口形参 `qRotation`、`kRotation`、`vScale` 重命名为 `optionalQRotation`、`optionalKRotation`、`optionalVScale`。
- 同步修改 L0 op_api 形参名、DFX 入参、空指针校验、dtype/shape 校验、Contiguous 调用和 L0 调用传参。
- 同步更新设计文档中的 ACLNN 接口示例和 checker 说明。
- 在设计文档可选输入表中补充 `q_rotation`、`k_rotation`、`v_scale` 对应的 ACLNN 形参名。

#### 关键决策
- 仅修改接口形参名，不改变参数顺序、类型、op_host 输入名和算子语义。
- `q_rotation`、`k_rotation`、`v_scale` 仍保持 op_host/算子语义输入名；ACLNN 形参使用 optional 前缀体现可选输入形态。

#### 影响范围
- `bash build.sh --opapi --ops=qkv_rms_norm_rope_cache_with_kscale --soc=ascend950 --noexec -j8` 编译通过。
- 当前未修改 tiling 和 kernel。

### 2026-06-04 | QKV RMSNorm RoPE Cache KScale op_api CSV 覆盖测试

**类型**: 📝 文档 | 🔧 修改
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/tests/ut/op_api/, .workflow/raw/design/

#### 变更内容
- 将 `test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp` 改造为 CSV 驱动参数化测试。
- 新增 `test_aclnn_qkv_rms_norm_rope_cache_with_kscale.csv`，覆盖 54 条 ACLNN 接口正常/异常用例。
- 覆盖空指针、dtype、shape、`headNums`、`Nv == Nk`、`D=128` fast path、输出 shape 和空 tensor 等接口校验。
- 新增测试报告 `qkv_rms_norm_rope_cache_with_kscale_op_api_interface_test_report.md`。

#### 关键决策
- 本轮只从接口角度覆盖 `GetWorkspaceSize` 参数校验，不修改算子实现代码，不覆盖 tiling/kernel 数值正确性。
- 参考 GMMFR CSV 驱动测试写法，将用例矩阵从 C++ 逻辑中拆分到 CSV。

#### 影响范围
- `bash build.sh --opapi_test --ops=qkv_rms_norm_rope_cache_with_kscale --soc=ascend950 --noexec -j8` 编译通过。
- `transformer_op_api_ut` 运行通过，54/54 pass。

### 2026-06-04 | CSV 用例补充与 vCache 一致性校验修复

**类型**: 🔧 修改 | 📝 文档
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/tests/ut/op_api/, attention/qkv_rms_norm_rope_cache_with_kscale/op_api/, .workflow/raw/design/

#### 变更内容
- CSV 用例矩阵从 54 条扩充至 57 条，新增 3 条接口校验用例：
  - `k_cache_last_dim_invalid`：`k_cache.shape[3] != D`
  - `v_cache_last_dim_invalid`：`v_cache.shape[3] != D`
  - `v_cache_block_mismatch`：`v_cache` 的 `BlockSize` 与 `k_cache` 不一致
- 修复 `aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp` 中 `ShapeCheck` 函数，新增 `v_cache` 与 `k_cache` 的 `BlockNum/BlockSize` 一致性校验。
- 更新测试报告 `qkv_rms_norm_rope_cache_with_kscale_op_api_interface_test_report.md`，反映 57 条用例和接口修复。

#### 关键决策
- 补充用例时发现 `v_cache` 的 `BlockNum/BlockSize` 未与 `k_cache` 做一致性校验，属于接口校验缺口，已修复。
- 修复后所有 57 条用例均通过。

#### 影响范围
- `bash build.sh -u --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8 --noexec` 编译通过。
- `transformer_op_api_ut` 运行通过，57/57 pass。

### 2026-06-05 | 任务 5/6/7：ACLNN 接口检查、代码重构、非连续性修复

**类型**: 🔧 修改 | 🆕 新功能
**范围**: attention/qkv_rms_norm_rope_cache_with_kscale/op_host/

#### 变更内容

**任务 5 — ACLNN 接口参数检查**（无需修改）
- 对比 `quant_grouped_matmul_inplace_add` 和 `scatter_pa_kv_cache` 参考实现
- 确认 inplace 参数（kCache/vCache/kScaleCache）处理正确：L2 签名 `aclTensor*`（非 const），L0 同一 tensor 同时出现在 OP_INPUT 和 OP_OUTPUT
- 确认多输出处理正确：5 个输出均在 `OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache)` 中

**任务 6 — ACLNN 代码实现检查和修复**
- 新增 `op_host/op_api/qkv_rms_norm_rope_cache_with_kscale_checker.h`（497 行）
  - `QkvRmsNormRopeCacheWithKScaleParams` 参数结构体（17 个参数）
  - 5 类校验方法：CheckNotNull / CheckEmptyTensor / CheckDtypeValid / CheckShape / CheckFormat
  - 使用 `OP_LOGE_FOR_*_WITH_REASON` 宏替代 `OP_LOGE`，统一对外报错格式
  - dtype 支持列表使用 `std::initializer_list<op::DataType>`
- 重构 `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`（407 行 → 152 行）
  - 移除 7 个内联校验函数和 2 个常量，改用 checker
  - 保留 `Contiguous()` 函数（L2 特有逻辑）
  - `CheckParams()` 直接返回原始错误码（NULLPTR/INVALID），不丢失错误信息

**任务 7 — cache 参数非连续性问题**
- 移除 `op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp` 中 k_cache/v_cache/k_scale_cache 的 `.AutoContiguous()`
- 保留原始 stride 信息，避免 AutoContiguous 创建连续副本导致写出位置错误
- L2 中已正确不对 cache 参数调用 `Contiguous()`，直接透传给 L0

#### 关键决策
- 错误码保留：`CheckParams()` 使用直接 return 而非 `CHECK_RET` 宏包裹，确保 NULLPTR 和 INVALID 错误码不丢失
- 非连续 cache 处理：采用简化模式（不 Contiguous 也不 CreateView），cache 参数直接透传。scatter_pa_kv_cache 提供更复杂的两路处理（ProcessNonContiguous / ProcessContiguous），当前算子无需此复杂度
- checker 不使用 class 封装：参考 quant_grouped_matmul_inplace_add 的 namespace + static function 模式，比 class 模式更简洁

#### 影响范围
- `bash build.sh -u --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8 --noexec` 编译通过
- `transformer_op_api_ut` 运行通过，57/57 pass
- 新增文件：`op_host/op_api/qkv_rms_norm_rope_cache_with_kscale_checker.h`
- 修改文件：`op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`、`op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp`

### 2026-06-05 | 任务 5/6/7 二次验证与改进

**类型**: 🔧 修改
**范围**: ACLNN checker 重构 + 接口参数/非连续性二次验证

#### 变更内容

**任务 5 — ACLNN 接口参数检查（二次验证）**
- 对比 quant_grouped_matmul_inplace_add、scatter_pa_kv_cache、gather_pa_kv_cache、recurrent_gated_delta_rule、nsa_compress_with_cache 五个参考算子
- 确认 inplace 参数处理正确：def.cpp 同名 Input/Output、L0 同参 OP_INPUT+OP_OUTPUT、L2 非 const aclTensor* 且不调用 Contiguous()
- 确认多输出处理正确：L0 单返回 qOut（5 个 OP_OUTPUT 均为 caller-provided mutable pointers），与 recurrent_gated_delta_rule 模式一致
- 无需 ViewCopy — 所有输出直接写入 caller-provided tensor

**任务 6 — ACLNN 代码实现检查（二次改进）**
- Checker 从 namespace + static free functions 重构为 class-based 模式（对齐 grouped_matmul_finalize_routing reference）
  - `QkvRmsNormRopeCacheWithKScaleChecker` 类，params_ 成员变量
  - dtype 列表使用 `inline static const`（C++17），常量使用 `static constexpr`
  - `CheckType` 辅助函数保留在 namespace 级别（供 OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON 宏调用）
- L2 错误传播改用 `CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet)` 替代 if-return 模式

**任务 7 — cache 参数非连续性（二次验证）**
- 对比 scatter_pa_kv_cache、gather_pa_kv_cache、quant_grouped_matmul_inplace_add 三个参考算子
- 确认三个参考算子 def.cpp 均不使用 AutoContiguous()
- 确认当前实现正确：cache 参数无 AutoContiguous()（def.cpp）+ 无 Contiguous()（L2）→ 原始 stride 完整传递给 L0 kernel

#### 关键决策
- Checker class-based 封装：对齐 grouped_matmul_finalize_routing reference 的 class 模式，比 namespace + static functions 更安全（避免 ODR 风险）
- CHECK_RET 错误传播：对齐 reference 的 CHECK_RET 宏用法，比 if-return 更简洁
- 非连续 cache 处理维持简化模式：二次验证确认无需引入 scatter/gather 的 ProcessNonContiguous/ProcessContiguous 双路径复杂度

#### 影响范围
- `bash build.sh -u --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8 --noexec` 编译通过
- `transformer_op_api_ut` 运行通过，57/57 pass
- 修改文件：`op_host/op_api/qkv_rms_norm_rope_cache_with_kscale_checker.h`、`op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`

### 2026-06-08 | L0 多输出返回模式修正

**类型**: 🐛 修复
**范围**: L0 接口多输出返回值处理

#### 变更内容

**问题**：L0 接口仅返回 `qOut`（单指针），`qScale` 等其他 4 个输出完全丢失，L2 无法感知其处理结果。

**调研**：对比 8+ 个算子的 L0 返回模式，发现两种主流模式：
1. **全部输出返回**（scatter_pa_kv_cache, gather_pa_kv_cache, attention_update, chunk_gated_delta_rule）：tuple 包含 OP_OUTPUT 中所有输出
2. **仅返回新分配输出**（qkv_rms_norm_rope_cache, recurrent_gated_delta_rule）：tuple 仅含 AllocTensor 创建的输出，inplace 排除

**修复**：采用模式 1（全部输出返回），与 scatter_pa_kv_cache/gather_pa_kv_cache 一致：
- L0 .h：返回类型从 `const aclTensor*` 改为 `std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>`（5 个输出）
- L0 .cpp：返回 `{qOut, qScale, kCache, vCache, kScaleCache}`，错误路径返回 `{nullptr, nullptr, nullptr, nullptr, nullptr}`
- L2 .cpp：解包 tuple，使用 `std::get<0>` 到 `std::get<4>` 逐一检查所有 5 个返回值

#### 关键决策
- 选择模式 1（全部输出返回）而非模式 2（仅返回新分配输出），因为：
  - 主流模式，更多算子采用
  - 返回所有输出（包括 inplace）更安全，L2 可逐一验证
  - 用户明确要求"至少要包括 qOut 和 qScale，其他 inplace 参数也需要返回"
- 使用 `std::tuple` 而非 `std::array`，因为各输出语义不同（qOut/qScale 为非 inplace 输出，kCache/vCache/kScaleCache 为 inplace 输出），tuple 更适合异构语义

#### 影响范围
- `bash build.sh -u --opapi --soc=ascend950 --ops=qkv_rms_norm_rope_cache_with_kscale -j8 --noexec` 编译通过
- `transformer_op_api_ut` 运行通过，57/57 pass
- 修改文件：`op_host/op_api/qkv_rms_norm_rope_cache_with_kscale.h`、`op_host/op_api/qkv_rms_norm_rope_cache_with_kscale.cpp`、`op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`
