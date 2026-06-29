# GMM A16W4 PerGroup ND Review 整改方案

> 输入来源：`.workflow/raw/review/review.json`
> 当前阶段：代码 review 整改方案设计
> 约束：本文件记录修改方案与执行结果。
> 日期：2026-06-29

## 总体处理原则

1. 保持本任务范围只覆盖 arch35 tiling + kernel，不引入 Aclnn 通路修改。
2. 对已 resolved 的意见不再设计新增改动，只记录后续保持要求。
3. 对 `tilingKey` 中 bias 编码相关意见合并整改：`optionInputSituation/OFFSET_OR_BIAS_EXIT` 仅表达 antiquant offset 是否存在，bias 继续使用 `tilingData_.gmmWeightQuantParam.hasBias` 和 kernel 内 `gmmBaseTiling_->hasBias` 判断。
4. 对 UT 相关意见合并整改：优先收敛为 CSV 驱动，成功用例必须断言关键字段，单独 `TEST_F` 仅保留无法自然表驱动或作为回归语义更清晰的用例。

## 整改优先级

| 优先级 | review id | 主题 |
|---|---:|---|
| P0 | 177628833, 177632946, 177638649, 177633115 | tilingKey bias 编码和 kernel 分支去重 |
| P0 | 177637651, 177637964 | UT 正例覆盖不足和输出 shape 构造错误 |
| P1 | 177629060, 177630470, 177623968, 177655959 | 规范、注释、日志格式 |
| P1 | 177632338, 177637990, 177652233, 177644673, 177645893 | 可维护性和测试结构 |
| 记录 | 177629513, 177634275, 177645601 | 已 resolved，保持当前方向 |

## 逐条整改方案

### 177623968 - 魔鬼数字用注释解释一下

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:249`

**问题理解**：当前 `scaleDimNum != 2 && scaleDimNum != 3` 中的 `2/3` 表达 perchannel/pergroup 的 rank，但直接写数字可读性弱。附近同类逻辑还包含多多多 `1/2`、MX `4` 等维度判断。

**修改方案**：
- 在 tiling header 或 cpp 匿名命名空间补充语义常量，例如：
  - `A16W4_SINGLE_PER_CHANNEL_SCALE_DIM = 2`
  - `A16W4_SINGLE_PER_GROUP_SCALE_DIM = 3`
  - `A16W4_MULTI_PER_CHANNEL_SCALE_DIM = 1`
  - `A16W4_MULTI_PER_GROUP_SCALE_DIM = 2`
- 替换 `CheckTensorDimSingleXSingleWeightSingleY`、`CheckTensorDimMultiXMultiWeightMultiY`、`DeriveGroupSizeSingle`、`DeriveGroupSizeMulti` 中新增 A16W4 ND rank 判断的裸数字。
- 注释说明场景：单单单 scale/offset 为 `[E,N]` 或 `[E,G,N]`；多多多 scale/offset 为 `[N_i]` 或 `[G_i,N_i]`。

**验证**：编译 tiling UT，确认日志字符串不变或只做可接受的文字优化。

### 177628833 - hasbias 直接用 tilingdata 不要用 tiling Key

**位置**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h:194`

**问题理解**：新增 PER_GROUP tiling key 把 bias 是否存在编码进 `OFFSET_OR_BIAS_EXIT`，但现有数据流的 bias 通常由 tilingData/kernel runtime 判断，不应扩展模板 key 组合。

**修改方案**：
- host 侧 `SetTilingKey` 中，A16W4 ND pergroup 不再执行 `(hasAntiquantOffset_ << 1) | hasBias_`，改为与其他路径一致：`optionInputSituation = static_cast<uint8_t>(hasAntiquantOffset_) << 1`。
- `weight_quant_tiling_key.h` 删除 PER_GROUP 下 `WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST` 和 `WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_EXIST` 两类组合，只保留：
  - no transB + no offset
  - no transB + offset
  - transB + no offset
  - transB + offset
- kernel 继续通过 `gmmBaseTiling_->hasBias` 初始化 basic block，不通过 tiling key 判断 bias。

**影响范围**：
- PER_GROUP tiling key 组合由 8 个降为 4 个。
- CSV 正例中带 bias 和不带 bias 的同 offset/transB 用例应期望相同 tilingKey。

**验证**：
- `bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul -j8 --noexec`
- `bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`
- 更新并运行 pergroup tiling CSV UT，重点覆盖 `hasBias=true/false` tilingKey 相同但 `tilingData.hasBias` 不同。

### 177629060 - 按照 xtype-weighttype 的写法全文排查

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:1145`

**问题理解**：新增错误信息和注释中出现 `A16W4ND`、`A16W4 pergroup` 等缩写，和文件既有 `x-weight is bf16/fp16-int4/int32` 风格不一致。review 要求全文统一为 `x_dtype-weight_dtype` 描述。

**修改方案**：
- 全文排查新增文案中的 `A16W4ND`、`A16W4 pergroup`、`bf16/fp16-int4/int32` 等表述。
- 对用户可见日志统一写成 dtype 对，例如：
  - `When x-weight is bf16/fp16-int4/int32 ...`
  - 或按 review 177630470 进一步明确为 `bf16/fp16-int4`、`bf16/fp16-int32`。
- 内部函数名 `IsA16W4ND()` 可保留，不属于用户可见文案；如需注释，可写成 “B16 x with INT4/INT32 ND weight”。

**验证**：`rg -n "A16W4|A16W4ND|bf16/fp16-int4/int32" grouped_weight_quant_batch_matmul_tiling.cpp`，确认只保留函数名或必要内部标识。

### 177629513 - `__VEC_SCOPE__` 替换为 `__simd_vf__`

**状态**：已 resolved。

**处理方案**：
- 不再追加整改。
- 后续新增 VF 函数保持当前已采用的 `__simd_vf__` / `__simd_callee__` 风格。
- 若继续触碰旧的非本 PR 代码，不做无关大范围替换，避免扩大风险。

### 177630470 - A16W4 写成 x_dtype-weight_dtype 样式

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:295`

**问题理解**：错误信息 `In multi-multi-multi A16W4ND` 不符合 reviewer 要求。

**修改方案**：
- 将新增日志中的 `A16W4ND` 改为 dtype pair 描述，例如：
  - `When x-weight is bf16/fp16-int4/int32 and groupType is -1, ...`
- 对 `groupType=0` 和 `groupType=-1` 分别保留明确场景说明，避免只写 A16W4。
- 与 177629060 一起全文排查和统一。

**验证**：同 177629060。

### 177632338 - 增加注释说明应用场景

**位置**：`gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp:57`

**问题理解**：`VEC_ANTIQUANT_CONFIG_6 = {2, 256}` 是本次新增 A16W4 ND pergroup 使用的 vec antiquant 配置，但名字和数字无法体现应用场景。

**修改方案**：
- 在 `VEC_ANTIQUANT_CONFIG_6` 旁增加简短注释，说明其用于 A16W4 ND PerGroup数据流。

**验证**：添加注释即可。

### 177632946 - 简化实现

**位置**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h:195`

**问题理解**：当前 PER_GROUP tiling key 因 bias 参与编码膨胀为 8 个组合，且 `grouped_matmul_apt.cpp` 中对应分支重复。简化方向与 177628833、177638649 一致。

**修改方案**：
- tiling key模板的参数支持正交参数的组合，因为他们输入是ASCENDC_TPL_UI_LIST，表示后面能够把可选的输入都填入，自动正交构造组合。
- OFFSET_OR_BIAS_EXIT和WQ_B_TRANS是主要组合的值。

**验证**：检查生成的 tilingKey 仍能命中 4 个 PER_GROUP kernel 分支。

### 177633115 - 去掉不必要的 constexpr 表达，减少代码量

**位置**：`gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp:420`

**问题理解**：A16W4 ND pergroup 分支中重复定义局部 `static constexpr WqmmConfig wqmmCfg`，而 `basic_block_config.h` 已有同名全局配置。

**修改方案**：
- tiling key提供的模板参数可以直接用于编译期构造WqmmConfig参数，只需要对基本的数据流进行判断，确保调用 `INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, ...)`即可。

**验证**：后续人工验证。

### 177634275 - 用 simd_vf 的新标准写

**状态**：已 resolved。

**处理方案**：
- 不再追加整改。
- 后续 VF 新增代码继续使用 `AscendC::VF_CALL<...>` 配合 `__simd_vf__` / `__simd_callee__`。

### 177637651 - CSV 驱动 UT 中输出 y shape 构造错误

**位置**：`gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.cpp:197`

**问题理解**：当前表驱动用例输出 y 使用 `MakeGertStorageShape({m}, {n})`，origin/storage rank 都不是真实 `{m,n}`，可能掩盖输出维度校验问题。

**修改方案**：
- 单单单 `groupType=0`：输出 y 应构造成 2D `{m,n}`，即 `MakeGertStorageShape({m, n}, {m, n})`。
- 多多多 `groupType=-1`：当前表驱动只用一组 `m/k/n` 描述所有 tensor，若仍只构造一个输出实例，则也应使用 `{m,n}`；若后续扩展多实例字段，需要每个输出实例分别使用 `{m_i,n_i}`。
- 不增加输出 rank/N 维异常 CSV 用例：out shape 由 infershape/框架代码推导，不作为 tiling 用户输入校验项。
- 若代码中已引入 `CheckOutputShape` 或 `yShapeMode` 类测试辅助，需要删除，避免 tiling 重复校验推导输出。

**验证**：运行既有 host UT，确认正例仍成功；不要求 y shape 异常例失败。

### 177637964 - CSV 成功正例缺少关键字段断言

**位置**：`gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.csv:3`

**问题理解**：26 条成功正例 `blockDim=0`、`tilingKey=0`、`tilingData` 为空，测试遇到 `expectTilingData.empty()` 直接返回，只覆盖“能成功”，不能覆盖本 PR 的核心行为。

**修改方案**：
- CSV 字段扩展或复用现有字段，至少为成功正例补齐：
  - `blockDim`
  - `tilingKey`
  - `groupSize`
  - `hasBias`
  - `singleX/singleWeight/singleY`
  - `groupType/groupListType`
  - `kSize/nSize`
- 建议不要继续比对完整二进制 `tilingData` 字符串作为主要方式，改为解析 `GMMWeightQuantTilingData` 后按字段断言，减少结构 padding 或无关 tiling 字段变化导致的脆弱性。
- 保留 `expectTilingData` 可选字段用于少量白盒完整比对；当为空时也不能直接返回，应至少断言上述关键字段。
- 对 177628833 的改动补充专项断言：
  - 同 transB、同 offset 下，bias true/false 的 `tilingKey` 相同。
  - `tilingData.gmmWeightQuantParam.hasBias` 分别为 1/0。
  - offset true/false 的 `tilingKey` 不同。
  - pergroup/perchannel 的 `W_QUANT_TYPE` 不同。

**验证**：pergroup CSV 成功用例至少覆盖 32/64/128/256、transB true/false、offset true/false、bias true/false、groupType 0/-1 的关键字段。

### 177637990 - 可以和其他的保持一致在外面设

**位置**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h:92`

**问题理解**：`biasBaseOffset_ = 0` 是一个成员变量，而其他offset都是函数局部变量，把bias改为局部变量传参，与其他offset的写法一致。

**修改方案**：
- 将成员声明改为函数内定义局部变量 `uint64_t biasBaseOffset = 0;`，后续传参调用。

**验证**：后续人工验证。

### 177638649 - tilingKey 里只表示 offset，bias 在 kernel 用 tilingData 判断

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:772`

**问题理解**：与 177628833 相同，是 host 侧生成 tilingKey 的根因位置。

**修改方案**：
- 将 A16W4 ND pergroup 分支中的 `optionInputSituation` 改为只编码 antiquant offset。
- 不再根据 `hasBias_` 改变 tilingKey。
- 确认 `SetTilingData` 已写入 `gmmWeightQuantParam.hasBias`，kernel `wqmmBasicBlock_.Init(gmmBaseTiling_->hasBias, ...)` 已使用该字段。

**验证**：同 177628833 和 177637964。

### 177644673 - 超大函数

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:1217`

**问题理解**：`DeriveGroupSizeMulti` 同时负责首个 scale 判断、perchannel 分派、pergroup groupSize 推导、遍历所有 tensor、一致性校验、N 维校验和 offset 校验，函数职责偏多。

**修改方案**：
- 拆分为小函数，建议：
  - `GetA16W4MultiScaleShape(context, idx, shape)`
  - `DeriveA16W4MultiPergroupSizeFromScale(scaleShape, k, idx, groupSize)`
  - `CheckA16W4MultiPergroupScaleShape(context, idx, expectedDim, expectedGroupSize)`
  - `CheckA16W4MultiPergroupScaleOffsetShapes(context, idx, scaleShape)`
- `DeriveGroupSizeMulti` 保留主流程：
  1. 获取 `scaleShape[0]`。
  2. 判断 perchannel/pergroup。
  3. perchannel 调用 `CheckMultiA16W4PerChannelShape`。
  4. pergroup 设置 `groupSize_` 并循环校验各 tensor。
- 注意拆分时不要改变日志语义和失败返回路径。

**验证**：host UT 全量通过；`git diff --check` 通过。

### 177645601 - 调度不要写在 vf 里面，写在 vec_compute.h 里面

**状态**：已 resolved。

**处理方案**：
- 不再追加整改。
- 保持分派逻辑位于 `weight_quant_vec_compute.h`，VF 文件只保留具体计算函数。

### 177645893 - UT 已有 CSV 表驱动，为什么又引入单独 TEST_F

**位置**：`gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.cpp:547`

**问题理解**：新增多个 `TEST_F` 白盒/回归用例，与 CSV 表驱动模式并存，增加维护成本。

**修改方案**：
- 优先把以下 `TEST_F` 场景迁移到 CSV 或 CSV 扩展字段：
  - 多多多 m/k/n tilingData 写入验证。
  - perchannel groupSize=0。
  - pergroup groupSize 不一致失败。
  - `K == groupSize` 时 pergroup/perchannel tilingKey 区分。
- 如 CSV 暂时无法表达“两个相同 shape 下 pergroup 与 perchannel tilingKey 不同”的对比型断言，可保留一个专门 `TEST_F`，但需在注释中说明这是 cross-case comparison，不适合单行 CSV。
- 其余单 case 断言应统一进表驱动，减少重复构造 `TilingContextPara` 的代码。

**验证**：迁移后 `TEST_F` 数量明显减少，CSV case 覆盖不降低。

### 177652233 - `2` 改成 `AIC_AIV_CORE_RATIO`

**位置**：`gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_pergroup_tiling.cpp:113`

**问题理解**：UT 中 `coreNum * 2` 与 host tiling 中 `AIC_AIV_CORE_RATIO` 语义一致，应复用常量提升可读性。

**修改方案**：
- 当前测试文件已 include `grouped_weight_quant_batch_matmul_tiling.h`，可直接使用 `AIC_AIV_CORE_RATIO`。
- 将：
  - `coreNum * 2` 改为 `coreNum * optiling::AIC_AIV_CORE_RATIO` 或实际命名空间下可见的 `AIC_AIV_CORE_RATIO`。
  - 默认 `64` 可改为 `32 * AIC_AIV_CORE_RATIO`，避免一处改比率另一处遗漏。
- 同步检查 `MakeAscend950CompileInfo()` 中的 `32/64`，可加注释或同样使用常量表达。

**验证**：UT 编译通过。

### 177655959 - `groupListType` 类型和格式符不匹配

**位置**：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:1470`

**问题理解**：host 成员 `groupListType_` 是 `int64_t`，`PrintInputParam` 当前用 `%u` 打印存在格式不匹配风险。`PrintTilingResult` 中的 `tilingData_.gmmWeightQuantParam.groupListType` 是 `uint8_t`，使用 `%u` 在默认提升后可接受，但也可显式 cast。

**修改方案**：
- `PrintInputParam` 中 `groupListType_` 的格式符从 `%u` 改为 `%lld`，实参改为 `static_cast<long long>(groupListType_)`。
- 顺手检查 `splitItem_`、`groupType_` 等 int64/int8 字段的格式符和 cast 是否一致。
- `PrintTilingResult` 可保持 `%u`，但建议对 `uint8_t` 字段使用 `static_cast<uint32_t>(...)`，避免小整数类型日志歧义。

**验证**：host 编译无格式告警。

## 合并整改建议

### 修改批次 1：tilingKey 和 kernel 分支收敛

覆盖 review：177628833、177632946、177633115、177638649。

预期改动：
- `grouped_weight_quant_batch_matmul_tiling.cpp`
  - `optionInputSituation` 只编码 offset。
- `weight_quant_tiling_key.h`
  - PER_GROUP key 从 8 个缩减到 4 个。
- `grouped_matmul_apt.cpp`
  - 删除 bias 维度重复分支。
  - 使用 `basic_block_config.h` 中已有 A16W4_ND_PG 配置。

重点验证：
- bias true/false 不改变 tilingKey。
- offset true/false 改变 tilingKey。
- kernel 仍通过 tilingData 识别 bias。

### 修改批次 2：UT 有效性修复

覆盖 review：177637651、177637964、177645893、177652233。

预期改动：
- y shape 改为真实 `{m,n}`。
- out shape 由代码推导，不做 tiling 校验；不保留输出维度异常用例。
- CSV 成功正例补齐关键字段断言。
- 单 case `TEST_F` 尽量迁移 CSV。
- core ratio 使用 `AIC_AIV_CORE_RATIO`。

重点验证：
- pergroup 正例不再只检查成功返回。
- `groupSize`、`tilingKey`、`hasBias`、`singleX/singleWeight/singleY` 等关键字段被断言。
- antiquantOffset 与 antiquantScale 的 shape/rank 不一致能被 UT 捕获。

### 修改批次 3：规范和可维护性

覆盖 review：177623968、177629060、177630470、177632338、177637990、177644673、177655959。

预期改动：
- rank 魔鬼数字语义化。
- 用户可见日志统一为 `x_dtype-weight_dtype` 写法。
- `VEC_ANTIQUANT_CONFIG_6` 补应用场景注释。
- `biasBaseOffset_` 改成类内默认初始化或更符合现有风格的位置初始化。
- 拆分 `DeriveGroupSizeMulti`。
- 修复日志格式符。

重点验证：
- host/kernel 编译无新增告警。
- 文案 grep 排查通过。
- UT 无回归。

## 最终验证清单

- [x] `git diff --check`
- [x] `bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul -j8 --noexec`
- [x] `ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_GLOBAL_LOG_LEVEL=0 BUILD_PATH=$(pwd)/build ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut`
- [ ] `bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`（按约束暂不执行）
- [x] A16W4 ND pergroup 手工检查：single/multi、transB true/false、offset true/false、bias true/false、groupSize 32/64/128/256。

## 整改执行结果（2026-06-29）

### 代码整改

- `CheckOutputShape` 已从 host tiling 中删除，`CheckEveryTensor` 不再校验 y shape。
- A16W4 ND pergroup 的 `optionInputSituation` 已收敛为只编码 antiquantOffset，bias 由 tilingData/runtime 判断。
- `weight_quant_tiling_key.h` 中 PER_GROUP key 分支已收敛为 offset × transB 的正交组合，不再按 bias 展开。
- `grouped_matmul_apt.cpp` 中 A16W4 ND pergroup 分支已按 offset/transB 合并，bias 不参与模板分派。
- `DeriveGroupSizeMulti` 已拆分辅助函数；单单单和多多多均补充 antiquantOffset 与 antiquantScale shape 一致性校验。
- `biasBaseOffset_` 已改为 `Process` 内局部变量并通过参数传递。
- AIC/AIV ratio、rank 魔鬼数字、日志格式符和新增注释已按 review 要求处理。

### UT 整改

- CSV 输出 y shape 已统一修正为 `{m,n}`。
- 已删除 `yShapeMode` 字段以及依赖 out shape 校验的 `bad_rank/bad_n` 负例。
- CSV 成功正例已补齐 `blockDim`、`tilingKey`、`groupSize`、`hasBias`、`singleX/singleWeight/singleY`、`groupType/groupListType`、`kSize/nSize` 等关键字段断言。
- 已新增/保留 antiquantOffset shape/rank mismatch 负例，用于看护单单单 offset 与 scale 维度一致性。
- 保留少量 `TEST_F`：仅覆盖多实例、跨 case 对比、perchannel/pergroup key 对比、INT32 packed weight 逻辑 N 推导等 CSV 单行无法自然表达的场景。

### 验证结果

- `git diff --check`：通过。
- `bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul -j8 --noexec`：通过。
- `transformer_op_host_ut`：386/386 通过。
- kernel 编译：按本文档“关于kernel的编译”约束，本轮未执行，后续在其他机器验证。

## 附加说明、验证和修改项

### 关于kernel的编译

所有需要编译kernel的验证项都不用进行编译，后续再其他机器上验证。

### 关于环境

编译UT需要souce cann包，并激活conda环境

``` shell
source ~/Ascend/cann/set_env.sh
conda activate ascend
```

### 单单单场景antiquantScale与antiquantOffset维度匹配校验

在多多多场景下对每个tensor的维度一致性校验进行了拦截，请检查一下单单单场景是否有antiquantScale与antiquantOffset的一致性校验，并检查UT用例是否存在相应看护用例，如果没有请修改或添加。
