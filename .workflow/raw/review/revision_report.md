# GMM A16W4 PerGroup ND Review 整改复核报告

> 输入来源：`.workflow/raw/review/review.json`
> 复核对象：当前工作区代码与 UT
> 复核日期：2026-06-29
> 范围：arch35 tiling + kernel + op_host UT，不包含 Aclnn 通路

## 复核结论摘要

- Review 意见共 18 条。
- 已整改 / 无需追加处理：14 条。
- 部分整改：3 条。
- 未整改：1 条。
- 复核中新发现 1 个实现风险：`groupSize=64` 的 NK pergroup 调度当前调用了 `Gs32` VF，需后续修正或确认。

## 逐条复核

| review id | 结论 | 证据 | 残留风险 |
|---:|---|---|---|
| 177623968 | 已整改 | `grouped_weight_quant_batch_matmul_tiling.h` 已定义 `A16W4_SINGLE_PER_CHANNEL_DIM`、`A16W4_SINGLE_PER_GROUP_DIM`、`A16W4_MULTI_PER_CHANNEL_DIM`、`A16W4_MULTI_PER_GROUP_DIM` 并附 shape 注释；tiling cpp 使用这些常量做 rank 判断。 | 无。 |
| 177628833 | 已整改 | `SetTilingKey()` 中 `optionInputSituation = static_cast<uint8_t>(hasAntiquantOffset_) << 1`，bias 不再参与 A16W4 ND pergroup tilingKey；kernel 仍通过 `gmmBaseTiling_->hasBias` 初始化。 | 无。 |
| 177629060 | 部分整改 | 新增用户可见错误信息已统一为 `When x-weight is bf16/fp16-int4/int32 ...` 或 `When the dtype of x-weight is ...`。 | 代码标识符和部分注释仍保留 `A16W4` / `A16W4ND`。若 reviewer 的“全文排查”包含内部命名和注释，还需继续处理；若仅指用户可见日志，则已满足。 |
| 177629513 | 无需追加处理 | review 已 resolved；新增 pergroup VF 主函数使用 `__simd_vf__`，专用 helper 使用 `__simd_callee__`。 | 文件中旧的通用 B8/INT4 函数仍有 `__VEC_SCOPE__`，不属于本次 pergroup 新增整改范围。 |
| 177630470 | 已整改 | `grouped_weight_quant_batch_matmul_tiling.cpp` 中相关日志已从 `A16W4ND` 改为 `x-weight is bf16/fp16-int4/int32` 样式。 | 无。 |
| 177632338 | 已整改 | `grouped_matmul_apt.cpp` 中 `VEC_ANTIQUANT_CONFIG_6 = {2, 256}` 后已添加 `// A16W4 ND PerGroup` 应用场景注释。 | 无。 |
| 177632946 | 已整改 | `weight_quant_tiling_key.h` 中 PER_GROUP 组合使用一个 `ASCENDC_TPL_ARGS_SEL`，`OFFSET_OR_BIAS_EXIT` 仅列出 no offset / offset，`WQ_B_TRANS` 列出 no trans / trans，形成正交组合。 | 无。 |
| 177633115 | 部分整改 | `grouped_matmul_apt.cpp` 中 A16W4 ND pergroup 已由多分支收敛为一个分支，并用模板参数构造 `wqmmCfg`。 | 该分支仍保留局部 `static constexpr WqmmConfig wqmmCfg`。该对象可能是当前宏/模板参数形态所需，但若 reviewer 要求完全移除 constexpr 表达，还需继续确认。 |
| 177634275 | 无需追加处理 | review 已 resolved；pergroup 新增 VF 使用 `__simd_vf__` / `__simd_callee__`。 | 同 177629513，旧公共函数仍保留原写法。 |
| 177637651 | 已整改 | CSV 驱动 UT 中输出 y 已改为 `MakeGertStorageShape({m, n}, {m, n})`；未再保留 out shape 异常模式。 | 无。 |
| 177637964 | 已整改 | CSV 成功用例已补齐 `blockDim`、`tilingKey`；UT 成功路径断言 `groupSize`、`hasBias`、`singleX/singleWeight/singleY`、`groupType/groupListType`、`kSize/nSize` 等关键字段。 | `tilingData` 全量字符串仍是可选字段；当前关键字段断言已覆盖 review 指出的核心行为。 |
| 177637990 | 未整改 | review 指向 `grouped_matmul_weight_quant_basic_controller.h` 中 `groupListGm_.SetGlobalBuffer`；当前仍在 `if (groupList != nullptr)` 内设置。 | 与“可以和其他的保持一致在外面设”的建议不一致。需确认该建议是否要求无条件在外层设置，或当前 nullptr 保护是否必须保留。 |
| 177638649 | 已整改 | host tilingKey 只编码 antiquant offset；CSV 中同 transB/offset 下 bias true/false 的期望 tilingKey 相同。 | 无。 |
| 177644673 | 已整改 | `DeriveGroupSizeMulti()` 已拆分出 `GetA16W4MultiScaleShape()`、`DeriveA16W4MultiPergroupSizeFromScale()`、`CheckA16W4MultiPergroupScaleShape()`、`CheckMultiA16W4PerChannelShape()`、`CheckAntiquantOffsetMatchScale()`。 | 无。 |
| 177645601 | 已整改 | pergroup groupSize 调度位于 `weight_quant_vec_compute.h::AntiQuantProcessNdInt4()`；`basic_block_vf_nd.h` 只保留具体 VF 函数。 | 复核发现 `groupSize=64` 分支目前调用 `AntiQuantInt4PerGroupNdNkGs32`，见“新增风险”。 |
| 177645893 | 部分整改 | UT 文件已说明保留的 `TEST_F` 是多实例输入或跨 case 对比，CSV 单行不能自然表达；普通 CSV 正例已增加关键字段断言。 | 仍保留 9 个 `TEST_F`。其中部分场景是否必须保留为独立测试，可继续与 reviewer 确认。 |
| 177652233 | 已整改 | UT 中 `aivNum` 已使用 `DEFAULT_AIC_NUM * optiling::AIC_AIV_CORE_RATIO` 和 `coreNum * optiling::AIC_AIV_CORE_RATIO`。 | 无。 |
| 177655959 | 已整改 | `PrintInputParam()` 中 `groupListType` 格式符已改为 `%lld`；`PrintTilingResult()` 中 tilingData 的 `groupListType` 是小整数并已显式 cast 到 `uint32_t` 后用 `%u` 打印。 | 无。 |

## 新增风险

### NK pergroup `groupSize=64` 调度疑似错误

**位置**：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h`

**代码事实**：
- `basic_block_vf_nd.h` 已实现 `AntiQuantInt4PerGroupNdNkGs64`。
- `AntiQuantProcessNdInt4()` 中 `antiQuantGroupSize_ == PERGROUP_SIZE_64` 分支当前调用的是 `AntiQuantInt4PerGroupNdNkGs32`，不是 `AntiQuantInt4PerGroupNdNkGs64`。

**影响判断**：
- 该问题不属于 `review.json` 中某条意见的直接整改状态，但会影响 A16W4 ND pergroup、`bTrans=true`、`groupSize=64` 的 kernel 计算路径。
- 建议后续将该分支改为调用 `AntiQuantInt4PerGroupNdNkGs64`，并在可编译环境执行 kernel 编译与精度验证。

## UT 与验证状态

- 本次复核未重新执行 host UT 或 kernel 编译，仅做静态代码阅读和 `rg` 证据核对。
- `.workflow/raw/review/revision.md` 中已有历史记录：`git diff --check` 通过、ophost 编译通过、`transformer_op_host_ut` 386/386 通过；kernel 编译按项目约束暂未执行。
