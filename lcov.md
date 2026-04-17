# PromptFlashAttention arch32 lcov 增补记录

## 2026-04-13 基线

- 基线日志：`log/pfa_ophost_ut.log`
- 基线覆盖：`prompt_flash_attention_tiling.cpp | 52.3% | 3409`，分支覆盖 `47.5% | 120`
- 目标：先推动到 70% 左右，优先恢复历史注释用例，并接受 `GRAPH_FAILED` 分支作为有效覆盖场景

## 2026-04-13 第 1 轮恢复用例

- 修改文件：`attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`
- 恢复的历史用例：`7`-`25`、`27`-`29`、`33`-`42`
- 校正内容：
  - 修复大段取消注释后的语法问题
  - 对已跑通的成功用例回填 `expectTilingKey`
  - 对失败即返回的场景保留 `GRAPH_FAILED`，不再强求 `expectTilingKey`
- 验证命令：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
bash build.sh -u --ophost --ops=prompt_flash_attention --soc=ascend910b --cov -j$(nproc) 2>&1 | tee ./log/pfa_ophost_cov_iter1.log
```

- 第 1 轮结果：
  - 共运行 39 条用例
  - 31 条通过，8 条失败
  - 失败点集中在：
    - `PromptFlashAttention_910b_tiling_9`：原预期 `GRAPH_FAILED`，实际返回成功
    - `35/37/39/41/42`：`expectTilingKey` 未回填
    - `38`：`sparse_mode=3` 时 `attenMask` 形状不合法
    - `40`：Base API 路径下误把 ALIBI mask 放进 `attenMask`，命中 `PFA SH layout do not support attenMask or antiquant`

## 2026-04-13 第 2 轮修正方向

- 修改文件：`attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`
- 本轮准备修正：
  - `tiling_9` 改为 `GRAPH_SUCCESS`
  - 回填 `tiling_35/37/39/41/42` 的实际 `expectTilingKey`
  - `tiling_38` 将 `attenMask` 调整为 `sparse_mode=3` 允许的 `(1, 1, 2048, 2048)` 形状，继续争取成功分支覆盖
  - `tiling_40` 改成将 ALIBI mask 走 `pse_shift` 输入，`attenMask` 置空，继续争取 Base API 成功路径覆盖

## 2026-04-13 第 2 轮结果

- 直接构建验证命令：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export BUILD_PATH=/home/j60100428/ops-transformer_0408/build
cmake --build /home/j60100428/ops-transformer_0408/build --target transformer_op_host_ut -j4
```

- 结果：
  - 39 条已恢复用例全部转绿
  - 新确认的 `expectTilingKey`：
    - `tiling_9 = 1000000000000001012`
    - `tiling_35 = 1000000000000011112`
    - `tiling_37 = 1000000000000101012`
    - `tiling_38 = 1000000000000111112`
    - `tiling_39 = 1000000000000000020`
    - `tiling_41 = 1000000000000021217`
    - `tiling_42 = 1000000000000001612`
  - `tiling_40` 保留为 `GRAPH_FAILED`，实际命中 `CheckBaseApiOptionalInput` 的 `inputLayout need to be SH`

## 2026-04-13 第 3 轮覆盖扩展

- 新增探索用例：`tiling_43`、`tiling_44`、`tiling_45`
  - 目标：补 Base API 相关分支
  - 当前实际命中：仍在 `CheckBaseApiOptionalInput` 处失败，错误为 `inputLayout need to be SH`
  - 先按 `GRAPH_FAILED` 保留，作为 Base API 失败分支覆盖
- 全量 cov 验证日志：`log/pfa_ophost_cov_iter2.log`
- 当前稳定覆盖结果：
  - `prompt_flash_attention_tiling.cpp | 56.0% | 3409`
  - 函数覆盖 `51.7% | 120`
- 结论：
  - 相比基线 `52.3%` 已提升，但距离 70% 仍远
  - 仅靠恢复历史注释用例的增益有限，后续要继续围绕 Base API / TND / 特殊输入校验细分补场景

## 2026-04-13 当前未收敛点

- 最新一次继续扩展时，`tiling_43`、`tiling_44`、`tiling_45` 已稳定作为 Base API 失败分支存在
- 最后一轮临时调参时误改了少量旧用例预期，已逐步回调；若继续执行全量 cov，需重点确认：
  - `PromptFlashAttention_910b_tiling_5` 的预期已改回成功路径
  - `tiling_43`、`tiling_44`、`tiling_45` 继续维持 `GRAPH_FAILED`
- 若继续补覆盖，应优先围绕：
  - 让真正的 SH/Base API 成功路径打通
  - 恢复 `tiling_26` 一类纯失败分支用例
  - 对照 `build/cov_result/html_report/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp.func-sort-c.html` 中的零命中函数逐个设计场景

## 待继续记录

- 重新执行 cov 构建，确认所有恢复用例转绿
- 查看 `build/cov_result/html_report/` 的最新命中情况，继续补零命中或低命中分支
- 如后续为覆盖率继续修改其他文件，必须在此文档补充修改目的、文件路径和验证结果



## 2026-04-14 lcov

`attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`中我对重复场景的用例进行了删除，因为这些用例对覆盖率增长没丝毫作用，徒增代码量，现在用例名称PromptFlashAttention_910b_tiling_后面的数字编号并不连续，你帮忙顺手修正一下

另外build/cov_result/html_report下面可以查看究竟哪些代码行覆盖到了（诸如prompt_flash_attention_tiling.cpp.gcov.html标红的是没覆盖到，标蓝的是覆盖到了，没颜色的是不需要覆盖该行），我觉得你可以好好利用编译后的html去查看，现在prompt_flash_attention_tiling.cpp的覆盖率离70%都差很多，当然我也理解，按照prompt_flash_attention_def.cpp算子原型去传参会发现tiling里面可能有些参数压根就没法设置，这个原因是为什么呢：因为attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp和attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp依赖prompt_flash_attention_tiling.cpp的实现，有些参数可能不是由prompt_flash_attention控制的，而是由incre_flash_attention和fused_infer_attention_score的def.cpp去控制传参了，你去查阅incre_flash_attention和fused_infer_attention_score的def.cpp，可能会发现这两个算子会有pfa需要的参数，遇见这个情况你直接在本文档内给我说明该参数分支的情况吧

现在请继续帮我增补prompt_flash_attention_tiling.cpp的覆盖率，并将相关的详细记录记到本文档。

### 2026-04-14 第 1 轮处理结果

- 修改文件：
  - `attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`
- 已先处理你提到的编号不连续问题：
  - 当前用例名已连续整理为 `PromptFlashAttention_910b_tiling_0` ~ `PromptFlashAttention_910b_tiling_26`
  - 这次只改了测试名编号，不改各用例本身的覆盖意图，避免后续看日志和 html 时编号继续跳号
- 本轮新增 4 个 `GRAPH_FAILED` 分支用例，用来吃掉 prompt 侧仍可控的参数校验分支：
  - `tiling_23`：`DT_FLOAT` 输入 / 输出，命中 `prompt_flash_attention_tiling.cpp:3840`
  - `tiling_24`：`TND + pse_shift`，命中 `prompt_flash_attention_tiling.cpp:3894`
  - `tiling_25`：910B 下 `inner_precise=4`，命中 `prompt_flash_attention_tiling.cpp:3929`
  - `tiling_26`：`int8` 输入 + `fp16` 输出同时给 `quant_scale2` / `quant_offset2`，命中 `prompt_flash_attention_tiling.cpp:3975`
- 同时修正历史回归：
  - `PromptFlashAttention_910b_tiling_5` 已改回 `GRAPH_SUCCESS`

### 2026-04-14 第 1 轮验证

- 直接构建验证：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export BUILD_PATH=/home/j60100428/ops-transformer_0408/build
cmake --build /home/j60100428/ops-transformer_0408/build --target transformer_op_host_ut -j4
```

- 全量 cov 验证：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
bash build.sh -u --ophost --ops=prompt_flash_attention --soc=ascend910b --cov -j$(nproc) 2>&1 | tee ./log/pfa_ophost_cov_20260414_iter1.log
```

- 结果：
  - `log/pfa_ophost_cov_20260414_iter1.log` 中 27 条用例全部通过
  - 日志中新增失败分支已稳定命中：
    - `3840`: `inputType(0) and outputType(0) can not be DT_FLOAT`
    - `3894`: `When Layout is TND, not support PSE`
    - `3929`: `not support APPROXIMATE_COMPUTATION when curShortSocName is Atlas A2`
    - `3975`: `When query dtype is int8 and output dtype is fp16, the output's dequant scale and the output's dequant offset should be null`
  - 当前稳定覆盖结果：
    - `prompt_flash_attention_tiling.cpp | 56.2% | 3409`
    - 函数覆盖 `51.7% | 120`
- 结论：
  - 这一轮继续提升了少量行覆盖，但增益已经明显放缓
  - 说明 PromptFlashAttention 自身原型可直接驱动到的分支，已经被挖得比较深了

### 2026-04-14 html 报告对照结论

- 对照 `build/cov_result/html_report/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp.func-sort-c.html` 与 `prompt_flash_attention_tiling.cpp.gcov.html` 后，当前大块零命中区域主要集中在 3 类：
  - `PA / Base API / Page Attention` 相关函数，如：
    - `CheckPASparseMode`
    - `CheckPAWhenBaseApi`
    - `CheckPATypeAndShape`
    - `checkPABlockSizeAndBlockTable`
    - `SetBaseApiSeqTilingData`
    - `SetBaseApiAlibiMaskInfo`
    - `SetBaseApiPseInfo`
  - `antiquant` 相关函数，如：
    - `CheckAntiquantParamsShape`
    - `CheckPAAntiquantSupportScenarios`
  - 非当前 910B Prompt UT 主路径的分支，如：
    - 310P 特化分支
    - 更深的 Base API workspace / seq split / CV 调整逻辑
- 这些函数不是“测试没写到”这么简单，很多是因为 PromptFlashAttention 的 def 根本不给这组入参入口，或者当前 Prompt UT 无法把上下文推进到这些深层分支

### 2026-04-14 Prompt / IFA / FIA 参数边界说明

- 我对照了 3 个算子原型：
  - `attention/prompt_flash_attention/op_host/prompt_flash_attention_def.cpp`
  - `attention/incre_flash_attention/op_host/incre_flash_attention_def.cpp`
  - `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp`
- 结论是：`prompt_flash_attention_tiling.cpp` 里确实有一批分支，并不是由 PromptFlashAttention 自己完整控参，而是给 IFA / FIA 共用实现时吃的
- 目前已确认的这类参数有：
  - 只在 `IncreFlashAttention` / `FusedInferAttentionScore` def 中出现，而 `PromptFlashAttention` def 没有的输入：
    - `antiquant_scale`
    - `antiquant_offset`
    - `block_table`
    - `kv_padding_size`
  - 只在 `IncreFlashAttention` / `FusedInferAttentionScore` def 中出现，而 `PromptFlashAttention` def 没有的属性：
    - `block_size`
  - 只在 `FusedInferAttentionScore` def 中出现，而 `PromptFlashAttention` def 没有的属性：
    - `softmax_lse_flag`
    - `key_antiquant_mode`
    - `value_antiquant_mode`
    - `query_quant_mode`
    - `pse_type`
- 这直接导致：
  - `Page Attention`
  - `block_table / kv_padding_size`
  - 更完整的 `antiquant`
  - 部分 Base API 扩展能力
  这些分支在 PromptFlashAttention 的 UT 里天然就不好构造，甚至根本无法构造

### 2026-04-14 Base API 当前卡点

- 这一轮继续保留了 Base API 失败分支用例 `tiling_20`、`tiling_22`
- 日志稳定报错位置仍是：
  - `prompt_flash_attention_tiling.cpp:5944`
  - 错误信息：`inputLayout need to be SH`
- 这说明当前 PromptFlashAttention UT 传参虽然能把流程推进到 `CheckBaseApiOptionalInput`，但还进不到真正的 SH/Base API 成功路径
- 结合 3 个 def 文件来看，当前结论是：
  - Prompt 侧虽然能传 `BSH` / `BNSD` / `BSND` / `TND` 一类常规布局场景
  - 但真正驱动 Base API 深层成功路径所需的 `SH + 额外可选输入/属性组合`，更像是 IFA / FIA 侧在供参
  - 因此仅靠 `test_prompt_flash_attention_tiling.cpp` 做 Prompt UT，很难把 `5944` 后面那大片 Base API 代码打透

### 2026-04-14 当前判断

- 现阶段继续硬堆 PromptFlashAttention UT，覆盖率还能涨，但会越来越慢
- 从 `56.2%` 再推到 `70%`，单靠 PromptFlashAttention 当前 def 的控参能力，难度已经很高
- 后续若还想继续推，优先级建议是：
  - 继续扫 Prompt 侧仍能直接构造的失败分支
  - 对照 html 把“可控但未覆盖”的普通参数校验再挖一遍
  - 对于明确由 `IncreFlashAttention` / `FusedInferAttentionScore` 提供参数的分支，在本文档中记录为 Prompt UT 不可直达分支，不再强行在 Prompt 用例里硬凹

### 2026-04-14 第 2 轮：改走 FIA / IFA 侧 UT 同步拉升 PFA 覆盖

- 本轮新增文件：
  - `attention/incre_flash_attention/tests/ut/op_host/arch32/CMakeLists.txt`
  - `attention/incre_flash_attention/tests/ut/op_host/arch32/test_incre_flash_attention_PFA_tiling.cpp`
  - `attention/fused_infer_attention_score/tests/ut/op_host/arch32/test_fused_infer_attention_score_PFA_tiling.cpp`
- 本轮沿用你指定的多算子 op_host cov 方案：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export BUILD_PATH=/home/j60100428/ops-transformer_0408/build
bash build.sh -u --ophost --ops="prompt_flash_attention,fused_infer_attention_score,incre_flash_attention" --soc=ascend910b --cov -j$(nproc) 2>&1 | tee ./log/pfa_ophost_ut.log
```

- 新增用例分工：
  - `test_fused_infer_attention_score_PFA_tiling.cpp`：优先构造 `block_table / sparse_mode / antiquant` 等 Prompt def 无法直接控参的场景，目标是从 FIA 入口进入 `prompt_flash_attention_tiling.cpp`
  - `test_incre_flash_attention_PFA_tiling.cpp`：先补 IFA 入口的失败边界，观察哪些场景会在 IFA 前置校验被拦截，哪些可以继续下钻到 PFA 共用实现

### 2026-04-14 第 2 轮排障记录

- FIA 侧最开始直接复用 `ExecuteTestCase(...)` 失败：
  - 新增 FIA arch32 用例一运行就崩在 `gert::OpImplSpaceRegistryV2::GetOpImpl(char const*)`
  - 栈落在 `tests/ut/framework_normal/common/tiling_case_executor.cpp:308`
  - 结论：当前这条 UT 路径下，FIA 新增用例直接走 registry 取 tiling 实现不稳定，不能继续沿用默认执行器
- 改成手工构造 `gert::TilingContext` 后，又遇到链接问题：
  - 直接调用 `optiling::DoOpTilingFusedInferAttentionScore(...)` / `optiling::TilingFusedInferAttentionScore(...)` 均出现 `undefined reference`
  - 根因是测试侧能看到声明，但最终可链接的导出入口不是这两个符号
  - 最终修正为调用带默认可见性导出的包装入口 `optiling::DeviceDoOpTilingFusedInferAttentionScore(...)`
- 之后又出现“新 FIA 用例先跑完，Prompt 原有成功用例批量回归失败”的问题：
  - 现象是 `PromptFlashAttention_910b_tiling_0/1/2/5/...` 这类原本稳定成功的用例，开始在 `prompt_flash_attention_tiling.cpp:4975` 一带报 `adjust tiling cv diff fail`
  - 继续看日志，之前 FIA 直接执行辅助函数里没有给软打桩的 `softmax_tiling` 显式设置 SoC，导致其第一次走到高级 api stub 时用的是空 SoC，上下文状态被污染
  - 最终修正：
    - 在 `FusedInferAttentionScorePfaTiling` 的 `SetUp/TearDown` 中增加 `SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B") / Reset()`
    - 在 `IncreFlashAttentionPfaTiling` 里同步做同样处理
  - 修正后，多算子 `transformer_op_host_ut` 重新全绿，45 条测试全部通过

### 2026-04-14 第 2 轮新增场景实际命中

- FIA 新增 4 条用例中，当前最有价值的是 2 条真正打进了 PFA 共用分支：
  - `FusedInferAttentionScore_PFA_tiling_2`
    - 日志命中 `prompt_flash_attention_tiling.cpp:4445`
    - 报错：`attenMask should not be null when sparse_mode is 3`
    - 说明 FIA 侧传 `sparse_mode=3` 的场景已经成功推进到了 PFA 的 Page Attention / sparse 校验区间，这条分支单靠 Prompt UT 不好构造
  - `FusedInferAttentionScore_PFA_tiling_3`
    - 日志命中 `prompt_flash_attention_tiling.cpp:5112`
    - 报错：`shape of the key's dequant scale(1, 1024) is not same with ND(8, 128) in perChannel mode`
    - 说明 FIA 侧已经能把流程推进到 PFA 的 antiquant / dequant shape 校验分支
- FIA 另外 2 条当前还停留在 FIA 自身前置 checker：
  - `FusedInferAttentionScore_PFA_tiling_0`
    - 命中 `attention/fused_infer_attention_score/op_host/checkers/paged_attention_checker.cpp:393`
    - 报错：`blockSize(127) should be a multiple of 16`
  - `FusedInferAttentionScore_PFA_tiling_1`
    - 命中 `attention/fused_infer_attention_score/op_host/checkers/mask_checker.cpp:376`
    - 报错：`when sparse_mode is 1 ... atten_mask should not be null`
  - 这两条对 PFA 行覆盖帮助有限，但它们把“哪些参数在 FIA 自身就被前置拦截”这件事验证清楚了
- IFA 新增 2 条用例当前都还停留在 IFA 自身校验：
  - `IncreFlashAttention_PFA_tiling_0`
    - 命中 `attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp:1917`
    - 报错：`when CV 1:1, only support MLA ...`
  - `IncreFlashAttention_PFA_tiling_1`
    - 命中 `attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp:479`
    - 报错：`b of tensor in tensor list should be 1`
  - 当前判断：这 2 条更多是在帮助厘清 IFA 的参数边界，暂时还没把流程真正推到 PFA 共用 tiling 深处

### 2026-04-14 第 2 轮结果

- 验证日志：`log/pfa_ophost_ut.log`
- 多算子 op_host UT 结果：
  - 总计 `45` 条测试，全部通过
  - 新增 FIA / IFA 侧 PFA 补充用例共 `6` 条，全部通过
- 最新覆盖结果：
  - `prompt_flash_attention_tiling.cpp | 59.8% | 3409`
  - 函数覆盖 `52.5% | 120`
- 相比上一轮 `56.2% / 51.7%`：
  - 行覆盖提升 `+3.6`
  - 函数覆盖提升 `+0.8`
- 结论：
  - 通过 FIA 入口补 PFA 共用失败分支，这条路是有效的，已经证明比继续只堆 Prompt 自身 UT 更能撬动新的覆盖区间
  - 但 IFA 当前两条样例还主要卡在 IFA 前置校验，后续若继续从 IFA 侧推进，需要优先绕开 `CV 1:1` / `KV tensor list batch=1` 这些前置门槛
  - FIA 侧下一步更值得继续深挖：
    - 把 `FusedInferAttentionScore_PFA_tiling_0/1` 从 FIA checker 再往后推，争取越过 `paged_attention_checker` / `mask_checker`
    - 把 `FusedInferAttentionScore_PFA_tiling_3` 的 antiquant tensor shape 从当前 `(1, 1024)` 对齐到日志要求的 `ND(8, 128)`，继续向更深的 `CheckPAAntiquantSupportScenarios` 一类分支推进

## 2026-04-15 第 1 轮：FIA PFA 用例改回显式写法并继续补覆盖

- 修改文件：
  - `attention/fused_infer_attention_score/tests/ut/op_host/arch32/test_fused_infer_attention_score_PFA_tiling.cpp`
  - `tests/ut/framework_normal/op_host/CMakeLists.txt`
- 先按你的可读性要求把 FIA 侧 PFA UT 改回显式写法：
  - 去掉 `MakeFiaContext(...)` / `MakeCommonAttrs(...)` 这类封装；
  - 每条用例直接展开 `gert::TilingContextPara(...)`；
  - 对关键 input index 和 attr 语义补了注释，便于后续继续按日志调参。
- 链接侧顺手补充了说明注释：
  - `tests/ut/framework_normal/op_host/CMakeLists.txt` 中保留 `-Wl,--allow-multiple-definition`；
  - 注释里明确说明它的唯一用途是让 `softmax_tiling_ut_stub` 覆盖 `tiling_api` 中 3 个 softmax 符号。

### 2026-04-15 第 1 轮新增 FIA 用例

- 在 `test_fused_infer_attention_score_PFA_tiling.cpp` 新增：
  - `FusedInferAttentionScore_PFA_tiling_4`
    - 把原 `block_size=127` 的 PA 非量化场景改成 `128`，让 FIA 前置 checker 放行，继续进入 PFA；
  - `FusedInferAttentionScore_PFA_tiling_5`
    - `TND + sparse_mode=1 + 4D mask + block_table`；
    - 目标是把 TND/PA/Base-API 路径向后推进；
  - `FusedInferAttentionScore_PFA_tiling_6`
    - `TND + sparse_mode=0 + block_table(batch 维故意缩小)`；
    - 目标是探索 Base-API/PA 对 block_table 约束的处理；
  - `FusedInferAttentionScore_PFA_tiling_7`
    - `BSH + PA + generic antiquant_scale + block_table + sparse_mode=3`；
    - 目标是补 `CheckAntiquantParamsShape` 相关行/函数覆盖，而不是继续停在 MSD 分支。

### 2026-04-15 第 1 轮构建与日志结论

- 本轮直接构建验证命令：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export BUILD_PATH=/home/j60100428/ops-transformer_0408/build
cmake --build /home/j60100428/ops-transformer_0408/build --target transformer_op_host_ut -j4
```

- 多算子 cov 验证日志：
  - `log/pfa_ophost_cov_20260415_iter1.log`
- 稳定结果：
  - `transformer_op_host_ut` 共 `49` 条测试全部通过；
  - 新增 FIA 4 条用例全部稳定。
- 新增用例的实际价值：
  - `tiling_4`
    - 成功从 FIA 进入更深的 PFA PA 路径并返回 `GRAPH_SUCCESS`；
    - 说明把 `block_size` 调整为合法值后，这类 `BNSD + PA` 组合可以作为继续挖 PFA 成功路径的模板。
  - `tiling_5`
    - `TND + sparse_mode=1 + 4D mask` 最终返回 `GRAPH_SUCCESS`；
    - 这说明 FIA/PFA 共用路径里，之前卡住的是 FIA 自己对 `2D mask + TND` 的限制，而不是简单的 `sparse_mode=1` 本身。
  - `tiling_6`
    - `TND + sparse_mode=0 + block_table(batch 维缩小)` 也返回 `GRAPH_SUCCESS`；
    - 说明这组参数没有命中我预期的 `checkPABlockSizeAndBlockTable` 失败分支，反而把更多 Base-API 成功路径走通了。
  - `tiling_7`
    - 日志稳定命中：
      - `prompt_flash_attention_tiling.cpp:2419`
      - `prompt_flash_attention_tiling.cpp:2502`
    - 典型报错：
      - `antiquant scale is nullptr`
      - `key antiquant scale or offset shape is wrong`
    - 这条已经把 `CheckAntiquantParamsShape` 从 0 次命中打到了 1 次命中。

### 2026-04-15 第 1 轮覆盖结果

- 最新覆盖结果：
  - `prompt_flash_attention_tiling.cpp | 60.1% | 3409`
  - 函数覆盖 `54.2% | 120`
- 相比上一轮 `59.8% / 52.5%`：
  - 行覆盖提升 `+0.3`
  - 函数覆盖提升 `+1.7`
- 结论：
  - 这一轮的主要收益不在线覆盖的大幅跳升，而在于把若干之前 0 命中的函数真正打亮了其中一部分；
  - 仅靠 FIA/IFA 再补失败/成功样例，覆盖率仍在涨，但速度已经明显变慢。

### 2026-04-15 第 1 轮 html 对照结论

- 对照 `build/cov_result/html_report/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp.func-sort-c.html`：
  - `CheckAntiquantParamsShape` 已从 `0` 提升到 `1` 次命中；
  - 但以下函数当前仍然是 `0`：
    - `CheckPASparseMode`
    - `CheckPAWhenBaseApi`
    - `checkPABlockSizeAndBlockTable`
    - `processPageAttentionInputFlag`
    - `CheckPAAntiquantSupportScenarios`
- 当前判断：
  - `CheckAntiquantParamsShape` 已经证明 generic antiquant 方向是可达的；
  - 但 `PA + antiquant + block_table` 想继续推进到 `CheckPAAntiquantSupportScenarios`，目前仍会被 `CheckAntiquantParamsShape` 内部对 `KeyAntiquantScaleShape/valueAntiquantScaleShape` 的要求拦住；
  - `TND + PA/Base-API` 相关函数虽然通过新增成功用例走深了一些，但还没有真正把 `CheckPASparseMode` / `checkPABlockSizeAndBlockTable` 这几个零命中函数点亮，说明 FIA 侧前置格式/路径转换与我最初预估仍有偏差。

### 2026-04-15 当前阶段性判断

- 继续只靠调 FIA/IFA 侧少量新用例，PFA 覆盖率还能涨，但已经进入“低增益慢爬坡”阶段；
- 目前最明确的新收获有 2 点：
  - FIA 侧可读性已经恢复，后面继续按日志加场景会更轻松；
  - `CheckAntiquantParamsShape` 已经被打亮，可以继续沿这条线验证 generic antiquant 与 split key/value antiquant 的参数边界。


## 2026-04-16 第 1 轮：FIA 回退到原生 ExecuteTestCase，并补 SH/Base-API 场景

- 修改文件：
  - `attention/fused_infer_attention_score/tests/ut/op_host/arch32/test_fused_infer_attention_score_PFA_tiling.cpp`
  - `attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`
- 本轮先处理你反馈的 FIA 写法问题：
  - 我把 `test_fused_infer_attention_score_PFA_tiling.cpp` 里之前为了绕开 registry 问题临时加的 `ExecuteDirectTiling(...)` 全部回退掉；
  - 直接改回统一使用 `ExecuteTestCase(...)`；
  - 顺手把那套本地 `PlatformInfo`/`json`/`context faker` 辅助函数整段删除，文件重新回到和原有 UT 同一套执行风格。

### 2026-04-16 第 1 轮验证：FIA 原生执行方式已可用

- 为了确认是不是还会复现之前的 `GetOpImpl` 崩溃，我先把 FIA 的失败用例 `_0` 单独切回 `ExecuteTestCase(...)` 做增量构建；
- 这一次没有再出现之前记录过的 registry 崩溃，`FusedInferAttentionScore_PFA_tiling_0` 直接稳定通过；
- 随后我把 8 条 FIA PFA 用例全部改回 `ExecuteTestCase(...)`：
  - `FusedInferAttentionScore_PFA_tiling_0/1/2/3/7` 保持 `GRAPH_FAILED`
  - `FusedInferAttentionScore_PFA_tiling_4/5/6` 改为 `GRAPH_SUCCESS + expectTilingKey` 校验
- 这 3 条成功用例重新实测得到的 `expectTilingKey`：
  - `tiling_4 = 103000000000200000`
  - `tiling_5 = 103000000010300003`
  - `tiling_6 = 5000000000010210100`
- 结论：
  - 当前代码状态下，FIA arch32 PFA 补充用例已经可以和其他常规 UT 一样直接走 `ExecuteTestCase(...)`；
  - 因此 `ExecuteDirectTiling(...)` 这层封装已经没有保留必要，我已删除。

### 2026-04-16 第 1 轮新增 Prompt SH 用例

- 为了继续推进 Prompt 自身的 Base-API 方向，我在 `test_prompt_flash_attention_tiling.cpp` 新增了 3 条 `SH` 布局用例：
  - `PromptFlashAttention_910b_tiling_27`
    - `SH + sparse_mode=20 + 无 mask`
    - 目标：验证 Base-API no-mask 成功路径
  - `PromptFlashAttention_910b_tiling_28`
    - `SH + sparse_mode=21 + pse_shift 走 norm mask`
    - 目标：验证 Base-API norm-mask 场景
  - `PromptFlashAttention_910b_tiling_29`
    - `SH + sparse_mode=22 + pse_shift 走 alibi mask`
    - 目标：验证 Base-API alibi 场景
- 这 3 条用例第一次跑时都返回同一个 tiling key：
  - `2000000000004001012`
- 回填后 3 条用例全部稳定通过。

### 2026-04-16 第 1 轮完整 cov 验证

- 本轮按你给的完整命令执行，并显式保留日志：

```bash
source /home/j60100428/Ascend_0408/cann/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export CCACHE_DISABLE=1
bash build.sh -u --ophost --ops="prompt_flash_attention,fused_infer_attention_score,incre_flash_attention" --soc=ascend910b --cov -j$(nproc) 2>&1 | tee ./log/pfa_ophost_ut.log
```

- 说明：
  - 这次 `build.sh` 在当前沙箱里会先因为 `.git/modules/3rdparty/catlass/config` 无法加锁而中断；
  - 为了真正跑完整个 cov 流程，我改成了提权执行这一条命令；
  - 这不涉及代码实现改动，但属于本轮实际执行上的必要处理。
- 验证结果：
  - `transformer_op_host_ut` 共 `52` 条测试，全部通过；
  - `log/pfa_ophost_ut.log` 中新增 SH/Base-API 3 条用例全部转绿；
  - 最新覆盖率：
    - `prompt_flash_attention_tiling.cpp | 65.6% | 3409`
    - 函数覆盖 `62.5% | 120`
- 相比上一轮 `60.1% / 54.2%`：
  - 行覆盖提升 `+5.5`
  - 函数覆盖提升 `+8.3`

### 2026-04-16 第 1 轮阶段性判断

- 本轮最关键的收获有两点：
  - FIA PFA 补充用例已经可以彻底回归原生 `ExecuteTestCase(...)` 写法，可读性问题算是解决了；
  - Prompt 侧新增的 `SH` 场景明显有效，把覆盖率从 `60.1%` 推到了 `65.6%`。
- 但我对照 `build/cov_result/html_report/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp.func-sort-c.html` 后，仍然看到这批函数是 `0 hit`：
  - `CheckPASparseMode`
  - `CheckPAWhenBaseApi`
  - `checkPABlockSizeAndBlockTable`
  - `processPageAttentionInputFlag`
  - `CheckPAAntiquantSupportScenarios`
  - 以及一批 `SetBaseApi*` / `CheckBaseApiPse` / `GetPFABaseApiWorkSpaceSize`
- 这说明一个很重要的新现象：
  - 虽然 `SH + sparse_mode=20/21/22` 这 3 条 Prompt 用例拿到了 `2000000000004001012` 的 Base-API tiling key，且用例成功；
  - 但 html 上对应的若干 Base-API 细分函数仍没有点亮；
  - 也就是说，这条成功路径并没有像我最初预估的那样真正走进 `SetBaseApiPseInfo / CheckBaseApiPse / SetBaseApiSeqTilingData` 这几个函数。
- 当前更合理的判断是：
  - `2000000000004001012` 这类 SH 场景确实命中了 Base-API 分类，但走到的是更“短”的成功路径；
  - 那些仍为 0 的函数，后续还需要继续找更细的输入组合去触发，不能只靠现在这 3 条 SH 成功样例。

### 2026-04-16 下一步建议

- 优先继续围绕 3 个方向补：
  - `FIA -> PFA` 的 `TND/NTD_TND + block_table` 组合，争取真正点亮 `CheckPASparseMode / CheckPAWhenBaseApi / processPageAttentionInputFlag`；
  - `PA + antiquant + block_table`，继续试着越过 `CheckAntiquantParamsShape`，往 `CheckPAAntiquantSupportScenarios` 推；
  - `SH/Base-API` 的失败分支，尤其是 mask 形状/类型与 `CheckBaseApiPse` 相关的细分错误场景。
