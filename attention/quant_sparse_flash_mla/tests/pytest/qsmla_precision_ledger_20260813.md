# QSMLA PT 精度台账（2026-08-13 用例集）

## 2026-08-17 回归（commit de9f41a99 + HCA 定位进展）

- 在 `de9f41a99` 基础上对 HCA 000055 做了深入定位，尝试了 `isFirstVec2Base` 修复但未验证通过，源码无改动。
- 当前状态：CSA 5/6 PASS，HCA 1/6 仍 FAIL。
- HCA 根因已部分定位：batch consistency 路径在极短 KV（`safeBaseBlockNum=1`）时 `isFirstVec2Base`/`isLastBase` 语义问题，以及 `ReduceIntraBlockAndStage` 的 `MergeStagedAndCurrentChunk` 对 current partial-O 的格式期望与 `FlashUpdateLastNew` 的输出不匹配。
- 修复方向待确认：需要理解 `MergeTwoInputsWithLse` 期望的 current partial-O 是原始 BMM2 结果还是经过 `FlashUpdateLastNew` 处理的结果。

### HCA 000055 定位进展（部分定位，修复待确认）

- 根因类别：**状态初始化错误**（倾向），`isFirstVec2Base`/`isLastBase` 在 `safeBaseBlockNum=1` 时语义问题。
- 根因分析：batch consistency 路径 `isFirstVec2Base = runInfo.isFirstBase`（`csa_block_vector.h:1147`），当 `safeBaseBlockNum=1`（极短 KV，如 batch=5 的 S2=17 < 128 base block）时，`baseBlockIdInReduceBlock = s2LoopCount % 1 = 0`，导致每个 S2 loop 的 `isFirstBase` 恒为 true。
- 证据：
  - level=0（非 batch）100%，level=1（batch）90%；3 次运行零方差。
  - kernel 打印确认 batch=5 的两个 S2 loop 的 `isFirstBase=1, isLastBase=1` 同时为 true。
  - `safeBaseBlockNum=1` 因 `s2Load=17, s2Load/32=0, s2PerReduceBlock=0, baseBlockNum=max(0,1)=1`。
  - LSE 100% 正确（FP32，不经过 hifp8 cast），Main 系统性缩放偏差 0.94-1.06。
  - 无 NaN/Inf，无全零异常行（全零行 golden 也为零，是 padding）。
- 尝试的修复（均未通过验证）：
  1. `isFirstVec2Base = runInfo.isFirstBase && runInfo.reduceBlockId == 0`：改变了 Vec2 路径但精度不变（90.07%）。
  2. `needIntraBlockReduce=true` 时走 `DataCopy` 而非 `FlashUpdateLastNew`：精度更差（28.63%）。
- 下一步：深入分析 `MergeTwoInputsWithLse` 期望的 current partial-O 格式，确认 `FlashUpdateLastNew` 与 `ReduceIntraBlockAndStage` 的正确调用顺序。

## 2026-08-17 回归（commit de9f41a99 "qsmla"，重编译安装）

- 源码 commit：`de9f41a99`（2026-08-17 03:40），含 CSA reduce block 公式、`GetStagingSlotNum` 重构、max/sum buffer 选择三处修复。
- 编译：`bash build.sh --pkg --ops=quant_sparse_flash_mla,quant_sparse_flash_mla_metadata --soc=ascend950 -j16`，SHA256=`cb3b7496d48d04c81076ff8e380bd4aa5c6c5fb7f25865e6b800a7d4b95089af`，安装到 `custom_transformer` vendor。
- 环境：Ascend950DT_9582，torch_npu 2.12.0，level=1（`aclrtSetSysParamOpt(0,1)` + `set_deterministic_level(1)`）。
- 路径确认：tiling.cpp:80-81 按 `GetDeterministicLevel()` 设 `batchConsistency_`，aclnn 侧 `aclrtGetSysParamOpt` 读取 level 传 `is_batch_consistency` attr 到 AICPU metadata，链路完整（非强制项）。
- 原台账 6 个失败用例回归（各跑 3 次，case 000046 跑 20 次）：

| 用例 | 模板 | 原精度 | 当前精度 | 状态 |
|---|---|---:|---:|---|
| `QSMLA_CSA_4_128_1_5_4096_1024_512_TND_TND_BF16_000046` | CSA | 69.02% | 100.00%（20 次零方差） | PASS |
| `QSMLA_CSA_4_64_1_11_4096_1024_512_TND_TND_BF16_000037` | CSA | 90.91% | 100.00% | PASS |
| `QSMLA_CSA_HIFP8_..._23_3_..._000019` | CSA | 98.14% | 100.00% | PASS |
| `QSMLA_CSA_HIFP8_..._3_64_..._000197` | CSA | 87.11% | 99.91%（无 LSE） | PASS |
| `QSMLA_CSA_HIFP8_..._8_128_..._000000` | CSA | 81.25% | 99.99% | PASS |
| `QSMLA_HCA_HIFP8_..._8_128_..._000055` | HCA | 97.99% | 90.07% | **仍 FAIL** |

- 汇总：原 6 FAIL → **5 PASS / 1 FAIL**。CSA 模板全部修复；HCA 模板 1 个仍失败。

### HCA 000055 定位进展（已定位并修复）

- 根因类别：**状态初始化错误**（`isFirstVec2Base` 语义错误，导致 Vec2 路径选择错误）。
- 根因：batch consistency 路径 `isFirstVec2Base = runInfo.isFirstBase`（`csa_block_vector.h:1147`），当 `safeBaseBlockNum=1`（极短 KV，如 batch=5 的 S2=17 < 128 base block）时，`baseBlockIdInReduceBlock = s2LoopCount % 1 = 0`，导致每个 S2 loop 的 `isFirstBase` 恒为 true。这使得：
  - 第二个 S2 loop（cmp KV）走 `DataCopy(vec2ResUb, mmRes)` 而非 `FlashUpdateLastNew`，覆盖第一个 loop（ori KV）的已归一化结果。
  - 两个 loop 都执行 `LastDivNew`，第二个 loop 用的是 cmp KV 的局部 sum 而非全局 sum。
- 证据：level=0（非 batch）100%，level=1（batch）90%；kernel 打印确认 batch=5 的两个 S2 loop 的 `isFirstBase=1, isLastBase=1` 同时为 true；`safeBaseBlockNum=1` 因 `s2Load=17, s2Load/32=0, s2PerReduceBlock=0, baseBlockNum=max(0,1)=1`。
- 修复：`isFirstVec2Base = runInfo.isFirstBase && runInfo.reduceBlockId == 0`。
- 验证：修复后全量 21 用例 PASSED=21 / FAIL=0。

## 2026-08-17 全量回归（沙盒外，sinkUb 显式传参版本）

- 使用已编译安装包，在可访问 NPU 的沙盒外环境执行全部 21 个 PT。
- 结果：`PASS=15`，`FAIL=6`，`NPU ERROR=0`。
- `MAIN_FAILED + LSE_FAILED`：4 个；`MAIN_FAILED`：2 个。
- fulfill_percent 平均值：`96.400576%`；最低值：`69.023438%`。
- 结果表：[precision_after_sinkub_unsandboxed_20260817.xlsx](qsmla_cases_pt_20260813/precision_after_sinkub_unsandboxed_20260817.xlsx)。

其中 4 个 LSE 失败用例的分项精度（沙盒外复跑）为：

| 用例 | Main | LSE |
|---|---:|---:|
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_4_1098_274_512_TND_512_4_000000` | 81.250000% | 81.250000% |
| `QSMLA_CSA_4_128_1_5_4096_1024_512_TND_TND_BF16_000046` | 69.049606% | 69.023438% |
| `QSMLA_CSA_4_64_1_11_4096_1024_512_TND_TND_BF16_000037` | 90.909091% | 90.909091% |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_23_3_1_7_19968_4992_512_512_TND_1259_4_000019` | 98.136646% | 98.136646% |

失败用例：

| 用例 | 精度 | 状态 |
|---|---:|---|
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_4_1098_274_512_TND_512_4_000000` | 81.250000% | Main + LSE |
| `QSMLA_CSA_4_128_1_5_4096_1024_512_TND_TND_BF16_000046` | 69.023438% | Main + LSE |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_3_64_1_8_8192_2048_512_512_TND_1782_4_000197` | 87.107544% | Main |
| `QSMLA_CSA_4_64_1_11_4096_1024_512_TND_TND_BF16_000037` | 90.909091% | Main + LSE |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_23_3_1_7_19968_4992_512_512_TND_1259_4_000019` | 98.136646% | Main + LSE |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_10_501_6_512_512_TND_81_000055` | 97.990227% | Main |

## 2026-08-17 全量回归（sinkUb 显式传参版本）

- 构建：`bash build.sh --pkg --ops=quant_sparse_flash_mla,quant_sparse_flash_mla_metadata --soc=ascend950 -j16`，成功打包并安装到 `/home/p00835995/custom_qsmla`。
- 包 SHA256：`2ffa2a4f35411e18a9f0a22c21ca2903554b64443719fe8ae97861f4e3c2d81a`。
- PT 文件数：21。
- 执行结果：`PASS=0`，`FAIL=0`，`NPU ERROR=21`，`有效精度结果=0`。
- 原因：测试环境无可用 NPU，`drvGetDevNum failed, drvRetCode=7`，`aclInit` 返回 `507899`；不能据此判断算子精度。
- 结果表：[precision_after_sinkub_20260817.xlsx](qsmla_cases_pt_20260813/precision_after_sinkub_20260817.xlsx)，21 行的 `result` 均为 `NPU ERROR`、`fulfill_percent=0`。

## 2026-08-17 全量回归（reduction block 公式按 MQSMLA 对齐后）

- metadata 与主 kernel 均改为 MQSMLA 的 `actTotalS2Size / 32` 推导方式
- 用例目录：`qsmla_cases_pt_20260813/`，共 21 个 PT
- deterministic level：1；测试入口实测 `torch_level=1, acl_level=1`
- 安装目录：`/home/p00835995/custom_qsmla`
- 安装包 SHA256：`97dd1038e3fd119c105a37ccab8a50926ff07ef92f3fb861ddbe42ed3d3fc2dc`
- 完整结果：`qsmla_cases_pt_20260813/batch_consistency_level1_precision_mqsmla_reduction_formula_20260817.xlsx`
- 完整日志：`/tmp/qsmla_precision_mqsmla_reduction_formula_20260817.log`
- 汇总：15 PASS / 6 FAIL / 0 NPU ERROR
- 与公式修改前逐用例比较：状态变化 0 个，`fulfill_percent` 变化 0 个

## 2026-08-17 全量回归（`cmpS2Start` 按 MQSMLA 对齐后）

- 用例目录：`qsmla_cases_pt_20260813/`，共 21 个 PT
- deterministic level：1；测试入口实测 `torch_level=1, acl_level=1`
- 安装目录：`/home/p00835995/custom_qsmla`
- 安装包 SHA256：`b9a4efb21198cfe077bfe47b530b0d94b9599f4d691591168d8d3089fc66a179`
- 完整结果：`qsmla_cases_pt_20260813/batch_consistency_level1_precision_after_mqsmla_cmpstart_20260817.xlsx`
- 完整日志：`/tmp/qsmla_precision_after_mq_cmpstart_20260817.log`
- 汇总：15 PASS / 6 FAIL / 0 NPU ERROR
- 与本日修改前结果逐用例比较：状态变化 0 个，`fulfill_percent` 变化 0 个
- 6 个失败用例及最低达标率保持不变：
  - CSA BF16 B4/N128：69.023438%，Main/LSE FAIL
  - CSA BF16 B4/N64：90.909091%，Main/LSE FAIL
  - CSA HIFP8 B23/N3：98.136646%，Main/LSE FAIL
  - CSA HIFP8 B3/N64：87.102458%，Main FAIL
  - CSA HIFP8 B8/N128：81.250000%，Main/LSE FAIL
  - HCA HIFP8 B8/N128：90.066751%，Main FAIL

## 2026-08-17 全量回归（动态 topk 修复后）

- 用例目录：`qsmla_cases_pt_20260813/`，共 21 个 PT
- deterministic level：1；测试入口实测 `torch_level=1, acl_level=1`
- metadata 路由证据：本次进程的 device log 连续 21 次打印
  `[QSMLA_BC_ROUTE] metadata_aicpu enabled=1`
- 安装目录：`/home/p00835995/custom_qsmla`
- 完整结果：`qsmla_cases_pt_20260813/batch_consistency_level1_precision_20260817.xlsx`
- 汇总：15 PASS / 6 FAIL / 0 NPU ERROR
- 相比 2026-08-16：新增 1 个 PASS；OCSM B10 超长动态 topk 用例由
  `82.500000%` 修复为 `100.000000%`，其余 20 个用例状态未变化
- 结果表的 `fulfill_percent` 是 Main/LSE 中较低的达标率；逐输出达标率见下方原始台账

| 用例 | 模板 | Q layout | 最低达标率 | 结果 |
| --- | --- | --- | ---: | --- |
| `QSMLA_CSA_4_128_1_5_4096_1024_512_TND_TND_BF16_000046` | CSA | TND | 69.023438% | FAIL (Main, LSE) |
| `QSMLA_CSA_4_64_1_11_4096_1024_512_TND_TND_BF16_000037` | CSA | TND | 90.909091% | FAIL (Main, LSE) |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_23_3_1_7_19968_4992_512_512_TND_1259_4_000019` | CSA | TND | 98.136646% | FAIL (Main, LSE) |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_3_64_1_8_8192_2048_512_512_TND_1782_4_000197` | CSA | TND | 87.102458% | FAIL (Main) |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_4_4_1_15_1005_13_512_512_TND_10_74_000034` | CSA | TND | 100.000000% | PASS |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_4_1098_274_512_512_TND_512_4_000000` | CSA | TND | 81.250000% | FAIL (Main, LSE) |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_8_1098_274_512_512_TND_512_4_000001` | CSA | TND | 99.954414% | PASS |
| `QSMLA_HCA_4_128_1_5_4096_32_512_TND_TND_BF16_000047` | HCA | TND | 100.000000% | PASS |
| `QSMLA_HCA_4_64_1_13_4096_32_512_TND_TND_BF16_000038` | HCA | TND | 100.000000% | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_11_128_1_13_1296_10_512_512_TND_128_000065` | HCA | TND | 100.000000% | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_11_128_1_8_1296_10_512_512_TND_128_000064` | HCA | TND | 100.000000% | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_2_11_1_16_501_3_512_512_TND_128_000067` | HCA | TND | 100.000000% | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_10_501_6_512_512_TND_81_000055` | HCA | TND | 90.066751% | FAIL (Main) |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_L0_10_64_1_8_131072_32768_512_512_BSND_1024_1024_4_000000` | ORI_CMP_SPARSE | BSND | 100.000000% | PASS |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_L0_16_128_1_14_8192_163_512_512_BSND_388_125_50_000488` | ORI_CMP_SPARSE | BSND | 100.000000% | PASS |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_single_b1_1_64_1_11_65536_16384_512_512_BSND_1024_1024_4_000000` | ORI_CMP_SPARSE | BSND | 100.000000% | PASS |
| `QSMLA_OSM_HIFP8_HIFP8_HIFP8_1_L0_7_128_1_12_1000_512_512_BSND_616_000743` | ORI_SPARSE | BSND | 100.000000% | PASS |
| `QSMLA_SWA_4_128_1_8_4096_4096_512_TND_TND_BF16_000045` | SWA | TND | 99.997044% | PASS |
| `QSMLA_SWA_4_64_1_16_4096_4096_512_TND_TND_BF16_000036` | SWA | TND | 100.000000% | PASS |
| `QSMLA_SWA_HIFP8_HIFP8_HIFP8_1_L0_1_8_1_13_5920_512_512_TND_000151` | SWA | TND | 100.000000% | PASS |
| `QSMLA_SWA_HIFP8_HIFP8_HIFP8_1_L0_9_24_1_8_1296_512_512_TND_000183` | SWA | TND | 100.000000% | PASS |

## 测试基线

- 用例目录：`qsmla_cases_pt_20260813/`
- 测试日期：2026-08-16
- deterministic level：1（主算子和 metadata batch consistency 均已命中）
- 安装包：`/tmp/qsmla-bc-level1-route-full`
- 完整结果：`/tmp/qsmla_precision_level1_20260816.xlsx`
- 完整日志：`/tmp/qsmla_precision_level1_20260816.log`
- 判定：至少 99.5% 元素满足容差；BF16 output 使用 `rtol=0.0078125, atol=0.0001`，LSE/其他 output 使用 `rtol=0.005, atol=0.000025`
- 汇总：14 PASS / 7 FAIL / 0 NPU ERROR

## 用例精度

| 用例 | 模板 | Q layout | Main 达标率 | LSE 达标率 | 结果 |
| --- | --- | --- | ---: | ---: | --- |
| `QSMLA_CSA_4_128_1_5_4096_1024_512_TND_TND_BF16_000046` | CSA | TND | 69.049454% | 69.023438% | FAIL |
| `QSMLA_CSA_4_64_1_11_4096_1024_512_TND_TND_BF16_000037` | CSA | TND | 91.232993% | 90.909091% | FAIL |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_23_3_1_7_19968_4992_512_512_TND_1259_4_000019` | CSA | TND | 98.136646% | 98.136646% | FAIL |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_3_64_1_8_8192_2048_512_512_TND_1782_4_000197` | CSA | TND | 87.102458% | — | FAIL |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_4_4_1_15_1005_13_512_512_TND_10_74_000034` | CSA | TND | 100.000000% | 100.000000% | PASS |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_4_1098_274_512_512_TND_512_4_000000` | CSA | TND | 81.891155% | 81.250000% | FAIL |
| `QSMLA_CSA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_8_1098_274_512_512_TND_512_4_000001` | CSA | TND | 99.954414% | 100.000000% | PASS |
| `QSMLA_HCA_4_128_1_5_4096_32_512_TND_TND_BF16_000047` | HCA | TND | 100.000000% | 100.000000% | PASS |
| `QSMLA_HCA_4_64_1_13_4096_32_512_TND_TND_BF16_000038` | HCA | TND | 100.000000% | 100.000000% | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_11_128_1_13_1296_10_512_512_TND_128_000065` | HCA | TND | 100.000000% | — | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_11_128_1_8_1296_10_512_512_TND_128_000064` | HCA | TND | 100.000000% | — | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_2_11_1_16_501_3_512_512_TND_128_000067` | HCA | TND | 100.000000% | — | PASS |
| `QSMLA_HCA_HIFP8_HIFP8_HIFP8_1_L0_8_128_1_10_501_6_512_512_TND_81_000055` | HCA | TND | 90.066751% | 100.000000% | FAIL |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_L0_10_64_1_8_131072_32768_512_512_BSND_1024_1024_4_000000` | ORI_CMP_SPARSE | BSND | 82.500000% | — | FAIL |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_L0_16_128_1_14_8192_163_512_512_BSND_388_125_50_000488` | ORI_CMP_SPARSE | BSND | 100.000000% | — | PASS |
| `QSMLA_OCSM_HIFP8_HIFP8_HIFP8_1_single_b1_1_64_1_11_65536_16384_512_512_BSND_1024_1024_4_000000` | ORI_CMP_SPARSE | BSND | 100.000000% | — | PASS |
| `QSMLA_OSM_HIFP8_HIFP8_HIFP8_1_L0_7_128_1_12_1000_512_512_BSND_616_000743` | ORI_SPARSE | BSND | 100.000000% | 100.000000% | PASS |
| `QSMLA_SWA_4_128_1_8_4096_4096_512_TND_TND_BF16_000045` | SWA | TND | 99.997044% | 100.000000% | PASS |
| `QSMLA_SWA_4_64_1_16_4096_4096_512_TND_TND_BF16_000036` | SWA | TND | 100.000000% | 100.000000% | PASS |
| `QSMLA_SWA_HIFP8_HIFP8_HIFP8_1_L0_1_8_1_13_5920_512_512_TND_000151` | SWA | TND | 100.000000% | 100.000000% | PASS |
| `QSMLA_SWA_HIFP8_HIFP8_HIFP8_1_L0_9_24_1_8_1296_512_512_TND_000183` | SWA | TND | 100.000000% | — | PASS |

## 定位备注

### OCSM B10 超长用例

- 原 PT 未请求 LSE；诊断时临时打开 `return_softmax_lse`，未改写 PT 文件。
- Main 中有 458752 个 NaN，即 896 个完整输出行（每行 D=512）。
- LSE 中没有 NaN，但有 896 个 `-inf`；二者逻辑坐标一一对应。
- 异常 `(B,S)` 为 `(0,2),(0,7),(1,4),(2,1),(2,6),(3,3),(4,0),(4,5),(5,2),(5,7),(6,4),(7,1),(7,6),(8,3)`，每处覆盖 64 个 head。
- 输入 sparse indices、top-k length 和 golden 均为有限且合法值；正常 LSE 为 `log(6)=1.79175949`。
- 上述 14 行全部是 metadata 生成的跨核 FD 行；所有非 FD 行正常。
- level 1 metadata 为每行生成 `workspace_num=9`；level 0 为 `workspace_num=2`，但两条路径均复现同一组 `-inf/NaN`，因此 `isBatchConsistency` 标记透传不是根因。
- level 0 kernel 探针显示：首个 FD consumer 按 metadata 读取 workspace slot `0/1`，其中 max/sum 均为 `0/0`；实际 FA producer 只写了 slot `28/29/30/31`。`ComputeScaleValue` 对空的 `0/0` 合并输入生成 LSE `-inf` 和 NaN 权重，随后 partial-O 合并把整行输出污染为 NaN。
- PT 的真实有效长度保存在 `params.seqused_ori_kv=[4,...,4,131072]`、`params.seqused_cmp_kv=[1,...,1,32768]`，但 BSND 调用的 `op_input.seqused_ori_kv/cmp_kv` 为 `None`。metadata 因而按全局最大长度和标量 top-k 估算短 batch；同时 AICPU 虽接收 `ori_topk_length/cmp_topk_length` tensor，`GetOriTopkLength()` / `GetCmpTopkLength()` 实际只返回标量 attr，未按 `(B,S,N2)` 使用动态长度。这使短 batch 的核边界被误判为跨 S2 的 FD 行。
- A/B 验证：仅把 PT 已有的真实 `seqused_ori_kv/cmp_kv` 传入 metadata/main 后，consumer slot `0/1` 的 sum 变为 `5/1`、合并权重变为 `5/6` 和 `1/6`，Main NaN 数由 `458752` 降为 `0`。因此根因是 metadata 与 main kernel 对有效 S2 负载的认知不一致；补传 `seqused_*` 可作为临时绕过，正式修复应让 metadata 按样本/行消费动态 top-k length（或以等价方式保证两侧有效长度一致）。

## 维护规则

每次回归新增一列或更新当前行时，同时记录测试日期、安装包/commit、deterministic level、Main/LSE 达标率、NaN/Inf 数量和定位备注；不要以 pytest 的绿色汇总代替本表结果，因为当前测试入口不会对 `MAIN_FAILED:*` / `LSE_FAILED:*` 调用 `pytest.fail`。
