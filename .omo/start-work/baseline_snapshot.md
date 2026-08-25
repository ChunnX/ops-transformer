# Todo 2 基线回归快照（改动前，记录型）

- **计划**: `.omo/plans/mla-align-flash-attn.md` Todo 2（:73-78）
- **日期**: 2026-08-22（UTC 06:00）
- **性质**: 记录型任务，**零 git 提交**
- **范围**: 只跑只记；未改任何产品代码（attention/flash_mla_with_kvcache 等目录零改动）

---

## 1. 运行环境摘要

| 项 | 值 |
|---|---|
| 设备 | Ascend950PR ×1（NPU ID 0，Bus 0000:91:00.0，Health OK） |
| 驱动/npu-smi | 25.7.rc1.6 |
| HBM | 131072 MB（总），空闲 HBM 占用正常 |
| CANN | /home/developer/Ascend/cann-9.1.0（ASCEND_HOME_PATH） |
| bisheng | /home/developer/Ascend/cann-9.1.0/bin/bisheng（PATH 中） |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4（pytests 依赖） |
| cann_ops_transformer | 1.0.0（pip，/opt/buildtools python3.12.9 site-packages），已含 flash_mla_with_kvcache/flash_mla_metadata Python 绑定 |
| 自定义算子包 | Todo 1 产物重建后安装（见 §5 安装说明） |
| 源码基线 | 与 Todo 1 编译通过的 commit 相同，本次未改动任何源码 |

**环境修复记录（运行前发现并解决的历史遗留问题，非本次改动引入）**：
1. `/root/.cache/torch_extensions/py312_cpu/moe_init_routing/lock`（0 字节，2026-08-21 残留）导致 torch `cpp_extension` 的 FileBaton 跨进程编译锁永久等待——pytests import 卡死。删除残留锁后恢复（详见 learnings.md）。

---

## 2. UT 基线（`bash build.sh -u --ophost --ops=flash_mla_with_kvcache --soc=ascend950 -j20`）

- **命令 exit code**: `1`（build.sh 因 UT 断言失败以非零退出；UT 本体编译 PASS）
- **汇总**: 4 suites / **43 用例 = 36 PASS + 7 FAIL**
- 覆盖: `attention/flash_mla_with_kvcache/tests/ut/op_host/`（tiling arch35 + dtype infershape + shape infershape）

### 2.1 失败用例（7）与原因（逐条）

| # | 用例 | 断言位置 | 实测 vs 期望 | 根因归类 |
|---|---|---|---|---|
| 1 | FlashMlaWithKvcacheArch35Tiling.MetadataPassthroughConstantFill | tiling.cpp:123 | `tilingInfo.tilingKey=3672322` ≠ `EncodeMlaTilingKey(false,false)=1575170` | **KEY-1** |
| 2 | FlashMlaWithKvcacheArch35Tiling.SeqLensDecodeFlagAndWorkspace | tiling.cpp:167,182 | ① prefillInfo.tilingKey=3672322 ≠ 期望 1575170；② decodeInfo.workspaceSizes[0]=34078720 ≠ prefillWorkspace+expectedDelta=51380224 | **KEY-1 + WS-2** |
| 3 | TND_Prefill（csv 参数化） | tiling_case_executor.cpp:345 | workspaceSizes=34078720 ≠ csv 期望 16777216 | **KEY-1（key 派生 ws）** |
| 4 | TND_Prefill_MaskCausal（csv） | 同上 | 34078720 ≠ 16777216 | 同上 |
| 5 | TND_Prefill_LSE（csv） | 同上 | 34078720 ≠ 16777216 | 同上 |
| 6 | TND_Prefill_BF16（csv） | 同上 | 34078720 ≠ 16777216 | 同上 |
| 7 | MetadataMultiSectionPassthrough（csv） | 同上 | 34078720 ≠ 16777216 | 同上 |

**根因分析（KEY-1，统一根因）**：
- CSV/UT 固化期望：prefill 系列 `expectTilingKey=1575170`（0x180902，bit21=0）；decode 系列 `3672322`（0x380902，bit21=1）。
- UT 实测：host 侧 **prefill 路径产出的 tilingKey 恒为 3672322（bit21=1，与 decode 相同）** → prefill key 断言失败；同时 key 影响 workspace 计算（key=3672322 对应 workspace=34078720，恰为 CSV 中 decode 的期望值），故 5 个 csv 参数的 prefill 系列 workspace 全部得到 34078720 ≠ 期望 16777216。
- decode 系列（TND_Decode / _LSE / _MaskCausal）全部 PASS（host key/ws 与 CSV 期望一致）。
- **结论：改动前基线下，host 侧 prefill 路径的 tilingKey bit21 与 UT/CSV 固化期望不一致（实测恒 3672322）**。该差异为 Todo 14 行为门的 before 基准要点，详见 `before/tiling_key_values.txt`。

### 2.2 通过用例（36）清单

```
PASS FlashMlaWithKvcacheArch35Tiling.NeedInitShapeDerivedAndEmptyRejected
PASS FlashMlaWithKvcacheArch35Tiling.MetadataMissingRejected
PASS .../TND_Decode, TND_Decode_LSE, TND_Decode_MaskCausal
PASS .../NoMetadata, MaskModeInvalid_1, NoMaskWithAttnMask, HeadDimVInvalid_256,
      QLastDimInvalid_512, KvLayoutNotPaNz, OutLayoutInvalid_TND, KvHeadNumNot1,
      EmptyTensor, NoBlockTable, NoCuSeqlensQ, CuSeqlensQWrongSize, MetadataDtypeInvalid,
      QKvDtypeMismatch, MaskShapeInvalid, BlockTableDim0Mismatch
PASS FlashMlaWithKvcacheInferDTypeTest: inferdtype_FP16/_FP16_LSE/_BF16/_BF16_LSE（4）
PASS FlashMlaWithKvcacheInferShapeTest: TND_PANZ_NTD_FP16/_FP16_LSE/_BF16/_CAUSAL_FP16,
      q_last_dim_512_FP16, k_cache_last_dim_512_FP16, head_dim_v_448_FP16, head_dim_v_0_FP16,
      q_rank_2d_TND_FP16, layout_q_invalid_FP16, layout_out_invalid_FP16（11）
```
（35 条 PASS 明细可在运行日志 `/tmp/opencode/ut_run.log` 溯源；上表为 CTRL 摘要。）

---

## 3. pytests 基线（NPU 上板，改动前）

- **入口**: `attention/flash_mla_with_kvcache/tests/pytests/test_flash_mla_with_kvcache.py`（无 pytest 框架，直接 python3 运行；含 functional_stc + functional_nc 两模块）
- **命令 exit code**: `1`（4 个用例失败 → harness 汇总 fail_cnt=4）
- **汇总**: attn 维度 **10 PASS / 4 FAIL**（14 行 = 11 stc + 2 nc + 1 复用计数，见下表）；LSE 维度全部 PASS；NC 自洽维度全部 PASS；section 断言（SECTION_*）全部通过

### 3.1 失败用例（4）与原因

| 用例 | dtype | MaxAbsErr | FailRatio（阈 0.5%） | 原因 |
|---|---|---|---|---|
| PREFILL_BF16 | bf16 | 0.03125 | 1.8467% | **BF16-1**：bf16 attn_out 精度超阈（kernel 内 bf16 累计误差所致，与 examples bf16 FAIL 同因） |
| DECODE_CAUSAL_BF16 | bf16 | 0.03125 | 1.6724% | BF16-1 |
| NC_DECODE_BF16 | bf16 | 0.03125 | 1.6724% | BF16-1（NC 自洽列 PASS：连续/非连续输入输出位级一致） |
| DECODE_LSE | fp16 | 0.00390625 | 0.5127% | **FP16-EDGE**：fp16 decode 超阈 0.0127pp（0.5127% vs 0.5%），边界超阈；LSE 列 PASS |

**规律**：fp16 全部通过（除 DECODE_LSE 贴线超阈 0.013pp）；bf16 全部超阈（1.67%–1.85%）。与 examples 的 bf16-causal FAIL（2.002%）同数量级同根因。**基线现状：bf16 attn_out 上板精度不满足阈值**——后续 todo 需以此对比，禁止把基线失败记为"新引入"。

### 3.2 通过用例（10）清单

```
NC_PREFILL_FP16 (fp16, 0.2949%), PREFILL_NO_MASK (fp16, 0.2949%), PREFILL_CAUSAL (fp16, 0.2598%),
PREFILL_CAUSAL_LSE (fp16, 0.3109%, LSE PASS), PREFILL_LSE (fp16, 0.2949%, LSE PASS),
DECODE_FD (fp16, 0.2991%), SECTION_64B (sections=3), SECTION_16B_2K (sections=2),
VARLEN_NO_MASK (fp16, 0.2923%), VARLEN_CAUSAL (fp16, 0.3099%)
```

---

## 4. examples 基线（`examples/test_aclnn_flash_mla_with_kvcache.cpp`，NPU 上板）

- 命令: 手动 g++ 编译（build.sh `--run_example ... cust` 分支缺 `-I aclnn` include 的环境级缺陷，见 §6）+ 直接运行
- **exit code**: `1`（summary pass=2 fail=1）
- 结果:

| case | attn_out | softmax_lse | 结论 |
|---|---|---|---|
| fp16-causal-b2 | fail 0.1831% maxAbs=1.654e-4 | 0.0000% maxAbs=6.3e-8 | **PASS** |
| bf16-causal-b2 | fail 2.0020% maxAbs=1.328e-3 | 0.0000% maxAbs=1.1e-7 | **FAIL（BF16-1 同因）** |
| fp16-causal-b8-multisection | fail 0.0000% maxAbs=2.9e-6 | 0.0000% | **PASS**（metadata sectionNum=8 ≥2 断言满足） |

---

## 5. 安装说明（before 产物采集前安装了什么）

按仓内规范（build-install.md）补齐运行依赖：
1. **重建算子包**: `rm -rf build build_out` 后 `bash build.sh --ops=flash_mla_with_kvcache,flash_mla_metadata --soc=ascend950 --pkg -j20`（exit 0；单包含两算子 aclnn 符号，4 个 `aclnnFlashMla*`；因 `--run_example` 触发的 clean 弄脏 build 目录，需先全量重建，metadata 单独构建曾报 autogen 缺失）
2. **安装**: `bash build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=/home/developer/Ascend/custom_opp --force --quiet`（exit 0）→ `/home/developer/Ascend/custom_opp/vendors/custom_transformer/{op_api,op_impl,op_proto,bin}`
3. **环境**: `source /home/developer/Ascend/custom_opp/vendors/custom_transformer/bin/set_env.bash`（设 ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH）

> 注：Todo 1 产物 build_out/*.run 仅含 flash_mla_with_kvcache；examples 依赖 flash_mla_metadata 的 aclnn API，故重建双算子包。

---

## 6. 降级与说明项

- **examples 经 build.sh `--run_example flash_mla_with_kvcache eager cust --soc=ascend950` 触发时编译失败**（build.sh cust 分支编译命令缺 `-I ${ACLNN_INCLUDE_PATH}`，`bfloat16.h:18` 找不到 `opdev/fp16_t.h`）——属 build.sh 环境级 include 缺陷，非产品代码问题。按任务要求"自行读 build.sh 找正确形态"，手工组等价 g++ 命令（补 `-I .../include/aclnn`）编译并运行成功。**before 产物基于手工编译的二进制**（运行时链接 `-lcust_opapi` + rpath）。
- **pytests 首次运行卡死**：torch cpp_extension 残留锁（moe_init_routing/lock）导致 `file_baton.wait` 死等；清锁后恢复（非算子问题，详见 learnings.md）。
- 无其它降级：UT（无 NPU 需求）与上板（exmaples + pytests）均真实执行，产物无伪造。

---

## 7. before 产物清单（供 Todo 4/6/7/8/9 行为门与 Todo 14 key 比对的 before 基准）

```
.omo/start-work/before/
├── example_output.txt        # examples 3 用例完整输出（PASS/FAIL + maxAbs/failRatio 数值摘要）
├── pytests_result.csv        # pytests 14 行逐用例结果（attn/lse/meta_section/nc 各维）
└── tiling_key_values.txt     # tiling key 值（CSV 固化 8 项 + UT 实测 prefill/decode 差异 + bit21/bit18 观察）
```

## 8. 关键结论（供后续 todo）

1. **UT 基线 36/43**，7 个失败统一根因 = host prefill tilingKey bit21 与固化期望不一致（实测 3672322 vs 期望 1575170）。
2. **pytests 上板 10/14**：bf16 全挂（1.67–1.85%）、fp16 全过（DECODE_LSE 贴线 0.5127%）。
3. **examples 2/3**：bf16-causal FAIL（2.002%），与 pytests bf16 同因。
4. **before key 值**: host 实测 prefill=3672322（bit21=1）、decode=3672322 且与 CSV 一致 → Todo 14 的 FastEncodeTilingKeyDirect 等价断言必须以 host 实测（或 UT 固化 3672322 常量）为准，注意 UT/CSV 中 prefill 期望 1575170 与 host 不一致本身即基线漂移点。