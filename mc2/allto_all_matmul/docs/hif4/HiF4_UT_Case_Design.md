# AlltoAllQuantMatmul HiF4 特性 UT 用例设计

| 项目 | 内容 |
| :--- | :--- |
| 被测特性 | 47b981b42 `alltoallmatmul新增hif4全量化特性` |
| 测试层级 | op_host tiling UT / op_api UT / apace 公共组件 UT（新增） |
| 关联设计 | HiF4_Apace_Template_Design.md（§5 §6 §7.6） |
| 用例总数 | 71（正向 36 + 反向 35） |

## 1. 测试范围与框架

### 1.1 测试范围

| 层 | 被测对象 | 覆盖内容 |
| :--- | :--- | :--- |
| op_host | `AllToAllMxQuantMatmulTilingBase`（arch35，socVersion=9201） | IsCapable 准入、校验链（dtype/shape/groupSize/transpose/通路）、SWAT tiling 触发、tilingKey、workspace |
| op_host | `Mc2TilingUtils::InferGroupSize`（isHif4 分支） | HiF4 scale shape → groupSize 推断 |
| op_api | `allto_all_quant_matmul_base.cpp` HiF4 校验链 | CheckNotNull / CheckScaleShape / Check3DScaleShape / CheckHiF4QuantDtypesValidA5 / x2 非连续处理 |
| apace | `QuantMatmulTilingSwat<DT_HIFLOAT4,...>` + `apace_common_utils.h` 泛化函数 | baseK clamp、scale 字节预算、尾块分裂、stepK/scaleKL1、L1 预算合法性 |
| op_kernel | allto_all_matmul_apt.cpp HiF4 分支 | 由编译期模板/static_assert 保证，UT 不直接覆盖（ST 直调覆盖） |

### 1.2 框架与执行

| 项 | 说明 |
| :--- | :--- |
| tiling UT | 复用 `tests/ut/op_host/arch35/test_allto_all_matmul_tiling.cpp` CSV 参数化框架（`AlltoAllMatmulTilingUtParam` + `tiling_case_executor`），新增 HiF4 用例行 |
| op_api UT | 复用 `tests/ut/op_api/test_aclnn_allto_all_quant_matmul.cpp` + CSV，新增 HiF4 用例行 |
| apace UT | **新增** `mc2/common/op_kernel/apace/tests/ut/`（gtest + host 编译），直接调用 host 可编译的 tiling/常量函数 |

**框架前置依赖**：

1. UT 框架 `tiling_case_executor` 需支持 `socVersion=9201`（ascend960dt）平台 mock：`GetCurNpuArch()==DAV_9201`、aicNum/l1Size/l0aSize/l0cSize 平台参数；
2. op_api mock 框架需支持 `soc=Ascend960DT` 与 `DT_HIFLOAT4 / DT_HIFLOAT4_SCALE` dtype 枚举；
3. `expectTilingKey` 数值由 `GET_TPL_TILING_KEY(4, X2TRANSPOSE, 1, 0, COMMTYPE, 1, 0)` 计算，首轮执行时捕获后固化到 CSV（下表以字段向量表达）。

**HiF4 仅 A6 支持（执行状态说明，2026-09）**：HiF4 仅在 A6(ascend960dt) 芯片支持，当前 CANN 包（A5 及之前）缺少 `ge::DT_HIFLOAT4_SCALE(43)` / `ACL_HIFLOAT4(42)` 枚举，op_host/op_api 侧 HiF4 代码无法编译。因此：

- §4 apace 组件 UT（SWAT tiling + 泛化常量）为 host 纯 tiling 代码（仓内 `mm::DataType` 枚举），**A5 包可编译执行，已实测 15/15 通过**（`test_allto_all_matmul_hif4_swat_tiling.cpp`，断言值基于平台 mock AIC=32/L1=1MB/L0C=256KB 校准）；
- §2/§3 端到端用例以**带运行时门控的独立文件**交付（`test_allto_all_matmul_hif4_tiling.{csv,cpp}`、`test_aclnn_allto_all_quant_matmul_hif4.{csv,cpp}`）：参数层做 dtype 桥接（42/43 计划值），A5 包（`ge::DT_MAX<=43`）与框架 SOC_VERSION 未收录 Ascend960DT 时整体 GTEST_SKIP，不阻塞存量用例；A6 devkit + 框架 dtype/soc 表补齐后自动生效；
- 框架侧需同步补充（待 A6 devkit 就位后提交）：`op_host_csv_case_loader.h` GE_DTYPE 表 + `HIFLOAT4/HIFLOAT4_SCALE`、`tiling_case_executor.cpp` archToSoc/socToArch + `9201/Ascend960DT`、`op_api_csv_case_loader.h` ACL_DTYPE 表 + 上述 dtype 与 SOC_VERSION 表 + `Ascend960DT`。

### 1.3 通用参数约定

设 rankDim=R、每卡输出 M=m、总 K 维=K（x2 的 K 轴）、N=n、Ka=K/R：

```
x1      = [m×R, Ka]            HIFLOAT4        ND 连续
x2(transB)  = [N, K]           HIFLOAT4        （DN 布局）
x2(nonTrans)= [K, N]           HIFLOAT4        （ND 布局）
x1Scale = [m×R, Ka/64, 1]      HIFLOAT4_SCALE
x2Scale(transB)   = [N, K/64, 1]    HIFLOAT4_SCALE
x2Scale(nonTrans) = [K/64, N, 1]    HIFLOAT4_SCALE
bias    = [N]                  FLOAT（可选）
y       = [m, N]               BF16（默认，变体 FP16/FLOAT）
attr: x1QuantMode=8, x2QuantMode=8, groupSize=4295032896=(1,1,64),
      transposeX1=false, commMode=ccu（默认）, alltoalloutFlag=false
基准小 shape：R=2, m=128, N=256, K=256（Ka=128）
```

## 2. op_host tiling UT 用例

### 2.1 正向用例（expectStatus=SUCCESS）

| 编号 | 变更点（相对基准） | 覆盖目标 | 期望 tilingKey 字段 |
| :--- | :--- | :--- | :--- |
| UT-TH-01 | 基准（R2/m128/N256/K256/transB/ccu/无bias/y=BF16） | HiF4 最小合法规格端到端 tiling 成功；Ka=128 下限 | (QUANTMODE=4, X2TRANS=1, BIAS=1, K0, COMM=CCU, APACE=1, ADD=0) |
| UT-TH-02 | transposeX2=**false**（ND）：x2=[K,N]=[256,256]，x2Scale=[K/64,N,1]=[4,256,1] | ND 布局 tiling 放行（CheckX2Transpose HiF4 分支）；ND 的 baseN 对齐 64 | (4, X2TRANS=**0**, 1, 0, CCU, 1, 0) ⚠ **已知问题 R1**：当前 CheckMxQuantShapeInfo 按 [N,K] 假定校验 x2Scale 会误拒（报 shape[0] of x2Scale must be K），本用例预期值钉住修复后行为；修复前 expectStatus 暂记 FAILED 并标记 XFAIL |
| UT-TH-03 | R=4，K=512（Ka=128），m=64（x1=[256,128]） | 4 卡准入；mValue 整除 | 同 01 |
| UT-TH-04 | bias=[256] FLOAT | hasBias 传递 SetBiasInfo(true)；tilingKey DTYPEBIAS 恒 1（无变化） | 同 01 |
| UT-TH-05 | y=FLOAT16 / y=FLOAT（拆 2 条） | 输出 dtype 三支持（与 BF16 共 3 条） | 同 01 |
| UT-TH-06 | commMode=**aicpu** | 通信引擎枚举分支；tilingKey COMMTYPE=AICPU | (4,1,1,0,**AICPU**,1,0) |
| UT-TH-07 | m=272（x1=[544,128]，m%256=16 尾块） | M 尾块 tiling（mBlockCnt=2, mTail=16）；AlignCutResForKernel 余数路径 | 同 01 |
| UT-TH-08 | Ka=256（K=512, R2，x2=[256,512]，x2Scale=[256,8,1]） | baseK=256 上限分支（min(Ka,256)） | 同 01 |
| UT-TH-09 | R=8，K=1024，m=32（x1=[256,128]） | rankDim=8 边界（hcomm 上限内） | 同 01 |
| UT-TH-10 | ~~R=3 非 2 幂 rank~~ **已移除**：公共校验 `SUPPORT_RANK_SIZE={2,4,8,16}` 直接拒绝非 2 幂卡数（tiling 层 GetAndValidateRankSize），aclnn 通路无 R=3 用例；R=3 仅 ST 直调框架可测（见 ST 文档 §2.1 注） | — |
| UT-TH-11 | m=16（x1=[32,128]） | baseM=16 最小块 | 同 01 |
| UT-TH-12 | N=272（x2=[272,256]，y=[128,272]） | N 尾块（baseN=256, nTail=16） | 同 01 |

> workspace 断言：`expectWorkspaces` 按 §6.5 公式计算（commLen=AlignUp(m×K/2,512) 等），UT-TH-01 基准值首轮捕获后固化。

### 2.2 反向用例（expectStatus=FAILED）

| 编号 | 构造 | 期望拦截点（错误信息关键片段） |
| :--- | :--- | :--- |
| UT-TH-E01 | alltoalloutFlag=true 且 all2allOut=[128,256] HIFLOAT4 | InitTilingContextParameters："only supports the hcomm path: all2all_out must be empty and rankDim must be no more than 8" |
| UT-TH-E02 | R=16（worldSize=16） | 同上（rankDim 超限） |
| UT-TH-E03 | Ka=192（K=384, R2；x1=[256,192]） | CheckGroupSize："must be 128-aligned (Ka % 128 == 0)" |
| UT-TH-E04 | groupSize=4295032864（MX 的 (1,1,32)） | CheckGroupSize：期望 groupSizeK=64 |
| UT-TH-E05 | x1 dtype=FLOAT8_E4M3FN（mode 8/8） | CheckMxQuantTensorDataType："dtype of x1 must be hifloat4 in HIFLOAT4 quant mode" |
| UT-TH-E06 | x2 dtype=FLOAT4_E2M1（mode 8/8） | 同上（x2） |
| UT-TH-E07 | x1Scale dtype=FLOAT8_E8M0 | "dtype of x1Scale must be DT_HIFLOAT4_SCALE" |
| UT-TH-E08 | x2Scale dtype=FLOAT8_E8M0 | 同上（x2Scale） |
| UT-TH-E09 | x1Scale=[256, 2, **2**]（末维 2，MX 形态） | CheckMxQuantShapeInfo："shape [2] of x1Scale must be 1" |
| UT-TH-E10 | x1Scale=[256, **1**, 1]（dim1≠Ka/64） | "shape [1] of x1Scale must be 2" |
| UT-TH-E11 | x2Scale=[256, 4, **2**] | "shape [2] of x2Scale must be 1" |
| UT-TH-E12 | x1QuantMode=6, x2QuantMode=8（混合） | IsCapable 不命中（6/8 非 MX 非 HiF4）→ tiling 无实现，FAILED |
| UT-TH-E13 | ~~MX 模式 + hif4 dtype~~ **已移除**：实测推演该组合可通过 tiling dtype 列表（MX 列表已含 DT_HIFLOAT4）且后续校验不一定拦截，行为不确定（关联设计 R3），暂不钉用例；待 R3 补显式互斥校验后增加 | — |
| UT-TH-E14 | bias=[256] FLOAT16 | "dtype of bias must be FLOAT" |
| UT-TH-E15 | y=[128,256] INT32 | y dtype 不在 {FP16,BF16,FP32} |
| UT-TH-E16 | transposeX1=true | 公共 CheckAttrsInfo 拒绝（HiF4 不支持 A 转置） |
| UT-TH-E17 | x1Scale 2D=[256, 2] | CheckMxQuantShapeInfo："shape dim of x1Scale must be 3D" |
| UT-TH-E18 | x2（transB）K 维=192：x2=[256,192]（K%64≠0） | "dim(k) of x2 must be divisible by 64" |
| UT-TH-E19 | x1Scale format=FRACTAL_NZ | CheckMxTensorFormat："scale format must be ND" |
| UT-TH-E20 | x1 首维 255（M_total 不能整除 R=2） | 公共 shape 校验（m 整除）失败 |

## 3. op_api UT 用例

框架：`test_aclnn_allto_all_quant_matmul.csv` 新增 HiF4 行，`soc=Ascend960DT`，`x1_quant_mode=8, x2_quant_mode=8, group_size=4295032896`。基准 shape 同 §1.3。

### 3.1 正向（expect_result=SUCCESS）

| 编号 | 变更点 | 覆盖目标 |
| :--- | :--- | :--- |
| UT-API-01 | 基准（transB=true, y=BF16, 无 bias, x1Scale 非空） | HiF4 基本校验链全通过 |
| UT-API-02 | y=FLOAT16 / y=FLOAT（2 条） | 输出 dtype 组合 |
| UT-API-03 | bias=[256] FLOAT | bias 可选输入合法 |
| UT-API-04 | x2 传入 .t() 非连续视图 + transposeX2=false | 自动转连续 + transposeX2 置 true（§5.5） |
| UT-API-05 | transposeX2=false，x2=[K,N] 连续，x2Scale=[K/64,N,1] | op_api 层 ND 放行（CheckScaleShape HiF4 分支；不报 "must be transposed"——MX 才有该限制） |

### 3.2 反向

| 编号 | 构造 | 期望结果 / 拦截点 |
| :--- | :--- | :--- |
| UT-API-E01 | x1ScaleOptional=nullptr | PARAM_NULLPTR："input x1ScaleOptional should not be null" |
| UT-API-E02 | Ka=192（x1=[256,192]） | PARAM_INVALID："Ka (the second dim of x1) must be 128-aligned" |
| UT-API-E03 | x1Scale=[256,2,**2**] | PARAM_INVALID："last dim of x1scale should be 1" |
| UT-API-E04 | x1Scale=[256,**4**,1]（dim1=4≠Ka/64=2） | PARAM_INVALID："second dim of x1Scale should be CeilDiv(Ka, 64)" |
| UT-API-E05 | x2Scale=[**128**,4,1]（N 轴 128≠256） | PARAM_INVALID："n-axis of x2 and x2ScaleDim should be same" |
| UT-API-E06 | x1 dtype=BF16（mode 8/8） | PARAM_INVALID：x1 不在 HIFLOAT4 支持列表 |
| UT-API-E07 | x1=HIFLOAT4, x2=FLOAT4_E2M1 | PARAM_INVALID："x1 and x2 must have same dtype" |
| UT-API-E08 | x1Scale dtype=FLOAT8_E8M0 | PARAM_INVALID：scale 支持列表 |
| UT-API-E09 | y=INT32 | PARAM_INVALID：output 支持列表 |
| UT-API-E10 | bias=FLOAT16 | PARAM_INVALID：bias 支持列表 |
| UT-API-E11 | all2allOut=[128,256] FLOAT16（≠x1 dtype） | PARAM_INVALID："same dtype"（OP_CHECK_DTYPE_NOT_SAME） |
| UT-API-E12 | x2 非连续（.t() 视图）+ transposeX2=**true** | PARAM_INVALID："x2 not contiguous, and set x2 transpose, it is error!" |
| UT-API-E13 | x1Scale 2D=[256,2] | PARAM_INVALID："shape of x1Scale must be 3D" |
| UT-API-E14 | x1QuantMode=6, x2QuantMode=6 + x1/x2=HIFLOAT4 + scale=E8M0 | 校验链不命中 HiF4/MX dtype 组合（MX A5 列表含 HIFLOAT4 但 scale E8M0→ MX 校验链行为），预期 PARAM_INVALID（结合 UT-TH-E13 交叉钉住） |
| UT-API-E15 | x1=[256,128] 但 y=[100,256]（m 不符） | 公共 CheckShapeAAMM 拦截 |

## 4. apace 公共组件 UT（新增）

新增目录 `mc2/common/op_kernel/apace/tests/ut/`，host 侧 gtest（不依赖 AICORE 编译环境的目标函数为 constexpr/普通函数）。

### 4.1 常量与泛化函数（apace_common_utils.h / apace_constant.h）

| 编号 | 断言 | 期望 |
| :--- | :--- | :--- |
| UT-AP-01 | `GetPerGroupSize<DT_HIFLOAT4>()` | 64（MX 各 dtype=32） |
| UT-AP-02 | `GetScaleUnitPerGroup<DT_HIFLOAT4>()` | 4（MX=1） |
| UT-AP-03 | `GetAlignSize<DT_HIFLOAT4>()` | 1（MX=2） |
| UT-AP-04 | `GetShapeWithDataType<DT_HIFLOAT4>(32)` | 64（元素数=字节×2） |
| UT-AP-05 | `GetSizeWithDataType<DT_HIFLOAT4>(65)` | 33（⌈65/2⌉） |
| UT-AP-06 | 常量一致性：HIF4_GROUP_SIZE/HIF4_SCALE_UNIT/TILING_HIF4_MULTI_BASE_SIZE | 64/4/1 |

### 4.2 SWAT tiling（QuantMatmulTilingSwat<DT_HIFLOAT4, DT_HIFLOAT4, DT_FLOAT>）

以 mock `fe::PlatFormInfos`（aicNum/l1Size/l0aSize/l0cSize 按 960dt 配置）驱动 `GetTilingData`：

| 编号 | 输入 (m, n, Ka, transB) | 断言 |
| :--- | :--- | :--- |
| UT-SW-01 | (128, 256, 128, true) | baseK=128；baseM=128（min→align16）；baseN=256；scaleKL1≤K；usedCoreNum≤aicNum |
| UT-SW-02 | (128, 256, 512, true) | baseK=**256**（min(512,256) clamp，4-bit 档） |
| UT-SW-03 | (128, 300, 128, false) | baseN=Align(256,64)=256（ND 对齐 64，nTail=44 触发尾块）；transB=true 时同输入 baseN=256 对齐 16、nTail=44 |
| UT-SW-04 | (272, 256, 128, true) | mTailSize=16；mBlockCnt=2；尾块分裂 mTailTile≥1 |
| UT-SW-05 | (128, 256, 1024, true) | stepK≥1 且 ≤4；K 多 L1 深度循环可行（depthA1×baseK ≥ 预算内） |
| UT-SW-06 | L1 预算合法性（各用例通用断言） | depthA1×(baseA+baseScaleA) + depthB1×(baseB+baseScaleB) + biasL1 + scale 窗口 ≤ l1Size；其中 HiF4 baseScaleA=⌈baseK/64⌉×4×baseM |
| UT-SW-07 | hasBias=true（(128,256,128,true)+bias） | biasL1Size=2×Align(baseN,16)×4；tilingData 其余字段不回归 |
| UT-SW-08 | dbL0c | baseM×baseN×8 ≤ l0cSize 时 dbL0c=2，否则 1（选两组 shape 触发两分支） |
| UT-SW-09 | 复用同一 tiling 对象连续 GetTilingData 两次（不同 shape） | 第二次结果正确（args_/runInfo_ 清零复用，回归 GetTilingData 状态清理） |

### 4.3 InferGroupSize（mc2_tiling_utils，isHif4 分支）

| 编号 | 构造 | 断言 |
| :--- | :--- | :--- |
| UT-IG-01 | isMxfp=true, isHif4=true, x1Scale=[m, Ka/64, 1] | groupSizeK=64（scaleLastDim=1 路径） |
| UT-IG-02 | isMxfp=true, isHif4=false, x1Scale=[m, Ka/64, 2] | groupSizeK=32（MX 回归） |

## 5. 用例汇总与执行状态

| 套件 | 正向 | 反向 | 合计 | 交付形式 | 执行状态（2026-09, A5 包） |
| :--- | :-: | :-: | :-: | :--- | :--- |
| op_host tiling（§2，UT-TH-05 拆 2 条，移除 TH-10/E13） | 12（UT-TH-02 暂 XFAIL） | 19 | 31 | `test_allto_all_matmul_hif4_tiling.csv/.cpp`（门控） | 门控 SKIP（待 A6 devkit） |
| op_api（§3，UT-API-02 拆 2 条，E12 非连续用例需代码级构造暂缓） | 6 | 12 | 18 | `test_aclnn_allto_all_quant_matmul_hif4.csv/.cpp`（门控） | 门控 SKIP（待 A6 devkit） |
| apace 组件（§4：UT-AP 6 + UT-SW 9） | 15 | — | 15 | `test_allto_all_matmul_hif4_swat_tiling.cpp` | **15/15 通过（实测）** |
| **合计** | **33** | **31** | **64** | — | — |

> 注：§1 表头原计数 71 为初版规划；落地时因 SUPPORT_RANK_SIZE={2,4,8,16} 移除 R=3 用例、因 R3 行为不确定移除 E13、op_api E12 暂缓，最终 64 条。

执行顺序建议：UT-AP（纯常量）→ UT-IG → UT-SW → op_host tiling → op_api。全部反向用例需断言错误码与日志关键片段（`OP_LOGE_FOR_INVALID_*` 场景文案），避免仅断言 FAILED 状态导致误拦截（拦截在更早的公共校验）。

### 已知问题联动

- **R1（ND 布局 tiling 误拒）**：UT-TH-02 钉住正确行为（SUCCESS），修复前标记 XFAIL 并在用例名后缀 `//issue-R1`，防止用例被静默跳过；
- **R3（MX+hif4 错误信息不直接）**：UT-TH-E13 / UT-API-E14 当前仅能断言"失败"，修复增强后补断言具体文案。
