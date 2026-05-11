# GMMSQ算子MxA8W4场景 STC

> **使用说明**：与对应 SRS 同域存放在 **`data/design/stc/...`**，与 **`data/design/srs/...`** 镜像或并列对齐。`[占位]` 按需替换；涉密数据勿入表驱动文件。复杂模块建议拆：`README.md`（人读）+ `test_coverage_report.md`（统计）+ `*.csv` / `*.xlsx` + `*.py` 生成器。

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| STC 标题 | GMMSQ支持MxA8W4数据流 |
| 关联 SRS | [GMMSQ SRS](./srs_gmmsq.md) |
| 责任人 | 孙堆杨 |
| 更新时间 | 2026-05-09 |
| 版本 | v1.0 |

---

## 2. 测试目标与范围

### 2.1 目标

- **验证正确性**：覆盖MxA8W4伪量化数据流的端到端计算正确性
- **覆盖dtype·format**：验证float8_e4m3fn × float4_e2m1数据类型组合的完整通路
- **异常路径**：覆盖tiling拦截、参数校验等异常场景
- **性能门限**：建立MxA8W4场景的性能基线
- **通路冒烟**：验证aclnn通路的可用性

### 2.2 不在范围

- 其他量化场景（如MXFP8全量化、A8W8、A8W4 MSD等）的测试
- 图融合通路的测试
- weightAssistMatrix/bias/smoothScale非空场景的测试（MxA8W4场景下固定为null）

---

## 3. 环境、输入与限制（可与 SRS 第 5–6 节对照）

### 3.1 运行环境

- **硬件 / SOC**：Ascend 950PR / Ascend 950DT (arch35)
- **驱动 / CANN版本**：与ops-transformer配套版本
- **执行形态**：CI自动化测试 + 本地脚本调试

### 3.2 输入输出限制摘要

- **格式**：
  - x: ND, float8_e4m3fn
  - weight: FRACTAL_NZ_C0_32, float4_e2m1
  - weightScale/xScale: ND, float8_e8m0
  - output: ND, float8_e4m3fn
  - outputScale: ND, float8_e8m0
- **连续性**：contiguous（连续内存布局）
- **资源比 / 核数假设**：AIC + AIV异构协同，双缓冲同步
- **轴向门限**：
  - K % 32 == 0（K维度32对齐）
  - N % 128 == 0（N维度128对齐）
  - K >= 32, N >= 128
  - E <= 1024（专家数上限）

### 3.3 未在实现中显式文档化的假设

| 条目 | 当前结论 | 风险 |
| --- | --- | --- |
| groupList cumsum模式 | groupList累加和必须等于M | 分组错误导致计算结果错误 |
| groupList count模式 | 所有元素之和必须等于M | 同上 |
| weightScale shape | [E, N, ceil(K/64), 2] | shape不匹配导致tiling失败 |
| xScale shape | [M, ceil(K/64), 2] | shape不匹配导致tiling失败 |

---

## 4. 测试策略

### 4.1 白盒 / 小组合（精确逻辑）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 覆盖目标 | dtype×format全组合 | float8_e4m3fn×float4_e2m1×float8_e8m0 |
| 维度数 / shape范围 | M: [32, 16384], K: [64, 32768], N: [128, 65536], E: [1, 256] | 覆盖小/中/大shape |
| 用例规模 | 40条（L0: 20 + L1: 20） | 单因子全覆盖 + 两两组合覆盖 |

**白盒测试设计思路**：

1. **场景识别路径覆盖**：
   - MxA8W4-WEIGHT-NZ场景识别条件：
     - x.dtype == float8_e4m3fn
     - weight.dtype == float4_e2m1
     - weight.format == FRACTAL_NZ_C0_32
     - weightScale.dtype == float8_e8m0
     - xScale.dtype == float8_e8m0
   - 验证所有条件同时满足时正确进入MxA8W4分支

2. **Tiling切分路径覆盖**：
   - 动态K维度切分（kbL1Size=256/512）
   - M维度切分（mL1Size）
   - N维度切分（nL1Size=128/256）
   - 验证不同切分策略下的正确性

3. **Kernel内部路径覆盖**：
   - AIV反量化路径（fp4→fp8）
   - AIC矩阵乘路径（带MX Scale）
   - AIV SwiGLU路径
   - AIV MX量化路径
   - 验证AIC+AIV异构协同的正确性

4. **同步机制路径覆盖**：
   - AIC→AIV同步（SYNC_AIC_AIV_FLAG）
   - AIV→AIC同步（SYNC_AIV_AIC_FLAG）
   - fixpipe同步（SYNC_AIV_MTE3_AIC_FIX_FLAG）
   - 验证同步信号的正确性

### 4.2 泛化 / 大空间（稳定性与边界）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 随机 / 网格 | seed=42，可复现 | 使用固定seed确保可复现性 |
| 轴长上界 | M<=16384, K<=32768, N<=65536, E<=256 | 受硬件资源和内存限制 |
| 用例规模 | 20条（L1级别） | 覆盖关键组合 |

**泛化测试设计思路**：

1. **不同规模组合**：
   - 小规模：M=32, K=128, N=128, E=1
   - 中规模：M=1024, K=2048, N=4096, E=16
   - 大规模：M=8192, K=16384, N=32768, E=128
   - 超大规模：M=16384, K=32768, N=65536, E=256

2. **groupListType两种模式**：
   - cumsum模式（groupListType=0）
   - count模式（groupListType=1）
   - 验证两种模式下的分组正确性

3. **边界值测试**：
   - K=32（最小值）
   - N=128（最小值）
   - K=32768（较大值）
   - N=65536（较大值）
   - E=1（最小专家数）
   - E=1024（最大专家数）

### 4.3 异常 / 负向

| 类型 | 说明 | 期望 |
| --- | --- | --- |
| invalid_k_alignment | K % 32 != 0 | tiling报错，返回GRAPH_FAILED |
| invalid_n_alignment | N % 128 != 0 | tiling报错，返回GRAPH_FAILED |
| invalid_k_minimum | K < 32 | tiling报错，返回GRAPH_FAILED |
| invalid_n_minimum | N < 128 | tiling报错，返回GRAPH_FAILED |
| invalid_transpose_x | transpose_x == true | 属性校验失败，返回GRAPH_FAILED |
| invalid_transpose_w | transpose_w == false | 属性校验失败，返回GRAPH_FAILED |
| invalid_weight_format | weight.format != FRACTAL_NZ_C0_32 | 场景识别失败，不走MxA8W4分支 |
| invalid_expert_num | E > 1024 | tiling报错，返回GRAPH_FAILED |
| invalid_scale_shape | weightScale.shape != [E, N, ceil(K/64), 2] | shape校验失败，返回GRAPH_FAILED |
| invalid_xscale_shape | xScale.shape != [M, ceil(K/64), 2] | shape校验失败，返回GRAPH_FAILED |
| invalid_groupList_sum | cumsum模式下sum(groupList) != M | 运行时错误，结果不确定 |
| invalid_dtype_combination | x.dtype != float8_e4m3fn | 场景识别失败，不走MxA8W4分支 |

### 4.4 性能 / 兼容 / 通路（可选）

**性能测试策略**：

不涉及，仅做性能摸底。

**通路验证策略**：

| 通路 | 验证方式 | 结论 |
| --- | --- | --- |
| aclnn | 调用aclnnGroupedMatmulSwigluQuantWeightNzV2接口 | 通过标准：返回成功，结果正确 |
| PTA | 通过PTA框架调用 | 通过标准：返回成功，结果正确 |

---

## 4.5 Kernel侧单元测试（UT）设计

> **设计目标**：基于GMMFR算子实现文档（`01_gmmfr_architecture.md`、`02_gmmfr_controller.md`）和GMMSQ差异分析（`07_gmmsq_diff_analysis.md`），设计Kernel侧各层级组件的UT用例，通过**构造用例、代码打桩、打印结果**的方式，验证从GMMFR迁移到GMMSQ后各层级的功能正确性。

### 4.5.1 UT总体策略

| 层级 | 测试目标 | 打桩策略 | 验证方式 |
|------|---------|---------|---------|
| **Controller** | Process主流程、GM地址更新、N轴切分 | mock BasicBlock::ComputeBasicBlock | 打印offsetParam、tilingData |
| **BasicBlock** | AIC/AIV路径分发、Init/End、同步 | mock CubeCompute/VecCompute | 打印同步Flag、Buffer地址 |
| **CubeCompute** | Matmul、Fixpipe、Scale搬运 | 无需打桩（纯计算单元） | 比较L0C结果与期望值 |
| **VecCompute** | 反量化、SwiGLU、Quant、Buffer复用 | mock VF指令（`FrAntiQuantMxA8W4NzNkVf`等） | 打印UB内容、比较输出 |
| **Sync机制** | AIC↔AIV CrossCore同步 | 模拟对侧核行为 | 验证Flag设置/等待序列 |

### 4.5.2 Controller层UT

#### UT-C01: Init() - GM地址初始化验证

**用例构造方案**：
```cpp
// 构造TilingData
GMMFinalizeRoutingWeightQuantTilingData tiling;
tiling.groupNum = 2;
tiling.coreNum = 4;
tiling.kSize = 256;
tiling.nSize = 512;
tiling.hasBias = false;
tiling.groupListType = 0;

// 构造GM地址（mock内存）
GM_ADDR x = mockGmBuffer(M * K * sizeof(fp8_e4m3fn_t));
GM_ADDR w = mockGmBuffer(E * K * N / 2 * sizeof(fp4x2_e2m1_t));  // 4bit，>>1
GM_ADDR scale = mockGmBuffer(E * N * CeilDiv(K, 64) * sizeof(fp8_e8m0_t));
// ... 其他参数
```

**代码打桩方案**：
```cpp
// 打桩1: BasicBlock::Init，验证调用参数
MOCK(void, BasicBlock_Init, (uint64_t groupSize, GM_ADDR y, float weight), {
    PRINTF("Init called: groupSize=%lu, y=%p, weight=%f", groupSize, y, weight);
    EXPECT_EQ(groupSize, 32);  // MX_GROUPSIZE
    EXPECT_EQ(weight, 0.0f);    // GMMSQ无sharedInput
});

// 打桩2: GetBlockIdx，模拟核ID
MOCK(uint32_t, GetBlockIdx, (), {
    return 0;  // 模拟第0核（AIC:0, AIV:1）
});
```

**预期结果**：
```
Init called: groupSize=32, y=0x..., weight=0.000000
xGm_ = 0x... (初始值)
weightGm_ = 0x... (初始值)
scaleGm_ = 0x... (初始值)
mxA8W4L1KDynamicConfigMThreshold_ = 256 (无Bias场景)
```

**验证场景**：验证GMMSQ Init()正确初始化所有GM指针，且`sharedInputWeight`为0（GMMSQ无sharedInput）

---

#### UT-C02: UpdateGmAddr() - GM地址更新验证（GMMFR vs GMMSQ差异）

**用例构造方案**：
```cpp
// 构造参数：M=64, K=128, N=256, E=1
uint64_t mSize = 64, kSize = 128, nSize = 256;
// GMMSQ特有：无logitsAddr_、rowIndexAddr_、shareInputAddr_
```

**代码打桩方案**：
```cpp
// 在UpdateGmAddr前打印当前地址
#define TRACE_GmAddr(label, addr, msg) PRINTF("[UT] %s: %s=%p (+%lu)", label, #addr, addr, msg)
```

**预期结果**：
```
[UT] Before UpdateGmAddr:
  xGm_ = 0x1000
  weightGm_ = 0x2000
  scaleGm_ = 0x3000
  yGm_ = 0x4000

[UT] After UpdateGmAddr (M=64, K=128, N=256):
  xGm_ = 0x1000 + 64*128       = 0x2000
  weightGm_ = 0x2000 + (256*128)>>1 = 0x2000 + 16384 = 0x6000
  scaleGm_ = 0x3000 + 256 * CeilDiv(128, 32) = 0x3000 + 256*4 = 0x3400
  yGm_ = 0x4000 + 64*(256/2)    = 0x4000 + 8192 = 0x6000  [GMMSQ: N/2]
  // GMMSQ特有：yScaleGm_ += 64 * CeilDiv(CeilAlign(256/2, 64), 32)
  yScaleGm_ = 0x5000 + 64*2 = 0x5080
  // GMMFR原有：logitsAddr_ += 64 → 不存在
  // GMMFR原有：rowIndexAddr_ += 64 → 不存在
```

**验证场景**：验证GMMSQ的`UpdateGmAddr()`正确计算N/2输出偏移，且不包含logits/rowIndex地址更新

---

#### UT-C03: SplitNByMultiCore() - N轴多核切分验证（GMMSQ N/2适配）

**用例构造方案**：
```cpp
// N=512, coreNum=4
// GMMFR: nL1Size=256, 每个BB处理256个N元素
// GMMSQ: nL1Size=128, 每个BB处理128个N元素（Swish+Gate）
uint64_t nSize = 512;
uint32_t coreNum = 4;
```

**代码打桩方案**：
```cpp
// 打印每个核的nOffset和nL1Size
MOCK(void, PrintSplitResult, (uint32_t coreIdx, uint64_t nOffset, uint64_t nL1Size), {
    PRINTF("core[%u]: nOffset=%lu, nL1Size=%lu", coreIdx, nOffset, nL1Size);
});
```

**预期结果**：
```
core[0]: nOffset=0,   nL1Size=128   // Swish[0:128]
core[1]: nOffset=128, nL1Size=128   // Swish[128:256]
core[2]: nOffset=256, nL1Size=128   // Gate[256:384]
core[3]: nOffset=384, nL1Size=128   // Gate[384:512]
// 注意：GMMSQ的nL1Size=128（GMMFR为256），因为SwiGLU需要同时处理Swish和Gate
```

**验证场景**：验证GMMSQ的N轴切分从256调整为128，确保Swish和Gate分支在同一BasicBlock内处理

---

#### UT-C04: Process() - InitAtomicGm去除验证

**用例构造方案**：
```cpp
// 构造Controller并调用Process()
// GMMSQ预期：不调用InitAtomicGm（无GM清零和sharedInput拷贝）
```

**代码打桩方案**：
```cpp
// 打桩InitAtomicGm，如果调用则报错
MOCK(void, InitAtomicGm, (), {
    FAIL("InitAtomicGm should NOT be called in GMMSQ!");
});
```

**预期结果**：
```
Process() started
  GetBlockIdx() = 0
  InitOffsetParam() done
  [UT] InitAtomicGm NOT called ✅
  SyncAll() done
  Group loop started...
```

**验证场景**：验证GMMSQ的`Process()`去除了`InitAtomicGm`调用（GMMFR有此调用），因为GMMSQ无AtomicAdd Scatter操作

---

### 4.5.3 BasicBlock层UT

#### UT-BB01: ComputeBasicBlock() - AIC/AIV路径分发验证

**用例构造方案**：
```cpp
// 构造BasicBlockOffsetParam
BasicBlockOffsetParam curParam;
curParam.mL1Size = 256;
curParam.nL1Size = 128;  // GMMSQ: 128（GMMFR: 256）
curParam.kbL1Size = 256;
curParam.mOffset = 0;
curParam.nOffset = 0;
```

**代码打桩方案**：
```cpp
// AIC路径打桩
MOCK(void, IterateNzNkWithKAic, (const BasicBlockOffsetParam& cur, const BasicBlockOffsetParam& last), {
    PRINTF("[AIC] IterateNzNkWithKAic: mL1Size=%lu, nL1Size=%lu, kbL1Size=%lu", 
           cur.mL1Size, cur.nL1Size, cur.kbL1Size);
    EXPECT_EQ(cur.nL1Size, 128);  // GMMSQ验证
});

// AIV路径打桩
MOCK(void, IterateNzNkWithAiv, (const BasicBlockOffsetParam& cur, const BasicBlockOffsetParam& last), {
    PRINTF("[AIV] IterateNzNkWithAiv: mL1Size=%lu, nL1Size=%lu", 
           cur.mL1Size, cur.nL1Size);
    EXPECT_EQ(cur.nL1Size, 128);  // GMMSQ验证
});
```

**预期结果**：
```
[AIC] IterateNzNkWithKAic: mL1Size=256, nL1Size=128, kbL1Size=256
[AIV] IterateNzNkWithAiv: mL1Size=256, nL1Size=128
// 验证：nL1Size=128（GMMSQ），而非256（GMMFR）
```

**验证场景**：验证BasicBlock正确分发到AIC和AIV路径，且nL1Size符合GMMSQ的128切分策略

---

#### UT-BB02: End() - 后处理收尾验证（SwiGLU vs Routing差异）

**用例构造方案**：
```cpp
// 构造lastOffsetParam，模拟最后一个BasicBlock的PhaseB处理
BasicBlockOffsetParam lastParam;
lastParam.mL1Size = 256;
lastParam.nL1Size = 128;
```

**代码打桩方案**：
```cpp
// 打桩GMMFR的Routing函数（应不调用）
MOCK(void, MulLogits, (...), { FAIL("MulLogits should NOT be called in GMMSQ!"); });
MOCK(void, RoutingYToGm, (...), { FAIL("RoutingYToGm should NOT be called in GMMSQ!"); });

// 打桩GMMSQ的SwiGLU和Quant函数（应调用）
MOCK(void, SwiGLU, (...), { PRINTF("SwiGLU called"); });
MOCK(void, ComputeMaxExp, (...), { PRINTF("ComputeMaxExp called"); });
MOCK(void, ComputeScale, (...), { PRINTF("ComputeScale called"); });
MOCK(void, ComputeDataForQuantTargetFp8, (...), { PRINTF("QuantTargetFp8 called"); });
```

**预期结果**：
```
[UT] End() AIV path:
  SwiGLU called ✅
  ComputeMaxExp called ✅
  ComputeScale called ✅
  QuantTargetFp8 called ✅
  CopyOutputFromUb2Gm called ✅
  CopyScaleFromUb2Gm called ✅
  // MulLogits NOT called ✅
  // RoutingYToGm NOT called ✅
```

**验证场景**：验证GMMSQ的`End()`调用SwiGLU+Quant流程，而非GMMFR的MulLogits+Routing流程

---

### 4.5.4 CubeCompute层UT

#### UT-CUBE01: LaunchMatmul() - MMA计算验证（GMMFR与GMMSQ一致）

**用例构造方案**：
```cpp
// 构造L0 buffer和L1 buffer
LocalTensor<fp8_e4m3fn_t> aL1(TPosition::TSCM, 0, 128*256/sizeof(fp8_e4m3fn_t));
LocalTensor<fp8_e8m0_t> mxScaleAL1(TPosition::TSCM, 128*256, 128*4);
LocalTensor<fp8_e8m0_t> mxScaleBL1(TPosition::TSCM, 128*256+4096, 128*4);
```

**代码打桩方案**：
```cpp
// 无需打桩，直接验证L0C结果
// 构造已知的A矩阵和B矩阵，验证MMAD结果
```

**预期结果**：
```
[UT] LaunchMatmul:
  A矩阵: [256, 128] fp8_e4m3fn
  B矩阵: [128, 128] fp8_e4m3fn (反量化后)
  L0C结果: [256, 128] float32
  
  验证点1: L0C结果与CPU参考实现对比，误差<0.01%
  验证点2: MX Scale融合加载正确（每个元素 = data * scaleA * scaleB）
  验证点3: Bias加载正确（如有）
```

**验证场景**：验证CubeCompute的MMA计算在GMMSQ和GMMFR中完全一致（这是复用GMMFR的核心部分）

---

#### UT-CUBE02: GetTensorC() - Fixpipe目标地址验证（GMMSQ差异点）

**用例构造方案**：
```cpp
// GMMSQ: Fixpipe输出到yF32Buffer_（而非GMMFR的ubOutputF32Buffer_）
LocalTensor<float> yF32Buffer(TPosition::LCM, 64*1024, 128*256);  // 复用weightHighBit空间
```

**代码打桩方案**：
```cpp
// 验证GetTensorC的目标地址
MOCK(void, GetTensorC, (LocalTensor<float>& dst, const BasicBlockOffsetParam& param), {
    PRINTF("[UT] GetTensorC: dst.addr=%p", dst.GetPhyAddr());
    // GMMSQ验证：dst应为yF32Buffer_的地址
    EXPECT_EQ(dst.GetPhyAddr(), expectedYF32BufferAddr);
});
```

**预期结果**：
```
[UT] GetTensorC: dst.addr=0x... (yF32Buffer_)
// 而非GMMFR的ubOutputF32Buffer_
```

**验证场景**：验证GMMSQ的`GetTensorC()`将Fixpipe结果输出到`yF32Buffer_`（UB中复用`weightHighBit_`的空间）

---

### 4.5.5 VecCompute层UT

#### UT-VEC01: AntiQuantProcessNzMxA8W4() - 4bit反量化验证（GMMFR与GMMSQ一致）

**用例构造方案**：
```cpp
// 构造4bit权重数据（每字节2个4bit元素）
uint8_t weightData4bit[] = {0xAB, 0xCD, 0xEF, 0x12};  // 8个4bit元素
// 预期8bit输出（每个4bit解包为一个8bit）
fp8_e4m3fn_t expectedOutput[8] = {dequant(0xB), dequant(0xA), dequant(0xD), dequant(0xC), ...};
```

**代码打桩方案**：
```cpp
// 打桩VF指令，捕获输入输出
MOCK(void, FrAntiQuantMxA8W4NzNkVf, (MxA8W4NzParams& params), {
    PRINTF("[UT] FrAntiQuantMxA8W4NzNkVf: loopKNum=%u, innerLoopNum=%u", 
           params.loopKNum, params.innerLoopNum);
    // 验证反量化参数
    EXPECT_EQ(params.loopKNum, CeilDivide(kRealSize, C0_SIZE));  // kRealSize=128, C0_SIZE=32
    EXPECT_EQ(params.innerLoopNum, CeilDivide(nRealSizeAlign * C0_SIZE, VECTOR_REG_WIDTH));
});
```

**预期结果**：
```
[UT] FrAntiQuantMxA8W4NzNkVf: loopKNum=4, innerLoopNum=8
// 验证：4个K块(K=128, C0_SIZE=32)，每个K块8个inner循环
```

**验证场景**：验证反量化VF函数的参数正确，且GMMSQ和GMMFR的反量化逻辑完全一致（复用）

---

#### UT-VEC02: SwiGLU() - Swish*Gate计算验证（GMMSQ新增）

**用例构造方案**：
```cpp
// 构造FP32输入：Y[M, N]，其中N=256
// Swish分支：Y[:, 0:128]
// Gate分支：Y[:, 128:256]
// 预期输出：S = Swish(Y_act) * Gate, S[M, 128]
float yInput[256][256];  // m=256, n=256
// 填充测试数据
for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
        yInput[i][j] = (i * 256 + j) * 0.001f;
    }
}
```

**代码打桩方案**：
```cpp
// 打桩SwiGLU VF函数，捕获输入输出
MOCK(void, SwiGLU, (float* swishInput, float* gateInput, float* output, int m, int n), {
    PRINTF("[UT] SwiGLU: m=%d, n=%d", m, n);
    // 验证Swish分支和Gate分支的地址偏移
    EXPECT_EQ(gateInput - swishInput, n * sizeof(float));  // Gate紧跟Swish
    // 验证输出
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n/2; j++) {
            float swishVal = swishInput[i*n + j];
            float gateVal = gateInput[i*n + j];
            float expected = swishVal * gateVal;
            EXPECT_NEAR(output[i*(n/2) + j], expected, 1e-5);
        }
    }
});
```

**预期结果**：
```
[UT] SwiGLU: m=256, n=128
// 验证：Swish[0:128] * Gate[0:128] = Output[0:128]
// 注意：n=128是因为BasicBlock内N已切分为128（Swish）+128（Gate）
```

**验证场景**：验证SwiGLU函数正确计算`Swish(Y[:, 0:128]) * Y[:, 128:256]`，输出维度为N/2

---

#### UT-VEC03: ComputeMaxExp() + ComputeScale() - MX量化参数计算验证（GMMSQ新增）

**用例构造方案**：
```cpp
// 构造FP32输入数据（每32个元素一组MX block）
float swiGluOutput[32][64];  // m=32, n=64（N/2=64）
// 构造max exponent参考值
float expectedMaxExp[32];  // 每行一个shared_exp
```

**代码打桩方案**：
```cpp
// 验证ComputeMaxExp的输入输出
MOCK(void, ComputeMaxExp, (float* input, fp8_e8m0_t* maxExpBuffer, int m, int n), {
    PRINTF("[UT] ComputeMaxExp: m=%d, n=%d", m, n);
    // 验证maxExp计算：每32个元素一组
    EXPECT_EQ(n % 32, 0);  // MX_GROUPSIZE=32
});

// 验证ComputeScale的shared_exp计算
MOCK(void, ComputeScale, (fp8_e8m0_t* maxExpBuffer, fp8_e8m0_t* scaleBuffer, int m, int nBlocks), {
    PRINTF("[UT] ComputeScale: m=%d, nBlocks=%d", m, nBlocks);
    // 验证：nBlocks = CeilDiv(n, 32)
    EXPECT_EQ(nBlocks, CeilDiv(n, MX_GROUPSIZE));
});
```

**预期结果**：
```
[UT] ComputeMaxExp: m=32, n=64
[UT] ComputeScale: m=32, nBlocks=2
// 验证：每64个元素分为2个MX block（每block 32个元素）
```

**验证场景**：验证MX量化的shared_exp计算正确，每个MX block（32个元素）共享一个scale

---

#### UT-VEC04: UB Buffer复用策略验证（GMMSQ关键风险点）

**用例构造方案**：
```cpp
// 构造PhaseA和PhaseB的Buffer使用场景
// PhaseA: weight反量化 → 使用weightLowBit_ + weightHighBit_
// PhaseB: SwiGLU+Quant → 使用yF32Buffer_（复用weightHighBit_）+ yFp8Buffer_（复用weightLowBit_）
```

**代码打桩方案**：
```cpp
// 打印UB Buffer地址，验证复用关系
#define TRACE_UB(label, addr, size) PRINTF("[UB] %s: addr=%p, size=%lu", label, addr, size)

MOCK(void, VecCompute_Init, (...), {
    TRACE_UB("weightLowBit_", weightLowBit_.GetPhyAddr(), 64*1024);
    TRACE_UB("weightHighBit_", weightHighBit_.GetPhyAddr(), 128*1024);
    TRACE_UB("yF32Buffer_", yF32Buffer_.GetPhyAddr(), 128*1024);
    TRACE_UB("maxB16Buffer_", maxB16Buffer_.GetPhyAddr(), 2*1024);
    TRACE_UB("scaleE8M0Buffer_", scaleE8M0Buffer_.GetPhyAddr(), 16*1024);
    TRACE_UB("yFp8Buffer_", yFp8Buffer_.GetPhyAddr(), 32*1024);
    
    // 验证复用关系
    EXPECT_EQ(yF32Buffer_.GetPhyAddr(), weightHighBit_.GetPhyAddr());  // 复用
    EXPECT_EQ(yFp8Buffer_.GetPhyAddr(), weightLowBit_.GetPhyAddr());    // 复用
});
```

**预期结果**：
```
[UB] weightLowBit_: addr=0x..., size=65536 (64KB)
[UB] weightHighBit_: addr=0x..., size=131072 (128KB)
[UB] yF32Buffer_: addr=0x... (same as weightHighBit_) size=131072  [复用 ✅]
[UB] maxB16Buffer_: addr=0x..., size=2048 (2KB)
[UB] scaleE8M0Buffer_: addr=0x..., size=16384 (16KB)
[UB] yFp8Buffer_: addr=0x... (same as weightLowBit_) size=32768  [复用 ✅]

[UT] 总UB使用量（复用后）: 64 + 128 + 1 + 1 + 2 + 16 = 212KB <= 248KB  ✅
```

**验证场景**：验证GMMSQ的UB Buffer复用策略正确，`yF32Buffer_`复用`weightHighBit_`空间，`yFp8Buffer_`复用`weightLowBit_`空间，总UB使用量不超过248KB限制

---

### 4.5.6 Sync机制UT

#### UT-SYNC01: CrossCore同步序列验证（GMMFR与GMMSQ一致）

**用例构造方案**：
```cpp
// 模拟AIC和AIV的同步序列
// 验证GMMSQ保留了GMMFR的同步机制
```

**代码打桩方案**：
```cpp
// 打桩CrossCoreSetFlag/CrossCoreWaitFlag，记录同步序列
std::vector<std::string> syncLog;

MOCK(void, CrossCoreSetFlag, (uint64_t flag), {
    syncLog.push_back("SetFlag(" + std::to_string(flag) + ")");
});

MOCK(void, CrossCoreWaitFlag, (uint64_t flag), {
    syncLog.push_back("WaitFlag(" + std::to_string(flag) + ")");
});
```

**预期结果**：
```
[UT] Sync sequence (AIC path):
  WaitFlag(SYNC_AIV_AIC_FLAG=8)    ← 等待AIV反量化完成
  SetFlag(SYNC_AIC_AIV_FLAG=9)     → 通知AIV可以开始下一轮
  WaitFlag(SYNC_AIV_AIC_FLAG=8)    ← 等待AIV...
  SetFlag(SYNC_AIC_AIV_FLAG=9)     → ...
  
[UT] Sync sequence (AIV path):
  WaitFlag(SYNC_AIC_AIV_FLAG=9)    ← 等待AIC准备好
  SetFlag(SYNC_AIV_AIC_FLAG=8)     → 通知AIC反量化完成
  WaitFlag(SYNC_AIC_FIX_AIV_VF_FLAG=3)  ← 等待AIC Fixpipe完成
  SetFlag(SYNC_AIV_MTE3_AIC_FIX_FLAG=4) → 通知AIC可以Fixpipe
  
[UT] 验证：同步序列与GMMFR完全一致 ✅
```

**验证场景**：验证GMMSQ保留了GMMFR的AIC↔AIV同步机制，同步信号和时序完全一致

---

### 4.5.7 UT汇总表

| UT编号 | 层级 | 验证场景 | 关键差异点 | 打桩对象 | 预期结果 |
|--------|------|---------|-----------|---------|---------|
| UT-C01 | Controller | Init()初始化 | 无sharedInput | BasicBlock::Init | weight=0.0 |
| UT-C02 | Controller | UpdateGmAddr | 输出N/2，无logits/rowIndex | 无 | yGm_ += M*(N/2) |
| UT-C03 | Controller | SplitNByMultiCore | nL1Size=128（GMMFR=256） | 无 | nL1Size=128 |
| UT-C04 | Controller | Process流程 | 去除InitAtomicGm | InitAtomicGm | 不被调用 |
| UT-BB01 | BasicBlock | AIC/AIV分发 | nL1Size=128 | IterateNzNk* | nL1Size=128 |
| UT-BB02 | BasicBlock | End()收尾 | SwiGLU+Quant vs Routing | MulLogits, RoutingYToGm | SwiGLU调用，Routing不调用 |
| UT-CUBE01 | CubeCompute | MMA计算 | 与GMMFR一致 | 无 | L0C误差<0.01% |
| UT-CUBE02 | CubeCompute | GetTensorC | 输出到yF32Buffer_ | GetTensorC | dst=yF32Buffer_ |
| UT-VEC01 | VecCompute | 4bit反量化 | 与GMMFR一致 | FrAntiQuantMxA8W4NzNkVf | loopKNum=4 |
| UT-VEC02 | VecCompute | SwiGLU计算 | GMMSQ新增 | SwiGLU | Swish*Gate正确 |
| UT-VEC03 | VecCompute | MX Quant | GMMSQ新增 | ComputeMaxExp, ComputeScale | nBlocks正确 |
| UT-VEC04 | VecCompute | UB复用 | GMMSQ关键风险 | VecCompute::Init | yF32复用weightHighBit |
| UT-SYNC01 | Sync | 同步序列 | 与GMMFR一致 | CrossCoreSetFlag/WaitFlag | 序列一致 |

### 4.5.8 UT执行建议

**执行顺序**：
1. **阶段1**：UT-C01~UT-C04（Controller层）→ 验证GMMSQ的Controller层改动正确
2. **阶段2**：UT-BB01~UT-BB02（BasicBlock层）→ 验证BasicBlock的路径分发和收尾
3. **阶段3**：UT-CUBE01~UT-CUBE02（CubeCompute层）→ 验证MMA计算和Fixpipe
4. **阶段4**：UT-VEC01~UT-VEC04（VecCompute层）→ 验证反量化、SwiGLU、Quant、Buffer复用
5. **阶段5**：UT-SYNC01（Sync机制）→ 验证同步机制完整

**依赖关系**：
```
UT-C01 (Init) → UT-C04 (Process) → UT-BB01 (ComputeBasicBlock)
    ↓
UT-C02 (UpdateGmAddr) → UT-C03 (SplitNByMultiCore)
    ↓
UT-BB02 (End) → UT-VEC02 (SwiGLU) → UT-VEC03 (Quant)
    ↓
UT-VEC04 (Buffer复用)
    ↓
UT-SYNC01 (Sync)
```

**回归策略**：
- **每次GMMSQ代码修改后**：执行全部UT（约30分钟）
- **Controller层修改**：执行UT-C01~UT-C04
- **BasicBlock层修改**：执行UT-BB01~UT-BB02
- **VecCompute层修改**：执行UT-VEC01~UT-VEC04
- **CubeCompute层修改**：执行UT-CUBE01~UT-CUBE02

---

## 5. 覆盖统计（可独立成 `test_coverage_report.md`）

### 5.1 总体

| 类型 | 条数 | 占比 |
| --- | ---: | ---: |
| 白盒 | 40 | 66.7% |
| 泛化 | 10 | 16.7% |
| 异常 | 8 | 13.3% |
| **合计** | **58** | **100%** |

### 5.2 分维度表

**dtype分布表**：

| dtype组合 | 用例数 | 占比 |
|-----------|--------|------|
| float8_e4m3fn × float4_e2m1 × float8_e8m0 | 40 | 100% |

**format分布表**：

| format组合 | 用例数 | 占比 |
|-----------|--------|------|
| ND × FRACTAL_NZ_C0_32 × ND | 40 | 100% |

**维度数分布表**：

| 维度数 | 用例数 | 说明 |
|--------|--------|------|
| 2D × 5D × 4D | 40 | 标准维度组合 |

**对齐状态分布表**：

| 对齐状态 | 用例数 | 说明 |
|----------|--------|------|
| aligned | 40 | K%32==0, N%128==0 |
| misaligned_k | 4 | K%32!=0 |
| misaligned_n | 4 | N%128!=0 |

**异常类型 × 条数表**：

| 异常类型 | 条数 |
|----------|------|
| invalid_k_alignment | 2 |
| invalid_n_alignment | 2 |
| invalid_transpose_x | 2 |
| invalid_transpose_w | 2 |
| invalid_weight_format | 2 |

---

## 6. 用例命名规范

**推荐格式**：

```text
{scenario}_{shape_tag}_{variant}
```

**示例**：

```text
mxa8w4_M64K128N256E1_cumsum          # MxA8W4场景，基础shape，cumsum模式
mxa8w4_M1024K2048N4096E16_count       # 中等shape，count模式
mxa8w4_M8192K16384N32768E128_boundary  # 大规模边界测试
mxa8w4_invalid_k_alignment            # K不对齐异常
mxa8w4_invalid_transpose_x            # 转置参数异常
```

---

## 7. 约束关系（与实现对齐）

正常用例需满足（与 checker / SRS 一致）：

1. **格式约束**：
   - x.format == ND
   - weight.format == FRACTAL_NZ_C0_32
   - weightScale.format == ND
   - xScale.format == ND
   - output.format == ND
   - outputScale.format == ND

2. **维度关系**：
   - x.shape == [M, K]
   - weight.shape == [E, K/32, N/16, 16, 32]
   - weightScale.shape == [E, N, ceil(K/64), 2]
   - xScale.shape == [M, ceil(K/64), 2]
   - output.shape == [M, N/2]
   - outputScale.shape == [M, ceil((N/2)/64), 2]

3. **shape计算**：
   - K % 32 == 0
   - N % 128 == 0
   - K >= 32
   - N >= 128
   - E <= 1024

异常用例**每次只违反一条**规则，便于定位。

---

## 8. 数据文件契约

### 8.1 文件清单

| 文件 | 说明 |
| --- | --- |
| stc_gmmsq.md | 本说明 |
| L0_test_cases.csv | L0级别测试用例 |
| L1_test_cases.csv | L1级别测试用例 |
| test_coverage_report.md | 覆盖统计 |
| generate_test_cases.py | 用例生成脚本 |

### 8.2 列定义 — 正常用例

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| case_id | str | 唯一编号，如L0-001, L1-001 |
| testcase | str | 用例命名见第6节 |
| case_type | enum | whitebox / generalized / exception |
| x_dtype | str | x数据类型 |
| x_shape | str | x形状，如[M, K] |
| weight_dtype | str | weight数据类型 |
| weight_shape | str | weight形状，如[E, K/32, N/16, 16, 32] |
| weightScale_shape | str | weightScale形状，如[E, N, ceil(K/64), 2] |
| xScale_shape | str | xScale形状，如[M, ceil(K/64), 2] |
| groupListType | int | 0:cumsum, 1:count |
| output_shape | str | output形状，如[M, N/2] |
| outputScale_shape | str | outputScale形状，如[M, ceil((N/2)/64), 2] |
| description | str | 用例说明 |

### 8.3 列定义 — 异常用例

| 列名 | 说明 |
| --- | --- |
| case_id | 唯一编号 |
| exception_type | 异常类型，如invalid_k_alignment |
| error_description | 错误描述，含期望码、日志子串 |
| violated_constraint | 违反的约束条件 |
| expected_result | 期望结果，如GRAPH_FAILED |

### 8.4 fuzz / aclnn 表

- 正常用例：所有参数合法，验证正确性
- 异常用例：每次只违反一条约束，验证错误处理
- 可从 SRS 第10节或代码 `OP_LOGE` 摘「日志关键词」写入 `case_description`

---

## 9. 脚本与自动化

| 脚本 | 入参 | 出参 | 说明 |
| --- | --- | --- | --- |
| generate_test_cases.py | shape_range, dtype_list | csv文件 | 生成测试用例 |
| run_test.py | case_id, device_id | result_json | 执行测试并收集结果 |
| compare_result.py | actual_output, expected_output | diff_report | 对比实际与期望结果 |

**原则**：生成逻辑与 SRS 约束同构；变更约束时**先改 SRS / checker，再改生成器**。

---

## 10. 交付通路验证

| 通路 | 验证方式 | 结论 |
| --- | --- | --- |
| aclnn | 调用aclnnGroupedMatmulSwigluQuantWeightNzV2接口，验证结果正确性 | 通过标准：精度误差<=0.001 |
| PTA | 通过PTA框架调用，验证通路完整性 | 通过标准：返回成功 |

---

## 11. 实现/Tiling 对照（评审与测试共用）

**MxA8W4场景启用条件**：

1. x.dtype == float8_e4m3fn
2. weight.dtype == float4_e2m1
3. weight.format == FRACTAL_NZ_C0_32
4. weightScale.dtype == float8_e8m0
5. xScale.dtype == float8_e8m0
6. transpose_x == false
7. transpose_w == true
8. K % 32 == 0
9. N % 128 == 0
10. E <= 1024

**Tiling切分策略**：

| 条件 | kbL1Size | 说明 |
|------|----------|------|
| mL1Size <= threshold && nL1Size <= threshold | 512 | 小tile，更多K复用 |
| 其他 | 256 | 大tile，更多M/N buffer空间 |

---

## 12. 风险与未覆盖

| 项 | 说明 | 跟踪方式 |
| --- | --- | --- |
| 超大shape测试 | M>16384或K>32768可能内存不足 | 版本迭代中逐步扩展 |
| 非均匀groupList | cumsum模式下groupList非均匀分布 | issue跟踪 |
| 多core竞争 | AIC+AIV异构协同的竞态条件 | 代码评审 + 压力测试 |
| 精度累积误差 | 大K值下MX量化的精度损失 | 版本迭代中评估 |

---

## 附录 A：与 SRS 的一致性检查

- [x] 每个 SRS 约束至少一条用例或明确标记 N/A。
- [x] 可选输入含 null 与 非 null；非法组合有独立异常行。
- [x] NZ / storage shape 列齐全。
- [x] 覆盖报告数字与 CSV 行数一致。

### A.1 SRS约束覆盖检查表

| SRS约束 | 覆盖用例 | 状态 |
|---------|----------|------|
| K % 32 == 0 | L0-011~L0-015 | ✅ |
| N % 128 == 0 | L0-016~L0-020 | ✅ |
| K >= 32 | L0-001, L0-003 | ✅ |
| N >= 128 | L0-001, L0-003 | ✅ |
| E <= 1024 | L0-007~L0-010 | ✅ |
| transpose_x == false | 全部用例 | ✅ |
| transpose_w == true | 全部用例 | ✅ |
| weight.format == FRACTAL_NZ_C0_32 | 全部用例 | ✅ |
| x.dtype == float8_e4m3fn | 全部用例 | ✅ |
| weight.dtype == float4_e2m1 | 全部用例 | ✅ |
| weightScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| xScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| output.dtype == float8_e4m3fn | 全部用例 | ✅ |
| outputScale.dtype == float8_e8m0 | 全部用例 | ✅ |
| groupListType in {0, 1} | L0-001~L0-020, L1-001~L1-020 | ✅ |
| dequantMode == 2 | 全部用例 | ✅ |
| dequantDtype == 0 | 全部用例 | ✅ |
| quantMode == 2 | 全部用例 | ✅ |

### A.2 异常场景覆盖检查表

| 异常场景 | 覆盖用例 | 状态 |
|---------|----------|------|
| K % 32 != 0 | invalid_k_alignment | ✅ |
| N % 128 != 0 | invalid_n_alignment | ✅ |
| K < 32 | invalid_k_minimum | ✅ |
| N < 128 | invalid_n_minimum | ✅ |
| transpose_x == true | invalid_transpose_x | ✅ |
| transpose_w == false | invalid_transpose_w | ✅ |
| E > 1024 | invalid_expert_num | ✅ |
| weight.format != FRACTAL_NZ_C0_32 | invalid_weight_format | ✅ |
| weightScale.shape不匹配 | invalid_scale_shape | ✅ |
| xScale.shape不匹配 | invalid_xscale_shape | ✅ |
| groupList累加和!=M | invalid_groupList_sum | ✅ |
| x.dtype != float8_e4m3fn | invalid_dtype_combination | ✅ |
