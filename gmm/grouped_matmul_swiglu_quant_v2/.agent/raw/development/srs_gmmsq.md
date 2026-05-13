# GMMSQ支持MxA8W4数据流 SRS

> **使用说明**：正文内 `[占位]`、`（可选）` 按实际删减；涉密名称、客户、版本号用中性描述替换。图放模块内 `figure/`，命名与正文引用一致。

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档标题 | GMMSQ支持MxA8W4数据流 |
| 迭代/特性名称 | GMMSQ支持MxA8W4数据流 |
| 算子类型 | 融合算子 |
| 主领域目录 | FakeQuantization |
| 责任人 | 孙堆杨 |
| 更新时间 | 2026-05-09 |
| 文档版本 | v1.0 |
| 关联 STC | [GMMSQ STC](./stc_gmmsq.md) |
| 关联里程碑/需求单 | 无 |

---

## 2. 需求背景

### 2.1 背景

GMMSQ融合算子提升小算子拼接性能，支持伪量化MxA8W4数据流，给网络提供更高加速比。

### 2.2 目标

精度要求：双千分之一
性能要求：性能摸底，不做性能要求

### 2.3 范围

- **包含**：Host tiling校验和拦截、tiling切分、算子原型增加新数据流支持、kernel入口新增加伪量化kernel调用分支、实现kernel代码、补充Host侧tiling UT、kernel调用UT、aclnn/PTA通路支持（本srs不涉及）。
- **不包含**：图融合（本srs不涉及）。

---

## 3. 方案简述

GMMSQ（GroupedMatmulSwigluQuant）MxA8W4数据流在已有MxFP8全量化数据流基础上，新增伪量化MxA8W4数据流支持。核心架构采用 **AIC（Cube）+ AIV（Vector）异构协同** 设计，整体数据流如下：

```
GM输入(float8_e4m3 x, float4_e2m1 weight, float8_e8m0 scale)
  → AIV反量化(weight: fp4→fp8)
  → AIC矩阵乘(带MX Scale)
  → AIV SwiGLU激活
  → AIV MX量化输出
  → GM输出(float8_e4m3 output + float8_e8m0 outputScale)
```

| 层次 | 要点 |
| --- | --- |
| Host | 新增MxA8W4场景识别（InferScenario），完成参数校验、入参整形 |
| Tiling | 基于GMMFR MxA8W4模板扩展，适配GMMSQ的SwiGLU切分需求；动态K切分；tiling key生成 |
| Kernel | AIV负责fp4→fp8反量化 + SwiGLU + MX Quant；AIC负责带MX Scale的MMA矩阵乘；双缓冲同步 |
| 测试/交付 | aclnn通路和PTA通路 |

**方案示意图（可选）**：`![说明](./figure/overview.png)`

---

## 4. 数学定义与语义

### 4.1 符号约定

| 符号 | 含义 |
| --- | --- |
| $X$ | 激活矩阵（左矩阵），float8_e4m3fn |
| $W$ | 权重矩阵（右矩阵），float4_e2m1 |
| $xScale$ | 激活反量化因子，float8_e8m0 |
| $wScale$ | 权重反量化因子，float8_e8m0 |
| $C$ | 矩阵乘结果 |
| $S$ | SwiGLU输出结果 |
| $Q$ | 量化后输出，float8_e4m3fn |
| $QScale$ | 输出量化因子，float8_e8m0 |
| $M$ | 总token数 |
| $K$ | 特征维度 |
| $N$ | 输出维度 |
| $E$ | 专家个数 |

### 4.2 计算公式

**步骤1：分组矩阵乘法（MxA8W4反量化过程）**

$$
C_i = (X_i \cdot dequant(W_i)) \odot xScale_{i\ Broadcast} \odot wScale_{i\ Broadcast}
$$

其中 $X_i$ 为float8_e4m3fn类型激活，$W_i$ 为float4_e2m1类型权重（计算时反量化为float8_e4m3），$xScale_i$ 为float8_e8m0类型激活反量化因子，$wScale_i$ 为float8_e8m0类型权重反量化因子。

**步骤2：SwiGLU激活**

$$
C_{i,act}, gate_{i} = split(C_i)
$$

$$
S_i = Swish(C_{i,act}) \odot gate_{i}
$$

其中 $Swish(x)=\frac{x}{1+e^{-x}}$


**步骤3：共享指数MX量化输出结果**

$$
shared\_exp = \left\lfloor \log_2(max_j(|S_{i,j}|)) \right\rceil - emax
$$

$$
QScale_i = 2^{shared\_exp}
$$

$$
Q_{i,j} = quantize\_to\_element\_format(S_{i,j} / QScale_i)
$$

其中 $emax$ 为对应数据类型的最大正则数的指数位：

|   DataType    | emax |
| :-----------: | :--: |
| FLOAT8_E4M3FN |  8   |

---

## 5. 输入输出规格

### 5.A 张量接口型

#### 5.A.1 输入 — 张量参数

| 参数名 | 必选/可选 | dtype | format | shape / 存储语义 | 说明 |
| --- | --- | --- | --- | --- | --- |
| x | 必选 | float8_e4m3fn | ND | (m, k) | 左矩阵（激活），固定不转置 |
| weight | 必选 | float4_e2m1 | FRACTAL_NZ_C0_32 | viewShape=(e, n, k)<br>oriShape=(e, ceil(k,32), ceil(n,16), 16, 32) | 右矩阵（权重），固定转置 |
| weightScale | 必选 | float8_e8m0 | ND | (e, n, ceil(k/64), 2) | MX weight 反量化缩放因子 |
| weightAssistMatrix | 不使用 | - | - | - | MxA8W4场景为null |
| bias | 不使用 | - | - | - | MxA8W4场景为null |
| xScale | 必选 | float8_e8m0 | ND | (m, ceil(k/64), 2) | MX x 反量化缩放因子 |
| smoothScale | 不使用 | - | - | - | MxA8W4场景为null |
| groupList | 必选 | int64 | ND | (e,) | 分组列表，表示各专家的token分布 |

**注**：weight的format格式为FRACTAL_NZ_C0_32而不是FRACTAL_NZ_C0_64，原因是weight以float4_e2m1类型加载权重，但类型转换为float8_e4m3进行计算，所以实际C0_SIZE以float8_e4m3进行计算，得到 `32B / sizeof(float8_e4m3) = 32`。

#### 5.A.2 输入 — 标量 / 属性

| 参数名 | 类型 | 取值 / 约束 | 说明 |
| --- | --- | --- | --- |
| dequantMode | int64 | {2} | 反量化计算类型，固定为2（MX量化） |
| dequantDtype | int64 | {0} | 中间计算数据类型，固定为0（DT_FLOAT） |
| quantMode | int64 | {2} | 量化计算类型，固定为2（MX量化） |
| groupListType | int64 | {0, 1} | 0: cumsum模式, 1: count模式 |
| tuningConfig | int64数组 | - | 调优参数，可选，MxA8W4场景传入空指针 |
| transpose_x | bool | {false} | 左矩阵不转置，固定false |
| transpose_w | bool | {true} | 右矩阵转置，固定true |

#### 5.A.3 输出

| 输出名 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- |
| output | float8_e4m3fn | ND | (m, n/2) | 量化后的输出矩阵 |
| outputScale | float8_e8m0 | ND | (m, ceil((n/2)/64), 2) | 输出量化缩放因子 |

### 5.B 迭代算子型

#### 5.B.1 输入规格

| 输入项 | 说明 |
| --- | --- |
| 量化场景 | 伪量化MxA8W4 |
| dtype | x: float8_e4m3fn; weight: float4_e2m1; scale/xScale: float8_e8m0 |
| shape | x: [M, K]; weight: [E, K/32, N/16, 16, 32]; xScale: [M, CeilDiv(K, 64), 2]; scale: [E, N, CeilDiv(K,64), 2] |
| transpose | x: false; w: true（固定） |
| format | x: ND; weight: FRACTAL_NZ_C0_32 |
| 其他 | groupList: int64 [E]; bias/weightAssistMatrix/smoothScale: null |

#### 5.B.2 输出规格

| 输出项 | 说明 |
| --- | --- |
| shape | output: [M, N/2]; outputScale: [M, CeilDiv(N/2, 64), 2] |
| dtype | output: float8_e4m3fn; outputScale: float8_e8m0 |
| format | ND |

### 5.C dtype × format × 场景矩阵

| dtype | src_format | dst_format | 场景/备注 |
| --- | --- | --- | --- |
| float8_e4m3fn | ND | ND | 激活输入x |
| float4_e2m1 | FRACTAL_NZ_C0_32 | FRACTAL_NZ_C0_32 | 权重输入weight |
| float8_e8m0 | ND | ND | scale/xScale |
| float8_e4m3fn | ND | ND | 输出output |
| float8_e8m0 | ND | ND | 输出outputScale |

**注**：C0_SIZE为32（float4_e2m1反量化为float8_e4m3后计算），weight格式为FRACTAL_NZ_C0_32。K维度32对齐，N维度128对齐。

---

## 6. 约束条件

| 约束项 | 条件 | 说明 / 校验失败表现 |
| --- | --- | --- |
| k对齐 | k % 32 == 0 | K维度需32对齐，否则tiling报错 |
| N对齐 | n % 128 == 0 | N维度需128对齐，否则tiling报错 |
| K最小值 | k >= 32 | K维度最小值为32 |
| N最小值 | n >= 128 | N维度最小值为128 |
| weight格式 | FRACTAL_NZ或FRACTAL_NZ_C0_32 | weight数据格式约束，格式不匹配报错 |
| 转置约束 | transpose_x == false | 左矩阵固定不转置 |
| 转置约束 | transpose_w == true | 右矩阵固定转置 |
| weightScale转置 | 与weight保持一致 | weightScale转置属性需与weight一致 |
| 专家数上限 | e <= 1024 | 专家数量不超过1024 |
| scale shape | [E, N, CeilDiv(K,64), 2] | MX scale维度约束 |
| xScale shape | [M, CeilDiv(K,64), 2] | per-token scale维度约束 |

---

## 7. 交付标准

| 交付范围 | 当前迭代 |
| --- | --- |
| 交付通路 | PTA / aclnn |
| 动静要求 | 不涉及 |
| 精度标准 | 双千分之一 |
| 性能标准 | 仅做摸底要求 |
| 客户场景索引 | 无 |
| 相关设计文档索引 | gmmsq_kernel_impl.md, gmmsq_op.md |
| 维测相关索引 | 无 |

---

## 8. 具体方案描述

### 8.1 通路 / 调用链

本srs不涉及通路部分

### 8.2 Tiling

#### 8.2.1 准入与校验

Tiling层负责在Host侧进行算子参数的校验和场景识别。MxA8W4场景识别条件：

```cpp
// 伪代码：MxA8W4场景识别
bool InferScenario() {
    if (xDtype == DT_FLOAT8_E4M3FN && 
        wDtype == DT_FLOAT4_E2M1 &&
        (wFormat == FORMAT_FRACTAL_NZ || wFormat == FORMAT_FRACTAL_NZ_C0_32) &&
        scaleDtype == DT_FLOAT8_E8M0 && 
        perTokenScaleDtype == DT_FLOAT8_E8M0) {
        return MX_A8W4_WEIGHT_NZ;
    }
}
```

属性校验要点：
- `transpose_x` 必须为 `false`
- `transpose_w` 必须为 `true`
- `groupListType` 必须为 0 或 1
- `offset` 必须为 `nullptr`

Shape校验要点：
- x shape: [M, K]，K % 32 == 0
- w shape (FRACTAL_NZ): oriShape=[E, N, K]，storageShape=[E, K/32, N/16, 16, 32]
- scale shape: [E, N, CeilDiv(K, 64), 2]
- xScale shape: [M, CeilDiv(K, 64), 2]
- groupList shape: [E]

#### 8.2.2 切分策略

Tiling Data 结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| coreNum | uint32_t | AIC核心数 |
| groupNum | int64_t | 专家数E |
| kSize | int64_t | K维度大小 |
| nSize | int64_t | N维度大小 |
| outputBs | int64_t | 输出batch size |
| initSize | int64_t | 初始化大小 = (outputBS - sharedInputLen) * nSize |
| groupSize | uint32_t | MX分组大小 = 32 |
| groupListType | uint32_t | 分组列表类型 |

动态K维度切分：
```cpp
uint64_t ChooseKbL1Size(uint64_t mL1Size, uint64_t nL1Size) {
    if (mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD &&
        nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) {
        return 512;
    } else {
        return 256;
    }
}
```

#### 8.2.3 Tiling Key

当前支持：`ATRANS=0, BTRANS=1`

#### 8.2.4 Workspace

默认workspace大小：16MB（16 * 1024 * 1024）

### 8.3 Kernel

> [kernel详细编码方案查看这里](./srs_coding_design.md)

#### 8.3.1 执行流程

GMMSQ的Kernel计算采用 **AIC + AIV异构协同** 模式，核心流程如下：

```
初始化阶段 (Init)
  ├── AIC侧: 初始化L1 buffer分配，设置AIC→AIV同步信号
  └── AIV侧: 初始化UB buffer，设置MTE2/MTE3同步事件

计算阶段 (ComputeBasicBlock)
  ├── AIV路径: IterateNzNkWithAiv
  │   ├── Pass 1: 预加载前K chunk到UB进行反量化
  │   │   ├── MTE2: GM→UB搬运4bit weight
  │   │   ├── V: 反量化计算(fp4→fp8)
  │   │   ├── MTE3: UB→L1搬运反量化后的weight
  │   │   └── 设置AIC同步信号
  │   ├── Phase B: 处理前一个basic block的SwiGLU+Quant输出（如果存在）
  │   │   ├── 设置AIC fixpipe信号
  │   │   ├── 等待AIC fixpipe完成
  │   │   ├── V: SwiGLU计算
  │   │   ├── V: MX量化计算
  │   │   └── MTE3: 写出量化结果到GM
  │   └── Pass 2: 继续加载剩余K chunk
  │       └── 同Pass 1流程
  └── AIC路径: IterateNzNkWithKAic
      ├── 等待AIV反量化完成信号
      ├── MTE1→MTE2: 搬运Scale到L1
      ├── MTE1: 搬运A矩阵到L1
      ├── MTE2→MTE1: L1→L0加载（带MX Scale）
      ├── M: MMA矩阵乘计算
      └── 设置AIV同步信号

结束阶段 (End)
  ├── AIC侧: 最后一次fixpipe，同步结束
  └── AIV侧: 处理最后一个block，清理同步信号
```

#### 8.3.2 关键逻辑

**AIV路径**：
1. **反量化阶段**：将4bit weight从GM搬运到UB，通过VF指令反量化为8bit，再搬运到L1供AIC使用
2. **SwiGLU阶段**：由其他成员负责
3. **MX Quant阶段**：由其他成员负责

**AIC路径**：
1. 等待AIV反量化完成信号
2. 搬运Scale和A矩阵到L1
3. L1→L0加载（带MX Scale）
4. MMA矩阵乘计算
5. Fixpipe L0C→UB

#### 8.3.3 Buffer 与同步

**L1 Buffer分配（无bias场景）**：

```
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x18000)  ScaleAL1_P0        32 KB
├─ [0x18000, 0x20000)  ScaleBL1_P0        32 KB
├─ [0x20000, 0x40000)  AL1_P0            128 KB
├─ [0x40000, 0x60000)  AL1_P1            128 KB
├─ [0x60000, 0x68000)  ScaleBL1_P1        32 KB
├─ [0x68000, 0x70000)  ScaleAL1_P1        32 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

**L0 Buffer分配**：

| Buffer | Position | Size | Purpose |
|--------|----------|------|---------|
| L0A | A2 | 64 KB | A-matrix (activation) tile |
| L0B | B2 | 64 KB | B-matrix (weight) tile |
| L0C | CO1 | 256×256 floats | MMA accumulation |
| BiasTable | C2 | 4 KB | Bias lookup table |

**UB Buffer分配**：

| Buffer | Offset | Size | Purpose |
|--------|--------|------|---------|
| `weightLowBit_` | 0 | 64 KB | Packed 4-bit weight input |
| `weightHighBit_` | 64 KB | 128 KB | Anti-quantized fp8 weight |
| `yF32Buffer_` | 64 KB | 128 KB | MMA fixpipe output (reuses weightHighBit_ space) |
| `bias_` | 192 KB | 1 KB | Bias input |
| `biasRescale_` | 193 KB | 1 KB | Rescaled bias |
| `maxB16Buffer_` | 194 KB | 2 KB | Per-block max exponent (ComputeMaxExp) |
| `scaleE8M0Buffer_` | 196 KB | 16 KB | MX E8M0 scales |
| `yFp8Buffer_` | 212 KB | 32 KB | Final quantized FP8/FP4 output |

**同步机制**：

| Flag ID | Direction | Pipe | Purpose |
|---------|-----------|------|---------|
| `SYNC_AIC_AIV_FLAG = 9` | AIC → AIV | MTE1 | Weight ready in L1 |
| `SYNC_AIV_AIC_FLAG = 8` | AIV → AIC | MTE3 | Anti-quantized weight ready |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG = 4` | AIV → AIC | MTE3 | Request fixpipe L0C→UB |
| `SYNC_AIC_FIX_AIV_VF_FLAG = 3` | AIC → AIV | V / FIX | UB output ready for VF |

### 8.4 代码片段

**场景识别伪代码**：
```cpp
bool InferScenario() {
    if (xDtype == DT_FLOAT8_E4M3FN && 
        wDtype == DT_FLOAT4_E2M1 &&
        (wFormat == FORMAT_FRACTAL_NZ || wFormat == FORMAT_FRACTAL_NZ_C0_32) &&
        scaleDtype == DT_FLOAT8_E8M0 && 
        perTokenScaleDtype == DT_FLOAT8_E8M0) {
        return MX_A8W4_WEIGHT_NZ;
    }
}
```

**Tiling切分伪代码**：
```cpp
ge::graphStatus DoOpTiling() {
    tilingData.coreNum = compileInfo.aicNum;
    tilingData.groupNum = inputParams.groupNum;
    tilingData.kSize = inputParams.kSize;
    tilingData.nSize = inputParams.nSize;
    tilingData.groupSize = MX_BLOCK_SIZE;  // 32
    tilingData.initSize = (inputParams.outputBS - inputParams.sharedInputLen) * inputParams.nSize;
    tilingData.groupListType = inputParams.groupListType;
    return SUCCESS;
}
```

---

## 9. 软件设计与接口总览

### 9.1 模块职责

| 模块 | 说明 |
| --- | --- |
| Host | 场景识别、参数校验、入参整形、Tiling调用 |
| Tiling | 准入校验、动态切分、TilingKey生成、Workspace计算 |
| Kernel | AIV反量化 + SwiGLU + MX Quant；AIC MMA矩阵乘（带MX Scale） |
| 测试 | 验证host tiling拦截和校验，切分大小 |

### 9.2 接口列表（与实现对齐）

| 接口项 | 说明 |
| --- | --- |
| Host API | GMMSQWeightQuantTiling |
| Tiling 数据结构 | GMMSwigluQuantWeightQuantTilingData |
| Kernel entry / 模板参数 | GMMSQWeightQuantVcvBasicBlock<xType, wType, ...> |
| 对外泄漏 API(ACLNN) | aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize, aclnnGroupedMatmulSwigluQuantWeightNzV2 |

---

## 11. 外部依赖

| 依赖项 | 内容 | 影响级别 |
| --- | --- | --- |
| CANN / 编译器 / 算子库版本 | 无 | 高/中/低 |
| 硬件型号 / SOC | Ascend 950PR/Ascend 950DT (arch35) | 高 |
| 上下游算子 / Pass | 无 | 无 |
| 测试框架与 golden | 无 | 无 |

---

## 13. 修订记录

| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v0.1 | 2026-05-09 | 孙堆杨 | 初稿 |

---

## 附录 A：与 STC 的交叉检查清单

- [ ] 每个「必选/可选」输入在 STC 中有至少一条用例（含 null / 非法路径）。
- [ ] dtype×format 矩阵中每一格要么有 happy path，要么在约束中明确禁止并有用例命中报错。
- [ ] Tiling 准入条件在 STC「环境/限制」或白盒用例中有对照。
- [ ] 精度/性能交付标准在 STC 或 CI 描述中可追溯到具体阈值或 job。
