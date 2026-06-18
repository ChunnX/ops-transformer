# VF大融合 → FD开发 API 学习手册

> 目标：掌握 mixed_quant_sparse_flash_mla 算子中使用的 **85+ 基础 API**，具备 FD 开发能力
> 适用：碎片时间学习（地铁/厕所/手机阅读），每章 5-10 分钟可读完
> 来源：从本算子 arch35 kernel 源码中逆向提取的真实 API 使用场景

---

# 学习路线图

```
第1章 数据搬运 ──→ 第2章 Buffer管理 ──→ 第3章 流水线同步
     │                  │                   │
     ↓                  ↓                   ↓
第4章 基础计算 ──→ 第5章 归约操作 ──→ 第6章 精度转换
     │                  │                   │
     ↓                  ↓                   ↓
第7章 MicroAPI VF ──→ 第8章 Cube侧Matmul ──→ 第9章 Tensor与核管理
                                          │
                                          ↓
                                    第10章 综合实战
```

| 章节 | 预计时间 | 核心知识点 | FD开发关联 |
|------|---------|-----------|-----------|
| 1 | 10min | GM↔UB搬运三板斧 | CopyIn/CopyOut必备 |
| 2 | 10min | TBuf/TQue/DoubleBuffer | Buffer规划是FD第一步 |
| 3 | 10min | 7种HardEvent+跨核同步 | 双轴流水必备 |
| 4 | 10min | Adds/Muls优化秘籍 | Softmax/FlashUpdate核心 |
| 5 | 8min | ReduceMax/ReduceSum | LSE计算必备 |
| 6 | 8min | Cast/混合精度 | FP8↔BF16↔FP32全链路 |
| 7 | 15min | RegTensor/LoadAlign/StoreAlign | VF大融合核心技术 |
| 8 | 10min | Mmad/LoadData/Fixpipe | Bmm1/Bmm2矩阵乘 |
| 9 | 8min | GlobalTensor/GetBlockIdx | 核间协作基础 |
| 10 | 15min | 全链路串讲 | FD实战演练 |

---

# 第1章 数据搬运：GM ↔ UB 三板斧

## 1.1 核心原则：永远用 DataCopyPad

```
DataCopy      ← ⛔ 禁止！不支持非对齐，隐蔽bug
DataCopyPad   ← ✅ 强制！统一处理对齐/非对齐
SetValue      ← ⛔ 禁止！逐元素写入，效率极低
GetValue      ← ⛔ 仅调试用！效率极低
```

## 1.2 DataCopyPad 四要素

```cpp
AscendC::DataCopyPad(dst, src, copyParams, padParams);
```

| 参数结构体 | 字段 | 含义 | 注意 |
|-----------|------|------|------|
| `DataCopyExtParams` | blockCount | 拷贝行数 | ≤4095 |
| | blockLen | 每行有效字节长度 | **可以非对齐** |
| | srcStride | GM侧：字节间隔 | 前块尾→后块头 |
| | dstStride | UB侧：32字节块间隔 | 前块尾→后块头 |
| `DataCopyPadExtParams<T>` | isPad | 是否指定填充值 | false=自动填 |
| | leftPadding | 左填充元素数 | 一般=0 |
| | rightPadding | 右填充元素数 | paddedCols-cols |
| | paddingValue | 填充值 | 0.0f常用 |

## 1.3 stride 的致命陷阱

**单位不同！**

| 位置 | stride单位 | 计算 |
|------|-----------|------|
| GlobalTensor(GM) | **字节** | `(paddedCols-cols)*sizeof(T)` |
| LocalTensor(UB) | **32字节块** | `(paddedCols-cols)*sizeof(T)/32` |

```cpp
// ❌ 致命错误：混淆单位
copyParams.srcStride = paddedCols * sizeof(T) / 32;  // 错位！

// ✅ 正确：stride是"间隔"不是"长度"
copyParams.srcStride = (paddedCols - cols) * sizeof(T) / 32;
```

## 1.4 CopyIn/CopyOut 模式（本算子实际用法）

**CopyIn（GM→UB）：**
```cpp
// KV稀疏拷入，支持非对齐padding
DataCopyExtParams copyParams{1, (uint32_t)(cols*sizeof(KV_T)), 0, 0};
DataCopyPadExtParams<KV_T> padParams{true, 0, paddedCols-cols, (KV_T)0.0};
DataCopyPad(kvUb, kvGm[offset], copyParams, padParams);
```

**CopyOut（UB→GM）：**
```cpp
// 自动丢弃dummy，无需padParams
DataCopyExtParams copyParams{1, (uint32_t)(cols*sizeof(T)), 0, 0};
DataCopyPad(outGm, outUb, copyParams);
```

## 1.5 本算子特殊搬运：Fixpipe（L0C→UB）

```cpp
// Bmm结果从L0C搬到UB，NZ→ND格式转换
FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
fixpipeParams.nSize = nAlign;
fixpipeParams.mSize = mAlign;
Fixpipe(ubTensor, l0cTensor, fixpipeParams);
```

## 1.6 本算子特殊搬运：CopyToL1Nd2Nz（GM→L1格式转换）

```cpp
// Query从GM搬到L1，ND格式→NZ格式（Cube需要NZ输入）
Nd2NzParams nd2nzParams;
nd2nzParams.ndDim = 0;      // 第0维切分
nd2nzParams.srcNdShape = {m, k};
nd2nzParams.dstNzShape = {mCeil, kCeil};
CopyToL1Nd2Nz(l1Tensor, gmTensor, nd2nzParams);
```

## 1.7 速记口诀

> **DataCopyPad是王道，stride单位要分清**
> **GM用字节UB用块，间隔不是总长度**
> **CopyIn要padParams，CopyOut框架自动丢**

---

# 第2章 Buffer管理：TBuf vs TQue

## 2.1 选择原则（5秒决策）

```
需要MTE搬运？ ──→ TQue<VECIN/VECOUT>, depth=2
纯Vector计算？ ──→ TBuf<VECCALC>, depth=1
```

## 2.2 TQue 生命周期（本算子stage1OutQue实例）

```cpp
// 声明：VECOUT位置，depth=1（本算子只用单buffer）
TQue<QuePosition::VECOUT, 1> stage1OutQue;

// Init
pipe.InitBuffer(stage1OutQue, 1, bufferSize);

// 使用流程：AllocTensor → 写数据 → EnQue → DeQue → 读数据 → FreeTensor
LocalTensor<Q_T> tensor = stage1OutQue.AllocTensor<Q_T>();  // 分配
// ... Cast写数据到tensor ...
stage1OutQue.EnQue(tensor);                                  // 入队（同步点）
LocalTensor<Q_T> out = stage1OutQue.DeQue<Q_T>();            // 出队（等待就绪）
// ... 读数据 ...
stage1OutQue.FreeTensor(out);                                // 释放
```

## 2.3 TBuf 使用（本算子softmaxMaxBuf等）

```cpp
// 声明
TBuf<TPosition::VECCALC> softmaxMaxBuf;

// Init
pipe.InitBuffer(softmaxMaxBuf, bufferSize);

// 使用：直接Get，无需EnQue/DeQue
LocalTensor<float> maxLocal = softmaxMaxBuf.Get<float>();
ReduceMax(maxLocal, src, tmp, count, false);
float maxVal = maxLocal.GetValue(0);  // 调试用GetValue，生产改DataCopyPad
```

## 2.4 Double Buffer本质

> **不是"用2块内存计算"，而是"搬入/计算并行"**

```
无DoubleBuffer(串行):  Row0[MTE2][VEC][MTE3] Row1[MTE2][VEC][MTE3]
有DoubleBuffer(并行):  Row0[MTE2-B0][VEC-B0][MTE3-B0]
                       Row1     [MTE2-B1][VEC-B1][MTE3-B1]
                              ↑ 并行！
```

| Buffer | depth | 原因 |
|--------|-------|------|
| TQue<VECIN> (MTE2搬入) | 2 | 搬运与计算并行 |
| TQue<VECOUT> (MTE3搬出) | 2 | 计算与搬出并行 |
| TBuf<VECCALC> (纯计算) | 1 | 不涉及MTE搬运 |

## 2.5 本算子Buffer全景

```
UB Buffer规划（Vector侧）:
├── commonTBuf          (VECCALC)  通用临时buffer
├── sinksBuf            (VECCALC)  Sink计算buffer
├── softmaxMaxBuf       (VECCALC)  Softmax最大值
├── softmaxSumBuf       (VECCALC)  Softmax求和
├── softmaxExpBuf       (VECCALC)  Softmax指数
├── dequantScaleBuff    (VECCALC)  反量化scale
├── stage0InBuf         (VECCALC)  Stage0输入
├── stage0OutBuf        (VECCALC)  Stage0输出
├── stage2OutBuf        (VECCALC)  Stage2输出
└── stage1OutQue[2]     (VECOUT,depth=1) Stage1输出队列

L1 Buffer规划（Cube侧）:
├── l1QBuffer           (单buffer) Query L1缓存
├── l1PBuffer           (双buffer) KV L1缓存
├── l1RightBuffer       (3buffer)  右矩阵L1预加载
├── l0ABuffer           (双buffer) L0A矩阵缓存
├── l0BBuffer           (双buffer) L0B矩阵缓存
├── l0CBuffer           (双buffer) L0C结果缓存
```

## 2.6 速记口诀

> **搬运用TQue depth=2，计算用TBuf depth=1**
> **EnQue标记就绪，DeQue阻塞等待**
> **DoubleBuffer不是双计算，是搬运与计算并行**

---

# 第3章 流水线同步：7种HardEvent + 跨核通信

## 3.1 为什么必须同步？

```
DataCopy/DataCopyPad是异步DMA！立即返回！
如果直接在搬运后做Vector计算 → 读到未完成的数据 → 输出随机值！
```

## 3.2 两种同步方式

| 方式 | 性能 | 推荐度 | 原理 |
|------|------|--------|------|
| EnQue/DeQue | 高 | ★★★★★ | buffer级同步，支持并行 |
| PipeBarrier | 低 | ★★ | 全流水线停顿，仅调试用 |

## 3.3 7种HardEvent（本算子全部使用）

| HardEvent | 含义 | 典型用法 |
|-----------|------|---------|
| `MTE1_MTE2` | L1搬运完成→通知MTE2 | L1到UB搬运前等L1就绪 |
| `MTE2_MTE1` | GM到L1完成→通知MTE1 | GM到L1搬运后信号 |
| `MTE2_V` | MTE2搬入完成→通知Vector | UB数据就绪→Vector开始 |
| `V_MTE2` | Vector完成→通知MTE2 | Vector写完→下一轮MTE2 |
| `V_MTE3` | Vector完成→通知MTE3 | UB数据就绪→搬出到GM |
| `MTE3_V` | MTE3搬出完成→通知Vector | 上一轮MTE3完成→Vector可复用buffer |
| `FIX_M` | Fixpipe完成→通知Cube | L0C搬到UB完成→Cube可复用 |
| `M_FIX` | Cube计算完成→通知Fixpipe | L0C就绪→Fixpipe开始 |
| `M_MTE1` | Cube完成→通知MTE1 | Cube用完L0→L1可开始下一轮 |

## 3.4 SetFlag/WaitFlag 使用模式

```cpp
// 1. Init阶段：分配事件ID
AscendC::AllocEventID<HardEvent::MTE2_V>();  // 分配

// 2. 搬运完成后：SetFlag（信号方）
DataCopyPad(xLocal, xGm, copyParams, padParams);
SetFlag<HardEvent::MTE2_V>();  // "我搬完了"

// 3. 计算开始前：WaitFlag（等待方）
WaitFlag<HardEvent::MTE2_V>();  // "等搬完再算"
Adds(yLocal, xLocal, scalar, count);
```

## 3.5 跨核同步：AIC ↔ AIV（FD开发核心）

```cpp
// AIC(Cube核) 设置flag → AIV(Vector核) 等待flag
// 这是双轴流水的核心通信机制！

CrossCoreSetFlag<0, PIPE_FIX>();    // AIC: "我算完了"
CrossCoreWaitFlag<0, PIPE_FIX>();   // AIV: "等Cube算完"

CrossCoreSetFlag<1, PIPE_MATMUL>(); // AIV: "我Softmax完了"
CrossCoreWaitFlag<1, PIPE_MATMUL>(); // AIC: "等VectorSoftmax完"
```

**FD开发关键**：AIC做Matmul，AIV做Softmax+Cast，两者必须通过CrossCore同步！

## 3.6 SyncAll（全核同步）

```cpp
// 所有核一起等，用于初始化阶段
SyncAll();  // 确保所有核初始化完毕再开始计算
```

## 3.7 速记口诀

> **搬运是异步的，必须同步！**
> **SetFlag是信号方，WaitFlag是等待方**
> **MTE2→V搬运到计算，V→MTE3计算到搬出**
> **跨核用CrossCore，AIC算AIV等**

---

# 第4章 基础计算：算术API优化秘籍

## 4.1 二元算术API（本算子全部使用）

| API | 功能 | 本算子使用场景 |
|-----|------|-------------|
| `Add` | dst=src0+src1 | FlashUpdate累加 |
| `Sub` | dst=src0-src1 | Sink计算: sink-max |
| `Mul` | dst=src0*src1 | FP8反量化: kv*scale |
| `Div` | dst=src0/src1 | Flash归约: result/expSum |
| `Exp` | dst=exp(src) | Softmax指数: exp(x-max) |
| `Log` | dst=log(src) | LSE输出: log(sum) |

## 4.2 标量优化：Adds/Muls（★★★★★ 重点）

**单行处理（AR模板）**：

```
传统方案: Duplicate(maxVal, rLength) + Sub(x, max) = 2条指令 + 大buffer
优化方案: Adds(x, -maxVal, rLength)              = 1条指令 + 32B buffer
```

```cpp
// ❌ 低效：Duplicate + Sub
Duplicate(broadcastLocal, maxVal, rLength);  // 指令1
Sub(yLocal, xLocal, broadcastLocal, rLength); // 指令2，需broadcastBuf

// ✅ 高效：Adds（1条指令）
Adds(yLocal, xLocal, -maxVal, rLength);  // 减法变加法，1条指令
```

**除法转乘法**：
```cpp
// ✅ 高效：Muls
T invSum = (T)1.0 / sumVal;  // CPU端算1/sum
Muls(yLocal, yLocal, invSum, rLength);  // 除法变乘法，1条指令
```

**节省量**：指令数-33%，Buffer-94%（512B→32B）

## 4.3 广播优化：src1RepStride=0（★★★★★ 重点）

**多行处理（ARA模板）**：

```cpp
// ❌ 低效：逐行循环
for (uint32_t r = 0; r < R; r++) {
    Sub(dst[r*cols], src[r*cols], scalar, cols);  // R次调用
}

// ✅ 高效：单次广播（R≤64）
Sub(dst, src, scalar, mask, R,
    {1, 1, 1, cols/8, cols/8, 0});  // src1RepStride=0, 1次调用

// ✅ 高效：分批广播（R>64）
for (batch...) {
    Sub(dst[offset], src[offset], scalar, mask, batchSize,
        {1, 1, 1, cols/8, cols/8, 0});
}
```

**BinaryRepeatParams解读**：
```
dstBlkStride=1     : 单次迭代内dst步1个block
src0BlkStride=1    : 单次迭代内src0步1个block
src1BlkStride=1    : 单次迭代内src1步1个block
dstRepStride=cols/8  : 每次迭代dst前进cols个元素
src0RepStride=cols/8 : 每次迭代src0前进cols个元素
src1RepStride=0      : 每次迭代src1不前进 → 广播！
```

## 4.4 repeatTime ≤ 255 限制

```cpp
// uint8_t类型，256溢出为0 → 不执行任何计算！
// 解决方案：分批处理，每批≤255
uint8_t batchSize = min(remainingRows, 255);
Sub(dst[offset], src[offset], scalar, mask, batchSize, repeatParams);
```

## 4.5 本算子中的Muls（MicroAPI VF版本）

```cpp
// VF版Muls：scalar * register tensor
// 本算子用于Softmax缩放和FP8反量化
Muls(vDequantRes, vKvData, scaleVal, mask);  // KV * scale
Muls(vSoftmaxIn, vInput, softmaxScale, mask); // input * 1/sqrt(d)
```

## 4.6 速记口诀

> **单行用Adds/Muls，1条指令胜2条**
> **多行用src1RepStride=0，广播不Duplicate**
> **除法转乘法，减法变加法**
> **repeatTime不超255，超了要分批**

---

# 第5章 归约操作：ReduceMax & ReduceSum

## 5.1 Level 2接口（逐行处理，Softmax/LayerNorm常用）

```cpp
// ReduceMax：找一行中的最大值
ReduceMax<T>(dst, src, tmpBuffer, count, false);
// dst: 1个元素的最大值
// count: 有效元素个数（rLength，不是rLengthAlign！）
// tmpBuffer: 类型必须与T相同！

// ReduceSum：求一行之和
ReduceSum<T, true>(dst, src, tmpBuffer, count);
// isSetMask=true: 自动处理mask
```

**致命易错点**：

| 参数位置 | 用rLength | 用rLengthAlign |
|---------|-----------|-----------------|
| Reduce count | ✓ | ✗ |
| Sub/Exp count | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer大小 | ✗ | ✓ |

## 5.2 Pattern接口（跨行批量）

```cpp
// AR Pattern：每行归约为1值 → 输出rows个值
uint32_t srcShape[] = {rows, alignedCols};  // alignedCols必须32B对齐
ReduceMax<float, Pattern::Reduce::AR>(dst, src, tmpLocal, srcShape, true);

// 推荐形式1：显式传入tmpLocal
LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
ReduceMax<float, Pattern::Reduce::AR, true>(dst, src, tmpLocal, srcShape, true);
```

## 5.3 tmpBuffer类型必须匹配

```cpp
// ❌ 编译错误：类型不匹配
LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();
ReduceMax(rowTmp, src, tmp, count);  // uint8_t与float不匹配！

// ✅ 正确：类型必须相同
LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
ReduceMax(rowTmp, src, reduceTmp, count);
```

## 5.4 本算子中的归约场景

```
Softmax流程:
1. ReduceMax → 找行最大值max
2. Adds(x, -max) → 数值稳定
3. Exp(x) → 指数化
4. ReduceSum → 求行和sum
5. Muls(x, 1/sum) → 归一化

LSE输出:
6. Log(sum) → log-softmax
```

## 5.5 速记口诀

> **逐行用Level2，跨行用Pattern AR**
> **count传rLength，offset用rLengthAlign**
> **tmpBuffer类型必须与T相同**
> **dst≠tmpBuffer，必须分开！**

---

# 第6章 精度转换：Cast全链路

## 6.1 Cast RoundMode选择（秒查表）

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half→float | `CAST_NONE` | 低→高精度，无损失 |
| float→half | `CAST_ROUND` | 高→低精度，需舍入 |
| bf16→float | `CAST_NONE` | 低→高精度 |
| float→bf16 | `CAST_ROUND` | 高→低精度 |
| int8→half/bf16 | `CAST_NONE` | 整数→浮点 |
| FP8→bf16/float | `CAST_RINT` | 本算子FP8反量化 |

## 6.2 混合精度模式（Softmax标准流程）

```
half/bf16输入 → Cast(FP32) → ReduceMax/Exp/ReduceSum(FP32) → Cast(half/bf16) → 输出
```

```cpp
// Step1: 低→高精度
Cast<float, half>(xFloat, xHalf, CAST_NONE, count);
// Step2: FP32计算
ReduceMax<float>(tmp, xFloat, reduceTmp, count, false);
Adds<float>(yFloat, xFloat, -maxVal, count);
Exp<float>(yFloat, yFloat, count);
ReduceSum<float, true>(tmp, yFloat, reduceTmp, count);
Muls<float>(yFloat, yFloat, invSum, count);
// Step3: 高→低精度
Cast<half, float>(yHalf, yFloat, CAST_ROUND, count);
```

## 6.3 本算子精度链路（★★★★★ FD核心）

```
KV存储: FP8_e8m0 (4bit/8bit)
    ↓ Cast(FP8→bf16/float)  [AntiquantVFImplFp8D448]
反量化: kv * scale → bf16
    ↓ Muls(kv_bf16, scale)
Softmax: bf16→float → Softmax → float→bf16
    ↓ Cast(bf16→float) + Softmax + Cast(float→bf16)
Bmm2输出: bf16→float → Add/Sub/Mul/Div → float→bf16
    ↓ Cast + FlashUpdate + Cast
```

## 6.4 ReinterpretCast（不转换数据，只换类型视角）

```cpp
// 本算子用于FP8反量化：KV_T(bf16)视为int8_t来处理
LocalTensor<int8_t> kvInt8 = kvUb.ReinterpretCast<int8_t>();
// 数据不变，只是类型标签变了
```

## 6.5 VF版Cast（MicroAPI CastTrait）

```cpp
// 本算子arch35使用的VF Cast
using CastTraitFp8ToBf16 = MicroAPI::CastTrait<bfloat16_t, fp8_e8m0_t,
    MicroAPI::RegLayout::ONE,   // src layout
    MicroAPI::SatMode::NO_SAT,  // 不饱和
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT>;       // FP8专用舍入模式

Cast<CastTraitFp8ToBf16>(vDst, vSrc, mask);
```

## 6.6 速记口诀

> **低→高用CAST_NONE，高→低用CAST_ROUND**
> **FP8反量化用CAST_RINT**
> **混合精度：输入Cast高，算完Cast低**
> **ReinterpretCast不改数据，只换视角**

---

# 第7章 MicroAPI VF体系（★★★★★ VF大融合核心）

## 7.1 VF大融合是什么？

> **VF = Vector Fusion**：将多个Vector操作融合到一次寄存器级流水线中
> **传统AscendC API**：基于UB LocalTensor，每次操作读写UB
> **MicroAPI VF**：基于寄存器RegTensor，Load→连续计算→Store，中间不回UB

```
传统流程(读写UB):  Load→UB → Add→UB → Mul→UB → Exp→UB → Store  (4次UB读写)
VF融合流程(寄存器): Load→Reg → Add→Reg → Mul→Reg → Exp→Reg → Store  (1次UB读写)
```

## 7.2 VF三步曲：Load → Compute → Store

```cpp
// Step1: Load（UB→寄存器）
MicroAPI::RegTensor<float> vSrc;
MicroAPI::LoadAlign<float, POST_MODE_UPDATE, DIST_NORM>(vSrc, ubAddr);

// Step2: Compute（寄存器间运算）
MicroAPI::RegTensor<float> vDst;
MicroAPI::Add(vDst, vSrc, vOther, mask);

// Step3: Store（寄存器→UB）
MicroAPI::StoreAlign<float, DATA_BLOCK_COPY>(vDst, ubAddr, mask);
```

## 7.3 RegTensor：VF的寄存器类型

```cpp
// 声明寄存器变量（不分配内存，映射到硬件寄存器）
MicroAPI::RegTensor<float> vResult;
MicroAPI::RegTensor<bfloat16_t> vKvData;
MicroAPI::RegTensor<fp8_e8m0_t> vFp8Data;

// 类型映射：fp8_e8m0_t → 4bit/8bit FP8格式
// bfloat16_t → BF16
// float → FP32
// half → FP16
```

## 7.4 MaskReg：VF的条件控制

```cpp
// 创建全mask（处理整行）
MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

// 创建半mask（处理前半行）
MicroAPI::MaskReg maskHalf = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();

// 动态更新mask（处理尾部不满一行）
MicroAPI::UpdateMask<float>(mask, tailCount);
```

## 7.5 LoadAlign详解（本算子核心操作）

```cpp
// 基础Load：从UB加载到寄存器
MicroAPI::LoadAlign<T, PostLiteral, LoadDist>(regTensor, ubAddr);
```

| LoadDist模式 | 含义 | 本算子使用 |
|-------------|------|-----------|
| `DIST_NORM` | 正常连续加载 | 通用加载 |
| `DIST_UNPACK4_B8` | 4组解包加载(8bit) | FP8反量化加载 |
| `DIST_BRC_B32` | 广播加载(32字节) | scale广播加载 |
| `DIST_E2B_B32` | 偶数解包到32B | FP8→bf16中间格式 |

| PostLiteral | 含义 |
|------------|------|
| `POST_MODE_UPDATE` | 加载后自动递增地址（流水线模式） |
| 无 | 单次加载，不递增 |

## 7.6 StoreAlign详解

```cpp
// 从寄存器写回UB
MicroAPI::StoreAlign<T, DataCopyMode, PostLiteral>(regTensor, ubAddr, mask);
```

| DataCopyMode | 含义 |
|-------------|------|
| `DATA_BLOCK_COPY` | 按块拷贝模式 |

| StoreDist | 含义 |
|----------|------|
| `DIST_NORM_B32` | 正常32字节对齐存储 |

## 7.7 VF计算API汇总（本算子使用的全部VF API）

| API | 功能 | 本算子场景 | 对应传统API |
|-----|------|-----------|-----------|
| `Mul` | 逐元素乘 | kv*scale, expMax*preTensor | AscendC::Mul |
| `Add` | 逐元素加 | Flash累加, expSum累加 | AscendC::Add |
| `Sub` | 逐元素减 | sink-max | AscendC::Sub |
| `Div` | 逐元素除 | result/expSum | AscendC::Div |
| `Exp` | 指数 | exp(sink-max) | AscendC::Exp |
| `Log` | 对数 | log(sum) LSE输出 | AscendC::Log |
| `Max` | 行最大值 | Softmax找max | AscendC::ReduceMax |
| `Muls` | 标量乘 | softmaxScale, FP8反量化 | AscendC::Muls |
| `ExpSub` | 融合exp(x-y) | softmax: exp(input-max) | AscendC::Exp + Sub |
| `FusedExpSub` | 融合exp(maxDiff) | FlashUpdate: exp(oldMax-newMax) | 无对应 |
| `Duplicate` | 填充标量 | 清零, 初始化 | AscendC::Duplicate |
| `DeInterleave` | 解交织 | FP8→bf16输出打包 | 无对应 |
| `Or` | 按位或 | bf16奇偶合并NZ输出 | AscendC::Or |
| `Select` | 条件选择 | 无效行处理 | AscendC::Select |
| `Compare` | 比较 | LSE条件判断 | AscendC::Compare |
| `Cast` | 精度转换(VF版) | FP8→bf16, bf16→fp32 | AscendC::Cast |
| `SoftmaxFlashV2` | Flash Softmax(VF版) | Softmax计算 | 无对应(禁止AscendC::Softmax) |

## 7.8 VF融合示例：Softmax SFA块（★★★★★）

```
// 本算子arch35 Softmax VF流程:
LoadAlign(bf16, DIST_NORM)        → 加载input到寄存器(vInput)
Muls(vInput, softmaxScale)        → 缩放: input * 1/sqrt(d)
SoftmaxFlashV2(vInput, vMax, vSum) → Softmax融合计算
Cast(bf16→bf16, NZ格式)           → CastTrait格式转换
DeInterleave(奇偶分离)             → NZ格式输出打包
Or(奇偶合并)                       → 组合NZ输出
StoreAlign(结果→UB)               → 写回UB
```

**VF融合优势**：7步操作只Load一次、Store一次，中间全在寄存器！

## 7.9 VF融合示例：FP8反量化（★★★★★）

```
LoadAlign(fp8, DIST_UNPACK4_B8)   → 加载FP8数据(4组解包)
Cast(fp8→bf16, CastTrait)         → FP8转为BF16
LoadAlign(scale, DIST_BRC_B32)    → 广播加载scale
Mul(kv_bf16 * scale)              → 反量化: kv*scale
DeInterleave(解交织)               → 输出格式调整
StoreAlign(结果→UB)               → 写回
```

## 7.10 VF融合示例：FlashUpdate（★★★★★ FD核心）

```
LoadAlign(preTensor)               → 加载前一轮softmax结果
Muls(preTensor, expOldMax)         → 乘以旧最大值的指数
LoadAlign(curTensor)               → 加载当前轮softmax
Add(expOld*pre + cur)              → 累加
StoreAlign(累加结果)               → 写回

LoadAlign(expSum)                  → 加载指数和
Div(result / expSum)               → 归一化
StoreAlign(最终结果)               → 写回
```

## 7.11 GetPhyAddr：VF与UB的桥梁

```cpp
// VF的LoadAlign/StoreAlign需要__ubuf__物理地址
auto ubAddr = MicroAPI::GetPhyAddr<ALIGN_128B>(ubTensor);
LoadAlign<float, POST_MODE_UPDATE, DIST_NORM>(vReg, ubAddr);
```

## 7.12 速记口诀

> **VF=寄存器级融合，Load→算→Store，中间不回UB**
> **RegTensor是寄存器，MaskReg是条件**
> **LoadAlign进，StoreAlign出，Dist模式选对**
> **FP8用UNPACK4_B8，广播用BRC_B32**
> **SoftmaxFlashV2是VF专属，禁止用AscendC::Softmax**

---

# 第8章 Cube侧Matmul体系

## 8.1 Cube引擎三级存储

```
GM(全局内存) → L1(本地缓存) → L0A/L0B(矩阵缓存) → L0C(结果缓存) → UB
```

| 存储层 | 容量 | 用途 |
|--------|------|------|
| L0A | 64KB | 左矩阵(Q)缓存 |
| L0B | 64KB | 右矩阵(KV)缓存 |
| L0C | 128KB | Matmul结果缓存 |
| L1 | 512KB | GM→L0中间缓存 |

## 8.2 Mmad：矩阵乘累加

```cpp
// 本算子Bmm1(Q*KV)和Bmm2(P*KV)的核心操作
MmadParams params;
params.m = mSize;           // 左矩阵行数
params.n = nSize;           // 右矩阵列数
params.k = kSize;           // 内积维度
params.cmatrixInitVal = 0;  // C矩阵初始值(0=不累加)
params.cmatrixSource = 0;   // C矩阵来源(0=从L0C)
params.unitFlag = 0;        // 通用标志

Mmad(l0cTensor, l0aTensor, l0bTensor, params);
```

## 8.3 LoadData：L1→L0A/L0B

```cpp
// 3D LoadData（arch35专用，支持MK/KN布局）
LoadData3DParamsV2<T> params;
params.startIndex = 0;
params.strideM = mAlign / 16;   // 行步长
params.strideK = kAlign / 16;   // 列步长
params.repeatTimes = repeat;     // 重复次数
params.baseBlockSize = block;    // 基础块大小

LoadData(l0aTensor, l1aTensor, params);  // L1→L0A
LoadData(l0bTensor, l1bTensor, params);  // L1→L0B
```

## 8.4 Fixpipe：L0C→UB（结果搬出）

```cpp
// Mmad结果从L0C搬到UB，NZ→ND格式转换
FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
params.nSize = nAlign;    // N维度对齐大小
params.mSize = mAlign;    // M维度对齐大小
params.srcStride = srcStride;
params.dstStride = dstStride;

Fixpipe(ubTensor, l0cTensor, params);
```

## 8.5 本算子Matmul流程（Bmm1 SCFA）

```
GM→L1: CopyToL1Nd2Nz(queryGM, queryL1)    // Query搬到L1(NZ格式)
L1→L0A: LoadData(queryL0A, queryL1)        // Query从L1到L0A
L1→L0B: LoadData(kvL0B, kvL1)              // KV从L1到L0B
Cube: Mmad(l0c, l0a, l0b, params)          // 矩阵乘累加
L0C→UB: Fixpipe(ubResult, l0c, params)     // 结果搬到UB
```

## 8.6 速记口诀

> **L0A存左矩阵Q，L0B存右矩阵KV**
> **Mmad做乘累加，Fixpipe做搬出转换**
> **LoadData是L1→L0，CopyToL1Nd2Nz是GM→L1格式转换**
> **GM→L1→L0A/L0B→Mmad→L0C→Fixpipe→UB，六级流水**

---

# 第9章 Tensor与核管理

## 9.1 GlobalTensor vs LocalTensor

| 类型 | 位置 | 用途 | 声明 |
|------|------|------|------|
| `GlobalTensor<T>` | GM | 访问全局内存 | `GlobalTensor<float> xGm;` |
| `LocalTensor<T>` | UB/L1/L0 | 访问片上内存 | `LocalTensor<float> xLocal;` |

## 9.2 SetGlobalBuffer：绑定GM地址

```cpp
// Init阶段：为GlobalTensor绑定GM物理地址
xGm.SetGlobalBuffer((__gm__ float*)addr + offset, size);
```

**本算子绑定的GM Tensor**：query, kvCache, blockTable, actualSeqLens, cuSeqlens, attentionOut, metadata等

## 9.3 GetValue：读取单个标量（仅调试）

```cpp
// 从GlobalTensor读1个值
float val = xGm.GetValue(idx);  // ⛔ 生产禁止，仅调试

// 从LocalTensor读1个值
float maxVal = reduceBuf.Get<float>().GetValue(0);  // ⛔ 仅调试
```

## 9.4 核管理API

| API | 功能 | 本算子用法 |
|-----|------|-----------|
| `GetBlockIdx()` | 当前核编号 | aicIdx/aivIdx分配 |
| `GetBlockNum()` | 总核数 | 单核输出大小计算 |
| `GetSubBlockIdx()` | 子块编号(0或1) | 双子块模式split work |
| `ASCEND_IS_AIC` | 是否是Cube核 | 条件编译分支 |
| `ASCEND_IS_AIV` | 是否是Vector核 | 条件编译分支 |

```cpp
// 本算子双轴架构
if ASCEND_IS_AIC {
    // Cube核执行Matmul
    CubeBlock.Process();
} else if ASCEND_IS_AIV {
    // Vector核执行Softmax/Cast
    VecBlock.Process();
}
```

## 9.5 GetUserWorkspace：获取工作空间

```cpp
// Kernel入口获取GM工作空间
auto workspace = GetUserWorkspace((__gm__ uint8_t*)workspaceGm);
```

## 9.6 速记口诀

> **Global是GM，Local是片上**
> **SetGlobalBuffer绑定地址，GetValue仅调试**
> **AIC做Cube，AIV做Vector，IS_AIC/IS_AIV分支**
> **GetSubBlockIdx拆双子块，GetBlockIdx知核号**

---

# 第10章 综合实战：从本算子到FD开发

## 10.1 本算子完整流水线（双轴）

```
AIC(Cube核)流水线:
  GM→L1(CopyToL1Nd2Nz) → L1→L0A(LoadData) → Mmad(Q*KV)
  → CrossCoreSetFlag("Bmm1完成") → Mmad(P*KV)
  → CrossCoreSetFlag("Bmm2完成")

AIV(Vector核)流水线:
  CrossCoreWaitFlag("Bmm1完成") → Fixpipe(L0C→UB)
  → SoftmaxFlashV2(VF融合) → CrossCoreSetFlag("Softmax完成")
  → CrossCoreWaitFlag("Bmm2完成") → FlashUpdate(VF融合)
  → Cast(bf16→输出) → DataCopyPad(UB→GM)
```

## 10.2 FD开发需要掌握的API清单

### 必须精通（★★★★★）

| API类别 | API | 原因 |
|---------|-----|------|
| 数据搬运 | DataCopyPad | 所有CopyIn/CopyOut |
| 数据搬运 | Fixpipe | L0C→UB核心 |
| 数据搬运 | CopyToL1Nd2Nz | GM→L1格式转换 |
| Buffer | TBuf/TQue | Buffer规划 |
| 同步 | SetFlag/WaitFlag | 流水线同步 |
| 同步 | CrossCoreSetFlag/WaitFlag | 双轴同步 |
| VF计算 | Mul/Add/Sub/Div/Exp | FlashUpdate核心 |
| VF计算 | Muls | Softmax缩放/反量化 |
| VF计算 | SoftmaxFlashV2 | Softmax融合计算 |
| VF IO | LoadAlign/StoreAlign | VF数据进出 |
| VF类型 | RegTensor/MaskReg | VF寄存器体系 |
| 精度 | Cast(CastTrait) | FP8/bf16/fp32全链路 |
| Cube | Mmad | 矩阵乘核心 |
| Cube | LoadData | L1→L0加载 |

### 需要了解（★★★）

| API | 原因 |
|-----|------|
| Duplicate | 初始化/清零 |
| DeInterleave/Or | NZ格式输出打包 |
| Select/Compare | 无效行处理 |
| ExpSub/FusedExpSub | Flash指数融合 |
| Log | LSE输出 |
| ReinterpretCast | 类型视角切换 |
| GetPhyAddr | VF地址提取 |
| SyncAll | 全核初始化同步 |

## 10.3 FD开发 vs 本算子的差异预判

| 维度 | 本算子(mixed_quant_sparse_flash_mla) | FD(FusedDecoder) |
|------|--------------------------------------|------------------|
| 输入格式 | FP8 KV + sparse block table | BF16/FP16 KV + paged attention |
| Softmax | SoftmaxFlashV2(VF版) | 同样使用VF版Softmax |
| Matmul | Mmad + SCFA(Cube侧) | 同样Mmad + 可能不同tiling |
| 反量化 | FP8→bf16 VF融合 | 可能BF16直入，无反量化 |
| CrossCore | AIC→AIV Bmm1/Bmm2通知 | 同样需要双轴同步 |
| 输出格式 | NZ→ND(Fixpipe) | 同样需要格式转换 |

## 10.4 速记口诀（全书总结）

> **搬运用DataCopyPad，stride单位分清**
> **Buffer搬运用TQue，计算用TBuf**
> **同步EnQue/DeQue，跨核CrossCore**
> **单行Adds/Muls，多行RepStride=0**
> **Reduce逐行Level2，count传rLength**
> **低→高CAST_NONE，高→低CAST_ROUND**
> **VF=Load→算→Store，RegTensor寄存器级融合**
> **Mmad做矩阵乘，Fixpipe做结果搬出**
> **AIC做Cube，AIV做Vector，双轴CrossCore同步**

---

# 附录A：API速查表（全85+ API）

## A.1 数据搬运类(8个)

| API | 位置 | 简述 |
|-----|------|------|
| DataCopyPad | GM↔UB | 主力搬运，支持非对齐 |
| DataCopy | GM↔UB | ⛔ 禁止，不支持非对齐 |
| DataCopyExtParams | 参数 | blockCount/blockLen/stride(字节) |
| DataCopyPadExtParams | 参数 | isPad/leftPad/rightPad/padVal |
| Fixpipe | L0C→UB | Cube结果搬出，NZ→ND |
| CopyToL1Nd2Nz | GM→L1 | ND格式→NZ格式转换 |
| LoadData | L1→L0 | Matmul输入加载 |
| Nd2NzParams/Dn2NzParams | 参数 | ND→NZ格式转换参数 |

## A.2 Buffer/队列类(6个)

| API | 简述 |
|-----|------|
| TBuf<VECCALC> | 纯计算buffer |
| TQue<VECIN/VECOUT> | 搬运buffer，depth=2 |
| AllocTensor | 分配tensor |
| FreeTensor | 释放tensor |
| EnQue | 入队(同步点) |
| DeQue | 出队(等待就绪) |

## A.3 同步类(5个)

| API | 简述 |
|-----|------|
| SetFlag<HardEvent> | 信号方设置flag |
| WaitFlag<HardEvent> | 等待方等待flag |
| CrossCoreSetFlag | AIC→AIV跨核信号 |
| CrossCoreWaitFlag | AIV等待AIC信号 |
| SyncAll | 全核同步 |

## A.4 基础计算类(AscendC版, 8个)

| API | 简述 |
|-----|------|
| Add/Sub/Mul/Div | 二元算术 |
| Exp/Log | 指数/对数 |
| Adds/Muls | 标量加/乘(★★★优化) |
| Duplicate | 填充标量 |

## A.5 归约类(2个)

| API | 简述 |
|-----|------|
| ReduceMax | 找最大值 |
| ReduceSum | 求和 |

## A.6 精度转换类(3个)

| API | 简述 |
|-----|------|
| Cast | 精度转换 |
| ReinterpretCast | 类型视角切换 |
| CastTrait | VF版Cast配置 |

## A.7 VF计算类(MicroAPI版, 16个)

| API | 简述 |
|-----|------|
| VF::Mul/Add/Sub/Div | 寄存器级算术 |
| VF::Exp/Log | 寄存器级指数/对数 |
| VF::Muls | 寄存器级标量乘 |
| VF::ExpSub | 融合exp(x-y) |
| VF::FusedExpSub | 融合exp(maxDiff) |
| VF::Duplicate | 寄存器级填充 |
| VF::DeInterleave | 解交织 |
| VF::Or | 按位或(NZ打包) |
| VF::Select | 条件选择 |
| VF::Compare/Compares | 比较 |
| VF::Cast | 寄存器级精度转换 |
| VF::SoftmaxFlashV2 | Flash Softmax融合 |
| VF::Max | 行最大值(VF版) |

## A.8 VF IO类(6个)

| API | 简述 |
|-----|------|
| RegTensor<T> | 寄存器tensor |
| MaskReg | 条件mask |
| CreateMask | 创建mask |
| UpdateMask | 动态更新mask |
| LoadAlign | UB→寄存器 |
| StoreAlign | 寄存器→UB |

## A.9 Cube类(4个)

| API | 简述 |
|-----|------|
| Mmad | 矩阵乘累加 |
| MmadParams | Mmad参数 |
| LoadData3DParamsV2 | 3D加载参数 |
| FixpipeParamsC310 | Fixpipe参数 |

## A.10 Tensor/核管理类(8个)

| API | 简述 |
|-----|------|
| GlobalTensor<T> | GM tensor |
| LocalTensor<T> | 片上tensor |
| SetGlobalBuffer | 绑定GM地址 |
| GetValue | 读标量(调试用) |
| GetPhyAddr | VF物理地址 |
| GetBlockIdx/GetBlockNum | 核编号/总数 |
| GetSubBlockIdx | 子块编号 |
| AllocEventID<HardEvent> | 分配事件ID |

---

# 附录B：7天学习计划

| 天 | 章节 | 碎片时间安排 | 练习目标 |
|----|------|-------------|---------|
| D1 | 第1章数据搬运 | 早地铁10min+晚10min | 理解stride单位，默写DataCopyPad参数 |
| D2 | 第2章Buffer+第3章同步 | 早地铁8min+午休8min+晚8min | 画DoubleBuffer时序图，默写SetFlag/WaitFlag |
| D3 | 第4章基础计算 | 早地铁10min+晚10min | 写Adds/Muls优化，算src1RepStride |
| D4 | 第5章归约+第6章精度 | 早8min+午8min+晚8min | 理解Level2 vs Pattern，默写Cast RoundMode |
| D5 | 第7章VF体系 | 早15min+午15min+晚15min | **重点日**：画VF融合流程，理解LoadAlign Dist模式 |
| D6 | 第8章Cube+第9章核管理 | 早10min+晚10min | 理解Mmad三级存储，画GM→L1→L0→L0C→UB流水 |
| D7 | 第10章综合实战 | 早15min+午15min+晚15min | 画双轴流水图，对照本算子源码验证 |

---

# 附录C：HardEvent速查表

| 事件 | 方向 | 含义 | 本算子使用 |
|------|------|------|-----------|
| MTE2_V | MTE2→V | 搬入完成→计算 | ✓ CopyIn→Compute |
| V_MTE2 | V→MTE2 | 计算完成→下一轮搬入 | ✓ Compute→CopyIn |
| V_MTE3 | V→MTE3 | 计算完成→搬出 | ✓ Compute→CopyOut |
| MTE3_V | MTE3→V | 搬出完成→可复用buffer | ✓ CopyOut→Compute |
| MTE1_MTE2 | MTE1→MTE2 | L1就绪→UB搬入 | ✓ L1→UB |
| MTE2_MTE1 | MTE2→MTE1 | GM到L1完成 | ✓ GM→L1 |
| FIX_M | Fixpipe→Cube | Fixpipe完成→Cube可复用L0C | ✓ Bmm结果搬出 |
| M_FIX | Cube→Fixpipe | Cube计算完成→Fixpipe开始 | ✓ L0C就绪 |
| M_MTE1 | Cube→MTE1 | Cube用完L0→L1可开始 | ✓ L0释放→L1加载 |

---

# 附录D：VF LoadDist模式速查

| LoadDist | 适用场景 | 本算子用法 |
|----------|---------|-----------|
| DIST_NORM | 正常连续数据 | 通用加载softmax输入 |
| DIST_UNPACK4_B8 | FP8 4组数据 | FP8 KV反量化加载 |
| DIST_BRC_B32 | 标量广播 | scale因子广播 |
| DIST_E2B_B32 | 偶数位置数据 | FP8→bf16中间步骤 |
| DIST_DINTLV_B32 | 解交织数据 | Softmax NZ输出 |