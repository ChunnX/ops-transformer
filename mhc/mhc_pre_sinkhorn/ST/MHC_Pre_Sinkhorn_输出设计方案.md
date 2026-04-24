# MHC_Pre_Sinkhorn 算子8输出实现方案

## 一、输出概述

根据文档和需求，算子需要支持8个输出：

| 序号 | 输出名 | 类型 | Shape | 是否必选 | 说明 |
|------|--------|------|-------|---------|------|
| 1 | hin | BF16 | (bs, seq_len, c) | 必选 | 原始输出y，Attention/MLP层输入 |
| 2 | hPost | FP32 | (bs, seq_len, n) | 必选 | 原始输出post，后处理变换矩阵 |
| 3 | hRes | FP32 | (bs, seq_len, n*n) | 必选 | 原始输出comb_frag，未做Sinkhorn |
| 4 | hPre | FP32 | (bs, seq_len, n) | 可选 | sigmoid(H_pre) + hc_eps 结果 |
| 5 | hcBeforeNorm | FP32 | (bs, seq_len, n*n + 2*n) | 可选 | x @ phi 矩阵乘法结果 |
| 6 | invRms | FP32 | (bs, seq_len, 1) | 可选 | RmsNorm的1/r结果 |
| 7 | sumOut | FP32 | (numIters*2, bs, seq_len, n) | 可选 | Sinkhorn迭代中间sum结果 |
| 8 | normOut | FP32 | (numIters*2, bs, seq_len, n, n) | 可选 | Sinkhorn迭代中间norm结果 |

---

## 二、计算公式与数据流

### 2.1 完整计算流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Stage1: 矩阵乘法阶段                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入 x (BF16) [bs, n, c]                                                   │
│       │                                                                     │
│       ├─→ Cast to FP32 ─→ x_fp32                                            │
│       │                    │                                                │
│       │                    ├─→ SquareSum ─→ square_sum [bs]                 │
│       │                    │                                                │
│       │                    └─→ Store to Workspace                           │
│       │                                                                     │
│       └─→ x_fp32 @ phi^T ─→ hcBeforeNorm [bs, n*n + 2*n]  ◄── 输出5        │
│                               │                                             │
│                               ├─→ mix0 [bs, n]                              │
│                               ├─→ mix1 [bs, n]                              │
│                               └─→ mix2 [bs, n*n]                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Stage2: 后处理阶段                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  square_sum ─→ rsqrt = 1/sqrt(square_sum/d + eps) ─→ invRms ◄── 输出6      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 分支1: mix0 → hPre → hin                                             │   │
│  │                                                                       │   │
│  │  mix0 * rsqrt * alpha[0] + bias[0:n]                                 │   │
│  │           │                                                           │   │
│  │           ├─→ sigmoid ─→ hPre [bs, n]  ◄── 输出4                      │   │
│  │           │              │                                            │   │
│  │           │              └─→ + hc_eps ─→ weights                      │   │
│  │           │                         │                                 │   │
│  │           │                         └─→ x * weights ─→ sum(dim=n)    │   │
│  │           │                                    │                      │   │
│  │           │                                    └─→ hin [bs, c] ◄── 输出1│
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 分支2: mix1 → hPost                                                  │   │
│  │                                                                       │   │
│  │  mix1 * rsqrt * alpha[1] + bias[n:2n]                                │   │
│  │           │                                                           │   │
│  │           └─→ 2 * sigmoid ─→ hPost [bs, n]  ◄── 输出2                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 分支3: mix2 → hRes → Sinkhorn迭代                                    │   │
│  │                                                                       │   │
│  │  mix2 * rsqrt * alpha[2] + bias[2n:]                                 │   │
│  │           │                                                           │   │
│  │           └─→ softmax ─→ hRes [bs, n*n]  ◄── 输出3                    │   │
│  │                        │                                              │   │
│  │                        └─→ Sinkhorn迭代 (numIters次)                  │   │
│  │                                    │                                  │   │
│  │                                    ├─→ sumOut [...]  ◄── 输出7        │   │
│  │                                    │                                  │   │
│  │                                    └─→ normOut [...] ◄── 输出8        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各输出的详细计算公式

#### 输出1: hin (原始输出y)
```
计算流程:
1. mix0 = x @ phi[0:n, :]^T
2. mix0_scaled = mix0 * rsqrt * alpha[0] + bias[0:n]
3. hPre = sigmoid(mix0_scaled) + hc_eps
4. hin = sum(x * hPre, dim=n)

Shape: (bs, seq_len, c)
数据类型: BF16
```

#### 输出2: hPost (原始输出post)
```
计算流程:
1. mix1 = x @ phi[n:2n, :]^T
2. mix1_scaled = mix1 * rsqrt * alpha[1] + bias[n:2n]
3. hPost = 2 * sigmoid(mix1_scaled)

Shape: (bs, seq_len, n)
数据类型: FP32
```

#### 输出3: hRes (原始输出comb_frag，未做Sinkhorn)
```
计算流程:
1. mix2 = x @ phi[2n:n*n+2n, :]^T
2. mix2_scaled = mix2 * rsqrt * alpha[2] + bias[2n:]
3. hRes = softmax(mix2_scaled, dim=-1) + hc_eps

Shape: (bs, seq_len, n*n)
数据类型: FP32
```

#### 输出4: hPre (sigmoid后的H_pre)
```
计算流程:
1. mix0 = x @ phi[0:n, :]^T
2. mix0_scaled = mix0 * rsqrt * alpha[0] + bias[0:n]
3. hPre = sigmoid(mix0_scaled) + hc_eps

Shape: (bs, seq_len, n)
数据类型: FP32
注意: 这是sigmoid后的结果，用于反向传播
```

#### 输出5: hcBeforeNorm (x @ phi的结果)
```
计算流程:
hcBeforeNorm = x @ phi^T

Shape: (bs, seq_len, n*n + 2*n)
数据类型: FP32
内容: [mix0, mix1, mix2] 拼接，未经过归一化缩放
```

#### 输出6: invRms (RmsNorm的1/r结果)
```
计算流程:
1. square_sum = sum(x^2, dim=-1)
2. invRms = 1 / sqrt(square_sum / d + normEps)

Shape: (bs, seq_len, 1)
数据类型: FP32
```

#### 输出7: sumOut (Sinkhorn迭代的sum结果)
```
计算流程:
对于每次迭代 i = 0, 1, ..., numIters-1:
  sumOut[2i]   = sum(normOut[2i-1], dim=-1) + hcEps   (row sum)
  sumOut[2i+1] = sum(normOut[2i], dim=-2) + hcEps     (col sum)

Shape: (numIters * 2, bs, seq_len, n)
数据类型: FP32
注意: 根据needGrad参数决定是否输出
```

#### 输出8: normOut (Sinkhorn迭代的norm结果)
```
计算流程:
初始化:
  normOut[0] = softmax(hRes, dim=-1) + hcEps

迭代:
  normOut[2i]   = normOut[2i-1] / sumOut[2i]
  normOut[2i+1] = normOut[2i] / sumOut[2i+1]

Shape: (numIters * 2, bs, seq_len, n, n)
数据类型: FP32
注意: 根据needGrad参数决定是否输出
```

---

## 三、代码修改方案

### 3.1 Tiling数据结构修改

**文件**: `op_host/op_tiling/mhc_pre_sinkhorn_tiling.h`

```cpp
BEGIN_TILING_DATA_DEF(MhcPreSinkhornTilingData)
// ... 原有字段 ...
TILING_DATA_FIELD_DEF(int64_t, needGrad);        // 新增: 是否需要输出额外梯度信息
TILING_DATA_FIELD_DEF(int64_t, sumOutOffset);    // 新增: sumOut在workspace中的偏移
TILING_DATA_FIELD_DEF(int64_t, normOutOffset);   // 新增: normOut在workspace中的偏移
TILING_DATA_FIELD_DEF(int64_t, hPreOffset);      // 新增: hPre在workspace中的偏移
TILING_DATA_FIELD_DEF(int64_t, hcBeforeNormOffset); // 新增: hcBeforeNorm偏移
TILING_DATA_FIELD_DEF(int64_t, invRmsOffset);    // 新增: invRms在workspace中的偏移
END_TILING_DATA_DEF;
```

### 3.2 算子定义修改

**文件**: `op_host/mhc_pre_sinkhorn_def.cpp`

```cpp
class MhcPreSinkhorn : public OpDef {
public:
    explicit MhcPreSinkhorn(const char *name) : OpDef(name)
    {
        // 输入参数 (保持不变)
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Input("phi").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("alpha").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        // 输出参数 (扩展为8个输出)
        this->Output("hin").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("hPost").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("hRes").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("hPre").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("hcBeforeNorm").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("invRms").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("sumOut").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("normOut").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        // 属性参数
        this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
        this->Attr("num_iters").AttrType(OPTIONAL).Int(20);
        this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("need_backward").AttrType(OPTIONAL).Bool(true);  // 控制可选输出
    }
};
```

### 3.3 Kernel入口函数修改

**文件**: `op_kernel/mhc_pre_sinkhorn.cpp`

```cpp
extern "C" __global__ __aicore__ void mhc_pre_sinkhorn(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
    GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
    GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
    GM_ADDR sumOut, GM_ADDR normOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    
    GM_ADDR userWs = GetUserWorkspace(workspace);
    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tiling_data_in, tiling);
        const MhcPreSinkhornTilingData *tilingData = &tiling_data_in;
        
        // Stage1: 矩阵乘法
        MhcPreSinkhorn::MhcPreSinkhornMembaseKSplitCorePart1<DTYPE_X> op;
        op.Init(x, phi, userWs, tilingData, &pipe);
        op.Process();
        pipe.Destroy();

        // Stage2: 后处理 (传入所有输出地址)
        TPipe pipeStage2;
        MhcPreSinkhorn::MhcPreSinkhornMembaseKSplitCorePart2<DTYPE_X> op2;
        op2.Init(x, alpha, bias, 
                 hin, hPost, hRes,      // 必选输出
                 hPre, hcBeforeNorm, invRms,  // 可选输出
                 sumOut, normOut,       // 可选输出
                 userWs, tilingData, &pipeStage2);
        op2.Process();
        pipeStage2.Destroy();
    }
}
```

### 3.4 Part2类修改 (核心修改)

**文件**: `op_kernel/mhc_pre_sinkhorn_m_k_split_core.h`

#### 3.4.1 类成员变量新增

```cpp
template <typename T>
class MhcPreSinkhornMembaseKSplitCorePart2 {
public:
    // ... 原有成员 ...

private:
    // 新增输出Global Tensor
    GlobalTensor<float> hPreGm_;           // 输出4: hPre
    GlobalTensor<float> hcBeforeNormGm_;   // 输出5: hcBeforeNorm
    GlobalTensor<float> invRmsGm_;         // 输出6: invRms
    GlobalTensor<float> sumOutGm_;         // 输出7: sumOut
    GlobalTensor<float> normOutGm_;        // 输出8: normOut

    // 新增Local Tensor缓冲
    TBuf<QuePosition::VECCALC> hPreBuf_;
    TBuf<QuePosition::VECCALC> hcBeforeNormBuf_;
    TBuf<QuePosition::VECCALC> invRmsBuf_;
    
    LocalTensor<float> hPreLocal_;
    LocalTensor<float> hcBeforeNormLocal_;
    LocalTensor<float> invRmsLocal_;

    // needGrad标志
    bool needGrad_ = false;
};
```

#### 3.4.2 Init函数修改

```cpp
__aicore__ inline void Init(
    GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase,
    GM_ADDR y, GM_ADDR post, GM_ADDR combFrag,      // 必选输出
    GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,  // 可选输出
    GM_ADDR sumOut, GM_ADDR normOut,                // 可选输出
    GM_ADDR workspace,
    const MhcPreSinkhornTilingData *tilingDataPtr, TPipe *pipePtr)
{
    pipe = pipePtr;
    tilingData = tilingDataPtr;
    needGrad_ = tilingData->needGrad;  // 从tiling获取needGrad标志

    // 原有初始化...
    xGm.SetGlobalBuffer((__gm__ T *)x);
    hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
    hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    postGm.SetGlobalBuffer((__gm__ float *)post);
    combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
    workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

    // 新增输出Global Buffer初始化
    if (needGrad_) {
        if (hPre != nullptr) {
            hPreGm_.SetGlobalBuffer((__gm__ float *)hPre);
        }
        if (hcBeforeNorm != nullptr) {
            hcBeforeNormGm_.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
        }
        if (invRms != nullptr) {
            invRmsGm_.SetGlobalBuffer((__gm__ float *)invRms);
        }
        if (sumOut != nullptr) {
            sumOutGm_.SetGlobalBuffer((__gm__ float *)sumOut);
        }
        if (normOut != nullptr) {
            normOutGm_.SetGlobalBuffer((__gm__ float *)normOut);
        }
    }

    // 原有Queue和Buffer初始化...
    // ...

    // 新增Buffer初始化 (用于存储中间结果)
    if (needGrad_) {
        pipe->InitBuffer(hPreBuf_, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(invRmsBuf_, tilingData->stage2RowFactor * sizeof(float));
        hPreLocal_ = hPreBuf_.Get<float>();
        invRmsLocal_ = invRmsBuf_.Get<float>();
    }
}
```

#### 3.4.3 Process函数修改

```cpp
__aicore__ inline void Process() 
{
    if ASCEND_IS_AIV {
        // ... 原有变量定义 ...
        
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            // ... 原有计算逻辑 ...

            // ============ 输出6: invRms ============
            if (needGrad_) {
                // rsqrtLocal 已经计算好，直接输出
                CopyOut(invRmsLocal_,
                        invRmsGm_[stage2BlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->stage2RowFactor],
                        curRowFactor, 1);
            }

            // ============ 输出5: hcBeforeNorm ============
            // hcBeforeNorm = mix0 + mix1 + mix2 (拼接)
            // 在累加后输出
            if (needGrad_) {
                // 输出 mix0 累加结果
                CopyOut(mxies01ReduceLocal,
                        hcBeforeNormGm_[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix + 
                                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMix],
                        curRowFactor, tilingData->hcMult);
                // 输出 mix1 累加结果
                CopyOut(mxies01ReduceLocal[tilingData->stage2RowFactor * tilingData->hcMultAlign],
                        hcBeforeNormGm_[... + tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                // 输出 mix2 累加结果
                CopyOut(mxies02ReduceLocal,
                        hcBeforeNormGm_[... + tilingData->hcMult * 2],
                        curRowFactor, tilingData->hcMult * tilingData->hcMult);
            }

            // ============ 原有ProcessPre计算 ============
            ProcessPre(mxies01ReduceLocal, mxies01ReduceLocal, hcBase0Local, rsqrtLocal, ...);

            // ============ 输出4: hPre ============
            if (needGrad_) {
                // mxies01ReduceLocal[0:curRowFactor*hcMult] 就是 hPre
                CopyOut(mxies01ReduceLocal,
                        hPreGm_[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                                rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
            }

            // ============ 原有hin计算 ============
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                // ... 原有ProcessY逻辑 ...
            }

            // ============ 原有hPost计算 ============
            // ... 原有ProcessPost逻辑 ...

            // ============ 原有hRes和Sinkhorn迭代 ============
            // Softmax计算得到 hRes
            SoftmaxFP32Perf(mxies02ReduceLocal, mxies02ReduceLocal, ...);
            
            // ============ 输出3: hRes (未做Sinkhorn) ============
            // 原有combFrag输出逻辑保持不变

            // ============ 输出7,8: sumOut和normOut ============
            if (needGrad_) {
                // 初始 normOut[0] 已经在 SoftmaxFP32Perf 中计算
                int64_t iterBaseOffset = stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                                         rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMult;
                
                // 输出初始 normOut[0]
                CopyOut(combFragLocal,
                        normOutGm_[0 * bs * seqLen * n * n + iterBaseOffset],
                        curRowFactor * tilingData->hcMult, tilingData->hcMult);

                // Sinkhorn迭代
                for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                    // Row sum
                    LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                    
                    // 输出 sumOut[2*iter]
                    if (needGrad_) {
                        CopyOut(reduceLocal,
                                sumOutGm_[(2*iter) * bs * seqLen * n + ...],
                                curRowFactor, tilingData->hcMult);
                    }
                    
                    // Row normalization
                    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, ...);
                    
                    // 输出 normOut[2*iter]
                    if (needGrad_) {
                        CopyOut(combFragLocal,
                                normOutGm_[(2*iter+1) * bs * seqLen * n * n + ...],
                                curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    }

                    // Col sum
                    ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                    
                    // 输出 sumOut[2*iter+1]
                    if (needGrad_) {
                        CopyOut(reduceLocal,
                                sumOutGm_[(2*iter+1) * bs * seqLen * n + ...],
                                curRowFactor, tilingData->hcMult);
                    }
                    
                    // Col normalization
                    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, ...);
                    
                    // 输出 normOut[2*iter+1]
                    if (needGrad_) {
                        CopyOut(combFragLocal,
                                normOutGm_[(2*iter+2) * bs * seqLen * n * n + ...],
                                curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    }
                }
            } else {
                // 原有Sinkhorn迭代逻辑 (不输出中间结果)
                for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                    // ... 原有迭代逻辑 ...
                }
            }
        }
    }
}
```

### 3.5 Tiling计算修改

**文件**: `op_host/op_tiling/mhc_pre_sinkhorn_tiling.cpp`

```cpp
ge::graphStatus MhcPreSinkhornTiling::CalcOpTiling()
{
    // ... 原有计算 ...

    // 新增: 计算可选输出的workspace需求
    if (needGrad_) {
        // sumOut: (numIters * 2, bs, seq_len, n)
        int64_t sumOutSize = iterTimes_ * 2 * bs_ * seqLen_ * hcMult_ * sizeof(float);
        
        // normOut: (numIters * 2, bs, seq_len, n, n)
        int64_t normOutSize = iterTimes_ * 2 * bs_ * seqLen_ * hcMult_ * hcMult_ * sizeof(float);
        
        // hPre: (bs, seq_len, n)
        int64_t hPreSize = bs_ * seqLen_ * hcMult_ * sizeof(float);
        
        // hcBeforeNorm: (bs, seq_len, n*n + 2*n)
        int64_t hcBeforeNormSize = bs_ * seqLen_ * (hcMult_ * hcMult_ + 2 * hcMult_) * sizeof(float);
        
        // invRms: (bs, seq_len, 1)
        int64_t invRmsSize = bs_ * seqLen_ * sizeof(float);

        // 更新workspace大小
        workspaceSize_ += sumOutSize + normOutSize + hPreSize + hcBeforeNormSize + invRmsSize;
    }

    tilingData_.needGrad = needGrad_ ? 1 : 0;
    
    return ge::GRAPH_SUCCESS;
}
```

---

## 四、needGrad参数处理逻辑

### 4.1 参数传递链路

```
用户调用 → aclnn接口 → op_def → tiling → kernel
    │           │          │         │         │
    │           │          │         │         └─ 根据needGrad决定是否输出
    │           │          │         └─ 计算workspace大小
    │           │          └─ 定义可选输出
    │           └─ 传递needGrad参数
    └─ 设置needGrad=true/false
```

### 4.2 条件输出逻辑

```cpp
// 在Process函数中
if (needGrad_) {
    // 输出 hPre
    // 输出 hcBeforeNorm
    // 输出 invRms
    // 输出 sumOut (每次迭代)
    // 输出 normOut (每次迭代)
}
```

### 4.3 内存优化策略

当 `needGrad = false` 时：
- 不分配 sumOut、normOut 的输出内存
- 不存储中间迭代结果
- 减少workspace使用

---

## 五、输出存储与返回机制

### 5.1 输出内存布局

```
必选输出 (始终分配):
├─ hin:     GM地址由框架分配，直接输出
├─ hPost:   GM地址由框架分配，直接输出
└─ hRes:    GM地址由框架分配，直接输出

可选输出 (needGrad=true时分配):
├─ hPre:         GM地址由框架分配，直接输出
├─ hcBeforeNorm: GM地址由框架分配，直接输出
├─ invRms:       GM地址由框架分配，直接输出
├─ sumOut:       GM地址由框架分配，直接输出
└─ normOut:      GM地址由框架分配，直接输出
```

### 5.2 输出时机

| 输出 | 输出时机 | 存储位置 |
|------|---------|---------|
| hin | ProcessY计算完成后 | 直接写入GM |
| hPost | ProcessPost计算完成后 | 直接写入GM |
| hRes | Softmax计算完成后 | 直接写入GM |
| hPre | ProcessPre计算完成后 | 直接写入GM |
| hcBeforeNorm | 累加完成后、归一化缩放前 | 直接写入GM |
| invRms | rsqrt计算完成后 | 直接写入GM |
| sumOut | 每次迭代计算后 | 直接写入GM |
| normOut | 每次迭代计算后 | 直接写入GM |

---

## 六、不影响原有逻辑的策略

### 6.1 向后兼容设计

```cpp
// 原有3输出模式
if (!needGrad_) {
    // 保持原有计算流程不变
    // 只输出 hin, hPost, hRes
}

// 新增8输出模式
if (needGrad_) {
    // 在原有计算流程中插入输出逻辑
    // 输出全部8个结果
}
```

### 6.2 计算复用策略

```
原有计算:
  mix0 → ProcessPre → weights → ProcessY → hin
  mix1 → ProcessPost → hPost
  mix2 → Softmax → Sinkhorn → hRes

新增输出 (复用中间结果):
  mix0 + mix1 → hcBeforeNorm (输出5)
  rsqrt → invRms (输出6)
  ProcessPre结果 → hPre (输出4)
  Sinkhorn中间结果 → sumOut, normOut (输出7, 8)
```

### 6.3 性能影响最小化

1. **条件执行**: 只在 `needGrad=true` 时执行额外输出逻辑
2. **数据复用**: 复用已有计算结果，不重复计算
3. **直接输出**: 使用 `DataCopy` 直接写入输出内存，不阻塞计算流水线

---

## 七、实现步骤

### 7.1 第一阶段：框架层修改

1. 修改 `mhc_pre_sinkhorn_def.cpp`，新增5个可选输出定义
2. 修改 `aclnn_mhc_pre_sinkhorn.h/cpp`，扩展接口参数
3. 更新 `mhc_pre_sinkhorn_tiling.h`，新增needGrad字段

### 7.2 第二阶段：Kernel层修改

1. 修改 `mhc_pre_sinkhorn.cpp` 入口函数，传递新参数
2. 修改 `mhc_pre_sinkhorn_m_k_split_core.h` Part2类：
   - 新增成员变量
   - 修改Init函数
   - 修改Process函数

### 7.3 第三阶段：Tiling计算修改

1. 修改 `mhc_pre_sinkhorn_tiling.cpp`，添加needGrad属性获取和设置逻辑
2. 传递needGrad参数到tiling数据

### 7.4 第四阶段：测试验证

1. 单元测试：验证8个输出正确性
2. 性能测试：对比needGrad=true/false的性能差异
3. 精度测试：与参考实现对比

---

## 八、总结

本方案在不改变原有计算逻辑的前提下，通过以下方式实现8输出功能：

1. **新增needGrad参数**: 控制可选输出的开关
2. **复用中间结果**: 从现有计算流程中提取所需输出
3. **条件输出**: 只在需要时输出额外结果，不影响原有性能
4. **向后兼容**: 原有3输出模式完全保持不变

---

## 附录：与aclnnMhcPreSinkhorn.md文档的对应关系

| 本方案输出 | 文档参数名 | Shape | 说明 |
|-----------|-----------|-------|------|
| hin | hin | (bs, seq_len, c) | 必选输出 |
| hPost | hPost | (bs, seq_len, n) | 必选输出 |
| hRes | hRes | (bs, seq_len, n*n) | 必选输出 |
| hPre | hPre | (bs, seq_len, n) | 可选输出 |
| hcBeforeNorm | hcBeforeNorm | (bs, seq_len, n*n + 2*n) | 可选输出 |
| invRms | invRms | (bs, seq_len, 1) | 可选输出 |
| sumOut | sumOut | (numIters*2, bs, seq_len, n) | 可选输出 |
| normOut | normOut | (numIters*2, bs, seq_len, n, n) | 可选输出 |