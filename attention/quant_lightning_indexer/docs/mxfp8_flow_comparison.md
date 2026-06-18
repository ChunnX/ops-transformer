# QLI 算子 MXFP8 新老计算流程对比图

> 使用 Mermaid 语法，可在 VS Code (Markdown Preview Mermaid Support 插件)、draw.io、mermaid.live 等工具中编辑和渲染。

---

## 一、老流程（非 MXFP8）

### 1.1 整体流水线

```mermaid
flowchart TD
    subgraph AIC["AIC (Cube核)"]
        A1["GM加载 Q<br/>fp8/int8/bf16"] --> A2["ND2NZ<br/>Q → L1"]
        A3["GM加载 K<br/>fp8/int8/bf16"] --> A4["ND2NZ<br/>K → L1"]
        A2 --> A5["L1→L0<br/>LoadData"]
        A4 --> A5
        A5 --> A6["Mmad<br/>Q × K → L0C<br/>(float/int32)"]
        A6 --> A7["Fixpipe<br/>L0C → UB<br/>(float)"]
    end

    subgraph AIV["AIV (Vector核)"]
        B1["GM加载 weights<br/>[G]"] --> B4
        B2["GM加载 qScale<br/>[G] (per-token)"] --> B4
        B3["GM加载 kScale<br/>[S2] (per-token)"] --> B5
        A7 -->|"CV同步"| B4["weights × qScale<br/>(element-wise)<br/>→ regW"]
        B4 --> B5["For each group i:<br/>regQK = QK[i]<br/>Relu(regQK)<br/>accum += regQK × broadcast(regW[i])"]
        B5 --> B6["accum × kScale<br/>(element-wise)"]
        B6 --> B7["Cast → bf16<br/>FloatToSortableKey<br/>→ score (uint16)"]
        B7 --> B8["score → scoreGm"]
        B8 --> B9["TopK<br/>(histogram-based)"]
        B9 --> B10["indices → GM输出"]
    end
```

### 1.2 Vector 核内部计算细节（老流程）

```mermaid
flowchart LR
    subgraph VecCompute["MulWeightAndReduceSum (VF指令)"]
        direction TB
        V1["Load weights [G]<br/>Load qScale [G]"] --> V2["regW = weights × qScale<br/>(G个标量)"]
        V2 --> V3["for i in 0..G step 2:"]
        V3 --> V4["Load QK[i], QK[i+1]<br/>[S2] each"]
        V4 --> V5["Relu(QK[i])<br/>Relu(QK[i+1])"]
        V5 --> V6["accum += QK[i] × brc(regW[i])<br/>accum += QK[i+1] × brc(regW[i+1])"]
        V6 --> V7["accum × kScale[0:64]<br/>accum × kScale[64:128]"]
        V7 --> V8["DeInterleave + Cast bf16<br/>FloatToSortableKey<br/>→ uint16 score"]
    end
```

### 1.3 老流程 qScale 形状

```
qScale shape: [B, S1, N1]  (per-token, per-head)
                ↓
            每个 group 对应 1 个标量
                ↓
            weights[G] × qScale[G] = regW[G]  (element-wise, 1:1)
```

---

## 二、新流程（MXFP8）

### 2.1 整体流水线

```mermaid
flowchart TD
    subgraph AIC["AIC (Cube核)"]
        A1["GM加载 Q<br/>fp8_e4m3"] --> A2["ND2NZ<br/>Q → L1"]
        A1s["GM加载 qScale<br/>fp8_e8m0<br/>[G×S1, D/32]"] --> A2s["DN2NZ<br/>qScale → L1B"]
        A3["GM加载 K<br/>fp8_e4m3"] --> A4["ND2NZ<br/>K → L1"]
        A3s["GM加载 kScale<br/>fp8_e8m0<br/>[S2, D/32]"] --> A4s["DN2NZ<br/>kScale → L1B"]
        A2 --> A5["L1→L0<br/>LoadData Mx<br/>(Q + qScale)"]
        A2s --> A5
        A4 --> A6["L1→L0<br/>LoadData Mx<br/>(K + kScale)"]
        A4s --> A6
        A5 --> A7["Mmad mx_fp8_e4m3_t<br/>Q×K×qScale×kScale → L0C<br/>(float)"]
        A6 --> A7
        A7 --> A8["Fixpipe<br/>L0C → UB<br/>(float)"]
    end

    subgraph AIV["AIV (Vector核)"]
        B1["GM加载 weights<br/>[G]"] --> B4
        A8 -->|"CV同步"| B4["For each group i:<br/>regQK = QK[i]<br/>(已含 qScale×kScale)"]
        B4 --> B5["Relu(regQK)<br/>accum += regQK × broadcast(weights[i])"]
        B5 --> B7["Cast → bf16<br/>FloatToSortableKey<br/>→ score (uint16)"]
        B7 --> B8["score → scoreGm"]
        B8 --> B9["TopK<br/>(histogram-based)"]
        B9 --> B10["indices → GM输出"]
    end
```

### 2.2 Vector 核内部计算细节（新流程）

```mermaid
flowchart LR
    subgraph VecCompute["MulWeightAndReduceSum_Mxfp8 (VF指令)"]
        direction TB
        V1["Load weights [G]"] --> V3
        V3["for i in 0..G step 2:"] --> V4["Load QK[i], QK[i+1]<br/>[S2] each<br/>(已含qScale×kScale)"]
        V4 --> V5["Relu(QK[i])<br/>Relu(QK[i+1])"]
        V5 --> V6["accum += QK[i] × brc(weights[i])<br/>accum += QK[i+1] × brc(weights[i+1])"]
        V6 --> V8["DeInterleave + Cast bf16<br/>FloatToSortableKey<br/>→ uint16 score"]
    end
```

### 2.3 新流程 qScale 形状

```
qScale shape: [B, S1, N1, D/32]  (per-block, 每32个D元素一个scale)
                ↓
            每个 group 对应 D/32 = 4 个 scale (D=128)
                ↓
            无法与 weights[G] 做 1:1 element-wise 乘法
                ↓
            解决方案: 将 qScale 融入 mmad_mx 硬件指令
```

---

## 三、新老流程对比表

| 阶段 | 老流程 | 新流程 (MXFP8) |
|------|--------|----------------|
| **Q 数据** | fp8/int8/bf16 | fp8_e4m3 |
| **K 数据** | fp8/int8/bf16 | fp8_e4m3 |
| **qScale** | `[G]` float/half (per-token) | `[G, D/32]` fp8_e8m0 (per-block) |
| **kScale** | `[S2]` float/half (per-token) | `[S2, D/32]` fp8_e8m0 (per-block) |
| **Cube: Mmad** | `Q × K` → float/int32 | `Q × K × qScale × kScale` → float (mmad_mx) |
| **Cube: L0加载** | 普通 LoadData | LoadData Mx (数据+scale一起加载) |
| **Vec: 预处理** | `weights × qScale` | 无 (scale已在mmad_mx中处理) |
| **Vec: 主循环** | `Relu(QK) → × weight → accum` | `Relu(QK) → × weight → accum` (相同) |
| **Vec: 后处理** | `accum × kScale` | 无 (scale已在mmad_mx中处理) |
| **Vec: 输出** | cast bf16 → sortable key | cast bf16 → sortable key (相同) |
| **TopK** | histogram-based | histogram-based (相同) |

---

## 四、关键差异总结

```mermaid
flowchart LR
    subgraph OLD["老流程 Scale 处理"]
        direction TB
        O1["qScale[G] × weights[G]"] --> O2["broadcast → × QK"]
        O2 --> O3["reducesum"]
        O3 --> O4["× kScale[S2]"]
    end

    subgraph NEW["新流程 Scale 处理"]
        direction TB
        N1["qScale[G,D/32] + kScale[S2,D/32]"] --> N2["mmad_mx 硬件融合"]
        N2 --> N3["QK (已含scale)"]
        N3 --> N4["Relu"]
        N4 --> N5["weights[G] broadcast → × QK"]
        N5 --> N6["reducesum"]
    end

    OLD -.->|"MXFP8 qScale轴变多<br/>无法1:1乘weights"| NEW
```

---

## 五、数据维度变化

```mermaid
flowchart TD
    subgraph OLD_DIM["老流程维度"]
        OD1["Q: [S1×G, D=128]<br/>fp8/int8"]
        OD2["K: [S2, D=128]<br/>fp8/int8"]
        OD3["qScale: [S1×G]<br/>float (1个/token)"]
        OD4["kScale: [S2]<br/>float (1个/token)"]
        OD5["weights: [G]<br/>float/bf16"]
        OD6["QK: [S1×G, S2]<br/>float"]
    end

    subgraph NEW_DIM["新流程维度 (MXFP8)"]
        ND1["Q: [S1×G, D=128]<br/>fp8_e4m3"]
        ND2["K: [S2, D=128]<br/>fp8_e4m3"]
        ND3["qScale: [S1×G, D/32=4]<br/>fp8_e8m0 (4个/token)"]
        ND4["kScale: [S2, D/32=4]<br/>fp8_e8m0 (4个/token)"]
        ND5["weights: [G]<br/>float/bf16 (不变)"]
        ND6["QK: [S1×G, S2]<br/>float (已含scale)"]
    end
```
