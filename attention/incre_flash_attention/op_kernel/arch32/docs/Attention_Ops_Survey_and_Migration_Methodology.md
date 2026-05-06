# Attention 算子倉全景分析與跨平台遷移方法論

> 基於 `ops-transformer/attention/` 目錄下 **45+ 個算子** 的全面代碼分析。從算子分類、實現邏輯、優化技術三個層次進行歸納，提煉出可跨平台複用的算子設計與優化方法論。

---

## 1. 算子全景分類

### 1.1 按計算模式分類

```
Attention 算子家族
├── 1. Flash Attention 前向 (訓練+推理)
│   ├── flash_attention_score           — 訓練前向 (最基礎版)
│   ├── prompt_flash_attention          — Prompt 階段推理前向
│   ├── incre_flash_attention           — Decode 階段增量推理前向
│   ├── fused_infer_attention_score     — 統一推理前向 (Prompt+Decode 合一)
│   ├── attention_pioneer               — 下一代統一前向
│   ├── fused_floyd_attention           — Floyd 優化前向
│   └── nsa_compress_attention          — NSA 壓縮注意力前向
│
├── 2. Flash Attention 反向 (訓練)
│   ├── flash_attention_score_grad      — 訓練反向 (最複雜)
│   ├── fused_floyd_attention_grad      — Floyd 優化反向
│   └── nsa_selected_attention_grad     — NSA 選擇注意力反向
│
├── 3. 稀疏/Sparse Attention
│   ├── sparse_flash_attention          — 稀疏注意力前向
│   ├── sparse_flash_attention_grad     — 稀疏注意力反向
│   ├── block_sparse_attention          — 塊稀疏注意力 (基於 CATLASS)
│   ├── block_sparse_attention_grad     — 塊稀疏注意力反向
│   ├── kv_quant_sparse_flash_attention — KV 量化稀疏注意力
│   └── kv_quant_sparse_flash_attention_pioneer — 下一代 KV 量化稀疏
│
├── 4. 選擇/索引 Attention
│   ├── nsa_selected_attention          — NSA 選擇注意力
│   ├── nsa_selected_attention_infer    — NSA 選擇注意力推理
│   ├── lightning_indexer               — Lightning Indexer 前向
│   ├── lightning_indexer_grad          — Indexer 反向
│   ├── quant_lightning_indexer         — 量化 Indexer
│   ├── dense_lightning_indexer_softmax_lse — Indexer LSE 計算
│   ├── dense_lightning_indexer_grad_kl_loss — Indexer KL 損失梯度
│   └── sparse_lightning_indexer_grad_kl_loss — 稀疏 Indexer KL 損失梯度
│
├── 5. 壓縮 Attention
│   ├── nsa_compress                    — KV 壓縮
│   ├── nsa_compress_grad               — 壓縮反向
│   ├── nsa_compress_with_cache         — 帶緩存壓縮
│   └── nsa_compress_attention_infer    — 壓縮注意力推理
│
├── 6. 線性/循環 Attention
│   ├── chunk_gated_delta_rule          — 分塊門控 Delta Rule (線性注意力)
│   ├── recurrent_gated_delta_rule      — 循環門控 Delta Rule
│   └── ring_attention_update           — Ring Attention 更新 (分佈式)
│
├── 7. MLA (Multi-head Latent Attention) 輔助
│   ├── mla_preprocess                  — MLA 預處理 (矩陣乘法)
│   ├── mla_prolog                      — MLA Prolog (RoPE+RMSNorm+量化)
│   └── mla_preprocess_v2/v3           — MLA 預處理演進版本
│
├── 8. KV Cache / Page Attention
│   ├── gather_pa_kv_cache              — PA Cache 收集 (離散→連續)
│   └── scatter_pa_kv_cache             — PA Cache 分散 (多種模式)
│
├── 9. Attention 結果合併/調度
│   ├── attention_update                — 注意力輸出更新 (含 LSE)
│   ├── attention_worker_combine        — Worker 合併 (B/S/H/K split)
│   └── attention_worker_scheduler      — Worker 調度 (純 Host)
│
├── 10. 其他專用 Attention
│   ├── swin_attention_score_quant      — Swin Transformer 注意力量化
│   ├── rain_fusion_attention           — Rain Fusion Attention
│   ├── fused_causal_conv1d             — 融合因果卷積 (Ascend 950)
│   └── scatter_pa_cache                — PA Cache 寫入 (佔位)
│
└── 11. 共享基礎設施
    └── common/                         — FIA 通用 Kernel (Cube/Vector 塊)
                                         Antiquant 處理器、Attention Mask、PSE
                                         矩陣乘法服務模塊
```

### 1.2 按架構代際分類

| 架構 | 代號 | 代表算子 | 特點 |
|------|------|---------|------|
| arch32 (DaVinci) | Ascend 910B | incre_flash_attention_preload_dd, prompt_flash_attention, flash_attention_score | 傳統 AIC+AIV 雙核模式，Vector+Cube 異步流水 |
| arch35 (Regbase/SIMT) | Ascend 910C/950 | flash_attention_score_kernel_infer, fused_infer_attention_score_v3 | Regbase 尋址，SIMT 指令集，更強向量能力，FP8 支持 |
| arch38 | 下一代 | prompt_flash_attention_tiling_arch38 | WIP/前瞻 |

### 1.3 按計算精度分類

| 精度模式 | 算子數量 | 典型算子 |
|---------|---------|---------|
| FP16/BF16 標準精度 | 主流 | flash_attention_score, prompt_flash_attention |
| INT8 量化 | ~15 個 | incre_flash_attention (ANTIQUANT), swin_attention_score_quant |
| INT4 量化 | ~5 個 | incre_flash_attention (KVINT4), kv_quant_sparse |
| FP32 高精度 | 內部計算 | 所有算子的 Softmax 內部計算 |

---

## 2. 共享基礎設施分析 (common/)

### 2.1 核心抽象層

```
common/
├── op_kernel/
│   ├── fia_public_define.h              — 統一模板參數定義 FIAType<>
│   ├── kernel_common.h                  — 通用 Kernel 工具 (CU/SG 分配)
│   └── kernel_operator_softmax_compute_nz.h — Softmax 計算 NZ 佈局
│
├── op_kernel/arch32/
│   ├── fia_kernel_nonquant.h            — 非量化 FIA Kernel 框架
│   ├── fia_kernel_nonquant_mla.h        — MLA 非量化 Kernel 框架
│   ├── fia_block_cube_nonquant.h         — Cube 側非量化塊處理
│   ├── fia_block_cube_nonquant_mla.h    — MLA Cube 側塊處理
│   ├── fia_block_cube_nonquant_gqa.h    — GQA Cube 側塊處理
│   ├── fia_block_vec_nonquant.h         — Vector 側非量化塊處理
│   ├── fia_block_vec_nonquant_mla.h     — MLA Vector 側塊處理
│   ├── fia_block_vec_flashdecode.h      — FlashDecode Vector 塊處理
│   └── fia_kernel_empty_tensor.h        — 空 Tensor 處理
│
├── op_kernel/arch35/
│   ├── flash_attention_score_kernel_*.h — arch35 統一 Kernel 層
│   ├── flash_attention_score_block_vec_*.h — Vector 側塊處理 (infer/train)
│   ├── flash_attention_score_block_cube*.h — Cube 側塊處理
│   ├── flash_attention_score_antiquant_*.h — 反量化處理器
│   ├── attenmask.h / dropmask.h / pse.h — 輔助處理
│   └── util_regbase.h / flash_attention_score_common_regbase.h
│
└── op_host/
    └── fia_tiling_*.h                   — 通用 Tiling 框架
```

### 2.2 FIAType 統一模板

所有推理側 Attention 算子共享 `FIAType` 模板抽象：

```cpp
template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T,
          bool PAGE_ATTENTION, bool FLASH_DECODE, uint32_t FIA_LAYOUT,
          uint32_t ANTIQUANT_MODE, bool SHARED_PREFIX, uint32_t KV_LAYOUT_T,
          bool SOFTMAX_WITH_BRC>
struct FIAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr uint32_t layout = FIA_LAYOUT;
    static constexpr uint32_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr uint32_t kvLayout = KV_LAYOUT_T;
};
```

**設計意義**：
- 每組模板參數組合在編譯期實例化一個 Kernel 特化版本
- 所有 `if constexpr` 分支在編譯期消除，零運行時分支開銷
- 每個特化版本可針對特定 dtype/layout/quant 組合做激進內聯優化
- 等價於 CUDA 的 template kernel + `-D` 編譯

---

## 3. 各類算子實現邏輯歸納

### 3.1 Flash Attention 前向推理 (核心路徑)

以 `incre_flash_attention` (Decode 階段) 和 `prompt_flash_attention` (Prompt 階段) 為代表。

**算法骨架**（Flash Attention V2）：
```
for each KV block (sInnerLoop):
    // Step 1: BMM1 — Q × K^T → S (Cube)
    S = Q[gSize, headDim] × K[sInnerSize, headDim]^T
    
    // Step 2: Softmax + Scale (Vector)
    S = S × attention_scale
    S = S + attention_mask
    S_new_max = max(S_old_max, rowmax(S))
    S_new_sum = S_old_sum × exp(S_old_max - S_new_max) + rowsum(exp(S - S_new_max))
    P = exp(S - S_new_max) / S_new_sum
    
    // Step 3: BMM2 — P × V → O (Cube)
    O_block = P[gSize, sInnerSize] × V[sInnerSize, headDim]
    
    // Step 4: Online Update (Vector)
    O = O_old × exp(S_old_max - S_new_max) × S_old_sum / S_new_sum + O_block / S_new_sum
```

**實現架構**：Vector+Cube 異構流水
- **Cube 側 (AIC)**：執行 BMM1 (Q×K^T) 和 BMM2 (P×V) 兩個矩陣乘法
- **Vector 側 (AIV)**：執行 Q 預處理、Softmax、P 後處理、O 累積、輸出
- **Tiling 側 (Host)**：計算分核策略、sInnerSize、Buffer 大小、Workspace 分配

**特徵**：
1. **三層嵌套循環**：Batch/Head (BN)→ KV Block (S2)→ M Tile (G)
2. **中間結果通過 Workspace 傳遞**：mm1ResGm → vec1ResGm → mm2ResGm → vec2ResGm
3. **CrossCore 同步**：4 個標誌點 (V0→C1, C1→V1, V1→C2, C2→V2)
4. **量化路徑分支**：ANTIQUANT / NON_ANTIQUANT / PER_CHANNEL / PER_TOKEN

### 3.2 稀疏/選擇 Attention

以 `sparse_flash_attention`, `nsa_selected_attention`, `lightning_indexer` 為代表。

**與標準 Attention 的關鍵差異**：
1. **輸入**：除了 Q/K/V，還有額外的 `indices`/`blockTable`/`W` (權重) 參數
2. **前處理**：需要根據 indices 從 K/V 中 gather 有效 token 或 block
3. **計算路徑**：標準 Attention 計算
4. **後處理**：部分算子需要 Top-k 選擇、ReLU 激活、KL Loss、Indexer Weight 梯度

```
LightningIndexer 流程:
  K' = gather(K, indices)                  // 按 index 選擇 K token
  V' = gather(V, indices)                  // 按 index 選擇 V token
  score = ReLU(Q × W) + bias              // Indexer 特有：含 W 權重和 ReLU
  indices = TopK(score, K)                 // 選擇 Top-K 索引
  output = FlashAttention(Q, K', V')       // 對選中的 token 做 Attention
```

### 3.3 MLA (Multi-head Latent Attention)

MLA 的核心是 KV 壓縮：
```
傳統: K = X × W_K  (headDim)
      V = X × W_V  (headDim)
MLA:  K = UpProj(DownProj(X))  (latentDim << headDim)
```

**MLA 預處理 (mla_prolog/mla_preprocess)**：
```
mla_prolog (單個 CORE):            mla_preprocess (全量矩陣乘):
  Q → RMSNorm → RoPE              Q_all → W_down → W_up (兩個 BMM)
  K → DynamicQuant → ScatterCache
```

### 3.4 Page Attention / KV Cache 管理

**Gather PA (`gather_pa_kv_cache`)**：
```
輸入: blockTable[i], KV_blocks[blockTable[i]]
輸出: 連續 KV 序列 [seqLen, headDim]
實現: 按 block 粒度循環 + Nd2Nz 搬運
```

**Scatter PA (`scatter_pa_kv_cache`)**：
```
輸入: KV_cache_old, new_KV, blockTable, slot_mapping
輸出: KV_cache_new (更新後的緩存)
模式: Normal / RoPE / Alibi / Omni / Nz / SISO
```

### 3.5 KV Cache Combine / Update

**`attention_update`**：
```
用於 FlashDecode 後結果合併:
  O_new = O_part1 × lse_part1 / lse_sum + O_part2 × lse_part2 / lse_sum
  LSE_sum = log(exp(lse_part1) + exp(lse_part2))
```

**`attention_worker_combine`**：
```
按 B/S/H/K 軸的分裂結果合併:
  CombineSplitBS — 合併 B/S 分裂結果
  CombineSplitH  — 合併 Head 分裂結果
  CombineSplitK  — 合併 KV 分裂結果
```

---

## 4. 優化技術全景圖

### 4.1 編譯期優化

| 技術 | 實現方式 | 適用場景 |
|------|---------|---------|
| **模板特化** | `FIAType<Q_T, KV_T, ...>` 多個模板參數組合，編譯期分支消除 | 所有算子 |
| **64-bit Tiling Key** | 將 dtype/layout/quant/prefill/pageAttention 等全部編碼到單一 64-bit key，用於 Kernel Dispatch | FusedInfer/PromptFlash/IncreFlash |
| **`if constexpr`** | 量化/非量化、ANTIQUANT_PER_CHANNEL/PER_TOKEN 等路徑在編譯期確定 | 所有算子 |

**跨平台對照 (CUDA)**：
```
Ascend: FIAType<Q_T, KV_T, ...>  →  每個組合生成獨立 .o
CUDA:  template<...>             →  -D編譯或 `__if_cuda_arch`
       或 Cutlass::Gemm UniversalMode → 單一 kernel + runtime dispatch
```

### 4.2 內存層次優化

| 技術 | Ascend 實現 | CUDA 等價 |
|------|-----------|----------|
| **雙重緩衝** | L0A/L0B/L0C ping-pong (2×32KB/2×32KB/2×64KB) | `__shared__` `double_buffer[2][...]` |
| **多重緩衝** | L1 KP 三重緩衝 (3×64KB), L1 V 四重緩衝 (4×64KB) | SMEM Pipeline Stages (Cutlass::Pipeline) |
| **常駐緩衝** | Q L1 常駐 (s2Idx==0)，V L1 常駐 (mCopyIdx==0) | FlashAttention-2 Q tile 常駐 SMEM |
| **佈局轉換** | ND→NZ 轉換 (Nd2NzParams) 在 MTE 搬運時做 | `__shared__` transpose 或 `cp.async` implicit transpose |
| **UB 分塊** | Vector 側按 `gSizeVector` 切分，每次處理 32KB 以內 | Register blocking + SMEM tiling |
| **Workspace 分配** | 按 `preLoadNum × mmResUbSize × actCoreNum` 計算 | cudaMalloc workspace + kernel arg offset |

### 4.3 異構流水優化

```
┌──────────────────────────────────────────────────────────┐
│  流水級別           Ascend 實現           CUDA 等價       │
├──────────────────────────────────────────────────────────┤
│  CrossCore 同步     CrossCoreSetFlag/WaitFlag    cudaStream + event  │
│  Core 內流水        MTE2/MTE1/Cube 三路重疊      cp.async + mma   │
│  任務級流水         TASK_NUM=2 深流水            warp specialization│
│  數據級流水         L0 ping-pong                 software pipeline  │
└──────────────────────────────────────────────────────────┘
```

**Preload-DD 的三級流水詳解**：

```
級別1: CrossCore (AIV ↔ AIC)
  AIV: QPre → SetFlag(V0_C1) → AIC: BMM1 → SetFlag(C1_V1)
  AIV: Vec1 → SetFlag(V1_C2)  → AIC: BMM2 → SetFlag(C2_V2) → AIV: Vec2

級別2: Core 內 MTE-Cube (AIC 內部)
  MTE2: K[N+1] → L1_KP    (寫)
  MTE1: K[N]   → L0B      (讀 L1_KP → 寫 L0B)
  Cube: Mmad(L0A, L0B[N-1]) (用前一塊 L0B)

級別3: L0 ping-pong (AIC 內部)
  LoadData aL0_ping → SetFlag → Mmad(ping) → LoadData aL0_pong → ...
```

### 4.4 計算融合

| 融合類型 | 具體實現 | 收益 |
|---------|---------|------|
| **Fixpipe DEQF16** | Cube Mmad 輸出 INT32 直接在 Fixpipe 反量化為 FP16 | 消除一次 AIV 反量化運算 |
| **FusedMulAdd** | 反量化 scale + offset 用單指令合併 | 減少 GM→UB 交互次數 |
| **Softmax Flash V2** | 在線更新 max/sum，無需存整個 S 矩陣 | O(n) 空間代替 O(n²) |
| **SelectWithBytesMask** | Attention Mask 通過硬件指令直接填充 | 消除 Mask 的 UB copy |
| **AntiquantIterExpand** | MSD 雙段量化使用迭代展開 (F16→INT8) | 高精度 INT8 量化 |
| **PostQuant** | BMM2 輸出直接在輸出前做 AscendQuant | 消除中間 F32 buffer |

### 4.5 Tiling 策略

| 參數 | 作用 | 選取原則 |
|------|------|---------|
| `sInnerSize_` | KV 序列的單次處理長度 | 平衡: UB 容量 vs 循環次數。GQA 場景為控制 workspace ≤32MB，按 `/5` 分檔 |
| `gSize/gSizeCube` | Q 矩陣的 M 維處理大小 | 基於 L1 Q buffer (64K) 和 L0A (32K) 容量反推 |
| `groupSplitSize_` | G 軸切分 (BNSD layout) | 基於 gMax_ (512/256/128)，保證 L1 不溢出 |
| `s1SplitSize_` | Q 的 S 軸切分 (BSH layout) | 確保 M 總量 ≤ gMax_ |
| `baseM / baseK` | Cube 側 M-tiling 和 K-tiling | baseM = 64/sizeof(KV_T), baseK = 128/sizeof(KV_T) — 對齊 L0C 容量 |
| `mmResUbSize` | BMM1 中間結果大小 | = sInnerSizeAlign × cubeMSize × msdIterNum |
| `bmm2ResUbSize` | BMM2 中間結果大小 | = headDimAlign × cubeMSize × msdIterNum |
| `preLoadNum` | Workspace 預加載 slot 數 | 默認 4 (PRE_LOAD_NUM)，DD 模式 6 (BUF_NUM)，MLA 模式 2 |

### 4.6 分核策略

```cpp
// 核總數 = AIV 數量（默認）或 AIC 數量（CUBE_VIEW_MM 模式）
// 總任務數 = batchSize × kvHeadNum × outer (outer = s1Outer 或 gOuter)

// 三種分核模式:
1. SplitBN_V0 (均衡分配):  每個核 ⌈total_tasks / coreNum⌉
2. Sparse Split (負載均衡): 按 actualSeqLen 權重貪心分配
3. SplitBalanced (MLA):     基於 gS1SplitNumOfFdHead 的精細分配
```

### 4.7 量化路徑優化

| 量化模式 | 特點 | 反量化位置 |
|---------|------|-----------|
| **NO_QUANT** | 原始類型直接計算 | 不需要反量化 |
| **QUANT** | Q=K=V=INT8, 對稱量化 | Fixpipe 標量反量化 (Preload: AIV側) |
| **ANTIQUANT_PER_CHANNEL** | Q≠KV, K per-channel scale | Preload: AIV側; DD: Fixpipe DEQF16 |
| **ANTIQUANT_PER_TOKEN** | Q≠KV, K per-token scale | AIV 側逐 token 反量化 (帶 RowSum) |
| **ANTIQUANT_PER_CHANNEL_TOKEN** | K per-channel, V per-token | 混合模式 |
| **FULL_QUANT** | 全量化路徑 (INT4/INT8 Q/K/V) | Fixpipe DEQF16 + AIV post-scale |

**MSD (Most Significant Digit) 雙段機制**：
- 每層量化拆為兩段：high bits + low bits
- 對應 scale: `quantScaleC1S1 = 1/1024`, `quantScaleC1S2 = 1/(1024×254)`
- 合併方式: Preload 在 AIV (AtomicAdd)，DD 在 Fixpipe (SetAtomicAdd)

---

## 5. 算子演化路徑分析

### 5.1 IncreFlashAttention 家族演化

```
incre_flash_attention_allvec_new.h  — 純向量版本 (最早期)
    ↓
incre_flash_attention_preload.h     — Vector+Cube 異構 + 4-task 預加載
    ↓
incre_flash_attention_preload_dd.h  — DD 深入優化 (本次關注)
    ↓
incre_flash_attention_preload_mla.h — MLA 支持
    ↓
incre_flash_attention (arch35)      — Regbase/SIMT 架構
```

每代演進的核心改進：
1. **allvec → preload**：引入 Cube 加速 BMM，GM 搬運 + 計算重疊
2. **preload → preload_dd**：L1 分區 + M-tiling + Fixpipe融合 + 深流水
3. **preload_dd → preload_mla**：適配 MLA 的 headDim 擴展和 G 軸切分變化
4. **arch32 → arch35**：Regbase 尋址替代指針運算，SIMT 指令提升向量吞吐

### 5.2 PromptFlashAttention 家族演化

```
prompt_flash_attention (arch32)
    ↓
prompt_flash_attention (arch35)     — arch35 SIMT 加速
    ↓
fused_infer_attention_score         — Prompt + Decode 合一
    ↓
attention_pioneer                   — 全新架構重構
```

### 5.3 從訓練到推理的收斂

```
flash_attention_score_grad (訓練)
    ↓                      ← 複雜度高: BMM1+BMM2 都需反向 (dQ/dK/dV)
    
flash_attention_score → prompt_flash_attention → incre_flash_attention
    ↓                        ↓                        ↓
    訓練前向               Prompt推理               Decode推理
                          (Q為多token)             (Q為單token)
    ↓                        ↓                        ↓
fused_infer_attention_score (統一推理) → attention_pioneer (下一代)
```

---

## 6. 跨平台遷移方法論

### 6.1 核心原則：「抽象先行，硬件適配」

從 Ascend 算子倉中可提煉出 **五層抽象模型**，這些抽象層次與具體硬件解耦：

```
第1層: 算法抽象 (Algorithm)
  ├── FlashAttention / Sparse Attention / MLA / Delta Rule
  └── 與硬件無關的算法流程

第2層: 計算模式抽象 (Compute Pattern)
  ├── BMM(矩陣乘) / Softmax / Reduction / Element-wise
  └── 可映射到任意平台的 TensorCore / CUDA Core

第3層: 內存層次抽象 (Memory Hierarchy)
  ├── Global / Shared / Register 三級模型
  └── Tiling 策略：每個數據塊在何處常駐、何時搬運

第4層: 同步抽象 (Synchronization)
  ├── CrossCore (跨核) / IntraCore (核內) / Pipeline (流水)
  └── 映射為 CUDA Stream / Barrier / Sync

第5層: 硬件加速抽象 (Hardware Accelerator)
  ├── MMA (矩陣乘) / Fixpipe (後處理) / MTE (DMA)
  └── 映射為 Tensor Core / Epilogue / cp.async
```

### 6.2 到 NVIDIA CUDA 的遷移映射表

| Ascend 概念 | CUDA 對等實現 | 注意事項 |
|------------|--------------|---------|
| **FIAType 模板特化** | CUTLASS `GemmUniversalTraits` + `-DSM=80/90` | CUDA 需增加 SM 版本條件編譯 |
| **MTE2/MTE1 DMA** | `cp.async` (SM80+) / `TMA` (SM90+) | TMA 的 descriptor 需要單獨分配 |
| **L1 Buffer 512KB** | `__shared__` 228KB (H100) / 164KB (A100) | 需要更細粒度的 Tiling，SMEM 分區要重新規劃 |
| **L0A/L0B/L0C ping-pong** | `wgmma.fence` + `warpgroupmma` | CUTLASS Epilogue 等同於 Fixpipe |
| **CrossCore 同步** | `cudaStreamWaitEvent` + `cudaStreamSynchronize` | 跨核同步延遲比 Ascend 更大 |
| **AIC/AIV 異構** | warp specialization (Tensor Core + CUDA Core) | 一個 warp 做 MMA，另一個 warp 做 Softmax |
| **QuantScale/DEQF16** | `__hfma` (fused multiply-add) 在 epilogue 中 | 無硬件等價，需軟件實現 Scale+Cast |
| **Nd2Nz 佈局轉換** | `__shared__` manual transpose 或 `cp.async` 自動 | CUTLASS 的 `ldmatrix` 自帶轉置 |
| **Tiling 分核** | CUTLASS Tile Scheduler / `cudaOccupancyMaxPotentialBlockSize` | 做法不同但目標一致 |
| **Workspace 預分配** | `cudaMalloc` workspace + `cudaMemPool` | 後者更靈活（Persistent Kernel 模式） |

### 6.3 可直接復用的優化模式

以下優化思路與硬件無關，可直接遷移到 CUDA/Metal/ROCm 等平台：

**模式 1: 空間換時間 —— 數據常駐**
```
原則: 在內層循環中不變的數據只加載一次，常駐在最靠近計算單元的緩存中
Ascend: Q L1 常駐 (s2Idx==0), V L1 常駐 (mCopyIdx==0)
CUDA:  FlashAttention-2 Q tile 常駐 SMEM, V tile 常駐 SMEM
Apple:  Q tile 常駐 threadgroup memory
任何平台: 找出在 K/V sequence loop 中不變的數據，常駐在最靠近計算的層級
```

**模式 2: 離線量化 × 在線反量化**
```
原則: 將 decompression 動作融合到計算管線中，避免獨立的 decompress pass
Ascend: Fixpipe DEQF16 (Cube 內聯反量化)
CUDA:  Epilogue fused scale + cast (用 __hfma 或 cutlass::epilogue)
任何平台: 反量化的 scale 與矩陣乘的累加器合併，省去記憶體往返
```

**模式 3: 多級 Tiling**
```
原則: 將矩陣乘沿 M 和 K 兩個方向切分，M-tile × K-tile 的組合
     確保每級 Tile 都在對應緩存層級的容量內
Ascend: M×K 雙層 tiling (baseM=64, baseK=128)
CUDA:  CUTLASS Threadblock Tile × Warp Tile × Thread Tile
任何平台: M 和 K 方向的 tile size = min(該層緩存容量 / 元素大小 / 對齊要求)
```

**模式 4: 雙重/多重緩衝掩蓋延遲**
```
原則: 一邊消費一邊生產，用額外的緩衝空間換取流水重疊
Ascend: L1 KP 三重緩衝, L1 V 四重緩衝
CUDA:  Pipeline Stages 3~5 (CUTLASS), cp.async commit groups
任何平台: 緩衝數量 = ceil(搬運延遲 / 計算時長)，最少 2 (ping-pong)，最多 4~5
```

**模式 5: Task-level 軟流水**
```
原則: 將多個 task 按流水線方式組織，每個階段處理不同 task 的數據
     保持計算單元盡可能不空閒
Ascend: TASK_NUM=2 深流水 + taskCtx 槽位輪轉
CUDA:  Persistent Kernel + Multi-stage software pipeline
任何平台: task 槽位數 = 流水深度，ExtraInfo 環形緩衝 = task 槽位數 × 2
```

**模式 6: 精度分級策略**
```
原則: 中間計算用高精度 (F32)，存儲用低精度 (FP16/INT8)，反量化在計算路徑中融合
Ascend: Softmax 內部 F32, BMM 輸入 INT8, 輸出 FP16 (DEQF16)
CUDA:  Tensor Core FP16 累加器 F32, Softmax F32, Epilogue 輸出 FP16/BF16
任何平台: 所有 reduction 類操作 (Softmax max/sum, dot-product) 保持 F32
          非 reduction 的路徑可使用低精度
```

### 6.4 遷移工作流建議

```
Step 1: 算法驗證 (平台無關)
  ├── Python/PyTorch 實現算法原型
  ├── 單元測試矩陣乘 + Softmax 的正確性
  └── 定義 input/output tensor shape/dtype 規範

Step 2: 計算模式拆分
  ├── 識別 BMM1(B×M×K×N)、Softmax(M×K)、BMM2(B×M×K×D)
  ├── 確定每個步驟的輸入/輸出 buffer 大小
  └── 標記可在循環間常駐的數據 (Q, V, scale, offset)

Step 3: 內存規劃 (目標平台相關)
  ├── 計算每個內存層級的容量 (HBM / L2 / SMEM / RF)
  ├── 反推每個層級可容納的最大 tile size
  └── 確定 M/K 方向的 tile 參數

Step 4: 流水設計
  ├── 設計 task-level 流水 (跨 kernel / 跨 stream)
  ├── 設計 core-level 流水 (內存搬運 × 矩陣乘 × 向量運算)
  └── 確定同步點和緩衝區數量

Step 5: 硬件加速編程
  ├── 矩陣乘: CUTLASS/wgmma/Tensor Core 指令
  ├── 後處理: Epilogue 融合 (Scale/Cast/Activation)
  └── 量化: fused quant/dequant 操作

Step 6: 驗證與調優
  ├── 數值精度對比 (FP32 參考 vs 優化版本)
  ├── 帶寬利用率計算 (理論 vs 實際)
  └── 端到端延遲基準測試
```

### 6.5 常見陷阱與規避

| 陷阱 | 風險 | 規避方法 |
|------|------|---------|
| **SMEM 容量不足** | CUDA SMEM 只有 Ascend L1 的 1/3~1/2 | 減小 tile size，增加 K 方向循環次數 |
| **Tensor Core 對齊要求不同** | MMA 的 K/N 維度對齊可能不同 | 檢查目標平台的 MMA 指令文檔，調整 tile 參數 |
| **同步開銷差異** | CUDA Stream sync 比 Ascend CrossCore 同步更重 | 盡量用 warp-level sync 替代跨 kernel sync |
| **workspace 分配策略** | Ascend 預分配機制不適用 CUDA | 改用 cudaMallocAsync 或 Persistent Kernel |
| **量化精度損失** | 不同平台的 FP16/INT8 轉換行為不同 | 在目標平台重新驗證精度，調校 quant scale |
| **Epilogue 融合能力** | Fixpipe DEQF16 無 CUDA 等價硬件 | 用 epilogue 軟件實現，或接受額外的 cast 指令開銷 |

---

## 7. 總結：算子設計的黃金法則

從 45+ 個 Attention 算子的代碼中提煉出的核心法則：

```
法則1: 數據流為王
       — 數據的搬運次數比計算次數更決定性能
       — 找出循環不變量，讓它常駐在最近的緩存層級
       
法則2: 分而治之
       — 沿著兩個方向同時 Tiling (M 和 K/N)
       — 每一級 Tile 都要精確 fit 該層緩存容量
       — 切得越細，流水氣泡越小 (但同步開銷越大，需求平衡)

法則3: 融合一切可融合的
       — 反量化 × 矩陣乘 = Fixpipe
       — Mask × Softmax = SelectWithBytesMask
       — Scale × Output = FusedMulAdd
       — 每次融合節省一個 GM↔SMEM 往返

法則4: 雙重緩衝是最低要求
       — 生產和消費永遠不能在同一個 buffer 上同時進行
       — N 重緩衝 (N≥2) 可以用空間換取流水深度
       — N = max(1, ceil(搬運延遲 / 計算時長))

法則5: 精度在關鍵路徑保持
       — Reduction (Softmax max/sum) → F32
       — 存儲和傳輸 → 最低可行精度 (FP16/INT8)
       — 累加器 → F32 或更高 (INT32 for INT8 MMA)
       
法則6: 編譯期而非運行期
       — 所有分支應在模板特化時消除
       — Runtime dispatch 僅用於跨算子選擇，不用於熱路徑
       — 每個 quant/layout/dtype 組合 = 一個獨立編譯單元

法則7: Tiling 決定一切
       — Tiling 既決定計算粒度，也決定 memory footprint
       — Workspace 大小 = f(tiling_params) — 先算 tiling 再算 space
       — 分核策略必須考慮 actualSeqLen 的不均勻性
```

---

## 附錄: 算子分類速查表

| 算子 | 分類 | 架構 | 核心特點 |
|------|------|------|---------|
| `flash_attention_score` | FA前向(訓練) | arch32/35 | 基礎版，含 PSE/DropMask |
| `prompt_flash_attention` | FA前向(Prompt) | arch32/35 | 最大 tiling 文件 (6.5K行) |
| `incre_flash_attention` | FA前向(Decode) | arch32/35 | **唯一含 DD variant**，最豐富模板 |
| `fused_infer_attention_score` | FA前向(統一) | arch32/35 | Prompt+Decode 合一 |
| `flash_attention_score_grad` | FA反向(訓練) | arch32/35 | 最複雜，多個 basic_module |
| `sparse_flash_attention` | 稀疏FA | arch32 | MLA support |
| `block_sparse_attention` | 塊稀疏FA | arch32/35 | CATLASS 模板庫 |
| `kv_quant_sparse_flash_attention` | 量化稀疏FA | arch35 | KV INT4/INT8 量化 |
| `nsa_selected_attention` | NSA選擇FA | arch32 | indexer + attention |
| `lightning_indexer` | 索引器 | arch32 | Top-K + ReLU(Q×W) |
| `chunk_gated_delta_rule` | 線性注意力 | arch32 | 分塊門控 Delta Rule |
| `recurrent_gated_delta_rule` | 循環注意力 | arch32/35 | 循環門控更新 |
| `mla_prolog` | MLA輔助 | arch32/35 | RoPE+RMSNorm+Dequant+Scatter (最複雜輔助) |
| `gather_pa_kv_cache` | PA Cache | arch32/35 | ND/NZ 兩種佈局 |
| `scatter_pa_kv_cache` | PA Cache | arch32/35 | 7 種模式 (Normal/RoPE/Alibi/...) |
| `ring_attention_update` | 分佈式 | arch35 | Ring Topology 合併 |
| `attention_update` | 結果合併 | arch32 | LSE-based online update |
| `swin_attention_score_quant` | Swin量化 | arch32 (310P) | Swin Transformer 專用 |
| `rain_fusion_attention` | 融合FA | arch32 | 專用融合變體 |
| `fused_causal_conv1d` | 卷積融合 | arch35 (950) | 因果卷積+Attention |
