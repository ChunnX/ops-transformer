# SFAG arch35 用例设计（TND 尾 padding + KV merge）

## 1. 文档用途

给后续补测用，不替代全量 ST 因子穷举。范围只覆盖这两次提交引入的行为：

| 提交 | 行为 |
|------|------|
| `9e44b0ee2` ARCH35 TND 尾 padding | `actual_seq_lengths_query[B-1] <= T1`，尾 token 不参与计算 |
| `840ffeb2e` dv 合并 | `value` 为空时 `dKey = dK + dV`，不写 `dValue` |

落地方式：一张 CSV 用例表 + Python 脚本（读表、造输入、算 golden、调 NPU 反向、比对、写回结果）。精度主路径走这套脚本，不要做成 Host UT / 手工 examples。

建议顺序：**P0 精度 → L1 组合 → 负例**。

---

## 2. 怎么跑

```text
CSV  --解析-->  单条用例参数
             -->  生成输入 + CPU golden（可只做到这一步）
             -->  调 SparseFlashAttentionGrad 反向
             -->  有效区间比对，写回 results/ 与日志
```

```bash
# 精度
python3 -u run_sfag.py --case ./data/sfag_arch35_tnd_kvmerge.csv \
    --device 0 --start-from 1 --end-at -1

# 只造 golden，不需要 NPU
python3 run_sfag.py --golden-only --case ./data/sfag_arch35_tnd_kvmerge.csv

# 单条
python3 run_sfag.py --case ./data/sfag_arch35_tnd_kvmerge.csv \
    --start-from 2 --end-at 3 --device 0
```

- `Enable=Disable` 的行跳过。
- `is_deter` 写 `true`/`false` 或 `1`/`0`。
- NPU 优先 `torch_npu.npu_sparse_flash_attention_grad`；若 PTA 尚未暴露，用 `aclnnSparseFlashAttentionGrad` 两段式。
- 精度门槛：超差占比 `< 0.5%`，fp16/bf16 单点绝对误差 `< 0.1`。通过后再按算子精度标准收紧。

---

## 3. 脚本结构

```text
tests/debug_tools/                   # 或独立目录
  run_sfag.py
  data/
    sfag_arch35_tnd_kvmerge.csv
  sfag_test/
    config.py        # 用例字段
    golden.py        # gather(K/V) → flash backward → scatter；merge 时 dK += dV
    pipeline.py      # 按物理 T1/T2 组 tensor；padding 填脏数据
    runner.py        # 调反向；kv_merge 时不传 value/dValue
    test_utils.py    # 读 CSV、有效区比对
    main.py          # 遍历表
```

脚本要点：

- TND 的 query 物理 shape 是 `[S1, N1, D]`，`S1` 是 packed 长度，**可以大于** `actual_seq_q` 的最后一个值。
- 下发给算子的 `actual_seq_lengths_query` / `actual_seq_lengths_kv` 是 **前缀和**，长度等于 `B`。
- `sparse_indices` 必选；用 `seed` 在 `[0, T2)` 内随机生成，padding token 的 indices 也要填合法随机值，不要全 0。
- `softmax_max` / `softmax_sum` / `out` 由 CPU 前向生成，T 维用物理 `S1`，不是 validT。
- `kv_merge=1`：NPU 不传 `value`、`dValue`；比对时不算 dv。

---

## 4. CSV 列

只保留本算子用得到的列。

| 列 | 含义 |
|----|------|
| `Testcase_Name` | 唯一 ID |
| `Enable` | `Enable` / `Disable` |
| `Dtype` | `bf16` / `fp16` |
| `Input_Layout` | `TND` / `BSND` |
| `B,N1,N2,S1,S2,D` | 见 4.1 |
| `sparse_mode` | `0` 或 `3` |
| `seqlens_list_q` / `seqlens_list_kv` | TND 必填，**前缀和**，长度 = `B` |
| `seed` | indices / 随机输入种子 |
| `is_deter` | 是否确定性计算 |
| `rope` | `0` / `1`；`1` 时额外生成 `Dr=64` 的 `query_rope` / `key_rope` |
| `selected_block_count` | indices 最后一维 |
| `sparse_block_size` | 当前必须为 `1` |
| `kv_merge` | `0` 有 value 和 dValue；`1` 两者都空 |
| `expect` | `pass` / `fail`（负例用 `fail`） |

### 4.1 TND 物理长 vs 有效长

`seqlens_list_q` 直接写算子要的前缀和（与 `actual_seq_lengths_query` 一致）：

```text
B=3, seqlens_list_q = [32, 96, 128]
validT1 = 128                          # 最后一个前缀和
S1 = 160                               # 物理 packed T1
padding 区间 = [128, 160)
```

| 列 | TND | BSND |
|----|-----|------|
| `S1` | 物理 T1（含 padding） | 每个 batch 的 query 序列长 |
| `S2` | 物理 T2（含 padding） | 每个 batch 的 KV 序列长 |
| `seqlens_list_q` | 长度 `B` 的前缀和，`last <= S1` | 空 |
| `seqlens_list_kv` | 长度 `B` 的前缀和，`last <= S2` | 空 |

无 padding：`S1` 等于 `seqlens_list_q` 的最后一项。有 padding：`S1` 更大，尾巴填非零脏数据。

### 4.2 rope

rope 是独立 tensor，`Dr=64`，主 `D` 仍是 512。`rope=1` 时脚本生成 `[T, N, 64]`。  
`kv_merge=1` 时 V 梯度只加到 `dKey` 的 512 维，不能进 `dKeyRope`。

---

## 5. arch35 合法域

写进 CSV 的值必须落在这里，否则 tiling 直接打回，测不到 kernel。

| 因子 | 合法值 |
|------|--------|
| layout | `TND` / `BSND` |
| dtype | `fp16` / `bf16` |
| `D` | **512**（rope 时另加 `Dr=64`） |
| `N2` | **1** |
| `N1` | `1,2,3,4,6,8,12,16,24,32,48,64,128` |
| `sparse_block_size` | **1** |
| `selected_block_count` | `1024,2048,...,8192` |
| `sparse_mode` | `0` 或 `3` |
| TND seq | 长度 `= B`，前缀和单调非降，`last <= S1/S2` |
| KV merge | `kv_merge=1` ⇒ `value` 与 `dValue` **同时**空 |

P0 默认底座：

```text
Dtype                 = bf16
Input_Layout          = TND
N1, N2                = 8, 1
D                     = 512
selected_block_count  = 1024
sparse_block_size     = 1
sparse_mode           = 3
scale                 = 1/sqrt(512)
aicNum 假设           = 32    # 用来构造 validT1 < usedCoreNum
```

TND 下 `usedCoreNum = min(T1, aicNum)`。要覆盖「部分核只有 padding 任务」，必须 `validT1 < usedCoreNum`，见 `K-TND-04`。

---

## 6. 因子与约束

### 6.1 TND 尾 padding

| 因子 | 取值 | CSV |
|------|------|-----|
| padding 落在哪侧 | 无 / 仅 Q / 仅 KV / 两侧 | `S1` 与 `seqlens_q` 末项，`S2` 与 `seqlens_kv` 末项 |
| batch | `1` / `>1` | `B` 与 seqlens 长度 |
| `validT1` vs `usedCoreNum` | `=` / `>` / `<` | `<` 才打到确定性 extra sync |
| `is_deter` | `false` / `true` | 非确定性只 skip；确定性还改 sync |

### 6.2 KV merge

| 因子 | CSV |
|------|-----|
| 正常输出 dv | `kv_merge=0` |
| 合并到 dk | `kv_merge=1` |
| rope | `rope=0/1` |
| layout | `TND` / `BSND` |

### 6.3 写用例前先过

| ID | 规则 | 违反时 |
|----|------|--------|
| C-TND-01 | `Layout=TND` ⇒ 两条 seqlens 都非空，长度 = `B` | 失败 |
| C-TND-02 | `seqlens_q` 末项 `<= S1`（kv 对 S2） | 当前 Host **不读 runtime 值**；末项 `> T1` 时 kernel 当无 padding |
| C-TND-03 | softmax_max/sum 的 T 维是**物理 S1**，不是 validT1 | Host 失败 |
| C-MRG-01 | `kv_merge=1` ⇔ value 与 dv 都空 | 文档要求；当前 tiling 只看 value |
| C-CMP-01 | padding 区不参与精度比较 | 误报 |
| C-CMP-02 | merge 的 golden：`dKey_ref = dK_ref + dV_ref`（`dK` 含 scale，`dV` 不含） | 对错公式 |
| C-CMP-03 | merge 不比较 dv | 无 dv tensor |

---

## 7. 推荐用例

`Testcase_Name` 用下面的 ID。P0 先 `Enable`。

未写出的列用默认：`Enable=Enable`，`Dtype=bf16`，`N1=8`，`N2=1`，`D=512`，`sparse_mode=3`，`seed=2`，`selected_block_count=1024`，`sparse_block_size=1`，`expect=pass`。

### 7.1 P0（必跑）

| Testcase_Name | Layout | B | S1 | S2 | seqlens_q | seqlens_kv | is_deter | rope | kv_merge | 比对 |
|---------------|--------|---|----|----|-----------|------------|----------|------|----------|------|
| K-TND-01 | TND | 1 | 64 | 128 | `[64]` | `[128]` | false | 0 | 0 | 全量 dq/dk/dv；无 padding 回归 |
| K-TND-02 | TND | 1 | 80 | 128 | `[64]` | `[128]` | false | 0 | 0 | 只比 query `[0,64)`；与 K-TND-01 有效区对齐 |
| K-TND-03 | TND | 3 | 160 | 128 | `[32,96,128]` | `[128]` | false | 0 | 0 | query `[0,128)`；对照 S1=128 且末项=128 |
| K-TND-04 | TND | 1 | 40 | 128 | `[16]` | `[128]` | **true** | 0 | 0 | `validT1=16 < usedCore=32`；有效区对齐 S1=16 |
| K-TND-04b | TND | 1 | 40 | 128 | `[16]` | `[128]` | false | 0 | 0 | 同一 shape 的非确定性，证明 skip 不依赖 extra sync |
| K-MRG-01 | BSND | 1 | 16 | 128 | （空） | （空） | false | 0 | **1** | `dk ≈ dK_ref+dV_ref`；不比 dv |
| K-MRG-02 | TND | 1 | 64 | 128 | `[64]` | `[128]` | false | **1** | **1** | `dk` 的 512 维 = dK+dV；`dk_rope` 等于非 merge 的 rope 梯度 |
| K-X-01 | TND | 1 | 80 | 128 | `[64]` | `[128]` | **true** | 0 | **1** | padding + merge + 确定性 |

`K-TND-02` 构造：

1. 有效区输入与 `K-TND-01` 相同。
2. `[64,80)` 的 query / dOut / out / softmax / indices 填**非零**脏数据。
3. 只比较 `[0,64)`。

`K-TND-04` 同理：有效输入按 T1=16 构造，物理 `S1=40` 的尾巴填脏数据。`is_deter=true` 卡死或超差都算失败。

`K-MRG-01` 用 BSND、`B=1,S1=16`，先把 merge 公式跑通，不要一上来很大。

### 7.2 P1（P0 过后再加，每条只翻一个因子）

| Testcase_Name | 相对 P0 的变化 |
|---------------|----------------|
| K-L1-01 | K-TND-02 改 `Dtype=fp16` |
| K-L1-02 | K-TND-02 加 `rope=1` |
| K-L1-03 | K-TND-02 改 `is_deter=true` |
| K-L1-04 | K-MRG-01 改 `Input_Layout=TND`，补 seqlens，`S1=16` |
| K-L1-05 | K-TND-01 改 `selected_block_count=2048` |
| K-L1-06 | K-TND-01 改 `sparse_mode=0` |
| K-L1-07a / 07b | `N1=1` 与 `N1=128` 各一条，其余同 K-TND-01 |

### 7.3 负例（P1）

`expect=fail`：tiling / GetWorkspaceSize 失败或 NPU 抛错即通过。

| Testcase_Name | 怎么写表 | 期望 |
|---------------|----------|------|
| E-01 | TND，`seqlens_list_q` 空 | fail |
| E-02 | TND，`seqlens_list_kv` 空 | fail |
| E-03 | value 空但 dValue 非空 | 文档要求 fail；**现状可能 pass**，先记基线 |
| E-04 | value 非空、dValue 空 | fail |
| E-05 | 两条 seqlens 长度不一致 | fail |
| E-06 | TND，`B=0` 或 seqlens 为空列表 | fail |

末项 `> S1`：**不要写成 Host 必失败**。当前 tiling 读不到 runtime 值。不要放进 P0。

### 7.4 P0 表内容

```csv
Testcase_Name,Enable,Dtype,Input_Layout,B,N1,N2,S1,S2,D,sparse_mode,seqlens_list_q,seqlens_list_kv,seed,is_deter,rope,selected_block_count,sparse_block_size,kv_merge,expect
K-TND-01,Enable,bf16,TND,1,8,1,64,128,512,3,"[64]","[128]",2,false,0,1024,1,0,pass
K-TND-02,Enable,bf16,TND,1,8,1,80,128,512,3,"[64]","[128]",2,false,0,1024,1,0,pass
K-TND-03,Enable,bf16,TND,3,8,1,160,128,512,3,"[32,96,128]","[128]",2,false,0,1024,1,0,pass
K-TND-04,Enable,bf16,TND,1,8,1,40,128,512,3,"[16]","[128]",2,true,0,1024,1,0,pass
K-TND-04b,Enable,bf16,TND,1,8,1,40,128,512,3,"[16]","[128]",2,false,0,1024,1,0,pass
K-MRG-01,Enable,bf16,BSND,1,8,1,16,128,512,3,,,2,false,0,1024,1,1,pass
K-MRG-02,Enable,bf16,TND,1,8,1,64,128,512,3,"[64]","[128]",2,false,1,1024,1,1,pass
K-X-01,Enable,bf16,TND,1,8,1,80,128,512,3,"[64]","[128]",2,true,0,1024,1,1,pass
```

结果列不用手写，脚本写回 `results/SparseFlashAttentionGrad_Result_<时间戳>.csv`。

---

## 8. Golden 与比对

### 8.1 非 merge（`kv_merge=0`）

CPU FP32：按 `sparse_indices` gather K/V → attention 反向 → scatter 回 K/V。NPU 输出转 FP32 再比。

### 8.2 merge（`kv_merge=1`）

1. 同一份 Q/K/dO/out/softmax/indices，CPU 仍按**有 V**计算（V 可等于 K，与 kernel 把 value 指到 key 一致），得到 `dK_ref`、`dV_ref`、`dQ_ref`。
2. NPU 不传 value/dValue。期望：
   - `dKey ≈ dK_ref + dV_ref`
   - `dQuery ≈ dQ_ref`
   - 不比较 `dValue`
3. `rope=1`：`dKeyRope ≈ dKRope_ref`（V 梯度不得加进 rope 维）。

### 8.3 padding

- 只比 query `[0, seqlens_q 末项)`、KV `[0, seqlens_kv 末项)`。
- 有效区应与「把 `S1` 收到 validT、seq 末项 = T」的对照行一致。
- padding 区不比数值，也不要求为 0（实现不保证清零）。
- padding token 填非零随机数。有效区仍对齐对照行，才能证明 skip 生效，而不是「尾巴碰巧是 0」。

---

## 9. 确定性为什么单独两行

确定性 scatter 会按 `ceil(validT1 / usedCoreNum)` 对齐核间同步。  
`validT1 < usedCoreNum` 时，一部分核没有有效 token，要靠 extra sync 对齐。

因此 `K-TND-04`（`is_deter=true`）和 `K-TND-04b`（`false`）成对出现：

- 确定性：卡死或超差都算失败
- 非确定性：证明 skip 不依赖这条 sync

建议：`S1=40`，`seqlens_q=[16]`，核数按 32 理解：核 0–15 各 1 个有效 token，核 16–31 全是 padding。

---

## 10. 实施顺序

1. 搭脚本：读 CSV、造 indices、CPU golden、调反向、有效区比对。
2. **K-TND-01** — 无 padding 回归，先证明脚本和门槛可用。
3. **K-TND-02** — skip 不影响有效区。
4. **K-MRG-01** — `dk = dK+dV`。
5. **K-TND-04 / 04b** — 确定性 sync 边界。
6. **K-X-01** — padding 与 merge 叠在一起。
7. 其余 P0，再 L1，再负例。

P0 共 **8 条**（含 04b）。这些过了再扩 L1，否则分不清是 padding 还是 merge 的问题。
