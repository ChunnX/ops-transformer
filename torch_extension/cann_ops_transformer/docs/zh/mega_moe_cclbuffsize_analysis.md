# mega_moe cclbuffsize 分析报告

## 背景

mega_moe 在 A2 和 A3（`arch22` 文件夹）都支持 **A16W16（BF16）**、**A8W8-INT**、**A8W4** 三种场景。

HCCL 通信域缓冲区分为 `windowsIn` 和 `windowsOut` 两块，**大小相等**，各 = `cclBufferSize`（=`HCCL_BUFFSIZE`=`winSize`=`m_segmentSize`）。总物理分配 = `2 × cclBufferSize`。

**四条数据流**：

| 阶段 | 方向 | 物理窗口 | offset |
|---|---|---|---|
| Dispatch **发送** | 本 rank 量化后写入待发数据 | windowsOut | offsetA |
| Dispatch **接收** | 对端 RDMA 写入 + 本 rank 自拷贝 | windowsIn | offsetA |
| Combine **发送** | GMM2 输出写入待发数据 | windowsOut | offsetD |
| Combine **接收** | 对端 RDMA 写入 + 本 rank 自拷贝 | windowsIn | offsetD |

由于 in/out 共用同一 layout，每段须按 `max(发送侧数据量, 接收侧数据量) = M_recv = maxRecvTokenNum` 取值。

---

## 符号说明

| 符号 | 含义 |
|---|---|
| `M` | `maxRecvTokenNum`（本 rank 接收 token 上限，按双侧 max 计） |
| `h` | `hiddenSize` |
| `EP` | `epWorldSize`（通信域大小） |
| `E` | `expertPerRank = moeExpertNum / EP` |
| `TPE` | `EP × CeilAlign(EP × E, 128) × 4B`（tokenPerExpert 实际需要） |

---

## 各段实际所需大小

### 每行字节数

| 场景 | dispatch 每行（offsetA） | combine 每行（offsetD） | per-token scale |
|---|---|---|---|
| A16W16 | `h × 2`（BF16，无 padding） | `h × 2`（BF16） | 无 |
| A8W8-INT | `(h + 512) × 1`（INT8 + 512B inline scale） | `h × 2`（BF16） | `4B`（float） |
| A8W4 | `(h + 512) × 1`（dispatch 阶段 peermem 存 INT8，接收侧再转 INT4） | `h × 2`（BF16） | `4B`（float） |

### A2

| 场景 | offsetA | offsetD | offsetPeerPerTokenScale | offsetFlag | offsetAllgatherFlag | offsetPeerTokenPerExpert |
|---|---|---|---|---|---|---|
| A16W16 | `M × h × 2` | `M × h × 2` | — | `max(EP×E×64B, 1MB)` | `max(EP×64B, 1MB)` | `max(TPE, 2MB)` |
| A8W8-INT | `M × (h + 512)` | `M × h × 2` | `max(M × 4, 1MB)` | `AlignUp(EP×E×4, 512)` | — | `max(TPE, 2MB)` |
| A8W4 | `M × (h + 512)` | `M × h × 2` | `max(M × 4, 1MB)` | `max(EP×E×64B, 1MB)` | — | `max(TPE, 2MB)` |

### A3

A3 三种场景共用同一套 PeermemInfo layout，无 offsetFlag / offsetAllgatherFlag（A3 不通过 peermem 推 flag）。BF16 场景下 layout 仍保留 1MB offsetPeerPerTokenScale 区（纯浪费）。

| 场景 | offsetA | offsetD | offsetPeerPerTokenScale | offsetPeerTokenPerExpert |
|---|---|---|---|---|
| A16W16 | `M × h × 2` | `M × h × 2` | —（layout 占 1MB 浪费） | `max(TPE, 2MB)` |
| A8W8-INT | `M × (h + 512)` | `M × h × 2` | `max(M × 4, 1MB)` | `max(TPE, 2MB)` |
| A8W4 | `M × (h + 512)` | `M × h × 2` | `max(M × 4, 1MB)` | `max(TPE, 2MB)` |

---

## 单侧 cclBufferSize 计算公式（实际最小值）

### 实际所需最小值

| 场景 | 平台 | 单侧 `cclBufferSize` 最小值 |
|---|---|---|
| A16W16 | A2 | `2 × M × h × 2 + max(EP×E×64B, 1MB) + max(EP×64B, 1MB) + max(TPE, 2MB)` |
| A16W16 | A3 | `2 × M × h × 2 + 1MB(wasted) + max(TPE, 2MB)` |
| A8W8-INT | A2 | `M × (h + 512) + M × h × 2 + max(M×4, 1MB) + AlignUp(EP×E×4, 512) + max(TPE, 2MB)` |
| A8W8-INT | A3 | `M × (h + 512) + M × h × 2 + max(M×4, 1MB) + max(TPE, 2MB)` |
| A8W4 | A2 | `M × (h + 512) + M × h × 2 + max(M×4, 1MB) + max(EP×E×64B, 1MB) + max(TPE, 2MB)` |
| A8W4 | A3 | `M × (h + 512) + M × h × 2 + max(M×4, 1MB) + max(TPE, 2MB)` |

### 典型精简公式（固定段取下限，M×4 < 1MB，TPE < 2MB 时）

| 场景 | 平台 | 单侧 `cclBufferSize` 最小值（M 大时主导项） |
|---|---|---|
| A16W16 | A2 | `4 × M × h + 4MB` |
| A16W16 | A3 | `4 × M × h + 3MB` |
| A8W8-INT | A2 | `3 × M × h + 512 × M + 3MB` |
| A8W8-INT | A3 | `3 × M × h + 512 × M + 3MB` |
| A8W4 | A2 | `3 × M × h + 512 × M + 4MB` |
| A8W4 | A3 | `3 × M × h + 512 × M + 3MB` |

---

## 当前 1/3 切分 layout 下所需 cclBufferSize

当前 layout（kernel 中 PeermemInfo 的硬编码切分）：
- offsetA = `SegSize / 3`
- offsetD = 剩余空间

需同时满足：

```
SegSize/3 ≥ offsetA_need
SegSize − offsetA − (固定段) ≥ offsetD_need
```

| 场景 | 平台 | offsetA_need | offsetD_need | 固定段总和 | 当前 layout 所需 SegSize（取 max） |
|---|---|---|---|---|---|
| A16W16 | A2 | `M×h×2` | `M×h×2` | 1MB+1MB+2MB=4MB | `max(3×M×h×2, 1.5×(M×h×2+4MB))` = **`6×M×h`** |
| A16W16 | A3 | `M×h×2` | `M×h×2` | 1MB+2MB=3MB | `max(3×M×h×2, 1.5×(M×h×2+3MB))` = **`6×M×h`** |
| A8W8-INT | A2 | `M×(h+512)` | `M×h×2` | 1MB+AlignUp(EP×E×4,512)+2MB≈3MB | `max(3×M×(h+512), 1.5×(M×h×2+3MB))` |
| A8W8-INT | A3 | `M×(h+512)` | `M×h×2` | 1MB+2MB=3MB | `max(3×M×(h+512), 1.5×(M×h×2+3MB))` |
| A8W4 | A2 | `M×(h+512)` | `M×h×2` | 1MB+1MB+2MB=4MB | `max(3×M×(h+512), 1.5×(M×h×2+4MB))` |
| A8W4 | A3 | `M×(h+512)` | `M×h×2` | 1MB+2MB=3MB | `max(3×M×(h+512), 1.5×(M×h×2+3MB))` |

---

## 浪费对比（当前 layout vs 理想切分）

| 场景 | 平台 | 理想 SegSize | 当前 1/3 切分 SegSize | 浪费比例 |
|---|---|---|---|---|
| A16W16 | A2 | `4×M×h + 4MB` | `6×M×h` | **~50%** |
| A16W16 | A3 | `4×M×h + 3MB` | `6×M×h` | **~50%** |
| A8W8-INT | A2 | `3×M×h + 512×M + 3MB` | `max(3×M×(h+512), 3×M×h+4.5MB)` | ~0%（h 大时） |
| A8W8-INT | A3 | `3×M×h + 512×M + 3MB` | 同上 | ~0%（h 大时） |
| A8W4 | A2 | `3×M×h + 512×M + 4MB` | `max(3×M×(h+512), 3×M×h+6MB)` | ~0%（h 大时） |
| A8W4 | A3 | `3×M×h + 512×M + 3MB` | 同上 | ~0%（h 大时） |

---

## 当前分配存在的问题

### 问题 1：A16W16 场景 1/3 切分严重失衡

offsetA 和 offsetD 实际都需要 `M × h × 2`（完全相等），1/3 切分导致 offsetA 溢出时 offsetD 空一半。需要约 1.5x 的 cclBufferSize 才安全。

**建议：改为 1:1 切分，`offsetA = offsetD = (SegSize − 固定段) / 2`。**

### 问题 2：A8W8-INT / A8W4 场景 1/3 切分大致合理，但有偏差

INT8 dispatch : BF16 combine ≈ 1:2，1/3 切分接近理想。但 h 较小时（如 h=1024），512B padding 占比上升，offsetA 偏紧。

**建议：按 `(h+512) : 2h` 比例动态切分。**

### 问题 3：固定段浪费严重

| 段 | 分配 | 典型需要 (EP=8, E=8) | 浪费 |
|---|---|---|---|
| offsetPeerTokenPerExpert | 2MB | KB 级 | 大 |
| offsetPeerPerTokenScale | 1MB | M×4B（M=8192 时 32KB） | 大 |
| offsetFlag (BF16/W4A8) | 1MB | EP×E×64B = 4KB | 大 |

**建议：改为 `max(实际需要, lowerBound)` 动态计算。**

### 问题 4：A3 BF16 场景保留 1MB perTokenScale 区

纯浪费，因 BF16 无 per-token scale。

### 问题 5：Python `npu_get_mega_moe_ccl_buffer_size` 与 tiling `leastCclBufferSize` 均有缺陷

- 用 `M_send = bs×topK` 而非 `M_recv`，低估接收侧/combine 发送侧需求。
- 漏算 offsetA（dispatch 数据区）项。
- 不按 weight dtype 区分每行字节数。
- 校验下界与 `cclBufferSize`（单侧）口径不一致。

---