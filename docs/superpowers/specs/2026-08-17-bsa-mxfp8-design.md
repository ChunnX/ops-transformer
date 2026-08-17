# BSA MXFP8 全量化设计说明（kvs128 修订版）

日期：2026-08-17  
基线：`mxfp8_0814` 上已有 MXFP8 脚手架；本文件**取代** `2026-08-14-bsa-mxfp8-design.md` 的几何与内存决策。  
状态：设计修订稿（相对 0814：缩小 kvs 基块，换整块 P / 少切 k）

## 1. 目标

在 BSA MXFP4 全量化路径旁共存 MXFP8 OCP 全量化，由 tiling key 选择。

相对 0814 稿的核心修订：

| | 0814 稿 | **本修订** |
| --- | --- | --- |
| 单次 QK → L0C | 128×128 | **64(kvs)×128(qs)** |
| Softmax 基块 | 256×128（两轮拼） | **128×128**（两轮 64 拼） |
| 组 G | 4（1024×128） | **8**（仍覆盖 **1024×128**） |
| `kvBaseTile` | 256 | **128** |
| Dpv / P 环 | 8 | **12** |
| UB P | 半基块双缓冲 | **整基块双缓冲** |
| L0 切 k | QK 切 D + PV 切 S2 | **默认不切**（见 §5） |

不变：

- `qs` 基块 128，D = 128  
- Q/K/V/P = `e4m3fn`，scale = `e8m0`，MX 组 32，**仅 OCP**  
- 操作数角色 **K→L0A、Q→L0B**（不可对调）  
- Softmax：**低精**（S = fp16）；rescale / oTmp：**fp32**

## 2. 术语

| 术语 | 含义 |
| --- | --- |
| QK 半面 | 一次 Cube mmad + fixpipe：`64×128` fp32→fp16 |
| 基块（base tile） | 一次 softmax / 一个 L1 P 槽单位：`128(kvs)×128(qs)`，由两轮 QK 半面拼成 |
| 组（tile group） | L0C 累加窗口 + group-max / pScale 作用域；**G = 8** 个基块，覆盖 S2 合计 1024（与 0814 的 256×4 同量级） |
| loop | 主循环全局基块序号 |
| Dpv | `PV_RE_DELAY_N`，PV 落后 softmax 的基块数 |
| G | `TILE_GROUP_N` |

数据流：

1. Cube 两次 QK（各 64 列 kvs）→ 同一 AIV 的 UB S（128×128 fp16）  
2. 该 AIV 整块 softmax → 整块 P（16KB）→ L1 槽 `loop % 12`  
3. 基块按 `loop % 2` 交替 AIV0/AIV1；一组 8 块中每 AIV 各 4 块；组尾经 L1 交换 localGlobalMax  

## 3. 流水与延迟

### 3.1 下界

同拍顺序仍为 **PV → pScale → 写 P**。组末块在 `gG+G-1` 出 S，再 `DELAY_PSCALE_N=3` 拍后 pScale 就绪；PV 消费组首需满足：

```
Dpv >= G + 3
```

G = 8 → 下界 **11**。取 **Dpv = 12**（+1 余量且为偶数，两 AIV 各 6 个 P 槽）。

环深 = Dpv：槽 `loop % 12` 在 `loop+12` 被 PV 读走；同拍先 PV 后写，**不需要 13 槽**。

### 3.2 常量

| 常量 | MXFP4 | 0814 MXFP8 | **本修订** |
| --- | --- | --- | --- |
| `TILE_GROUP_N` | 16 | 4 | **8** |
| `PV_RE_DELAY_N` | 20 | 8 | **12** |
| `DELAY_PSCALE_N` | 3 | 3 | 3 |
| `PRELOAD_TASK_CACHE_SIZE` | 21 | 9 | **13** |
| `L1_P_BUF_CNT` | 20 | 8 | **12** |
| `KVS_BASE_SIZE` | 256 | 256 | **128** |
| `QS_BASE_SIZE` | 128 | 128 | 128 |
| `UB_LOCAL_GROUP_MAX_CNT`（1K） | 20 | 8 | **12** |

组首/组尾频率：每 8 个基块一次（介于 MXFP4 的 16 与 0814 的 4 之间）。

### 3.3 尾部

`while` 条件中 delay 游标改为 delay12 / delay3；kv 基块数 &lt; 12 的短序列由同一 drain 逻辑覆盖。

## 4. L1 分配（512 KB）

规则：数据槽按 fp8 字节计；scale 分形与位宽无关（`yStep` 公式同 0814）。  
**KV 单槽仍取 32 KB**（Vᵀ+rowsum seed 镜像 18 KB &gt; 16 KB，不能把 KV 缩到 16 KB）。

| 区 | 偏移 | 单槽 | 槽数 | 合计 | 说明 |
| --- | --- | --- | --- | --- | --- |
| P 数据 | 0 | **16 KB** = 128×128×1B | **12** | **192 KB** | `loop % 12` |
| P-scale | 192 KB | **768 B** = 128×2×(128/64+1) | 12 | ≈9 KB | 随 kvs 变小 |
| Q 数据 | ≈201 KB | 16 KB | 2 | 32 KB | 128×128×1B |
| Q-scale | | 512 B | 2 | 1 KB | |
| KV 共享池 | | **32 KB** | 4 | 128 KB | 数据仅需 16KB；余量给 seed |
| KV-scale | | 2 KB | 4 | 8 KB | |
| localGlobalMax | | 512 B | 4 | 2 KB | |
| 空闲 | | | | **≈140 KB** | 多于 0814 |

## 5. L0 分配（各 64 / 256 KB）

基块 S2 = 128 后，**默认不再强制双切 k**：

| 操作数 | 几何 | 单槽 | 策略 |
| --- | --- | --- | --- |
| K | 64×128×1B | 8 KB | 双缓冲 |
| Q | 128×128×1B | 16 KB | **单缓冲**（避免 L0B 顶满） |
| Vᵀ | 144×128×1B | 18 KB | 双缓冲，**不切 S2** |
| Pᵀ | 128×128×1B | 16 KB | 双缓冲 |
| L0C QK | 64×128×4B | **32 KB** | 双缓冲 |
| L0C PV | 144×128×4B | 72 KB | 单缓冲 |

推荐布局：

| | 偏移 | 合计 |
| --- | --- | --- |
| L0A：K 2×8 + Vᵀ 2×18 | 0 / 16 KB | **52 KB**（余 12） |
| L0B：Q 1×16 + Pᵀ 2×16 | 0 / 16 KB | **48 KB**（余 16） |
| L0C：QK 2×32 + PV 72 | 0 / 64 KB | **136 KB**（余 120） |

每基块 mmad：**2 QK + 1 PV = 3 条**（回到 MXFP4 节奏；0814 为 6 条）。

可选退路（L0B 双缓 Q 时）：Q 沿 D 切 2×64 → Q 槽 8KB×2，L0B 仍 48KB，QK 变 4 条 mmad。

角色约束不变：不能把 Q 放 L0A。

## 6. UB 分配（256 KB）

S 变为 128×128 fp16 → 单槽 32KB；相对 0814 的 128KB S **省 64KB**，可恢复整块 P。

| 区 | 大小 | 说明 |
| --- | --- | --- |
| S | **64 KB**（2×32） | fp16，128×128，256B 交错 |
| oTmp | 32 KB | fp32 |
| rowSum | 512 B | |
| P | **32 KB**（2×16） | **整基块**双缓冲 |
| oTrans | 18 KB | 复用 P 区 |
| pScale | 12×768 B ≈ 9 KB | 复用 P 区尾 |
| attentionOut | 32 KB | |
| peer / softmaxMax | ≈1.25 KB | |
| localGroupMax | **12 KB** | = Dpv（1K 计） |
| localGlobalMax / dm / index | ≈2.25 KB | |
| **合计** | **≈185 KB** | 余 ≈70 KB |

## 7. 数据类型与量化

与 0814 相同：GM e4m3/e8m0；S fp16；P e4m3；P-scale e8m0；L0C fp32；Fixpipe **NO_QUANT**；seed `0x38`×`0x7F`。  
C0：`FP8_C0_ELEMS = 32`。  
组覆盖仍为 1024 S2，精度窗口与 0814 同量级（优于 MXFP4 的 4096）。

## 8. 主机侧

- `QuantMode::MXFP8_OCP_QUANT`、dtype、tiling key、分发：同 0814 方向。  
- **`CalcBaseTileTilingParams950`：MXFP8 强制 `kvBaseTile_ = 128`**（不再 256）。  
- sparse `blockShapeY` 对齐改为 **128**（或 64 的偶数倍）；workspace/idx 按新基块重算。  
- 验收尾块：kvs = 64 / 128 / 非对齐。

## 9. 文件清单

在 0814 已列 mxfp8 文件上**改几何与循环**，而非再开一套命名：

- 常量表：`KVS_BASE=128`，P 环 12，L0C QK 32KB，去掉默认 `QK_K_SPLIT`/`PV_K_SPLIT`  
- QK：`n` 方向按 64 出两轮，拼进 128 行 S  
- PV：单次 k=128；P 槽 `% 12`  
- Softmax VF：整块 128×128；删除半块 P 双槽语义  
- Kernel：`PV_RE_DELAY_N=12`，`TILE_GROUP_N=8`  
- Host：`kvBaseTile=128` + 对齐校验  

## 10. 风险

1. **Vᵀ dstStride 奇**（rowsum pad → 9）：仍在；兜底 pad 到 160，L0A 56KB / L0C 208KB 仍可。  
2. **Dpv=12**：PV 更滞后；同覆盖 1024 S2 时 **loop 次数约为 0814 的 2 倍**（基块更碎），Cube/同步开销需实测。  
3. **Q L0B 单缓冲**：Q 与 mmad 重叠变弱；若成为瓶颈再启用 Q 切 D 双缓冲。  
4. **与已实现的 256/G=4 脚手架不兼容**：需改常量与主循环，不能只热切换。  
5. P-scale **768 B** 分形需板上确认。

## 11. 验收

- L1/L0/UB `static_assert` 通过。  
- Golden：与 MXFP4 同口径阈值；覆盖 kvs 64/128/尾块、qs 128/64、短序列 drain。  
- 性能：对比 MXFP4、以及（若保留）0814 几何参考实现；标注瓶颈（短 loop vs 深 Dpv）。

## 12. 实施分期

1. **常量 + tiling**：`kvBaseTile=128`、G=8、Dpv=12、三张内存表。  
2. **Cube**：QK 64×128×2、PV 单次 k=128、L0 推荐布局；pScale 可先钉 `0x7F`。  
3. **Vec + 流水**：整块 softmax/P、group-max、delay 12/3、跨核。  
4. **收尾**：尾块、精度、性能与是否启用 Q 切 D。

## 13. 与 0814 / MXFP4 对照摘要

| | MXFP4 | 0814 MXFP8 | **本修订** |
| --- | --- | --- | --- |
| 基块 | 256×128 | 256×128 | **128×128** |
| 组 | ×16 | ×4 | **×8** |
| Dpv | 20 | 8 | **12** |
| 基块 mmad | 3 | 6 | **3** |
| UB P | 整块 | 半块 | **整块** |
| L1 P 字节 | 320KB | 256KB | **192KB** |
