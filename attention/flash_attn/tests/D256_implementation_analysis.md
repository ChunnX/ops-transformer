# Flash Attention D=256 实现分析（arch35 / config=4）

基于 `flash_attn-d256-perf` 分支（master @ `fbb042dad`）梳理 D=256 路径的切块、Buffer 分配与流水实现，作为后续性能优化的基线参考。

---

## 1. 入口分发与 config 选择

- Tiling key 由 4 字段组成：`InOutLayoutType / KvLayoutType / HasAttenMask / Config`（`op_kernel/arch35/flash_attn_template_tiling_key.h:40`）。
- D=256 固定映射到 **config=4**，对应模板常量（`op_kernel/utils/flash_attn_common_def.h:179-181`）：
  - `S1(mBaseSize)=Aligned64 → 64`
  - `S2(s2BaseSize)=Aligned256 → 256`
  - `D(dBaseSize)=Aligned256 → 256`
  - `DV(dVBaseSize)=Aligned256 → 256`
- Host 侧 `AdjustSinnerAndSouter` 对 `vHeadDim==256` 无条件取 `sOuter=32, sInner=256`（`op_host/fa_adjust_sinner_souter.h:78-81`），`UpdateTilingKeyConfig` 置 `config=4`（`op_host/arch35/flash_attn_tiling.cpp:144`）。
- `EnableSoftmaxDn<hasAttenMask,config>` 在 config=4 时返回 **false**（`op_kernel/flash_attn.cpp:86-93`），因此 D=256 **始终走 Nd 路径**（`FlashAttentionNoQuantGqaKernelNd`），不走 Dn。

> Host `sOuterFactor_=32` 与 kernel `mBaseSize=64` 一致：`mSize = sOuterFactor_ * cvRatio = 32*2 = 64`（`flash_attn_tiling.cpp:212`），即 cube 一次处理 M=64，再被 2 个 AIV sub-block 各分 32。

---

## 2. 切块（Tile）策略

| 维度 | 切块大小 | 说明 |
|---|---|---|
| M（gS1 外切，=g×S1） | **64**（cube），按 `(actMSize+1)>>1` 拆成 32+32 给 2 个 AIV sub-block | `block_cube_nd.h:95`、`kernel_nd.h:528` |
| S2（内切） | **256** | 每个内循环处理 256 个 KV token |
| D / DV | **256 / 256** | head dim 不再切（整 D 进单次 K） |
| BMM1 L0 tile | `MatmulN<64, 128, 256>` | M=64, **N=128×2**（S2=256 拆 2 次 N）, K=256（整 D 一次） | `block_cube_nd.h:354-356` |
| BMM2 L0 tile | `MatmulN<64, 128, 256>` | M=64, **N=128×2**（DV=256 拆 2 次 N）, K=256（整 S2 块一次） | `block_cube_nd.h:440-442` |

即 D=256 时，D/K 维度单次吃满（K=256），但 **N 方向（S2 与 DV）都被切成 128×2**，每个 BMM 实际触发 2 次 L0 matmul + 2 次 fixpipe。

核间切分：`CalcNumBlocks` 用 `CalcTschBlockDim(aivNum, aicNum, aivNum)`（`flash_attn_tiling.cpp:205`），按 BN2/gS1O/S2O 三级循环分配给 AIC，S2 可跨核切分（`isS2SplitCore`）以支撑长序列与 FlashDecode。

---

## 3. Buffer 分配

> 符号约定：`M=mBaseSize=64`（cube 一次处理的 Q 行数），`N=s2BaseSize=256`（KV 内切），`D=dVBaseSize=256`（head dim），`CV_RATIO=2`（1 AIC:2 AIV，cube 64 行拆 32+32 给两个 AIV）。元素字节：fp32=4B、bf16=2B、uint8=1B。Vec 侧实际只处理 `M/CV_RATIO=M/2` 行，但按行 reduce / 跨核传递的 buffer 仍按 `M` 行分配。

### 3.1 Cube 侧（`op_kernel/arch35/flash_attn_block_cube_nd.h`）

D=256 触发 `L1KV_IS_SINGLE = (s2BaseSize==256 && dBaseSize>128) = true`（`block_cube_nd.h:105`），KV 走“单 buffer + 2 槽”策略：

| 层 | Buffer | 单块公式 | 单块 | 槽数 | 总量公式 | 总量 | 备注 |
|---|---|---|---|---|---|---|---|
| UB | mm1Res | `(M/2)·N·4 = 2MN` | 32KB | 2 | `4MN` | 64KB | BMM1 结果 QKᵀ（fp32），cube→vec 跨核传 |
| UB | mm2Res | `(M/2)·D·4 = 2MD` | 32KB | 2 | `4MD` | 64KB | BMM2 结果 P'V（fp32），cube→vec 跨核传 |
| L1 | P | `M·N·2 = 2MN`（bf16） | 32KB | 3 | `6MN` | 96KB | VEC 写入供 BMM2 用 |
| L1 | Q | `M·D·2 = 2MD`（bf16） | 32KB | 2 | `4MD` | 64KB | 首个 S2 循环载入并跨 S2 复用 |
| L1 | KV | 固定 128KB/槽 | **128KB** | 2 | — | 256KB | `L1KV_IS_SINGLE` 时固定 |
| L0A | manager | — | 64KB / 32KB sub | — | — | 64KB | `BufferManager<L0A>` |
| L0B | manager | — | 64KB / 32KB sub | **single** | — | 64KB | `L1KV_IS_SINGLE` 时用 `SingleBuffer`（无 DB） |
| L0C | — | — | 64KB | 4 | — | 256KB | `L0C_BUFCNT=4` |

- Cube UB 合计：`4MN + 4MD = 8MN = 8MD = 128KB`（N=D=256）。
- L1 合计：`6MN + 4MD + 256KB = 416KB`（`static_assert <= 512KB`，`block_cube_nd.h:167`）。

### 3.2 Vec 侧（`op_kernel/arch35/flash_attn_block_vec_nd.h`）

L1：P 3×32KB=96KB（`6MN`，与 cube 共享同一物理 L1P 区域，跨核传递 P）。

UB（`UbLayout`，`block_vec_nd.h:202-216`，`static_assert <= 248KB`）。下表区分「当前分配」与「实际所需最小空间」：每个 AIV 只处理 `M/2=32` 行（`actVecMSize=(actMSize+1)>>1`，`kernel_nd.h:528`），故按行 reduce / broadcast 的 buffer 实际只需 `M/2` 行，而代码从 base 类继承的硬编码值仍按全 `M=64` 行分配，存在 2x 过配。

| Buffer | 当前分配单块 | 实际所需单块公式 | 实际所需 | 槽 | 实际所需总量 | 备注 |
|---|---|---|---|---|---|---|
| mm1Res | 32KB | `(M/2)·N·4 = 2MN` | 32KB | 2 | `4MN` = 64KB | BMM1 结果（fp32），cube 写满 M/2 行，无过配 |
| mm2Res | 32KB | `(M/2)·D·4 = 2MD` | 32KB | 2 | `4MD` = 64KB | BMM2 结果（fp32），cube 写满 M/2 行，无过配 |
| vec1Res | **33024B** | `(M/2)·N·2 + (M/2)·4 = MN+2M` | **16512B** | 2 | `2MN+4M` = 33024B | softmax 输出 P'(bf16) + VF fp32 临时行，纯 FA 紧凑，**过配 2x**（代码按 FD 共享 33024 分配） |
| vec2Res | 32KB | `(M/2)·D·4 = 2MD` | 32KB | 1 | `2MD` = 32KB | FlashUpdate accum（fp32），写满 M/2 行 |
| mask | 8KB | `(M/2)·N·1 = MN/2`（uint8） | 8KB | 2 | `MN` = 16KB | attenMask 拷入，写满 M/2 行 |
| lseOut | 2KB | `(M/2)·8·4 = 16M`（fp32，broadcast 扩 8 倍） | **1KB** | 2 | `32M` = 2KB | broadcast 32 行×8 fp32，**过配 2x** |
| softmaxSum | 256B | `(M/2)·4 = 2M`（fp32，按行 reduce） | **128B** | 3 | `6M` = 384B | 行 reduce 32 行，**过配 2x** |
| softmaxMax | 256B | `(M/2)·4 = 2M` | **128B** | 3 | `6M` = 384B | 行 max 32 行，**过配 2x** |
| softmaxExp | 256B | `(M/2)·4 = 2M` | **128B** | 3 | `6M` = 384B | exp 缩放因子 32 行，**过配 2x** |
| softmaxTmp | 512B | 固定 512B | 512B | 1 | 512B | VF softmax 中间缓存（常驻） |

- Vec UB 当前分配合计：`4MN + 4MD + 66048 + 2MD + MN + 64M + 3·12M + 512 = 5MN + 6MD + 100M + 66048 + 512 = 253184B ≈ 247KB`（≤ 248KB）。
- Vec UB 实际所需合计（纯 FA 紧凑）：`4MN + 4MD + (2MN+4M) + 2MD + MN + 32M + 3·6M + 512 = 7MN + 6MD + 54M + 512 = 216960B ≈ 212KB`。
- 共过配 `253184 - 216960 = 36224B`：vec1Res 过配 `2·(33024-16512) = 33024B`（代码按 FA/FD 共享的 33024 单块分配，纯 FA 紧凑只需 16512），sum/max/exp/lseOut 过配 `50M = 3200B`（按全 M 分配但实际只用 M/2），合计可作 UB 优化空间。

#### 公式推导（按 FA 流程）

1. **mm1Res / mm2Res = 2MN / 2MD**：cube 把 64 行 BMM 结果按 `(M/2)×N` 拆给 2 个 AIV，fp32(4B)，故 `(M/2)·N·4 = 2MN`；双 buffer 做 2 级流水（`PRELOAD_N=2`）。cube 写满 M/2 行，分配 = 实际所需，无过配。
2. **vec1Res = MN+2M（实际所需，纯 FA 紧凑），代码分配 33024B（过配 2x）**：
   - **FA 实际所需**：P'(bf16) 形状 `(M/2)×N`，`(M/2)·N·2 = MN = 16384B`；VF softmax 内 fp32 临时行 `(M/2)·4 = 2M = 128B`；单块合计 `MN+2M = 16512B`，2 槽 `2MN+4M = 33024B`。
   - **代码为何分配 33024B 单块**：该 buffer 与 FlashDecode 共享（`flash_attn_block_vec_flashdecode.h:191-201`）。FD mm2Res 需 fp32 `(M/2)·D·4 = 2MD = 32768B`（拆 2×16KB 槽 `fdMm2ResBuf1_/2_`），+256 对齐 = 33024B。代码按 `max(FA 16512, FD 32768) + 对齐 = 33024B` 单块硬编码（`block_vec_nd.h:114`）。纯 FA 路径下单块过配 `33024-16512 = 16512B`。
   - 注：VF 模板实际写出用 NZ 布局 stride=`M/2+1=33`（`vf_basic_block_unaligned256_no_update.h:346-351`），比紧凑多 512B gap；纯 FA 含 NZ gap 的单块最小为 17024B。此处取紧凑口径（不含 gap）。
3. **vec2Res = 2MD**：FlashUpdate 的 `accum`（fp32），形状 `(M/2)×D`；`vec2CalcSize = actVecMSize · dTemplateAlign64`，末轮 `Cast(attenOut, ubVec2Res_, ..., dSizeAligned64)` 转 bf16 输出（`block_vec_nd.h:663,713`）。写满 M/2 行，无过配。
4. **sum/max/exp = 2M（实际所需），代码分配 4M（过配 2x）**：按行 reduce，每行 1 个 fp32；VF 模板循环 `for (i=0; i<m; ++i)` 其中 `m=actVecMSize=M/2=32`（`vf_basic_block_unaligned256_no_update.h:136`），实际写 32 个 fp32 = 128B = `2M`。但代码从 base 类继承硬编码 256B（`block_vec_nd.h:119/122/125`，`flash_attention_noquant_block_vec_base.h:2150-2162` 注释 `[64,1]`），按全 `s1BaseSize=M=64` 分配，过配 2x。
5. **lseOut = 16M（实际所需），代码分配 32M（过配 2x）**：`BroadCastAndCopyOut` 把每行 sum/max broadcast 到 8 个 fp32（`fp32BaseSize=8`，`flash_attention_score_common_regbase.h:49`）；`BroadcastMaxSum` 内 `loopM = (vecS1RealSize+7)>>3 = 4`，每轮写 64 个 fp32，共 256 个 fp32 = 1024B = `16M`（`vf_mul_sel_softmaxflashv2_cast_nz_dn.h:3162-3182`，`calculateSize = actVecMSize·8`）。但代码硬编码 2048B（`block_vec_nd.h:129`），按全 M 取上限，过配 2x。
6. **mask = MN/2**：uint8 mask，`(M/2)×N` = 32×256 = 8192B，写满 M/2 行，无过配。

### 3.3 Workspace（`op_host/arch35/flash_attn_tiling.cpp:209-255`）

- `mSize = sOuterFactor_*cvRatio = 64`，`dVBasicBlock=256`。
- D=256 且非 dn：`bmm2Bytes = mSize * 256 * 4`，`vec2Bytes=0`（仅 `dVBasicBlock>256` 才有 vec2），×3（preload 2+1）×coreNum。
- FlashDecode：`accumOutSize = coreNum*2*mSize*dSize`，`logSumExpSize = coreNum*2*mSize*8`（sum+max 两份）。
- PA 场景额外 `coreNum*2*2*64` 对齐开销。

---

## 4. 流水（Pipeline）

### 4.1 总体结构

- `KERNEL_TASK_TYPE_MIX_AIC_1_2`：1 AIC : 2 AIV sub-block（`CV_RATIO=2`）。
- `PRELOAD_N=2`，`PRELOAD_TASK_CACHE_SIZE=4`（位掩码代模，`kernel_nd.h:54-61`）。`ExecuteTask` 在同一轮里跑 **task[i] 的 BMM1/Vec1** 与 **task[i-2] 的 BMM2/Vec2**（`kernel_nd.h:426-448`），形成 2 级软件流水，跨 S2 迭代重叠 BMM1 与 BMM2。

### 4.2 单个 S2 迭代的 4 级 CV 流水

跨核同步 `CROSS_CORE_SYNC_MODE=4`，flag：`CC_BMM1_0/1`、`CC_BMM2_0/1`、`CC_L1P_0/1/2`（cube 与 vec 各定义一致，`block_cube_nd.h:68-75` / `block_vec_nd.h:73-80`）。

```
CUBE BMM1 (task i):
  MTE2: K→L1 (Q 仅 isFirstS2Loop 载入, 跨 S2 复用)
  MTE1→M:  Q@Kᵀ → L0C  (MatmulN 64,128,256, N 拆 2 次)
  FIX:     L0C → UB mm1Res[i%2]
  set CC_BMM1[i%2]  ──────────────────────►  VEC
VEC Vec1 (task i):
  wait CC_BMM1[i%2]
  V:    softmax(scale,mask,max,sub,exp,cast bf16) on mm1Res → stage1Cast
  MTE3: DataCopy P → L1 l1P[i%3]
  set CC_L1P[i%3]   ──────────────────────►  CUBE
  (并发) UpdateExpSum/Max; 末尾 SoftmaxDataCopyOut
CUBE BMM2 (task i):
  wait CC_L1P[i%3]
  MTE2: V→L1
  MTE1→M:  P@V → L0C  (MatmulN 64,128,256, N 拆 2 次)
  FIX:     L0C → UB mm2Res[i%2]
  set CC_BMM2[i%2]  ──────────────────────►  VEC
VEC Vec2 (task i):
  wait CC_BMM2[i%2]
  V:    FlashUpdate(accum += exp*mm2 / rescale) → ubVec2Res
        (首 S2: DataCopy; 末 S2: FlashUpdateLast + LastDiv + Cast→bf16)
  set CC_BMM2[i%2]  (释放 mm2Res UB 给 cube)
  末 S2: Bmm2DataCopyOutTrans → GM  (或 isS2SplitCore → 写 workspace 交 FD)
```

核内用 `Mutex::Lock/Unlock<PIPE_*>` 在 MTE2/MTE1/M/FIX/V/MTE3 间保护多 buffer 槽（事件 ID 见 `block_cube_nd.h:78-91`、`block_vec_nd.h:84-91`）。

### 4.3 FlashDecode 阶段（`op_kernel/arch35/flash_attn_block_vec_flashdecode.h`）

`flashDecodeFlag_` 在 `SplitPolicy` 中恒置 true（`flash_attn_tiling.cpp:129`），`enableFlashDecode` 由 AICPU metadata[1] 决定。当某 S1 块的 S2 被跨核切分（`isS2SplitCore`）时：
- Vec2 末轮把 `accumOut / softmaxFDSum / softmaxFDMax` 写入 workspace GM（`Bmm2ResForFDCopyOut`、`BroadCastAndCopyOut`）。
- FD 阶段 `vecFdBlock_.FlashDecode(fdParams_)` 跨 S2-split 做 max/sum 归一化与加权累加，产出最终 `attentionOut`（`kernel_nd.h:580-591`，前有 `ICachePreLoad(2)` + `SyncAll`）。

---

## 5. D=256 性能特征与可关注点

1. **N 拆 2×128**：BMM1/BMM2 因 L0C N-tile=128，S2/DV=256 各多一轮 L0 matmul+fixpipe，cube 计算量与 L0C 周转翻倍。可评估是否允许 N-tile=256 的 L0C 模式以减半轮次（受 L0C 64KB 单块约束）。
2. **L0B 单 buffer**：`L1KV_IS_SINGLE` 下 L0B 用 `SingleBuffer` 策略（`block_cube_nd.h:124-128`），K/V 的 L0B 无 double buffer，MTE1 载入与 M 计算间易出 bubble。可考察 L1 KV 是否能压缩（如 D 分块）换回 L0B DB。
3. **L1 KV 占 256KB**（2×128KB），是 L1 大头；P 3×32KB 槽为跨核 P 传递流水服务。L1 已用 416/512KB。
4. **Vec UB 当前分配 247KB**（实际所需 212KB）：vec1Res 单块代码按 FA/FD 共享分配 33024B，纯 FA 紧凑只需 16512B（过配 33024B）；sum/max/exp/lseOut 按全 M 分配但实际只用 M/2（过配 3200B）。共可释放 36224B， UB 优化空间充足。
5. **softmax 走 `GT_128_AND_LTE_256` 通用路径**（`block_vec_nd.h:437-445`），非 `EQ_128` 快路径；尾块 ≤128 时才回落快路径。s2BaseSize=256 主路径的 VF softmax 是 vec 侧热点之一。
6. **M=64 偏小**：D=256 下 cube K=256、M=64，L0A 利用率有限；若 UB/L1 可调，增大 M 块能提升 cube 占用，但会推高 UB（mm1/mm2Res 线性增长）与 L1Q。
7. **PRELOAD_N=2**：已做 2 级跨 S2 重叠；task ring 用 4 槽位掩码索引避免取模（`kernel_nd.h:55-59`）。进一步加深流水受 UB 槽数（mm1/mm2Res 各 2 槽）限制。

后续优化可优先从 **L0B 单 buffer bubble**、**N=128 拆分轮次**、**s2=256 softmax VF 路径** 三处切入，并配合 msprof 确认 cube/MTE2/VEC 的实际空闲段。

---

## 关键文件索引

| 模块 | 文件 |
|---|---|
| Tiling key 定义 | `op_kernel/arch35/flash_attn_template_tiling_key.h` |
| Host tiling | `op_host/arch35/flash_attn_tiling.cpp` / `.h` |
| sOuter/sInner 调整 | `op_host/fa_adjust_sinner_souter.h` |
| Tiling data 结构 | `op_kernel/arch35/flash_attn_tiling_data.h` |
| Kernel 入口 | `op_kernel/flash_attn.cpp` |
| Nd Kernel 主类 | `op_kernel/arch35/flash_attn_kernel_nd.h` |
| Cube Block (Nd) | `op_kernel/arch35/flash_attn_block_cube_nd.h` |
| Vec Block (Nd) | `op_kernel/arch35/flash_attn_block_vec_nd.h` |
| FlashDecode Block | `op_kernel/arch35/flash_attn_block_vec_flashdecode.h` |
| FAType / 布局映射 | `op_kernel/utils/flash_attn_type.h` |
| 公共定义 | `op_kernel/utils/flash_attn_common_def.h` |
