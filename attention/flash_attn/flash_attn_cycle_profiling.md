# FlashAttention Kernel Cycle 耗时分析方案

## 一、Kernel 整体架构

FlashAttention kernel 采用 AIC（Cube核）+ AIV（Vector核）异构协同的方式执行，主要包含以下阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kernel 执行流程                              │
├─────────────────────────────────────────────────────────────────────┤
│  1. Init         → 初始化 (tiling解析, buffer分配, GM设置)          │
│  2. Process      → 主入口                                          │
│     ├─ 2.1 CrossCoreBufferInit  → 跨核buffer初始化                 │
│     ├─ 2.2 InitBuffers/AllocEventID → AIC/AIV各自buffer+事件初始化 │
│     ├─ 2.3 FlashAttention (per section)  → FA主循环                │
│     │    ├─ CreateTask    → 任务创建 + 参数计算                     │
│     │    └─ ExecuteTask   → 任务执行                                │
│     │         ├─ [AIC] ComputeMm1  → BMM1 (Q*K^T)                  │
│     │         ├─ [AIV] ComputeVec1 → Softmax + Scale + Mask        │
│     │         ├─ [AIC] ComputeMm2  → BMM2 (P*V)                    │
│     │         └─ [AIV] ComputeVec2 → FlashUpdate + CopyOut         │
│     ├─ 2.4 FlashDecode (per section, AIV only)  → FD规约           │
│     └─ 2.5 CrossCoreBufferUnInit → 跨核buffer回收                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 二、模块详细分析

### 2.1 初始化阶段 (`Init`)
**文件**: `flash_attn_kernel_base_gqa_template.h:158-247`

| 子模块 | 说明 |
|--------|------|
| `InitConstInfo` | 从tiling解析所有编译期/运行期常量 |
| `InitMMResBuf` | 初始化L1/UB/GM buffer管理器、分配BMM1/BMM2中间结果buffer |
| `InitQCuSeqLensParser` | 初始化Q序列长度解析器 |
| `InitKvCuSeqLensParser` | 初始化KV序列长度解析器 |
| `vecFaBlock.InitVecBlock` | AIV: 初始化向量计算所需的GM/UB/Queue |
| `cubeBlock.InitCubeBlock` | AIC: 初始化Cube计算所需的GM/L1 |
| `vecFdBlock.InitParams/InitGlobalTensor` | AIV: FlashDecode初始化 |

### 2.2 Buffer/Event初始化 (`Process` 前段)
**文件**: `flash_attn_kernel_base_gqa_template.h:827-837`

| 子模块 | 核 | 说明 |
|--------|-----|------|
| `CrossCoreBufferInit` | AIV | 建立跨核buffer同步 |
| `InitBuffers` | AIC | L0A/L0B/L0C/L1 buffer分配 |
| `AllocEventID` | AIC | 事件ID分配(实际为空实现) |
| `vecFaBlock.InitBuffers` | AIV | UB Queue分配 |
| `vecFaBlock.AllocEventID` | AIV | 事件ID分配 |
| `vecFaBlock.ClearOutput` | AIV | 输出tensor清零 |

### 2.3 FlashAttention 主循环 (`FlashAttention`)
**文件**: `flash_attn_kernel_base_gqa_template.h:385-438`

每个section内，按(bN2, gS1, s2)三层循环遍历任务：

#### 2.3.1 任务创建 (`CreateTask` → `CalcParams`)
- 计算当前任务的actMSize, actSingleLoopS2Size, 坐标等运行参数

#### 2.3.2 AIC: ComputeMm1 (BMM1: Q×K^T)
**文件**: `flash_attn_block_cube_noquant_gqa.h`

根据D/S2配置分发到不同实现：

| 函数 | 条件 | 说明 |
|------|------|------|
| `IterateBmm1NdL1SplitK` | `dBaseSize > 256` | L1切K, Q驻留, 每轮重搬K |
| `IterateBmm1DnSplitK` | `useDn && dBaseSize > 128` | DN模式L0切K |
| `IterateBmm1Dn` | `useDn && dBaseSize ≤ 128` | DN模式单次Matmul |
| `IterateBmm1NdL0Split` | 其他(默认) | L0切D, MatmulBase自动分发 |

**BMM1子步骤** (以`IterateBmm1NdL0Split`为例):
1. **CopyQ** → `CopyQueryTile`: GM→L1搬运Q (仅首次S2循环)
2. **CopyK** → `CopyKeyTile`: GM→L1搬运K
3. **Matmul** → `MatmulBase/MatmulN/MatmulK`: L1→L0A/L0B→Cube计算
4. **FixpipeMm1** → `FixpipeMm1`: L0C→UB搬运BMM1结果

#### 2.3.3 AIV: ComputeVec1 (Softmax处理)
**文件**: `flash_attn_block_vec_noquant_gqa.h:246-616`

| 子步骤 | 函数 | 说明 |
|--------|------|------|
| AttenMask搬入 | `AttenMaskCopyIn` | (可选)GM→UB搬运mask |
| Softmax+Scale | `ProcessVec1Vf` | VF融合指令: scale → mask → softmax → cast |
| P矩阵搬出 | `DataCopy UB→L1` | softmax结果从UB搬到L1 (BMM2左矩阵) |
| 累加更新 | `UpdateExpSumAndExpMax` | 非首次S2循环时更新全局max/sum |
| FD数据搬出 | `SoftmaxDataCopyOut` | (末次S2循环)max/sum搬到GM |

#### 2.3.4 AIC: ComputeMm2 (BMM2: P×V)
**文件**: `flash_attn_block_cube_noquant_gqa.h`

根据DV配置分发：

| 函数 | 条件 | 说明 |
|------|------|------|
| `IterateBmm2L1SplitN` | `dVTemplateType > 256` | L1切N, 多次搬V |
| `IterateBmm2l0Split` | 其他(默认) | 单次搬V, L0自动分发 |

**BMM2子步骤** (以`IterateBmm2l0Split`为例):
1. **CopyV** → `CopyValueTile`: GM→L1搬运V
2. **Matmul** → `MatmulBase/MatmulN/MatmulFull`: L1→L0A/L0B→Cube计算
3. **FixpipeMm2** → `FixpipeMm2PartialN`: L0C→UB/GM搬运BMM2结果

#### 2.3.5 AIV: ComputeVec2 (FlashUpdate + 输出)
**文件**: `flash_attn_block_vec_noquant_gqa.h:969-991`

| 子步骤 | 函数 | 说明 |
|--------|------|------|
| BMM2结果读取 | `ProcessVec2OnUb / ProcessVec2_C2GmUpdateUb` | UB/GM→UB搬运BMM2结果 |
| Flash全局更新 | `FlashUpdateVF / ReduceFinalRes_VF` | 用exp(max_old-max_new)缩放旧累加值+加新值 |
| Cast+CopyOut | `Bmm2ResCastAndCopyOut` | float→bf16/fp16 → DataCopy到GM |
| 无效行处理 | `RowInvalid` | mask场景刷0 |
| FD中间结果 | `Bmm2ResForFDCopyOut` | S2切分时搬到workspace |

### 2.4 FlashDecode (`FlashDecode`)
**文件**: `flash_attention_noquant_block_vec_flashdecode.h:472-567`
**执行条件**: `isFd == true` (仅AIV核)

| 子步骤 | 函数 | 说明 |
|--------|------|------|
| 初始化Buffer | `InitBuffers` | 复用SharedBuffer |
| LSE搬入 | `CopyLseIn` | 各S2分片的max/sum搬入UB |
| Scale计算 | `ComputeScaleValue` | 根据全局max/sum计算缩放系数 |
| 累加结果搬入 | `CopyAccumOutIn` | workspace中的P*V部分结果搬入UB |
| 规约计算 | `ReduceFinalRes` | 加权求和所有S2分片的结果 |
| 最终输出 | `CopyFinalResOut` | cast + DataCopy到GM |
| SoftmaxLse输出 | `DataCopySoftmaxLse*` | (可选)输出log-sum-exp |

### 2.5 收尾 (`Process` 后段)
| 子模块 | 核 | 说明 |
|--------|-----|------|
| `FreeEventID` | AIC/AIV | 释放事件ID |
| `CrossCoreBufferUnInit` | AIC | 跨核buffer回收 |

## 三、插桩点位设计

使用 `set_mark(0xAABBCCDD)` 在各模块首尾插入标记，通过instr_log搜索"COND"获取cycle值。

### 标记编码规则
```
0xAA__XXYY
  AA = 模块大类:
    0x10 = Init初始化
    0x20 = AIC BufferInit
    0x30 = AIV BufferInit  
    0x40 = AIC ComputeMm1 (BMM1)
    0x50 = AIV ComputeVec1 (Softmax)
    0x60 = AIC ComputeMm2 (BMM2)
    0x70 = AIV ComputeVec2 (FlashUpdate)
    0x80 = AIV FlashDecode
    0x90 = AIC/AIV Uninit
  XX = 子步骤编号 (01=CopyQ, 02=CopyK, 03=Matmul, 04=Fixpipe, ...)
  YY = 00=开始, FF=结束
```

### 完整插桩点位表

| 序号 | mark值 | 位置 | 说明 |
|------|--------|------|------|
| 1 | `0x10000000` | `Init` 开始 | 初始化开始 |
| 2 | `0x100000FF` | `Init` 结束 | 初始化结束 |
| 3 | `0x20000000` | AIC `InitBuffers` 开始 | AIC buffer初始化开始 |
| 4 | `0x200000FF` | AIC `InitBuffers` 结束 | AIC buffer初始化结束 |
| 5 | `0x30000000` | AIV `InitBuffers` 开始 | AIV buffer初始化开始 |
| 6 | `0x300000FF` | AIV `InitBuffers` 结束 | AIV buffer初始化结束 |
| 7 | `0x40000000` | `ComputeMm1` 开始 | BMM1整体开始 |
| 8 | `0x40010000` | `CopyQueryTile` 开始 | Q搬运开始 |
| 9 | `0x400100FF` | `CopyQueryTile` 结束 | Q搬运结束 |
| 10 | `0x40020000` | `CopyKeyTile` 开始 | K搬运开始 |
| 11 | `0x400200FF` | `CopyKeyTile` 结束 | K搬运结束 |
| 12 | `0x40030000` | `MatmulBase` 开始 | BMM1 Matmul开始 |
| 13 | `0x400300FF` | `MatmulBase` 结束 | BMM1 Matmul结束 |
| 14 | `0x40040000` | `FixpipeMm1` 开始 | L0C→UB搬运开始 |
| 15 | `0x400400FF` | `FixpipeMm1` 结束 | L0C→UB搬运结束 |
| 16 | `0x400000FF` | `ComputeMm1` 结束 | BMM1整体结束 |
| 17 | `0x50000000` | `ComputeVec1` 开始 | Vec1整体开始 |
| 18 | `0x50010000` | `AttenMaskCopyIn` 开始 | Mask搬入开始 |
| 19 | `0x500100FF` | `AttenMaskCopyIn` 结束 | Mask搬入结束 |
| 20 | `0x50020000` | `ProcessVec1Vf` 开始 | Softmax融合计算开始 |
| 21 | `0x500200FF` | `ProcessVec1Vf` 结束 | Softmax融合计算结束 |
| 22 | `0x50030000` | `DataCopy UB→L1` 开始 | P矩阵搬出开始 |
| 23 | `0x500300FF` | `DataCopy UB→L1` 结束 | P矩阵搬出结束 |
| 24 | `0x500000FF` | `ComputeVec1` 结束 | Vec1整体结束 |
| 25 | `0x60000000` | `ComputeMm2` 开始 | BMM2整体开始 |
| 26 | `0x60010000` | `CopyValueTile` 开始 | V搬运开始 |
| 27 | `0x600100FF` | `CopyValueTile` 结束 | V搬运结束 |
| 28 | `0x60020000` | `MatmulBase` (BMM2) 开始 | BMM2 Matmul开始 |
| 29 | `0x600200FF` | `MatmulBase` (BMM2) 结束 | BMM2 Matmul结束 |
| 30 | `0x60030000` | `FixpipeMm2` 开始 | BMM2 Fixpipe开始 |
| 31 | `0x600300FF` | `FixpipeMm2` 结束 | BMM2 Fixpipe结束 |
| 32 | `0x600000FF` | `ComputeMm2` 结束 | BMM2整体结束 |
| 33 | `0x70000000` | `ComputeVec2` 开始 | Vec2整体开始 |
| 34 | `0x70010000` | `ProcessVec2OnUb/C2Gm` 开始 | FlashUpdate计算开始 |
| 35 | `0x700100FF` | `ProcessVec2OnUb/C2Gm` 结束 | FlashUpdate计算结束 |
| 36 | `0x70020000` | `CopyOut` 开始 | 最终结果搬出开始 |
| 37 | `0x700200FF` | `CopyOut` 结束 | 最终结果搬出结束 |
| 38 | `0x700000FF` | `ComputeVec2` 结束 | Vec2整体结束 |
| 39 | `0x80000000` | `FlashDecode` 开始 | FD规约开始 |
| 40 | `0x80010000` | `CopyLseIn` 开始 | LSE搬入开始 |
| 41 | `0x800100FF` | `CopyLseIn` 结束 | LSE搬入结束 |
| 42 | `0x80020000` | `ReduceFinalRes` 开始 | 规约计算开始 |
| 43 | `0x800200FF` | `ReduceFinalRes` 结束 | 规约计算结束 |
| 44 | `0x80030000` | `CopyFinalResOut` 开始 | FD结果搬出开始 |
| 45 | `0x800300FF` | `CopyFinalResOut` 结束 | FD结果搬出结束 |
| 46 | `0x800000FF` | `FlashDecode` 结束 | FD规约结束 |
| 47 | `0x90000000` | `Uninit` 开始 | 收尾开始 |
| 48 | `0x900000FF` | `Uninit` 结束 | 收尾结束 |

## 四、预期耗时分布图

### 4.1 单次任务 (单个S2循环) Pipeline 时序

```
时间轴 ──────────────────────────────────────────────────────────────────────►

AIC (Cube核):
  ┌──────────┐     ┌─────┐┌──────────┐┌───────┐            ┌──────────┐┌───────────┐┌───────┐
  │ CopyQ(L1)│     │CopyK││ Matmul1  ││Fixpipe│   wait     │CopyV(L1) ││  Matmul2  ││Fixpipe│
  │  GM→L1   │     │GM→L1││ L0A/B→C  ││L0C→UB │  Vec1完成  │  GM→L1   ││ L0A/B→C   ││L0C→out│
  └──────────┘     └─────┘└──────────┘└───────┘            └──────────┘└───────────┘└───────┘
  ├─0x4001──┤     ├0x4002┤├──0x4003──┤├0x4004─┤            ├──0x6001──┤├───0x6002──┤├0x6003─┤
  ├──────────── BMM1 (0x40) ──────────────────┤            ├──────── BMM2 (0x60) ──────────┤

AIV (Vector核):
                                     wait                  ┌─────────────────────────┐
                                    BMM1完成               │     ProcessVec1         │
                                              ┌────┐┌─────┐│ Softmax+Scale+Mask      │
                                              │Mask││ VF  ││ DataCopy UB→L1          │
                                              │Copy││Comp ││                         │
                                              └────┘└─────┘└─────────────────────────┘
                                              ├5001┤├5002─┤├──────── 0x5003 ────────┤
                                              ├──────── Vec1 (0x50) ──────────────────┤

                                                                              wait
                                                                             BMM2完成
                                                                                      ┌──────────────────┐
                                                                                      │   ProcessVec2    │
                                                                                      │ FlashUpdate      │
                                                                                      │ Cast+CopyOut GM  │
                                                                                      └──────────────────┘
                                                                                      ├──Vec2 (0x70) ───┤
```

### 4.2 多S2循环 Pipeline 重叠 (PRELOAD_N=2)

```
时间轴 ──────────────────────────────────────────────────────────────────────────────────────►

AIC:  ┌─BMM1[0]──┐┌─BMM1[1]──┐┌─BMM2[0]──┐┌─BMM1[2]──┐┌─BMM2[1]──┐┌─BMM2[2]──┐
      │Q+K+Mat+FP││K+Mat+FP  ││V+Mat+FP  ││K+Mat+FP  ││V+Mat+FP  ││V+Mat+FP  │
      └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘
      ├──0x40────┤├──0x40────┤├──0x60────┤├──0x40────┤├──0x60────┤├──0x60────┤

AIV:             ┌──Vec1[0]──┐┌──Vec1[1]──┐┌Vec2[0]──┐┌──Vec1[2]──┐┌Vec2[1]──┐┌Vec2[2]──┐
                 │ Softmax   ││ Softmax   ││FlshUpd  ││ Softmax   ││FlshUpd  ││FlshUpd  │
                 └───────────┘└───────────┘└─────────┘└───────────┘└─────────┘└─────────┘
                 ├──0x50─────┤├──0x50─────┤├──0x70──┤├──0x50─────┤├──0x70──┤├──0x70───┤

Pipeline说明:
  - PRELOAD_N=2: 前2个task只执行Mm1+Vec1, 从第3个task开始同时执行 Mm1[i] + Mm2[i-2]
  - AIC交替执行 BMM1 和 BMM2
  - AIV交替执行 Vec1 和 Vec2
  - AIC-AIV通过 CrossCoreSync 同步BMM1/BMM2结果
```

### 4.3 完整Kernel各阶段占比示意 (典型Decode场景)

```
  阶段                           预估占比        标记范围
 ┌─────────────────────────────────────────────────────────────┐
 │ Init (Tiling解析+Buffer分配+GM设置)            ~5%         │ 0x10
 ├─────────────────────────────────────────────────────────────┤
 │ AIC InitBuffers + AIV InitBuffers              ~3%         │ 0x20/0x30
 ├─────────────────────────────────────────────────────────────┤
 │ ┌──── Section Loop ──────────────────────────────────┐     │
 │ │ FlashAttention 主循环 (多task pipeline)     ~80%    │     │
 │ │  ┌──── Task Loop ──────────────────────────┐       │     │
 │ │  │ AIC BMM1 (CopyQ+CopyK+Matmul+Fixpipe)  │ ~25%  │     │ 0x40
 │ │  │ AIV Vec1 (Mask+Softmax+CopyToL1)        │ ~20%  │     │ 0x50
 │ │  │ AIC BMM2 (CopyV+Matmul+Fixpipe)         │ ~20%  │     │ 0x60
 │ │  │ AIV Vec2 (FlashUpdate+Cast+CopyOut)      │ ~15%  │     │ 0x70
 │ │  └─────────────────────────────────────────┘       │     │
 │ └────────────────────────────────────────────────────┘     │
 ├─────────────────────────────────────────────────────────────┤
 │ FlashDecode (AIV规约, 仅S2切分时)              ~10%        │ 0x80
 ├─────────────────────────────────────────────────────────────┤
 │ Uninit (事件释放+跨核回收)                      ~2%        │ 0x90
 └─────────────────────────────────────────────────────────────┘
```

### 4.4 BMM1 / BMM2 内部子步骤细分

```
BMM1 内部 (IterateBmm1NdL0Split):
 ┌───────────┬──────────┬────────────────────┬────────────┐
 │ CopyQ     │ CopyK    │ MatmulBase         │ FixpipeMm1 │
 │ GM→L1     │ GM→L1    │ L1→L0A/B→L0C      │ L0C→UB     │
 │ ~15%      │ ~15%     │ ~50%               │ ~20%       │
 └───────────┴──────────┴────────────────────┴────────────┘
  0x40-0100   0x40-0200   0x40-0300            0x40-0400

BMM2 内部 (IterateBmm2l0Split):
 ┌──────────┬───────────────────┬────────────────────┐
 │ CopyV    │ MatmulBase        │ FixpipeMm2PartialN │
 │ GM→L1    │ L1→L0A/B→L0C     │ L0C→UB/GM          │
 │ ~20%     │ ~55%              │ ~25%               │
 └──────────┴───────────────────┴────────────────────┘
  0x60-0100   0x60-0200          0x60-0300
```

## 五、如何分析耗时数据

1. 跑case得到 `instr_log` 文件
2. 搜索关键字 `COND`
3. 对每个mark值，记录对应的cycle数
4. 同一模块的 `结束cycle - 开始cycle = 该模块耗时`
5. 注意区分AIC和AIV的日志 (不同core的instr_log文件)

### 分析示例
```
搜索到: COND = 0x40000000  cycle=1000   → BMM1开始
搜索到: COND = 0x40010000  cycle=1000   → CopyQ开始
搜索到: COND = 0x400100FF  cycle=1200   → CopyQ结束  → CopyQ耗时=200
搜索到: COND = 0x40020000  cycle=1200   → CopyK开始
搜索到: COND = 0x400200FF  cycle=1400   → CopyK结束  → CopyK耗时=200
搜索到: COND = 0x40030000  cycle=1400   → Matmul开始
搜索到: COND = 0x400300FF  cycle=2000   → Matmul结束  → Matmul耗时=600
搜索到: COND = 0x40040000  cycle=2000   → Fixpipe开始
搜索到: COND = 0x400400FF  cycle=2400   → Fixpipe结束  → Fixpipe耗时=400
搜索到: COND = 0x400000FF  cycle=2400   → BMM1结束   → BMM1总耗时=1400
```
