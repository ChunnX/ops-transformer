# MoeDistributeCombineA2::Process() 核心逻辑深度解析

> 源文件: `moe_distribute_combine_a2.h`
> 函数: `MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()`
> 运行核心: **AIV** (向量计算核心)
> 特点: **支持多种专家类型 (MoE/Zero/Copy/Const)**

---

## 一、全局背景：与 Layered 版本的定位对比

### 1.1 两个版本的功能对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MoE Combine A2 两个实现版本对比                            │
│                                                                             │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐         │
│  │   moe_distribute_combine_a2 │   │ moe_distribute_combine_a2   │         │
│  │        (基础版)              │   │     _layered (分层版)        │         │
│  ├─────────────────────────────┤   ├─────────────────────────────┤         │
│  │ ✅ 多种专家类型支持           │   │ ❌ 仅MoE专家                 │         │
│  │ ✅ Token/Expert Mask         │   │ ❌ 无Mask支持                │         │
│  │ ❌ 不支持量化                 │   │ ✅ INT8动态量化              │         │
│  │ ❌ 单层通信模式               │   │ ✅ 分层通信(IPC+RDMA)        │         │
│  │ ✅ 单服务器优化               │   │ ✅ 跨服务器优化              │         │
│  └─────────────────────────────┘   └─────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 专家类型支持

```
专家ID范围及处理方式:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  expId ∈ [0, moeExpertNum_)                    → MoE专家 (从Window读取)    │
│  expId ∈ [moeExpertNum_, moeExpertNum_+zero)    → Zero专家 (跳过)          │
│  expId ∈ [zero_start, zero_start+copy)          → Copy专家 (从oriX复制)    │
│  expId ∈ [copy_start, copy_start+const)         → Const专家 (alpha*x+v)    │
│                                                                             │
│  专家ID分布:                                                                 │
│  ├────────────────┼──────────┼──────────┼──────────┤                       │
│  │   MoE专家      │ Zero专家 │ Copy专家  │ Const专家 │                       │
│  │   (路由计算)    │  (空操作) │ (恒等复制)│ (线性变换) │                       │
│  └────────────────┴──────────┴──────────┴──────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、关键参数约定

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `axisBS_` | 本地Batch Size（Token数） | 256 |
| `axisH_` | Hidden Size（隐层维度） | 4096 |
| `axisK_` | TopK（每个Token选K个Expert） | 8 |
| `worldSize_` | EP总卡数 | 64 |
| `aivNum_` | AIV核心数量 | 40 |
| `moeExpertNum_` | MoE专家数量 | 512 |
| `localMoeExpertNum_` | 每卡MoE专家数 | 8 |
| `zeroExpertNum_` | Zero专家数量 | 0 |
| `copyExpertNum_` | Copy专家数量 | 0 |
| `constExpertNum_` | Const专家数量 | 0 |
| `isSingleServer_` | 是否单服务器 | worldSize <= 8 |
| `isInputTokenMaskFlag_` | 是否有Token Mask | false |
| `isInputExpertMaskFlag_` | 是否有Expert Mask | false |

---

## 三、Process() 整体流程

### 3.1 流水线概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Process() 五阶段流水线                              │
│                                                                             │
│   阶段1: CalXActiveMask                                                     │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 处理Token/Expert Mask，计算有效Token数量                      │    │
│   │ 输入: xActiveMask (Token Mask), expertMask (Expert Mask)          │    │
│   │ 输出: 更新后的axisBS_ (有效Token数)                                │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段2: AlltoAllDispatch                                                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 将本地Token数据分发到各Rank的WindowOut                        │    │
│   │ 通信: 单服务器→直接复制; 多服务器→HCCL BatchWrite                   │    │
│   │ 输入: expandXGlobal_, sendCountGlobal_                            │    │
│   │ 输出: WindowOut (各Rank的数据区)                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段3: Preload                                                            │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 预加载专家ID、Scale、IndexCounts，计算接收计数和偏移          │    │
│   │ 输入: expertIdsGlobal_, expandScalesGlobal_, expandIdxGlobal_     │    │
│   │ 输出: recvCountLocal_, expertWindowOffsetLocal_                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段4: WaitDispatch                                                       │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 等待各Rank的数据到达（轮询Flag）                              │    │
│   │ 同步: 轮询检查每个Rank的完成Flag                                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段5: LocalWindowCopy                                                    │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 从WindowIn读取数据，按Expert类型处理，加权聚合输出            │    │
│   │ 输入: localInWindow_, oriXGlobal_, constExpertVGlobal_            │    │
│   │ 输出: expandOutGlobal_ (最终聚合结果)                              │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心代码结构

```cpp
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        CalXActiveMask();      // 阶段1: 处理Mask
        AlltoAllDispatch();    // 阶段2: 数据分发
        Preload();             // 阶段3: 预加载
        WaitDispatch();        // 阶段4: 等待数据
        LocalWindowCopy();     // 阶段5: 聚合输出
        CopyPerformanceInfo(); // 性能信息回写
        hccl_.Finalize();      // HCCL清理
    }
}
```

---

## 四、逐阶段详细分析

### 阶段 1: CalXActiveMask() - Mask处理

#### 功能描述
处理Token Mask和Expert Mask，计算有效Token数量。

#### 详细流程

```
CalXActiveMask() 执行流程:
│
├── 1. Token Mask处理 (isInputTokenMaskFlag_ == true)
│   │
│   │  功能: 根据Token Mask计算有效Token数量
│   │
│   └── TokenActiveMaskCal():
│       ├── DataCopyPad: 读取xActiveMask到UB
│       ├── Cast: bool → int8 → fp16
│       ├── Sum: 求和得到有效Token数
│       └── 更新: axisBS_ = sum结果, bsKAlign_ = 新对齐值
│
└── 2. Expert Mask处理 (isInputExpertMaskFlag_ == true)
    │
    │  功能: 读取二维Expert Mask表 (axisBS × axisK)
    │
    └── DataCopyPad: 读取expertMask到expertMaskTensor_
```

#### Token Mask计算流程

```
TokenActiveMaskCal() 详细流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  xActiveMaskGlobal_ [axisBS个bool]                                         │
│         │                                                                   │
│         ▼ DataCopyPad                                                       │
│  xActiveMaskTensor [bool]                                                   │
│         │                                                                   │
│         ▼ ReinterpretCast                                                   │
│  xActiveMaskInt8Tensor [int8]                                               │
│         │                                                                   │
│         ▼ Cast (CAST_NONE)                                                  │
│  xActiveMaskHalfTensor [fp16]                                               │
│         │                                                                   │
│         ▼ Sum (沿axis=1求和)                                                │
│  sumOutTensor [1个fp16]                                                      │
│         │                                                                   │
│         ▼ GetValue + cast                                                   │
│  axisBS_ = 新的有效Token数                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段 2: AlltoAllDispatch() - 数据分发

#### 功能描述
将本地Token数据按专家分组，分发到各Rank的WindowOut区域。

#### 详细流程

```
AlltoAllDispatch() 执行流程:
│
├── 1. 加载sendCount到UB
│   └── DataCopy: sendCountLocal ← sendCountGlobal_
│
├── 2. 分支判断: 单服务器 vs 多服务器
│   │
│   ├── if (isSingleServer_):
│   │       │
│   │       └── SingleServerDispatch():
│   │           ├── for dstRankId in [startRankId, endRankId):
│   │           │       ├── 数据写入WindowOut
│   │           │       ├── 直接复制到目标WindowIn
│   │           │       └── 设置完成Flag
│   │           └── 更新bufferId
│   │
│   └── else (多服务器):
│           │
│           ├── ConstructBatchWriteInfo():
│           │       └── 构造BatchWriteItem结构体
│           │
│           ├── SyncAll()
│           │
│           └── MultiServerDispatch():
│                   ├── core 0: 调用hccl_.BatchWrite()
│                   ├── 本地数据: WindowOut → WindowIn
│                   └── 设置完成Flag
│
└── 3. SyncAll() - 确保所有核完成
```

#### 单服务器分发流程

```
SingleServerDispatch() 详细流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  for dstRankId = startRankId to endRankId:                                 │
│      │                                                                      │
│      ├── 1. 设置localOutWindow_地址                                        │
│      │       windowOutGM_ + dstRankId × rankSizeOnWin_                     │
│      │                                                                      │
│      ├── 2. GetRankTokenNumAndDataCopy2WindowOut()                         │
│      │       ├── 遍历本地专家 (0 ~ localMoeExpertNum_-1)                    │
│      │       ├── 计算每个专家的Token数量                                    │
│      │       └── DataCopy: expandXGlobal_ → localOutWindow_                │
│      │                                                                      │
│      ├── 3. 复制到目标WindowIn                                              │
│      │       dstGlobal = GetWindowsInAddr(dstRankId) + offset              │
│      │       DataCopy: localOutWindow_ → dstGlobal                         │
│      │                                                                      │
│      └── 4. 设置完成Flag                                                    │
│              flagGlobal_(0) = FLAG_VALUE (0xFFFFFFFF)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 多服务器分发流程

```
MultiServerDispatch() 详细流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ConstructBatchWriteInfo() - 构造批量写入信息:                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ BatchWriteItem结构 (32字节):                                         │   │
│  │   Offset 0-7:   srcAddr (本地WindowOut地址)                          │   │
│  │   Offset 8-15:  dstAddr (远端WindowIn地址)                           │   │
│  │   Offset 16-23: length (数据长度)                                    │   │
│  │   Offset 24-27: dataType (FP16/BF16)                                │   │
│  │   Offset 28-31: dstRankId                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Core 0 执行:                                                               │
│      hccl_.BatchWrite(workspaceGlobal_, worldSize_)                        │
│                                                                             │
│  各核并行执行本地复制:                                                        │
│      if (rankId_ in [startRankId, endRankId)):                             │
│          DataCopy: localOutWindow_ → localInWindow_                        │
│          设置完成Flag                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### BatchWriteItem 结构

```
BatchWriteItem 内存布局 (32字节):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset │ Size │ Field      │ Description                                  │
├────────┼──────┼────────────┼──────────────────────────────────────────────┤
│   0    │  8   │ srcAddr    │ 本地WindowOut物理地址                         │
│   8    │  8   │ dstAddr    │ 远端WindowIn物理地址                          │
│  16    │  8   │ length     │ 数据长度 (单位: 元素数)                        │
│  24    │  4   │ dataType   │ 数据类型 (HCCL_DATA_TYPE_FP16/BF16)          │
│  28    │  4   │ dstRankId  │ 目标Rank ID                                  │
└────────┴──────┴────────────┴──────────────────────────────────────────────┘
```

---

### 阶段 3: Preload() - 预加载与计数

#### 功能描述
预加载专家ID、Scale、IndexCounts，并计算每个专家的接收Token数量和偏移。

#### 详细流程

```
Preload() 执行流程:
│
├── 1. 初始化缓冲区
│   ├── recvCountLocal_: 接收计数 (复用rowTmpFloatBuf_)
│   └── expertWindowOffsetLocal_: 专家偏移 (复用batchWriteItemBuf_)
│
├── 2. 加载专家ID片段
│   ├── taskInfo_.SplitCore(axisBS × axisK, aivNum, coreIdx)
│   └── DataCopyPad: expertIdsSegLocal_ ← expertIdsGlobal_[片段]
│
├── 3. 计算接收计数
│   │
│   ├── 初始化: Duplicate(recvCountLocal_, 0, moeExpertNum_)
│   │
│   ├── if (有ExpertMask):
│   │       for 每个有效Expert:
│   │           if expertMask[i] == true:
│   │               recvCountLocal_[expId] += 1
│   │
│   └── else (无Mask):
│           for 每个Expert:
│               recvCountLocal_[expId] += 1
│
├── 4. 原子累加到全局
│   ├── SetAtomicAdd<int32_t>()
│   ├── DataCopyPad: 全局recvCount += 本地recvCount
│   └── SetAtomicNone()
│
├── 5. SyncAll()
│
├── 6. 读取全局计数
│   └── DataCopyPad: recvCountLocal_ ← expertRecvCountGlobal_
│
├── 7. 计算专家偏移 (前缀和)
│   │
│   │  for 每个Rank组:
│   │      prefixSum = 0
│   │      for 每个本地专家:
│   │          expertWindowOffsetLocal_[i] = prefixSum
│   │          prefixSum += recvCountLocal_[i]
│   │
│   └── DataCopyPad: 写入expertWindowOffsetGlobal_
│
├── 8. SyncAll()
│
├── 9. 读取全局偏移
│   └── DataCopyPad: expertWindowOffsetLocal_ ← expertWindowOffsetGlobal_
│
└── 10. 加载Scale和IndexCounts片段
    ├── 计算tokenBeginIndex_, tokenNumPerCore_
    ├── DataCopyPad: expandScalesSegLocal_ ← expandScalesGlobal_[片段]
    ├── DataCopyPad: indexCountsSegLocal_ ← expandIdxGlobal_[片段]
    └── DataCopyPad: expertIdsSegLocal_ ← expertIdsGlobal_[片段]
```

#### 接收计数计算示意

```
专家接收计数计算:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  Token 0: [expId=3, expId=7, ...]                                         │
│  Token 1: [expId=1, expId=3, ...]                                         │
│  ...                                                                       │
│  Token N: [expId=3, expId=5, ...]                                         │
│                                                                            │
│  recvCount 统计:                                                           │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                        │
│  │exp=0│exp=1│exp=2│exp=3│exp=4│exp=5│exp=6│exp=7│                        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤                        │
│  │  0  │  1  │  0  │  3  │  0  │  1  │  0  │  1  │                        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                        │
│                                                                            │
│  expertWindowOffset (前缀和):                                               │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                        │
│  │exp=0│exp=1│exp=2│exp=3│exp=4│exp=5│exp=6│exp=7│                        │
│  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤                        │
│  │  0  │  0  │  1  │  1  │  4  │  4  │  5  │  5  │                        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                        │
│        ↑              ↑              ↑                                     │
│       offset=0      offset=1      offset=4                                 │
│       (exp=1开始)   (exp=3开始)   (exp=5开始)                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段 4: WaitDispatch() - 等待数据到达

#### 功能描述
轮询等待各Rank的数据到达完成。

#### 详细流程

```
WaitDispatch() 执行流程:
│
├── 1. 检查是否有等待任务
│   └── if (startRankId_ >= worldSize_): SyncAll() 并返回
│
├── 2. 主循环 (轮询)
│   │
│   └── while (waitFlagNum < sendRankNum_):
│       │
│       └── for rankId in [startRankId_, endRankId_):
│           │
│           ├── 计算Flag地址
│           │   wAddr = windowInGM_ + rankSizeOnWin_ × rankId
│           │         + SKIP_OFFSET
│           │         + (recvCount + offset) × axisHExpandXTypeSize_
│           │
│           ├── 刷新Cache
│           │   DataCacheCleanAndInvalid(flagGlobal_)
│           │
│           ├── 读取Flag
│           │   flag = flagGlobal_(0)
│           │
│           ├── if (flag == FLAG_VALUE):
│           │       ├── waitFlagNum++
│           │       ├── flagGlobal_(0) = 0  // 清除Flag
│           │       ├── DataCacheCleanAndInvalid()  // 确保清零生效
│           │       └── 记录性能信息 (可选)
│           │
│           └── continue (继续轮询)
│
└── 3. SyncAll() - 确保所有核完成
```

#### Flag地址计算

```
Flag位置计算公式:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  flagAddr = windowInGM_                         // WindowIn基地址          │
│           + rankSizeOnWin_ × rankId             // 目标Rank的数据区偏移     │
│           + SKIP_OFFSET                         // 跳过32字节的头          │
│           + (recvCount[tokenIdx]                // 该专家的Token数量        │
│              + expertWindowOffset[tokenIdx])    // 该专家的起始偏移         │
│           × axisHExpandXTypeSize_               // 每个Token的字节数        │
│                                                                            │
│  示例: rankId=2, tokenIdx=7, recvCount[7]=5, offset[7]=20, H=4096         │
│  flagAddr = base + rankSize×2 + 32 + (5+20)×8192                          │
│           = base + rankSize×2 + 32 + 204800                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段 5: LocalWindowCopy() - 聚合输出

#### 功能描述
从WindowIn读取数据，按Expert类型处理，加权聚合后输出到`expandOutGlobal_`。

#### 详细流程

```
LocalWindowCopy() 执行流程:
│
├── 1. 检查是否有任务
│   └── if (tokenNumPerCore_ == 0): return
│
├── 2. 初始化
│   └── sumFloatLocal_ = sumFloatBuf_.Get<float>()
│
└── 3. 主循环 (遍历每个Token)
    │
    └── for i = 0 to tokenNumPerCore_-1:
        │
        ├── tokenIdx = tokenBeginIndex_ + i
        │
        ├── 初始化累加器
        │   Duplicate(sumFloatLocal_, 0.0f, axisH_)
        │
        ├── 遍历TopK专家
        │   │
        │   └── for topKIdx = 0 to axisK_-1:
        │       │
        │       ├── if (有ExpertMask && mask==false): continue
        │       │
        │       ├── 获取expId
        │       │   expId = expertIdsSegLocal_[tokentopKIdx - baseOffset]
        │       │
        │       ├── 根据专家类型分发:
        │       │   │
        │       │   ├── if (expId < moeExpertNum_):
        │       │   │       ProcessMoeAndCopyExpert()  // MoE专家
        │       │   │
        │       │   ├── elif (expId < moe + zero):
        │       │   │       continue  // Zero专家: 跳过
        │       │   │
        │       │   ├── elif (expId < moe + zero + copy):
        │       │   │       ProcessMoeAndCopyExpert()  // Copy专家
        │       │   │
        │       │   └── else:
        │       │           ProcessConstantExpert()    // Const专家
        │       │
        │
        ├── 转换输出格式
        │   Cast(sumBufLocal_, sumFloatLocal_, CAST_RINT, axisH_)
        │
        └── 写入结果
            DataCopy(expandOutGlobal_[tokenIdx × H], sumBufLocal_, axisH_)
```

#### 专家类型处理流程图

```
专家类型分发与处理:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                        expId 判断                                           │
│                           │                                                 │
│         ┌─────────────────┼─────────────────┐                              │
│         ▼                 ▼                 ▼                              │
│   expId < moeNum    moe <= expId < moe+zero   expId >= moe+zero            │
│         │                 │                     │                          │
│         ▼                 ▼                     ▼                          │
│   ┌───────────┐     ┌───────────┐        ┌─────────────────┐              │
│   │ MoE专家   │     │ Zero专家  │        │  Copy/Const专家  │              │
│   │           │     │           │        │                   │              │
│   │ 从Window  │     │ 跳过      │        │  ┌─────────────┐  │              │
│   │ 读取数据  │     │ (无操作)  │        │  │ < moe+zero+ │  │              │
│   │           │     │           │        │  │   copy?     │  │              │
│   │ scale×data│     │           │        │  └──────┬──────┘  │              │
│   │ + sum     │     │           │        │         │         │              │
│   └───────────┘     └───────────┘        │    ┌────┴────┐   │              │
│                                         │    ▼         ▼   │              │
│                                         │ Copy专家  Const专家│              │
│                                         │ 从oriX   alpha1×x+│              │
│                                         │ 复制     alpha2×v │              │
│                                         └─────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ProcessMoeAndCopyExpert() - MoE/Copy专家处理

```
ProcessMoeAndCopyExpert() 详细流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  输入: tokenIdx, topKIdx                                                    │
│                                                                             │
│  1. 计算参数:                                                               │
│     tokentopKIdx = tokenIdx × axisK + topKIdx                              │
│     scaleVal = expandScalesSegLocal_[tokentopKIdx - baseOffset]            │
│     expId = expertIdsSegLocal_[tokentopKIdx - baseOffset]                  │
│                                                                             │
│  2. 计算数据地址:                                                           │
│     if (expId < moeExpertNum_):  // MoE专家                                │
│         rank = expId / localMoeExpertNum_                                  │
│         wAddr = windowInGM_ + rankSizeOnWin_ × rank                        │
│               + expertWindowOffset[expId] × H                              │
│               + indexCounts[tokentopKIdx] × H                              │
│     else:  // Copy专家                                                      │
│         wAddr = oriXGM_ + tokenIdx × H                                     │
│                                                                             │
│  3. 数据搬运与计算:                                                         │
│     DataCopy: tmpUb_ ← rankWindow_ [H个元素]                               │
│     Cast: rowTmpFloatLocal ← tmpUb_ (fp16/bf16 → fp32)                     │
│     Muls: rowTmpFloatLocal *= scaleVal                                     │
│     Add: sumFloatLocal_ += rowTmpFloatLocal                                │
│                                                                             │
│  输出: 累加到sumFloatLocal_                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ProcessConstantExpert() - Const专家处理

```
ProcessConstantExpert() 详细流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Const专家计算公式:                                                         │
│     output = scale × (alpha1 × x + alpha2 × v)                             │
│                                                                             │
│  其中:                                                                      │
│     x = 原始输入Token (oriXGlobal_[tokenIdx])                              │
│     v = 常量向量 (constExpertVGlobal_[constExpertIdx])                     │
│     alpha1, alpha2 = 线性变换系数                                           │
│     scale = 专家权重                                                        │
│                                                                             │
│  详细步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 读取系数: alpha1, alpha2 (从constExpertAlpha1/2Global_)          │   │
│  │                                                                    │   │
│  │ 2. 读取数据:                                                       │   │
│  │    constVInUB ← constExpertVGlobal_[constExpertIdx × H]            │   │
│  │    constXInUB ← oriXGlobal_[tokenIdx × H]                          │   │
│  │                                                                    │   │
│  │ 3. 类型转换 (fp16/bf16 → fp32):                                    │   │
│  │    Cast: constXFloatLocal ← constXInUB                             │   │
│  │    Cast: constVFloatLocal ← constVInUB                             │   │
│  │                                                                    │   │
│  │ 4. 线性变换:                                                       │   │
│  │    Muls: constXFloatLocal *= alpha1                                │   │
│  │    Muls: constVFloatLocal *= alpha2                                │   │
│  │    Add:  constXFloatLocal += constVFloatLocal                      │   │
│  │                                                                    │   │
│  │ 5. 加权累加:                                                       │   │
│  │    Muls: constXFloatLocal *= scaleVal                              │   │
│  │    Add:  sumFloatLocal_ += constXFloatLocal                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、数据结构

### 5.1 核心类成员变量

| 变量名 | 类型 | 用途 |
|--------|------|------|
| `expandXGlobal_` | `GlobalTensor<ExpandXType>` | 输入Token数据 |
| `expertIdsGlobal_` | `GlobalTensor<ExpandIdxType>` | 专家ID表 |
| `expandIdxGlobal_` | `GlobalTensor<ExpandIdxType>` | Token索引表 |
| `sendCountGlobal_` | `GlobalTensor<ExpandIdxType>` | 发送计数 |
| `expandScalesGlobal_` | `GlobalTensor<float>` | 专家权重 |
| `expandOutGlobal_` | `GlobalTensor<ExpandXType>` | 输出聚合结果 |
| `localOutWindow_` | `GlobalTensor<ExpandXType>` | 本地发送窗口 |
| `localInWindow_` | `GlobalTensor<ExpandXType>` | 本地接收窗口 |
| `oriXGlobal_` | `GlobalTensor<ExpandXType>` | 原始输入(Copy/Const专家用) |
| `constExpertVGlobal_` | `GlobalTensor<ExpandXType>` | Const专家向量 |
| `xActiveMaskGlobal_` | `GlobalTensor<bool>` | Token Mask |
| `expertRecvCountGlobal_` | `GlobalTensor<uint32_t>` | 专家接收计数 |
| `expertWindowOffsetGlobal_` | `GlobalTensor<uint32_t>` | 专家窗口偏移 |

### 5.2 UB缓冲区布局

```
UB缓冲区分配 (BuffInit):
┌────────────────────────────────────────────────────────────────────────────┐
│ 缓冲区名称              │ 大小计算                              │ 用途      │
├─────────────────────────┼───────────────────────────────────────┼──────────┤
│ moeQueue_               │ 2 × axisH × sizeof(ExpandXType)       │ Token队列 │
│ expertIdsBuf_           │ max(expertIdsNumPerCore×4, 256)       │ 专家ID   │
│ expandScalesBuf_        │ expertIdsNumPerCore × 4               │ 权重值   │
│ tokenBuf_               │ max(axisH×4, moeExpertNum×4)          │ 临时缓冲 │
│ rowTmpFloatBuf_         │ max(axisH×4, moeExpertNum×4)          │ 行计算   │
│ sumFloatBuf_            │ max(axisH×4, moeExpertNum×4)          │ 累加器   │
│ indexCountsBuf_         │ max(expertIdsNumPerCore×4, 256)       │ 索引计数 │
│ batchWriteItemBuf_      │ max(32×worldSize, moeExpertNum×4)     │ 批写信息 │
│ expertMaskBuf_          │ RoundUp(axisBS×axisK×1, 32)           │ 专家Mask │
└────────────────────────────────────────────────────────────────────────────┘

expertIdsNumPerCore = Ceil(axisBS, aivNum) × axisK
```

### 5.3 TaskInfo 结构

```cpp
struct TaskInfo {
    uint32_t startTaskId;   // 起始任务ID
    uint32_t endTaskId;     // 结束任务ID
    uint32_t taskNum;       // 任务数量

    // 任务分片函数
    void SplitCore(taskNumTotal, aivNum, aivId) {
        // 将taskNumTotal个任务均匀分配到aivNum个核
        // 每个核处理 taskNum = taskNumTotal / aivNum (+1如有余数)
    }
};
```

---

## 六、内存排布

### 6.1 HCCL Window 布局

```
Window 布局 (Ping/Pong双缓冲):
┌────────────────────────────────────────────────────────────────────────────┐
│ WindowIn/WindowOut (halfWinSize = winSize / 2)                            │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ Offset 0: 数据区                                                        ││
│ │ ┌────────────────────────────────────────────────────────────────────┐ ││
│ │ │ Rank 0 数据区        │ Rank 1 数据区 │ ... │ Rank N 数据区        │ ││
│ │ │ rankSizeOnWin_       │               │     │                       │ ││
│ │ └────────────────────────────────────────────────────────────────────┘ ││
│ ├────────────────────────────────────────────────────────────────────────┤│
│ │ Offset dataSpaceSize_: 状态区                                           ││
│ │ ┌────────────────────────────────────────────────────────────────────┐ ││
│ │ │ Rank 0 状态 (512B) │ Rank 1 状态 │ ... │ Rank N 状态 │ BufferId    │ ││
│ │ │ STATE_OFFSET       │            │     │             │ (32B)       │ ││
│ │ └────────────────────────────────────────────────────────────────────┘ ││
│ ├────────────────────────────────────────────────────────────────────────┤│
│ │ Offset dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET: BatchWriteItem区      ││
│ │ ┌────────────────────────────────────────────────────────────────────┐ ││
│ │ │ Item[0] (32B) │ Item[1] (32B) │ ... │ Item[worldSize-1] (32B)      │ ││
│ │ └────────────────────────────────────────────────────────────────────┘ ││
│ └────────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│ dataSpaceSize_ = halfWinSize - STATE_SPACE_SIZE (1MB)                     │
│ rankSizeOnWin_ = dataSpaceSize_ / worldSize (32B对齐)                     │
│ stateOffsetOnWin_ = dataSpaceSize_ + rankId × STATE_OFFSET (512B)         │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 数据区布局

```
单个Rank的数据区布局:
┌────────────────────────────────────────────────────────────────────────────┐
│ Rank数据区 (rankSizeOnWin_ 字节)                                           │
│                                                                            │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ Expert 0 Token数据    │ Expert 1 Token数据 │ ... │ Flag (32B)          ││
│ │ [recvCount[0] × H]    │ [recvCount[1] × H]  │     │ FLAG_VALUE          ││
│ │ + SKIP_OFFSET (32B)   │                     │     │                     ││
│ └────────────────────────────────────────────────────────────────────────┘│
│                                                                            │
│ Token数据按expertWindowOffset顺序排列                                       │
│ Flag位于该Rank数据区的末尾                                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、时序流水同步信息

### 7.1 整体时序图

```
时间轴 ════════════════════════════════════════════════════════════════════════►

所有核并行执行:
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ CalXActiveMask  │ AlltoAllDispatch│    Preload      │  WaitDispatch   │Local│
│   (Mask处理)    │   (数据分发)     │   (预加载计数)   │   (等待数据)    │Copy │
├─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────┤
│                              SyncAll() 同步点                               │
│  1. Mask处理后                                                              │
│  2. AlltoAllDispatch后                                                      │
│  3. recvCount原子累加后                                                     │
│  4. expertWindowOffset计算后                                                │
│  5. WaitDispatch后                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 同步点总结

| 同步点 | 类型 | 目的 |
|--------|------|------|
| CalXActiveMask后 | 隐式同步 | 确保Mask处理完成 |
| AlltoAllDispatch后 | SyncAll | 确保数据分发完成 |
| recvCount原子累加后 | SyncAll | 确保计数可见 |
| expertWindowOffset后 | SyncAll | 确保偏移计算完成 |
| WaitDispatch后 | SyncAll | 确保所有数据到达 |

### 7.3 Flag同步机制

```
Flag同步流程:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  发送方:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 写入Token数据到WindowOut                                          │  │
│  │ 2. 复制到目标WindowIn (或通过HCCL BatchWrite)                        │  │
│  │ 3. 写入Flag = FLAG_VALUE (0xFFFFFFFF)                               │  │
│  │ 4. DataCacheCleanAndInvalid() 确保Flag生效                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  接收方:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ while (waitFlagNum < sendRankNum):                                  │  │
│  │     for each rank:                                                  │  │
│  │         DataCacheCleanAndInvalid() 刷新Cache                        │  │
│  │         flag = flagGlobal_(0)                                       │  │
│  │         if (flag == FLAG_VALUE):                                    │  │
│  │             waitFlagNum++                                           │  │
│  │             flagGlobal_(0) = 0  // 清除Flag                         │  │
│  │             DataCacheCleanAndInvalid()                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 八、与 Layered 版本的核心差异

### 8.1 通信模式对比

```
基础版 (a2.h) 通信模式:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  本地数据 ──────► WindowOut ──────► 直接复制/HCCL ──────► WindowIn (远端)  │
│                   (单层)            BatchWrite                               │
│                                                                            │
│  特点: 每个Token的完整数据直接发送到目标Rank                                 │
│  通信量: O(axisBS × axisK × H × worldSize)                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

分层版 (a2_layered.h) 通信模式:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  本地数据 ──► IPC共享 ──► 机内聚合 ──► WindowOut ──► RDMA ──► WindowIn    │
│             (机内)      (量化可选)                                         │
│                                                                            │
│  特点: 先机内聚合减少数据量，再跨服务器传输                                   │
│  通信量: O(axisBS × H × serverNum) [机间] + 机内IPC                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 功能差异对比表

| 特性 | 基础版 (a2.h) | 分层版 (a2_layered.h) |
|------|---------------|----------------------|
| **通信模式** | 单层AlltoAll | 分层(机内IPC + 机间RDMA) |
| **数据量化** | ❌ 不支持 | ✅ INT8动态量化 |
| **MoE专家** | ✅ 支持 | ✅ 支持 |
| **Zero专家** | ✅ 支持 | ❌ 不支持 |
| **Copy专家** | ✅ 支持 | ❌ 不支持 |
| **Const专家** | ✅ 支持 | ❌ 不支持 |
| **Token Mask** | ✅ 支持 | ❌ 不支持 |
| **Expert Mask** | ✅ 支持 | ❌ 不支持 |
| **适用场景** | 单服务器/小规模 | 大规模跨服务器 |
| **性能优化** | 通用 | 针对跨服务器优化 |

---

## 九、开发注意事项

### 9.1 内存对齐要求

| 类型 | 对齐粒度 | 常量 |
|------|----------|------|
| UB缓冲区 | 32B | `UB_ALIGN` |
| 状态空间 | 512B | `STATE_OFFSET` |
| 状态区总大小 | 1MB | `STATE_SPACE_SIZE` |
| Block大小 | 32B | `BLOCK_SIZE` |

### 9.2 Flag使用注意

```cpp
// 重要: 写入Flag后必须刷新Cache
flagGlobal_(0) = FLAG_VALUE;
DataCacheCleanAndInvalid<uint32_t, SINGLE_CACHE_LINE, CACHELINE_OUT>(flagGlobal_);

// 重要: 清除Flag后也必须刷新，避免重复计数
flagGlobal_(0) = 0;
DataCacheCleanAndInvalid<uint32_t, SINGLE_CACHE_LINE, CACHELINE_OUT>(flagGlobal_);
```

### 9.3 原子操作使用

```cpp
// 计算recvCount时使用原子累加
SetAtomicAdd<int32_t>();
DataCopyPad(expertRecvCountGlobal_, recvCountLocal_, ...);
SetAtomicNone();  // 必须恢复，避免影响后续操作
```

### 9.4 专家类型判断顺序

```cpp
// 正确的判断顺序 (按照ID范围)
if (expId < moeExpertNum_) {
    // MoE专家
} else if (expId < moeExpertNum_ + zeroExpertNum_) {
    // Zero专家: 跳过
} else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_) {
    // Copy专家
} else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
    // Const专家
}
```

### 9.5 常见问题排查

| 问题 | 可能原因 | 排查方法 |
|------|----------|----------|
| 数据不正确 | Mask未正确处理 | 检查axisBS_更新逻辑 |
| 死锁 | Flag未正确设置/清除 | 检查DataCacheCleanAndInvalid调用 |
| 计数错误 | 原子操作遗漏 | 确保SetAtomicNone被调用 |
| 偏移错误 | expertWindowOffset计算错误 | 检查前缀和逻辑 |

---

## 十、关键常量定义

```cpp
constexpr uint8_t BUFFER_NUM = 2;                       // Ping-Pong双缓冲
constexpr uint32_t STATE_OFFSET = 512;                  // 每个Rank状态空间大小
constexpr uint32_t STATE_SPACE_SIZE = 1024 * 1024;      // 状态区总大小 = 1MB
constexpr uint32_t UB_ALIGN = 32;                       // UB对齐粒度
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;      // 本卡状态空间偏移
constexpr uint32_t BATCH_WRITE_ITEM_OFFSET = 8 * 1024;  // BatchWriteItem偏移
constexpr uint32_t BATCH_WRITE_ITEM_SIZE = 32;          // 每个Item 32字节
constexpr uint32_t SKIP_OFFSET = 32;                    // Flag前的跳过字节数
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;             // Flag完成标志
constexpr uint32_t A2_RANK_NUM_PER_SERVER = 8;          // 每服务器8卡
```

---

## 十一、数据流端到端示意图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Combine A2 基础版 数据流                              │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │   输入Token      │  expandXGlobal_ [axisBS × H]                          │
│  │   (Expert结果)   │  expertIdsGlobal_ [axisBS × K]                        │
│  └────────┬────────┘  expandScalesGlobal_ [axisBS × K]                      │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │  CalXActiveMask │  处理Token/Expert Mask                                 │
│  │   (Mask处理)     │  计算有效Token数量                                     │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │AlltoAllDispatch │  按专家分组分发数据                                     │
│  │   (数据分发)     │  单服务器: 直接复制                                     │
│  │                 │  多服务器: HCCL BatchWrite                              │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │    Preload      │  预加载专家ID/Scale/Index                               │
│  │   (预加载)       │  计算recvCount和expertWindowOffset                     │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ WaitDispatch    │  轮询等待各Rank数据到达                                 │
│  │   (等待数据)     │  检查Flag == FLAG_VALUE                                │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ LocalWindowCopy │  遍历每个Token的TopK专家                                │
│  │   (聚合输出)     │  ├─ MoE专家: scale × WindowIn数据                      │
│  │                 │  ├─ Zero专家: 跳过                                      │
│  │                 │  ├─ Copy专家: scale × oriX                              │
│  │                 │  └─ Const专家: scale × (alpha1×x + alpha2×v)           │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   最终输出       │  expandOutGlobal_ [axisBS × H]                        │
│  │   (加权结果)     │  Σ scale[expert] × data[expert]                       │
│  └─────────────────┘                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

*文档生成时间: 2026-03-14*
*基于源文件: moe_distribute_combine_a2.h*
