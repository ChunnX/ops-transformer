# Flash Attention Kernel Architecture (arch35)

## 概述

Flash Attention Kernel采用单一Kernel类 + 入口分离调用架构。Cube和Vector共用同一个Kernel类和ctx（保证跨核通信），但入口函数通过编译宏调用不同的Init/Process方法。

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           flash_attn_kernel_run                              │
│                        (flash_attn_entry_regbase.h)                          │
│                                                                              │
│   #ifdef __DAV_C310_CUBE__              #else                                │
│   op.InitCubeKernel(...)                op.InitVectorKernel(...)             │
│   op.ProcessCube()                      op.ProcessVector()                   │
│   └──────────────────────────────────────────────────────────────────────────┘
│                                   │                                          │
│   ┌───────────────────────────────▼──────────────────────────────────────┐   │
│   │            FlashAttentionNoQuantGqaKernel (单一类)                    │   │
│   │  ┌────────────────────────────────────────────────────────────────┐   │   │
│   │  │            FlashAttentionGqaContext ctx                       │   │   │
│   │  │  ┌──────────────────────────────────────────────────────────┐  │   │   │
│   │  │  │ BufferManager (UB/L1/GM) - 跨核通信共享                  │  │   │   │
│   │  │  │ bmm1Buffers, bmm2Buffers - Cube写Vector读               │  │   │   │
│   │  │  │ GlobalTensor (GM指针), TilingData, ConstInfo            │  │   │   │
│   │  │  └──────────────────────────────────────────────────────────┘  │   │   │
│   │  └────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                        │   │
│   │  CubeProcessor cubeProc           VectorProcessor vectorProc          │   │
│   │  (Cube计算)                        (Vector计算)                        │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. FlashAttentionNoQuantGqaKernel（单一Kernel类）

**文件位置**: `flash_attn_kernel_base_gqa_template.h`

**职责**: 管理Flash Attention的完整计算流程，Cube和Vector共用

```cpp
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel {
public:
    FlashAttentionGqaContext ctx;  // 共享数据，跨核通信关键
    
    FlashAttentionCubeProcessor<CubeData> cubeProc;
    FlashAttentionVectorProcessor<VectorData> vectorProc;
    
    __aicore__ inline FlashAttentionNoQuantGqaKernel() : cubeProc(&ctx), vectorProc(&ctx) {}
    
    // 旧接口（兼容）
    __aicore__ inline void Init(...);      // 内部用ASCEND_IS_AIC/AIV分离
    __aicore__ inline void Process();      // 内部用ASCEND_IS_AIC/AIV分离
    
    // 新接口（入口分离调用）
    __aicore__ inline void InitCubeKernel(...);   // Cube专用Init
    __aicore__ inline void InitVectorKernel(...); // Vector专用Init
    __aicore__ inline void ProcessCube();         // Cube专用Process
    __aicore__ inline void ProcessVector();       // Vector专用Process
    
private:
    // 分离的初始化方法
    __aicore__ inline void InitQCuSeqLensParserCube();
    __aicore__ inline void InitKvCuSeqLensParserCube();
    __aicore__ inline void InitQCuSeqLensParserVector();
    __aicore__ inline void InitKvCuSeqLensParserVector();
    __aicore__ inline void InitCubeBlock(...);
    __aicore__ inline void InitVecBlock(...);
    
    // 兼容方法（内部用宏分离）
    __aicore__ inline void InitQCuSeqLensParser();
    __aicore__ inline void InitKvCuSeqLensParser();
    __aicore__ inline void InitCube(...);
    __aicore__ inline void InitVector(...);
};
```

**关键设计**:
- 单一类，ctx是成员变量，保证Cube和Vector共享同一个Buffer实例
- 跨核通信Buffer（bmm1Buffers、bmm2Buffers）通过ctx共享
- cubeProc和vectorProc通过`&ctx`指针访问共享数据

### 2. Kernel入口函数

**文件位置**: `flash_attn_entry_regbase.h`

```cpp
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, ...>
inline __aicore__ void flash_attn_kernel_run(...)
{
    // 类型推导
    using CubeBlock = BaseApi::FANoQuantGqaBlockCube<...>;
    using VecFaBlock = BaseApi::FANoQuantGqaBlockVec<...>;
    using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<...>;
    
    // Dummy类型用于编译分离
    using VecBlockDummy = BaseApi::VecBlockBase<...>;
    using CubeBlockDummy = BaseApi::CubeBlockBase<...>;
    
#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::FlashAttentionNoQuantGqaKernel<CubeBlock, VecBlockDummy, VecBlockDummy>;
    TPipe tPipe;
    Kernel op;
    op.InitCubeKernel(query, key, value, cuSeqLensQ, cuSeqLensKv, blocktable, workspace,
                      runtimeMetaData, sequsedQ, sequsedKv, tilingData, &tPipe);
    op.ProcessCube();
#else
    using Kernel = BaseApi::FlashAttentionNoQuantGqaKernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
    TPipe tPipe;
    Kernel op;
    op.InitVectorKernel(key, attenMask, cuSeqLensQ, cuSeqLensKv, learnableSink, softmaxLse, 
                        attentionOut, workspace, runtimeMetaData, sequsedQ, sequsedKv,
                        tilingData, &tPipe);
    op.ProcessVector();
#endif
}
```

**关键设计**:
- 单一Kernel类，通过不同模板参数区分Cube和Vector
- Dummy类型（VecBlockDummy/CubeBlockDummy）用于编译分离，避免编译不需要的代码
- 入口宏分离调用不同的Init/Process方法

### 3. Block类（无继承设计）

```cpp
class FANoQuantGqaBlockCube {  // 不继承CubeBlockBase
    static constexpr uint32_t mBaseSize = ...;  // 直接定义
    static constexpr LayOutTypeEnum LAYOUT = ...;
    using Q_T = INPUT_T;  // 直接定义类型别名
    ...
};

class FANoQuantGqaBlockVec {  // 不继承VecBlockBase
    static constexpr uint32_t mBaseSize = ...;
    static constexpr bool HAS_MASK = ...;
    ...
};
```

## 跨核通信机制

**关键**: Cube和Vector必须共享同一个Buffer实例

```
┌─────────────────────────┐        ┌─────────────────────────┐
│      Cube核             │        │      Vector核           │
│                         │        │                         │
│  ctx.bmm1Buffers        │◄──────►│  ctx.bmm1Buffers        │
│  (同一个实例)            │  同一   │  (同一个实例)            │
│                         │  实例   │                         │
│  写入MM1结果             │        │  读取执行Softmax        │
│                         │        │                         │
│  ctx.bmm2Buffers        │◄──────►│  ctx.bmm2Buffers        │
│  写入MM2结果             │        │  读取输出               │
└─────────────────────────┘        └─────────────────────────┘
```

**为什么必须使用单一Kernel类**:
- 两个独立Kernel类会有各自的ctx实例
- CubeKernel.ctx.bmm1Buffers ≠ VectorKernel.ctx.bmm1Buffers
- Cube写入自己的Buffer，Vector读自己的Buffer → 无法通信 → 卡死

## 执行流程

### Cube核执行流程
```
flash_attn_kernel_run (Cube路径)
  │
  ├─ Kernel op;                           // 同一Kernel类
  ├─ op.InitCubeKernel(...)               // Cube专用Init
  │    ├─ ctx.pipe = tPipe
  │    ├─ ctx.tilingData = tiling
  │    ├─ InitConstInfoWithPA()
  │    ├─ ctx.keyPtr = key
  │    ├─ SetGlobalBuffer (cuSeqLensGmQ, seqUsedGmQ, ...)
  │    ├─ InitQCuSeqLensParserCube()      // Cube专用
  │    ├─ InitKvCuSeqLensParserCube()     // Cube专用
  │    ├─ InitMMResBuf(workspace)         // 初始化共享Buffer
  │    ├─ ctx.sectionNum_ = ...
  │    └─ InitCubeBlock(query, key, value, blockTable)
  │
  └─ op.ProcessCube()                     // Cube专用Process
       └─ for each section:
            ├─ cubeProc.ProcessCube(sectionIdx)
            └─ CrossCoreBufferUnInit()    // 等待Vector完成
```

### Vector核执行流程
```
flash_attn_kernel_run (Vector路径)
  │
  ├─ Kernel op;                           // 同一Kernel类，共享ctx
  ├─ op.InitVectorKernel(...)             // Vector专用Init
  │    ├─ ctx.pipe = tPipe
  │    ├─ ctx.tilingData = tiling
  │    ├─ InitConstInfoWithPA()
  │    ├─ ctx.keyPtr = key
  │    ├─ SetGlobalBuffer (cuSeqLensGmQ, seqUsedGmQ, ...)
  │    ├─ InitQCuSeqLensParserVector()    // Vector专用
  │    ├─ InitKvCuSeqLensParserVector()   // Vector专用
  │    ├─ ctx.sectionNum_ = ...
  │    ├─ vectorData.isFd = ...
  │    └─ InitVecBlock(...)
  │
  └─ op.ProcessVector()                   // Vector专用Process
       ├─ CrossCoreBufferInit()           // 设置跨核同步
       └─ for each section:
            ├─ vectorProc.ProcessVector(sectionIdx)
```

## 文件结构

```
op_kernel/arch35/
├── flash_attn_entry_regbase.h          # Kernel入口，宏分离调用
├── flash_attn_kernel_base_gqa_template.h  # FlashAttentionNoQuantGqaKernel
│                                       # InitCubeKernel/InitVectorKernel
│                                       # ProcessCube/ProcessVector
├── flash_attn_kernel_base_gqa_common.h # FlashAttentionGqaContext共享数据
├── flash_attn_kernel_base_gqa_cube.h   # Cube Processor
├── flash_attn_kernel_base_gqa_vector.h # Vector Processor
├── flash_attn_block_cube_noquant_gqa.h # Cube Block（无继承）
├── flash_attn_block_vec_noquant_gqa.h  # Vector Block（无继承）
```

## 设计原则

### 1. 共享ctx原则（关键）
- 单一Kernel类，ctx是成员变量
- Cube和Vector通过`&ctx`指针访问共享数据
- 跨核通信Buffer必须共享同一个实例

### 2. 入口分离原则
- 入口函数通过`#ifdef __DAV_C310_CUBE__`宏分离
- Cube调用`InitCubeKernel`和`ProcessCube`
- Vector调用`InitVectorKernel`和`ProcessVector`

### 3. 编译分离原则
- 使用Dummy类型（CubeBlockDummy/VecBlockDummy）
- Cube路径编译时不需要Vector相关代码
- Vector路径编译时不需要Cube相关代码

### 4. 组合优于继承原则
- Kernel通过成员变量`ctx`组合Context
- Block类不继承基类

## 与旧架构对比

### 旧架构
```cpp
op.Init(...);   // 一个Init，内部用ASCEND_IS_AIC/AIV判断
op.Process();   // 一个Process，内部用ASCEND_IS_AIC/AIV判断
```

### 新架构
```cpp
#ifdef __DAV_C310_CUBE__
    op.InitCubeKernel(...);   // Cube专用Init，无宏判断
    op.ProcessCube();         // Cube专用Process，无宏判断
#else
    op.InitVectorKernel(...); // Vector专用Init，无宏判断
    op.ProcessVector();       // Vector专用Process，无宏判断
#endif
```

### 改进点
| 方面 | 旧架构 | 新架构 |
|------|--------|--------|
| Init调用 | 一个方法内部判断 | 入口分离调用不同方法 |
| Process调用 | 一个方法内部判断 | 入口分离调用不同方法 |
| 代码清晰度 | 内部混杂宏判断 | 入口分离，逻辑清晰 |
| 跨核通信 | 共享ctx（正确） | 共享ctx（正确） |
| Dummy类型 | 需要 | 需要（编译分离） |

**注意**: 两套方法共存，旧`Init/Process`保留用于兼容