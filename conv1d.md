## 基础知识
### PlatformAscendC
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00059.html

### TilingData
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00024.html

### TilingKey
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html

### 核函数API
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002564044541__table1199372172410

基础知识在后续要求引用时再读取，无需预先读取。

# conv1d算子设计文档
## 基本信息
|名称|属性|
|--|--|
|算子名|batch_to_space|
|算子类别|moe|
|平台|ascend950|

## 0. 前置准备
依据提供的算子的基本信息，调用`bash build.sh --genop=算子类别/算子名`生成算子目录。后续代码修改基于新生成的算子目录。

## 1. 计算公式描述
> 提供详细的计算公式描述，如果是对标torch或者tensorflow，直接提供torch或tensorflow的算子名。

参考TensorFLow官方文档
https://tensorflow.google.cn/api_docs/python/tf/batch_to_space

### 1.1 算子定义
```
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)
```

将算子定义放到算子目录的op_graph目录下，新建`算子名_proto.h`保存

### 1.2 ACLNN
无需生成ACLNN接口，修改op_host下cmake文件或者算子目录下的cmake文件，将传递aclnn类型改成aclnn_exclude

### 1.3 算子信息库文件
1. 生成算子的信息库文件--算子名_def.cpp, 其中opFile.value指定为batch_to_space_apt。
**重要**：kernel 核函数的 .cpp 文件名必须与 opFile.Value的值一致。无论文件名如何改变，核函数的名称（函数名）都必须固定为算子名。

2. **重要：DataType/Format配置的组合场景理解**
   - `DataType`数组中的每个元素代表一个独立的组合场景，不同输入/输出的数组按位置一一对应
   - 所有输入和输出的`DataType`、`Format`、`UnknownShapeFormat`数组长度必须完全一致
   - 每个位置代表一个完整的支持组合，例如：
     ```
     场景1：input1=float, input2=int32, output=float
     场景2：input1=float16, input2=int32, output=float16
     场景3：input1=int32, input2=int64, output=int32
     ```
   - 如果某个输入支持多种数据类型，需要为每种类型创建对应的组合场景
   - **示例**：若x支持9种DTYPE，crops支持int32和int64两种，则需要创建18个组合场景（9×2）
     ```cpp
     .Input("x")
         .DataType({DT_FLOAT, DT_FLOAT16, DT_INT32, ..., DT_DOUBLE,  // 前9个场景用int32 crops
                   DT_FLOAT, DT_FLOAT16, DT_INT32, ..., DT_DOUBLE}) // 后9个场景用int64 crops
     
     .Input("crops")
         .DataType({DT_INT32, DT_INT32, DT_INT32, ..., DT_INT32,     // 前9个场景
                   DT_INT64, DT_INT64, DT_INT64, ..., DT_INT64})     // 后9个场景
     
     .Output("y")
         .DataType({DT_FLOAT, DT_FLOAT16, DT_INT32, ..., DT_DOUBLE,  // 与x保持一致
                   DT_FLOAT, DT_FLOAT16, DT_INT32, ..., DT_DOUBLE})
     ```

3. 无需生成非指定平台的配置

## 2. 模板设计
模板设计的Tiling切分遵循几个基本规则：
1. Tiling切分需要优先满足CacheLine大小，即算子的输入shape从最内轴开始计算，ubFactor至少要大于一个CacheLine大小

### 2.0 Tiling公共处理
1. 只处理format为NHWC的格式
2. N必须能被block_size * block_size整除
3. H方向crop裁剪大小不能超过block_size * H, 同理W也是
4. 算子需要支持所有的DTYPE
5. **TilingKey实现要求**：
   - TilingKey参数只有一个：IsLargeC（判断尾轴C是否满足CacheLine要求）
   - 判断标准：IsLargeC = (C * sizeof(DataType) > CacheLineSize / 2)
   - **必须在实现前仔细阅读[##基础知识]章节的[###TilingKey]链接，严格按照链接中的接口和规范实现**
   - **错误警告**：不阅读链接直接实现会导致编译错误或运行时错误

### 2.1 模板1--纯搬移模板
> 提供模板详细的准入条件，描述模板主要是为了解决什么样的问题。

**模板说明**：
- **主要用途**：处理batch_to_space的空间重排操作，通过UB进行数据搬移
- **准入条件**：本模板适用于所有情况，通过TilingKey区分不同场景
  - 当 IsLargeC = TRUE 时：C轴足够大，可以直接高效搬运
  - 当 IsLargeC = FALSE 时：C轴较小，需要考虑其他优化策略（本设计聚焦IsLargeC=TRUE场景）
- **核心逻辑**：按输出shape切分，在UB内重排数据后搬出

### 2.2 TilingData设计
> 模板的TilingData设计
```
struct TilingData {
  int64_t inShape[4];
  int64_t outShape[4];
  int64_t blockSize;
  int64_t crops[2][2];
  uint64_t totalCount;
  uint64_t perCoreCount;
  uint8_t ubAxis;
  uint32_t ubFactor;
  uint32_t bufferSize;
}
```
### 2.3 Tiling
### 2.3.1 Tiling设计
> 提供模板的Tiling策略和内存分配策略
>

**前置条件**：
- 必须先阅读[##基础知识]章节的[###PlatformAscendC]和[###TilingKey]链接
- 必须先阅读[##基础知识]章节的[###TilingData]链接了解TilingData结构体定义规范

**Tiling流程**：

1. 获取平台信息，通过platform_util.h接口获取：
   - UB大小、核数、UBBlock大小、cacheLine大小
   - **参考**：[##基础知识]章节的[###PlatformAscendC]链接

2. 设置TilingKey（IsLargeC）：
   - 判断尾轴C是否满足模板条件
   - **实现规范**：严格按照[##基础知识]章节的[###TilingKey]链接中的接口定义
   - 调用SetTilingKey传递给kernel

3. 划分UB大小，UB一分为二，如果超过64K, 取到64K即可，作为bufferSize大小
4. 使用3确定的bufferSize大小切分原始的输出shape, 从最内轴开始遍历，找到满足bufferSize大小的位置，记下ubAxis和ubFactor(ubFactor表示的ubAxis切分轴在ub内的切分大小，不能比原始shape大)。
```
tmpbufferSize = bufferSize / sizeof(T);
for (auto axis = AXIS_COUNT - 1; axis >= 0; axis--) {
  uint64_t size = axis == AXIS_COUNT - 1 ? CeilAlign(shape[axis], UbBlockSize) : shape[axis];
  factor = tmpbufferSize / shape[axis];
  if (factor == 0) {
    // 记录ubAxis和ubFactor
  }
  tmpbufferSize = factor;
}
```
5. 切分轴UBFactor之外的部分与切分轴之外做累乘，记为totalCount，表示共有多少个切分块
6. 使用步骤5计算的totalCount，对核数做向上取整，得到perCoreCount
7. 实际使用的核数通过totalCount对perCoreCount向上取整得到
8. **设置运行时参数**：调用`context->SetTilingKey(tilingKey_)`将tilingKey传递给kernel

### 2.3.2 Tiling实现
1. **重要**：参考[##基础知识]章节的[###TilingData]链接，使用标准C++语法定义TilingData结构体（POD）
  
   **正确示例**：
   ```cpp
   struct BatchToSpaceTilingData {
       int64_t inShape[4];
       int64_t outShape[4];
       uint64_t totalCount;
   };
   ```
   **错误示例（禁止使用）**：
   ```cpp
   BEGIN_TILING_DATA_DEF(BatchToSpaceTilingData)
   TILING_DATA_FIELD_DEF_ARR(int64_t, inShape, 4);
   END_TILING_DATA_DEF;
   ```

2. Tiling的代码使用C++类封装实现，仅TilingFunc和TilingParseFunc可以通过C函数提供注册接口。
   **重要**：必须创建一个C++类（如class BatchToSpaceTiling, 无需继承TilingBaseClass），将所有tiling逻辑和参数封装在类中。
3. 参考[##基础知识]章节的[###PlatformAscendC]部分，使用文档中的接口获取平台核数核UB大小信息
    >使用platform_util.h的接口GetUbBlockSize获取硬件的UBBlock大小，可参考库上其他算子用法
    >使用platform_util.h的接口GetCacheLineSize获取硬件的cacheLine大小，可参考库上其他算子用法
    >使用math_util.h的接口CeilDiv计算向上取整
    >使用GetSizeByDataType接口获取DType的字节大小
    >属性获取时GetAttrs和GetAttrPointer的接口组合

### 2.4 Kernel
### 2.4.1 Kernel设计
> 提供模板的Vector计算主要涉及的指令

**前置条件**：
- 必须先阅读[##基础知识]章节的[###TilingKey]链接，了解kernel侧如何使用TilingKey模板参数
- 必须先阅读[##基础知识]章节的[###核函数API]链接，了解TBuf资源管理接口

**Buffer管理方案（关键）**：

使用**TBuf + Ping-Pong双缓冲**方式管理UB内存：

1. **Buffer申请**：
   - 使用TBuf申请一块完整的buffer，大小为`bufferSize * 2`（双倍）
   - 不使用TQue队列机制，直接通过Get<T>()获取buffer访问权限
   - 示例：
     ```cpp
     TBuf<TPosition::VECCALC> ubBuffer;
     pipe.InitBuffer(ubBuffer, bufferSize * 2);  // 申请双倍大小
     ```

2. **Ping-Pong轮流使用**：
   - 将双倍buffer分为两块：Buffer0（偏移0）和Buffer1（偏移bufferSize）
   - 每次迭代根据idx奇偶性选择不同buffer块
   - 计算偏移：
     ```cpp
     int64_t pingpongOffset = bufferSize / sizeof(T);  // 元素偏移量
     int64_t bufOffset = (idx % 2) * pingpongOffset;    // 根据idx选择buffer
     ```
   - 使用方式：
     ```cpp
     LocalTensor<T> ubLocal = ubBuffer.Get<T>();
     DataCopy(ubLocal[bufOffset], inputGM[offset], length);  // 偏移访问
     ```

3. **完整流程示意**：
   ```
   idx=0: CopyIn(buf[0]) → CopyOut(buf[0])
   idx=1: CopyIn(buf[1]) → CopyOut(buf[1])
   idx=2: CopyIn(buf[0]) → CopyOut(buf[0])  // 轮回buf[0]
   idx=3: CopyIn(buf[1]) → CopyOut(buf[1])  // 轮回buf[1]
   ```

**核心逻辑**：
该模板不存在计算逻辑，只需要找到对应的位置搬入和搬出。Kernel侧核心循环逻辑为：
```
beginIdx = blockIdx * perCoreCount;
endIdx = min((blockIdx + 1) * perCoreCount, totalCount)
pingpongOffset = bufferSize / sizeof(T)

for (idx = beginIdx; idx < endIdx; idx++)
{
    bufOffset = (idx % 2) * pingpongOffset
    
    CopyIn(bufOffset);
    
    CopyOut(bufOffset);
}
```

**地址计算详细流程（伪码）**：

1. **输出地址计算（CalculateOutputOffset）**：
```
输入: idx (当前处理块索引), ubAxisCopyLen (本次拷贝长度)
输出: currentOutputOffset (输出GM地址偏移)

// 步骤1: 将idx分解为切分轴索引和外部索引
ubAxisCount = CeilDiv(outShape[ubAxis], ubFactor)
ubAxisIdx = idx % ubAxisCount              // 切分轴内的块索引
outerIdx = idx / ubAxisCount               // 切分轴外的索引

// 步骤2: 计算切分轴内的偏移
currentUbAxisOffset = ubAxisIdx * ubFactor
ubAxisStart = currentUbAxisOffset
ubAxisEnd = ubAxisStart + ubAxisCopyLen

// 步骤3: 将outerIdx分解为各轴坐标（从切分轴前一轴开始）
offsets[AXIS_COUNT] = {0}
tmpOuterIdx = outerIdx
for (axis = ubAxis - 1; axis >= 0; axis--) {
    offsets[axis] = tmpOuterIdx % outShape[axis]
    tmpOuterIdx = tmpOuterIdx / outShape[axis]
}
offsets[ubAxis] = ubAxisStart

// 步骤4: 计算输出GM的线性地址偏移
currentOutputOffset = 0
for (axis = 0; axis < AXIS_COUNT; axis++) {
    axisStride = 1
    for (i = axis + 1; i < AXIS_COUNT; i++) {
        axisStride *= outShape[i]
    }
    currentOutputOffset += offsets[axis] * axisStride
}
```

2. **输入地址计算（CopyIn内部）**：
```
输入: idx, ubAxisCopyLen, bufOffset
输出: 执行DataCopy(inputGM[inOffset] → ubLocal[bufOffset])

// 步骤1: 根据idx获取输出坐标（NHWC格式）
ubAxisCount = CeilDiv(outShape[ubAxis], ubFactor)
ubAxisIdx = idx % ubAxisCount
outerIdx = idx / ubAxisCount

strides[AXIS_COUNT] = {0}
tmpOuterIdx = outerIdx
for (axis = ubAxis - 1; axis >= 0; axis--) {
    strides[axis] = tmpOuterIdx % outShape[axis]
    tmpOuterIdx = tmpOuterIdx / outShape[axis]
}
strides[ubAxis] = ubAxisIdx * ubFactor

outN = strides[0]  // 输出的batch维度
outH = strides[1]  // 输出的height维度
outW = strides[2]  // 输出的width维度

// 步骤2: BatchToSpace转换 - 将输出坐标映射到输入坐标
// Batch维度转换: outN → inN + block索引
inNIdx = outN
inHIdx = (outH + crops[0][0]) / blockSize
inWIdx = (outW + crops[1][0]) / blockSize
blockHIdx = (outH + crops[0][0]) % blockSize
blockWIdx = (outW + crops[1][0]) % blockSize

batchBlockIdx = blockHIdx * blockSize + blockWIdx
inN = inNIdx * (blockSize * blockSize) + batchBlockIdx

inH = inHIdx  // Height维度
inW = inWIdx  // Width维度

// 步骤3: 计算输入GM的线性地址偏移（NHWC格式）
inOffset = inN * inShape[1] * inShape[2] * inShape[3] +
           inH * inShape[2] * inShape[3] +
           inW * inShape[3] +
           strides[3]  // C维度保持不变

// 步骤4: 执行数据搬运
ubLocal = ubBuffer.Get<T>()
DataCopy(ubLocal[bufOffset], inputGM[inOffset], ubAxisCopyLen)
```

3. **输出数据搬运（CopyOut）**：
```
输入: idx, ubAxisCopyLen, bufOffset, currentOutputOffset
输出: 执行DataCopy(ubLocal[bufOffset] → outputGM[currentOutputOffset])

ubLocal = ubBuffer.Get<T>()
DataCopy(outputGM[currentOutputOffset], ubLocal[bufOffset], ubAxisCopyLen)
```

4. **BatchToSpace算子核心转换逻辑说明**：
   - 输入shape: [N, H, W, C]，其中 N 必须能被 blockSize^2 整除
   - 输出shape: [N/(blockSize^2), H*blockSize - cropH, W*blockSize - cropW, C]
   - 转换公式：
     ```
     输出坐标 (outN, outH, outW, C)
     → 输入坐标:
       inN = outN * blockSize^2 + blockH * blockSize + blockW
       inH = (outH + cropTop) / blockSize
       inW = (outW + cropLeft) / blockSize
       其中:
       blockH = (outH + cropTop) % blockSize
       blockW = (outW + cropLeft) % blockSize
     ```
   - 示例：blockSize=2, crop=[[1,1],[1,1]]
     ```
     输入: [4, 2, 2, 3] → 输出: [1, 3, 3, 3]
     输出(0,0,0,0) → 输入(0,0,0,0)  // blockH=1,blockW=1 → inN=0*4+1*2+1=3? 不对
     
     正确转换:
     输出outH=0: inHIdx=(0+crop[0][0])/blockSize = 1/2 = 0, blockHIdx=1%2=1
     输出outW=0: inWIdx=(0+crop[1][0])/blockSize = 1/2 = 0, blockWIdx=1%2=1
     输出outN=0: inN=0*4 + (1*2+1) = 3
     所以 输出(0,0,0,c) → 输入(3,0,0,c)
     ```

地址计算分输入和输出的，因为算子按输出切分，UB上保证按照输出排序完。因此先在CopyIn之前计算输出地址偏移，然后根据BatchToSpace转换规则计算输入地址偏移。

### 2.4.2 Kernel实现

#### 核心要求（必须严格遵守）

不要读库上已有的代码，只需参考kernel设计和提供的文档。

**1. kernel 文件名与函数名规则**：
- kernel 核函数的 `.cpp` 文件名必须与 `opFile.value` 的值一致（在算子信息库文件中设置）
- kernel 核函数的函数名必须固定为算子名，无论文件名如何改变
- 例如：opFile.value = "batch_to_space_apt"，则文件名是 `batch_to_space_apt.cpp`，函数名是 `batch_to_space`

**2. 模板化TilingKey的正确使用方式（关键）**：
- **必须先阅读[##基础知识]章节的[###TilingKey]链接，严格按照链接中的规范实现**
- **ASCENDC_TPL_ARGS_DECL 等宏只在 tiling_key.h 中使用**，用于定义模板参数和参数组合
- **kernel 核函数不使用任何 ASCENDC_TPL 相关宏**，而是使用标准的 C++ template 语法定义模板参数
- 模板参数的类型和名称必须与 tiling_key.h 中 ASCENDC_TPL_UINT_DECL/ASCENDC_TPL_BOOL_DECL/ASCENDC_TPL_DATATYPE_DECL 定义的参数一一对应

**正确示例**：
```cpp
// tiling_key.h 中定义（使用宏）：
// 严格按照[###TilingKey]链接中的接口规范

// kernel.cpp 中实现（不使用宏，使用标准 template）：
template <int IsLargeC>  // 对应 tiling_key.h 定义的参数
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    // 将模板参数传递给内部类
    NsBatchToSpace::BatchToSpace<DTYPE_X, IsLargeC> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

// batch_to_space.h 中类定义
template <typename T, int IsLargeC>
class BatchToSpace {
public:
    __aicore__ inline void Process() {
        if constexpr (IsLargeC == IS_LARGE_C_TRUE) {
            // kernel implementation
        }
    }
};
```

**错误示例（禁止使用）**：
```cpp
// 错误：在 kernel.cpp 中使用 ASCENDC_TPL_ARGS_DECL 等宏
ASCENDC_TPL_ARGS_DEF(BatchToSpace);  // ❌ 错误！这些宏不应在 kernel.cpp 中使用
template <ASCENDC_TPL_ARGS_DECL(BatchToSpace)>  // ❌ 错误！使用标准 template 语法
```

**3. 数据类型宏的使用**：
- kernel侧通过 `DTYPE_${输入名/输出名}` 获取数据类型，输入名/输出名必须在算子定义（REG_OP）中明确指定
- 拼接后的字符串全部大写，直接使用即可
- 例如：输入名为 x，则使用 `DTYPE_X`

**4. 重点接口参考链接**：
- 数据多维搬运接口 ： https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00242.html
- TBuf资源管理接口：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0110.html

**重要**：实现前必须阅读以下链接，理解接口使用方式：
- [##基础知识]章节的[###TilingKey]链接 - TilingKey定义和使用规范
- [##基础知识]章节的[###TilingData]链接 - TilingData结构体定义规范
- [##基础知识]章节的[###PlatformAscendC]链接 - 平台信息获取接口
- [##基础知识]章节的[###核函数API]链接 - TBuf资源管理和手动同步机制

**Buffer管理方案（核心）**：

使用**TBuf + Ping-Pong双缓冲 + 手动同步**方式管理UB内存：

1. **Buffer申请策略**：
   - 使用TBuf申请一块完整buffer，大小为`bufferSize * 2`（双倍大小）
   - 不使用TQue队列机制，避免队列管理开销
   - 通过Get<T>()直接获取buffer访问权限，通过偏移访问不同buffer块
   
2. **Ping-Pong双缓冲实现**：
   - 将双倍buffer划分为两块：Buffer0（偏移0）和Buffer1（偏移bufferSize）
   - 根据迭代idx奇偶性轮流使用：
     - idx偶数（0,2,4...）：使用Buffer0
     - idx奇数（1,3,5...）：使用Buffer1
   - 计算元素偏移：`bufOffset = (idx % 2) * (bufferSize / sizeof(T))`
   - 通过`ubLocal[bufOffset]`访问对应buffer块
   
3. **完整示例**：
   ```cpp
   // 初始化：申请双倍buffer
   TBuf<TPosition::VECCALC> ubBuffer;
   pipe.InitBuffer(ubBuffer, bufferSize * 2);
   
   // Ping-Pong轮流使用
   int64_t pingpongOffset = bufferSize / sizeof(T);
   for (int64_t idx = beginIdx; idx < endIdx; idx++) {
       int64_t bufOffset = (idx % 2) * pingpongOffset;
       
       // CopyIn: GM → UB[bufOffset]
       LocalTensor<T> ubLocal = ubBuffer.Get<T>();
       DataCopy(ubLocal[bufOffset], inputGM[offset], length);
       
       // CopyOut: UB[bufOffset] → GM
       DataCopy(outputGM[offset], ubLocal[bufOffset], length);
   }
   ```

4. **设计优势**：
   - 避免TQue队列管理的复杂性和额外buffer开销
   - Ping-Pong双缓冲最大化利用buffer资源
   - 符合Tiling设计的bufferSize双缓冲策略

## 3.编译验证
生成的算子，通过`bash build.sh --ops=算子名 --soc=平台 --pkg -j32`验证编译
编译之前，在用户家目录下，搜索setenv.bash文件，之后source该文件引入环境变量 
编译不过，不要试图修改TilingData的结构体实现方式。

## 备注
文档中标明重要的要严格执行，否则会有编译错误

**最常见错误总结**：
1. ❌ 不阅读链接直接实现 - 必须先阅读链接理解规范，否则导致编译/运行错误
2. ❌ 在 kernel.cpp 中使用 ASCENDC_TPL 相关宏 - 这些宏只在 tiling_key.h 中使用
3. ❌ kernel 函数名与 opFile.value 不一致 - 文件名与 opFile.value 一致，函数名固定为算子名
4. ❌ DataCopyPad 使用 count 参数 - 应使用 blockCount 和 blockLen
5. ❌ kernel 侧依赖 op_kernel/math_util.h - 直接实现简单运算即可
6. ❌ 忘记在 Tiling 中设置 tilingKey_ - 必须根据条件设置并调用 SetTilingKey
7. ❌ kernel 模板参数未传递给内部类 - 需要将 IsLargeC 作为模板参数传递给 BatchToSpace 类
8. ❌ 使用TQue而非TBuf - 应使用TBuf申请双倍buffer，通过偏移实现ping-pong双缓冲
9. ❌ Buffer大小设置错误 - 应申请bufferSize*2大小，而非单倍或多倍
10. ❌ Ping-Pong偏移计算错误 - bufOffset应为(idx%2) * (bufferSize/sizeof(T))，单位是元素而非字节
