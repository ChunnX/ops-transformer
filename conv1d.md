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
1. 算子名_def.cpp所有输入输出的DataType、Format、UnknownShapeFormat需要一一对应
2. 无需生成非指定平台的配置

## 2. 模板设计
模板设计的Tiling切分遵循几个基本规则：
1. Tiling切分需要优先满足CacheLine大小，即算子的输入shape从最内轴开始计算，ubFactor至少要大于一个CacheLine大小

### 2.0 Tiling公共处理
1. 只处理format为NHWC的格式
2. N必须能被block_size * block_size整除
3. H方向crop裁剪大小不能超过block_size * H, 同理W也是
4. 算子需要支持所有的DTYPE
5. 算子的DType不作为TilingKey，TilingKey的参数只有一个：IsLargeC, 仔细阅读[##基础知识]章节的[###TilingKey]的链接，使用文档中的接口完成TilingKey定义

### 2.1 模板1--纯搬移模板
> 提供模板详细的准入条件，描述模板主要是为了解决什么样的问题。

当尾轴C > CacheLine / 2时，使用该模板

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

1. 获取平台信息，基于基础知识中的硬件结构章节，通过提供的接口获取硬件的LocalMemory UB的大小、硬件的核数、硬件的UBBlock大小、硬件的cacheLine大小等信息
2. 判断C轴大小是否满足模板的准入条件，不满足不需要报错，只要返回即可
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
该模板不存在计算逻辑，只需要找到对应的位置搬入和搬出。Kernel侧核心循环逻辑为：
```
beginIdx = blockIdx * perCoreCount;
endIdx = min((blockIdx + 1) * perCoreCount, totalCount)
for (idx = beginIdx; idx < endIdx; idx++)
{
    CopyIn();

    CopyOut();
}
```

地址计算分输入和输出的，因为算子按输出切分，UB上保证按照输出排序完。因此先在CopyIn之前计算输出地址偏移：
1. 计算到切分轴的stride
2. 计算循环过程中每次切分轴的拷贝长度
```
// 计算切分轴的切分个数
ubFactorCount = CeilDiv(outShape[ubAxis], ubFactor)
// 切分轴每次的拷贝长度，可能存在尾块
ubAxisCopyLen = idx % ubFactorCount == ubFactorCount - 1 ? outShape[ubAxis] - ubFactor : ubFactor;
```
3. 循环计算每次拷贝的输出地址偏移

输出地址计算之后，调用CopyIn，因指令的限制，搬出如果跨block_size, 输入的stride步长不一样，无法一次搬入，需要通过循环搬入
1. 每次循环搬入地址计算需要考虑该维度对应的裁剪大小
2. 记录切分轴内每个轴的长度，作为DataCopyPad指令的参数

### 2.4.2 Kernel实现

#### 核心要求（必须严格遵守）

不要读库上已有的代码，只需参考kernel设计和提供的文档。

**1. kernel 文件名与函数名规则**：
- kernel 核函数的 `.cpp` 文件名必须与 `opFile.value` 的值一致（在算子信息库文件中设置）
- kernel 核函数的函数名必须固定为算子名，无论文件名如何改变
- 例如：opFile.value = "batch_to_space_apt"，则文件名是 `batch_to_space_apt.cpp`，函数名是 `batch_to_space`

**2. 模板化TilingKey的正确使用方式（关键）**：
- **ASCENDC_TPL_ARGS_DECL 等宏只在 tiling_key.h 中使用**，用于定义模板参数和参数组合
- **kernel 核函数不使用任何 ASCENDC_TPL 相关宏**，而是使用标准的 C++ template 语法定义模板参数
- 模板参数的类型和名称必须与 tiling_key.h 中 ASCENDC_TPL_UINT_DECL/ASCENDC_TPL_BOOL_DECL/ASCENDC_TPL_DATATYPE_DECL 定义的参数一一对应

**正确示例**：
```cpp
// tiling_key.h 中定义（使用宏）：
ASCENDC_TPL_ARGS_DECL(BatchToSpace,
    ASCENDC_TPL_UINT_DECL(IsLargeC, 1, ASCENDC_TPL_UI_LIST, IS_LARGE_C_FALSE, IS_LARGE_C_TRUE)
);

// kernel.cpp 中实现（不使用宏，使用标准 template）：
template <int IsLargeC>  // 对应 ASCENDC_TPL_UINT_DECL 定义的参数
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    if constexpr (IsLargeC == IS_LARGE_C_TRUE) {
        // kernel implementation
    }
}
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

## 3.编译验证
生成的算子，通过`bash build.sh --ops=算子名 --soc=平台 --pkg -j32`验证编译
编译之前，在用户家目录下，搜索setenv.bash文件，之后source该文件引入环境变量 
编译不过，不要试图修改TilingData的结构体实现方式。

## 备注
文档中标明重要的要严格执行，否则会有编译错误

**最常见错误总结**：
1. ❌ 使用 TBuf 而不是 TQue - TPipe::InitBuffer不支持TBuf
2. ❌ 在 kernel.cpp 中使用 ASCENDC_TPL 相关宏 - 这些宏只在 tiling_key.h 中使用
3. ❌ kernel 函数名与 opFile.value 不一致 - 文件名与 opFile.value 一致，函数名固定为算子名
4. ❌ DataCopyPad 使用 count 参数 - 应使用 blockCount 和 blockLen
5. ❌ kernel 侧依赖 op_kernel/math_util.h - 直接实现简单运算即可
