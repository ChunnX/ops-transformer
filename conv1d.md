# conv1d算子设计文档
## 基础知识
### 硬件结构
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html

### TilingData
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00024.html

### TilingKey
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html

### 核函数API
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0003.html#ZH-CN_TOPIC_0000002564044541__table1199372172410

## 基本信息
|名称|属性|
|--|--|
|算子名|BatchToSpace|
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

## 2. 模板设计
模板设计的Tiling切分遵循几个基本规则：
1. Tiling切分需要优先满足CacheLine大小，即算子的输入shape从最内轴开始计算，ubFactor至少要大于一个CacheLine大小

### 2.0 公共处理
1. 只处理format为NHWC的格式
2. N必须能被block_size * block_size整除
3. H方向crop裁剪大小不能超过block_size * H, 同理W也是
4. 算子需要支持所有的DTYPE

#### 2.0.1 TilingKey
算子的DType不作为TilingKey，TilingKey的参数只有一个，IsLargeC

### 2.1 模板1--纯搬移模板
> 提供模板详细的准入条件，描述模板主要是为了解决什么样的问题。

当尾轴C > CacheLine / 2时，使用该模板

### 2.2 TilingData
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
> 提供模板的Tiling策略和内存分配策略
>
在书写Tiling切分前，阅读基础知识的TilingData部分，使用该章节提供的接口书写。
Tiling的代码使用C++类封装实现，仅TilingFunc和TilingParseFunc可以通过C函数提供注册接口

1. 获取平台信息，基于基础知识中的硬件结构章节，通过提供的接口获取硬件的LocalMemory UB的大小、硬件的核数、硬件的UBBlock大小、硬件的cacheLine大小等信息
2. 判断C轴大小是否满足模板的准入条件，不满足不需要报错，只要返回即可
3. 划分UB大小，UB一分为二，如果超过64K, 取到64K即可，作为bufferSize大小
4. 使用3确定的bufferSize大小切分原始的输出shape, 从最内轴开始遍历，找到满足bufferSize大小的位置，记下ubAxis和ubFactor。
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

Tiling常用接口参考
|接口名|头文件|功能|
|--|--|--|
|GetUbBlockSize|platform_util.h|获取硬件的UBBlock大小|
|GetCacheLineSize|platform_util.h|获取硬件的cacheLine大小|
|CeilDiv|math_util.h|除法向上取整计算|

以上代码实现完优化下。

### 2.4 Kernel
> 提供模板的Vector计算主要涉及的指令

kernel代码书写前，阅读基础知识的TilingKey和核函数API部分，使用该章节提供的接口书写。
**重点掌握：**
>1. Buffer管理使用资源管理API列表的TQue接口
>2. CopyIn搬入使用Memory数据搬运的多维数据搬运接口
>3. Compute使用SIMD API的Memory矢量计算接口

Kernel部分负责计算，核间偏移, Kernel的搬入搬出逻辑可以按切分轴位置区分。
```
beginIdx = blockIdx * perCoreCount;
endIdx = min((blockIdx + 1) * perCoreCount, totalCount)
for (idx = beginIdx; idx < endIdx; idx++)
{
    CopyIn();

    CopyOut();
}
```
切尾轴时，ubAxis = 3:
在CopyIn之前计算输出地址偏移
```
// 计算切分轴的切分个数
ubFactorCount = CeilDiv(outShape[ubAxis], ubFactor)
// 切分轴每次的拷贝长度，可能存在尾块
ubAxisCopyLen = idx % ubFactorCount == ubFactorCount - 1 ? outShape[ubAxis] - ubFactor : ubFactor;
int64_t oOffset = 0;
int64_t outStride = 1;
int64_t inStride = 1;
for (auto axis = AXIS_COUNT - 1; axis >= ubAxis; axis--) {
  outStride = outShape[axis] * outStride;
  inStride = inShape[axis] * inStride;
}
for (auto axis = ubAxis; axis >= 0; axis--) {
    // 计算输出的offset偏移oOffset
    shape = axis == ubAxis ? ubFactorCount : outShape[axis];
    step = axis == ubAxis ? ubFactor : 1;
    oOffset += step * outStride * idx % shape;
    outStride = outShape[axis] * outStride;
    idx = idx / shape;
    if (idx == 0) {
      break;
    }
}
```
在CopyIn函数内计算输入地址偏移，输入偏移需要考虑crop裁剪

## 3.编译验证
生成的算子，通过`bash build.sh --ops=算子名 --soc=平台 --pkg`验证编译