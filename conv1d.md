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
|算子类别|conversion|
|平台|ascend950|

## 0. 前置准备
依据提供的算子的基本信息，调用`bash build.sh --genop=算子类别/算子名`生成算子目录。后续代码修改基于新生成的算子目录。

## 1. 计算公式描述
> 提供详细的计算公式描述，如果是对标torch或者tensorflow，直接提供torch或tensorflow的算子名。

参考TensorFLow官方文档
https://tensorflow.google.cn/api_docs/python/tf/batch_to_space

## 2. 模板设计
模板设计的Tiling切分遵循几个基本规则：
1. Tiling切分需要优先满足CacheLine大小，即算子的输入shape从最内轴开始计算，ubFactor至少要大于一个CacheLine大小

### 2.0 公共处理
1. 只处理format为NHWC的格式
2. N必须能被block_size * block_size整除
### 2.1 模板1--纯搬移模板
> 提供模板详细的准入条件，描述模板主要是为了解决什么样的问题。

当尾轴C > CacheLine / 2时，使用该模板

x 和 y输入shape一样时
### 2.2 TilingData
> 模板的TilingData设计
```
struct TilingData {
  int64_t N;
  int64_t H;
  int64_t W;
  int64_t C;
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

1. 获取平台信息，基于基础知识中的硬件结构章节，通过提供的接口获取硬件的LocalMemory UB的大小、硬件的核数、硬件的UBBlock大小、硬件的cacheLine大小等信息
2. 判断C轴大小是否满足模板的准入条件
3. 划分UB大小，UB一分为二，如果超过64K, 取到64K即可，作为bufferSize大小
4. 使用3确定的bufferSize大小切分原始的输出shape, 从最内轴开始遍历，找到满足bufferSize大小的位置，记下ubAxis和ubFactor。如果最内轴小于bufferSize，需要考虑UbBlock对齐
5. 切分轴UBFactor之外的部分与切分轴之外做累乘，记为totalCount，表示共有多少个切分块
6. 使用5计算的totalCount，对核数做向上取整，得到perCoreCount
7. 实际使用的核数通过totalCount对perCoreCount向上取整得到

对于硬件的UBBlock大小可以通过platform_util.h的GetUbBlockSize接口
对于硬件的cacheLine大小可以通过platform_util.h的GetCacheLineSize接口
这两个接口使用时可参考本仓算子，不是必须要找到函数实现

向上取整的接口可以通过math_util.h的CeilDiv接口获取

### 2.4 Kernel
> 提供模板的Vector计算主要涉及的指令

kernel代码书写前，阅读基础知识的TilingKey和核函数API部分，使用该章节提供的接口书写。

Buffer管理使用资源管理API列表的TQue接口

CopyIn搬入使用Memory数据搬运的多维数据搬运接口

Compute使用SIMD API的Memory矢量计算接口

Kernel部分负责计算，核间偏移, Kernel侧计算需要考虑尾不足UBFactor的场景，UBFactor每次遍历到outShape[ubAxis]结束就会出现尾块。
```
beginIdx = blockIdx * perCoreCount;
endIdx = min((blockIdx + 1) * perCoreCount, totalCount)
for (idx = beginIdx; idx < endIdx; idx ++)
{
    CopyIn();

    Compute();

    CopyOut();
}
```


## 3.编译验证
生成的算子，通过`bash build.sh --ops=算子名 --soc=平台 --pkg`验证编译