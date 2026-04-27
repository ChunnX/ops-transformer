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
|算子名|conv1d|
|算子类别|conv1d|
|平台|ascend950|

## 0. 前置准备
依据提供的算子的基本信息，调用`bash build.sh --genop=算子类别/算子名`生成算子目录。后续代码修改基于新生成的算子目录。

## 1. 计算公式描述
> 提供详细的计算公式描述，如果是对标torch或者tensorflow，直接提供torch或tensorflow的算子名。

z = x + y

## 2. 模板设计
### 2.1 模板1
> 提供模板详细的准入条件，描述模板主要是为了解决什么样的问题。

x 和 y输入shape一样时
### 2.2 TilingData
> 模板的TilingData设计

struct AddTilingData {
  int64_t size;
  int32_t ubFactor;
  int64_t ubTotalCount;
  int64_t ubCountPerCore;
}
### 2.3 Tiling
> 提供模板的Tiling策略和内存分配策略

1. 获取平台信息，基于基础知识中的硬件结构章节，通过提供的接口获取硬件的LocalMemory UB的大小、硬件的核数、硬件的UBBlock大小、硬件的cacheLine大小等信息
2. 判断x和y的shape是否一致
3. 合并x和y的shape，将他们合并成一个轴
4. 针对合轴的shape做tiling切分，tiling整体采用ping-pong流水的方式，即要求两个输入和输出都需要两份缓冲buffer
   4.1 切分时，优先保证CacheLine，即最小的切分块不能小于一个CacheLine大小
   4.2 在满足4.1的前提下，遍历shape大小，找到分核数最多的ubFactor, ubFactor尽可能保证UBBlock大小对齐
   4.3 根据4.2得出来的ubFactor计算ubTotalCount，即总的size对ubFactor向上取整
   4.4 根据4.3计算的ubTotalCount对硬件的核数向上取整，计算每个核处理的ubFactor个数ubCountPerCore
   4.5 最终实际的使用核数为ubTotalCount对ubCountPerCore向上取整
   4.6 所有的计算结果赋值给TilingData结构体

在书写Tiling切分前，阅读基础知识的TilingData部分，使用该章节提供的接口书写。

对于硬件的UBBlock大小可以通过platform_util.h的GetUbBlockSize接口
对于硬件的cacheLine大小可以通过platform_util.h的GetCacheLineSize接口
这两个接口可以参考当前库上的实现

### 2.4 Kernel
> 提供模板的Vector计算主要涉及的指令

Kernel部分负责计算，核间偏移
beginIdx = blockIdx * ubCountPerCore;
endIdx = min((blockIdx + 1) * ubCountPerCore, ubTotalCount)
for (idx = beginIdx; idx < endIdx; idx ++)
{
    coreOffset = beginIdx * ubFactor * sizeof(T);

    CopyIn();

    Compute();

    CopyOut();
}

kernel代码书写前，阅读基础知识的TilingKey和核函数API部分，使用该章节提供的接口书写。

Buffer管理使用资源管理API列表的TQue接口

CopyIn搬入使用Memory数据搬运的多维数据搬运接口

Compute使用SIMD API的Memory矢量计算接口

## 3.编译验证
生成的算子，通过`bash build.sh --ops=算子名 --soc=平台 --pkg`验证编译