# AIV直驱URMA

## 基础知识

URMA通信基于“Jetty抽象”[^1]，一个Jetty对应一个JFS-JFC资源对。

简单而言，JFS（Jetty For Send）中包括一个SQ队列和JFS Doorbell，前者用于写入WQE（Working Queue Element），后者用于通知硬件芯片处理WQE。WQE在部分地方也可能被称为SQE（Send Queue Element），因此SQ也会被称为WQ。

同理，JFC（Jetty For Completion）也包括一个CQ队列和JFC Doorbell，前者存放CQE（Completion Queue Element），后者用于将软件已经完成处理的CQE通知硬件芯片。

[^1]：https://mp.weixin.qq.com/s/2u47eHAsFX5j5wC5boezjQ

## URMA芯片的两个队列

URMA通信中有两个重要的队列，分别是SQ队列（Send Queue）和CQ队列（Completion Queue）。

两者都是循环队列，都包括一个PI指针（Producer index）和CI指针（Consumer Index）。

```
+--+ +--+      +--+
|  | |  | .... |  |
+--+ +--+      +--+
+--+           +--+
|  |   Send    |  | <-- CI
+--+           +--+
+--+           +--+
|  |   Queue   |  | <-- PI
+--+           +--+
+--+      +--+ +--+
|  | .... |  | |  |
+--+      +--+ +--+


+--+ +--+      +--+
|  | |  | .... |  |
+--+ +--+      +--+
+--+           +--+
|  |  C Queue  |  | <-- CI
+--+           +--+
+--+           +--+
|  |           |  | <-- PI
+--+           +--+
+--+      +--+ +--+
|  | .... |  | |  |
+--+      +--+ +--+
```

### SQ队列

对于SQ队列，图中每一个方块代表一个WQEBB（WQE Basic Block），而一个WQE可能包括多个WQEBB。

SQ队列的CI指针，由芯片（硬件）控制，表示当前位置之前的所有WQE都已经发送完成，但不保证发送成功；
PI指针，由用户（软件）控制，向SQ队列中拷贝WQE后，PI指针应指向写入的最后一个WQEBB的下一个位置。

当用户（软件）敲JFS Doorbell之后，芯片开始处理队列中从CI开始到PI结束的WQEBB（即**发送**操作）。

### CQ队列

对于CQ队列，是由芯片（硬件）通知用户（软件）相应WQE的发送情况，即芯片向队列中写入信息，用户从队列中读取信息。

CQ队列的PI指针，完全由芯片（硬件）控制，用户（软件）完全不感知，也无法获取。
CI指针，也由芯片（硬件）控制，用户（软件）需要记录CI值，并在需要时通过PollCQ操作轮询CQ队列读取是否存在相应CQE，同时记录最后一个CQE中的entry_idx，其表示指向SQE的最后一个WQEBB。

### WQE通信语义

通过WQE中的opcode字段可以设置Normal SQE的通信语义。URMA通信支持的通信语义包括`Send`、`Write`、`Write with notify`、`Read`等等。

当前DispatchSetup中使用`Write`语义，CombineSetup中使用`Write with notify`语义，其余语义暂未实现。

#### Write SQE

Write SQE结构体，占用一个WQEBB：

【图】

其中：
- cqe：表示当前WQE是否需要上报CQE
- odr：表示当前WQE是否有保序要求
- rmt_addr：写入的远端地址
- length：发送数据长度
- data_addr：源数据地址

#### Write with notify SQE

Write with notify SQE结构体，占用两个个WQEBB：

【图】

除了与Write SQE相同的字段外：
- notify_addr：notify写入的地址
- notify_data：notify写入的值

## URMA通信接口

相关通信接口定义在`mc2\moe_distribute_combine_setup\op_kernel\moe_distribute_base.h`

### 获取URMA Jetty信息

```cpp
__aicore__ inline void GetURMASqInfoTensor(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, GM_ADDR hcclContext, uint32_t dstRankId);
__aicore__ inline void GetURMACqInfoTensor(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, GM_ADDR hcclContext, uint32_t dstRankId);
```

从HCCL组件提供的建链信息hcclContext结构体中拷贝第dstRankId张卡的SQ或CQ的Jetty信息。

### 初始化WQE

```cpp
__aicore__ inline void GenerateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sqeCount = 1);
__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sqeCount = 1);
```

GenerateCommWriteSQE和GenerateCommWriteWithNotifySQE向sqeCount中初始化与dstRank无关的WQE基础信息。

### 更新WQE

```cpp
__aicore__ inline void UpdateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &sqInfoTensor);
__aicore__ inline void UpdateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &sqInfoTensor);
```

UpdateCommWriteSQE和UpdateCommWriteWithNotifySQE用于设置与dstRank有关的WQE信息。

```cpp
__aicore__ inline void SetCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr, uint64_t rmtAddr, uint32_t length, uint8_t cqe);
__aicore__ inline void SetCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr, uint64_t rmtAddr, uint32_t length, uint64_t notifyAddr, uint64_t notifyData, uint8_t cqe);
```

SetCommWriteSQE和SetCommWriteWithNotifySQE用于设置不同WQE的细节信息，包括：待发送数据的源地址、目的地址、发送长度以及是否需要上报CQE。

### 管理CQ

```cpp
__aicore__ inline void PollCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor, uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear);
```

PollCQ函数的两个主要作用：

1. 读取芯片写入CQE的位置，敲JFC Doorbell，更新芯片CQCI
2. 从CQE结构体中获取entry_idx，该字段含义为指向SQE的最后一个WQEBB，并更新软件SQCI

调用PollCQ函数的实际：

1. 希望检查之前发送通信是否成功的时候
2. 发送WQE前，当前sqDepth深度不足以放下所有WQE时，通过PollCQ更新SQCI以获取更大的SQ队列空间

### 敲Doorbell

```cpp
__aicore__ inline void SendJFSDoorBell(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t sqPi);
__aicore__ inline void SendJFCDoorBell(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, uint32_t cqCi)；
```

SendJFSDoorBell和SendJFCDoorBell分别用于敲JFS Doorbell和敲JFC Doorbell。

### 首次进入通信域清空CQ队列

```cpp
__aicore__ inline void InvalidateCqeStatus(const AscendC::LocalTensor<uint8_t> &cqeInfoTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor);
```

### 填充SQ队列

```cpp
__aicore__ inline void PutCommSQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sendSqeIdx, uint32_t sqPi);
__aicore__ inline void PutCommNotifySQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sendSqeIdx, uint32_t sqPi);
```

### 从win区获取指针/写回指针

```cpp
__aicore__ inline void GetPICI(const AscendC::LocalTensor<uint32_t> &piCiTensor, GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId);
__aicore__ inline void UpdatePICI(const AscendC::LocalTensor<uint32_t> &piCiTensor, GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId);
```

## 以CombineSetup算子为例，算子的流程图

【图】
