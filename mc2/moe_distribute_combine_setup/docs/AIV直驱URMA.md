# AIV直驱URMA

## 基础知识

URMA通信基于“Jetty抽象”[^1]，一个Jetty对应一个JFS-JFC资源对。

简单而言，JFS（Jetty For Send）中包括一个SQ队列和JFS Doorbell，前者用于写入WQE（Working Queue Element），后者用于通知硬件芯片处理WQE。WQE在部分地方也可能被称为SQE（Send Queue Element），因此SQ也会被称为WQ。

同理，JFC（Jetty For Completion）也包括一个CQ队列和JFC Doorbell，前者存放CQE（Completion Queue Element），后者用于将软件已经完成处理的CQE通知硬件芯片。

[^1]：https://mp.weixin.qq.com/s/2u47eHAsFX5j5wC5boezjQ

## URMA芯片的两个队列

URMA通信中有两个重要的队列，分别是SQ队列（Send Queue）和CQ队列（Completion Queue）。

两者都是循环队列，都包括一个PI指针（Producer index）和CI指针（Consumer Index）。

【picture】

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

通过WQE字段中的opcode可以设置Normal SQE的通信语义。当前支持的通信语义包括`Send`、`Write`、`Write with notify`、`Read`等等。

当前DispatchSetup中使用`Write`语义，CombineSetup中使用`Write with notify`语义，其余语义暂未实现。

#### Write SQE

Write SQE结构体，占用一个WQEBB：

