## 一、profiling分析

case:
query: (1, 4096, 64, 128)
cmp_key: (1, 1024, 64, 128)
topk=512
当前分核不均匀，0核和尾核性能差距比较大，**重新评估计算量修改分核**。

![image](./78f523e4-2906-4c26-8445-479730c54cb5.png)

## 二、流水分析

由于当前reduceSum对N1轴，softmax对K轴，当前复用原先V0/V1分别处理P和SY，当前SY去除计算操作，只保留softmax搬运，P保持原先计算逻辑。

主要问题：
当前实现P的计算量没有被切分到两个Vec核，切K轴会导致online softmax update，切N1轴会导致atomic_add累加引入确定性。

![image](./8bbc9874-3a4c-489e-811d-a86a44745064.png)

![image](./64e7e5a0-1cb2-4c1a-8cf2-361da63cb6b5.png)


