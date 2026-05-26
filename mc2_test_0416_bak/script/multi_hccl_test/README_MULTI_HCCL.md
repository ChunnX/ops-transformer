# 多通信域场景算子混合测试
该场景主要涉及在多通信域场景下，不同通信域分别执行mc2算子（包括基础通算融合类mc2算子和moe类mc2算子）及hccl算子，这里实现了两个调用demo，其中：
- test_multi_hcommResource_basic.py，为两个通信域内分别调用两个基础类通算融合mc2算子示例；
- test_multi_hcommResource_moe.py，为两个通信域内，分别调用一次moe类融合算子（dispatchv2+combinev2）及一个hccl纯通信算子的示例
## 代码适配
替换对应调用算子时，代码适配主要涉及如下内容：
- 通信域分配：
	- 若两个通信域内均调用单通信域算子，则参考test_multi_hcommResource_basic.py内通信域建立流程，无需做另外修改
    - 若涉及双通信域算子调用（dispatch/ combine/ combineAddRmsNorm/ alltoall_allgather_bmm/ bmm_reducescatter_alltoall），则参考test_multi_hcommResource_basic.py内group_moe通信域的建立流程
- 算子输入构造：
	参考对应算子资料pytorch接口调用示例内输入构造部分，进行对应适配。moe类算子需注意：
    - dev_num(worldsize)配置，需要小于总rank数，即配置为实际分配的group_moe内rank总数
    - 对应的专家数及专家卡数配置，需要满足算子约束
- 算子接口调用：
	参考对应算子资料pytorch接口调用示例内接口调用部分，进行对应适配