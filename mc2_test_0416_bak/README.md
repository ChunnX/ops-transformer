# mc2_torchapi

### 1、使用说明：
当前框架按照aclnn接口维度来区别算子，python文件，用例，CPP均通过aclnn接口名来唯一确定。
每个接口对应一个用例表格，表格内部通过页签来区别用例级别，level0 ,level1...等
另外david 投片阶段由于仿真model的特殊情况，部分代码会根据--soc 做一些特殊处理。
david 上当前需要增加 --soc 910D
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --scene cpu --sheet level0  --num 1 --soc 910D


**1、单算子精度测试：**

以excel/aclnnAllGatherMatmul.xlsx表中的AllGatherMatmul场景的第1个case用例为例，执行命令如下：

```python
# 生成cpu真值的output文件
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene cpu
# 通过三方比对进行mc2精度判断
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene npu

# 结果文件
aclnnAllGatherMatmul_result.csv 精度测试结果
```
测试程度正常结束截图如下 ，即有精度比较结果（PASS或FAIL）

![img](http://image.huawei.com/tiny-lts/v1/images/97a890a85d45aebae1a835e9b755fcd1_921x95.png)

**注：**
**1、allgather/allreduce_arn场景无需cpu结果，只需执行步骤二（--scene npu）**
**2、allreduce_arn场景精度比对需开启确定性计算（export HCCL_DETERMINISTIC=true）**
**3、allreduce_arn场景依赖AscendSpeed，请预先下载安装：https://gitee.com/ascend/AscendSpeed**
```
脚本参数说明：
optional arguments:
  -h, --help            show this help message and exit
  --xlsx XLSX           xlsx name (default: None)
  --sheet SHEET         测试表格sheet页名字：如level0,level1等
  --num NUM [NUM ...]   若num=x，则测试表格中的第x个用例；若num=i j，则测试表格中第i个到第j个的所有用例(仅支持one_hccl场景)
                        (default: None)
  --scene SCENE         仅支持cpu/npu/graph/performance/one_hccl/one_hccl_npu/one_hccl_graph/determin
                        (default: npu)
  --world_size WORLD_SIZE
                        当scene=one_hccl时，需要此参数 (default: 8)
```

多用例批跑精度测试，以执行aclnnAllGatherMatmul.xlsx内，level0页签下第1到200个case为例，执行命令如下：
```python
# 批量生成cpu真值的output文件
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 200 --scene one_hccl_cpu
# 批量进行mc2单算子模式：精度+性能比对
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 200 --scene one_hccl
# 批量进行mc2单算子模式：精度比对
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 200 --scene one_hccl_npu
# 批量进行mc2图模式：精度比对
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 200 --scene one_hccl_graph
# 双通信域场景(需指定ep域和tp域大小)
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 10 --scene one_hccl --ep 4 --tp 2 --world_size 8
```
**2、单算子性能测试+精度测试**

```
# 环境锁频
  1、下载testcase包，取包路径:与driver/firmware相同，包名：Ascend910B-testcase-xxx.tar.gz
  2、解压testcase包后拷贝Lock_Fre_8p.sh到test/lib/host目录下
  3、执行以下命令：bash Lock_Fre_8p.sh
# 同时采集小算子级联和MC2的性能/内存数据，另外也会进行精度测试
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene performance
# 910c上allgather_mm及mm_reducescatter算子性能测试(性能标杆计算方式与910B不同)：
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene performance --soc 910C

# 结果文件
aclnnAllGatherMatmul_result.csv 精度测试结果
aclnnAllGatherMatmul_performance_result.csv 性能测试汇总结果（含性能/内存）
hccl_aclnnAllGatherMatmul_pro_result.csv  仅小算子性能和内存数据
mc2_aclnnAllGatherMatmul_pro_result.csv 仅mc2性能和内存
```
采集profiling文件默认在执行脚本当前目录，脚本会同时采集小算子级联和MC2的profiling文件。

其中**hccl_mm**表示小算子级联的profiling，**mc2**表示mc2算子的profiling。

![img](http://image.huawei.com/tiny-lts/v1/images/a9f1c4d573227a71e5be3b651133778f_716x77.png)

备注：各算子性能标准
1、Ascend910_93上alltoall_allgather_bmm/bmm_reducescatter_alltoall算子性能标准为:
a、掩盖率=(小算子耗时-mc2耗时)/min(小算子级联中计算耗时，小算子级联中通信耗时)
b、大小shape定义：
大shape: 小算子级联耗时>2000us
小shape: 小算子级联耗时<2000us
c、性能达标标准：
客户shape: 大shape掩盖率>20%, 小shape掩盖率>10% 算pass
泛化shape: 相比小算子级联不劣化
2、除1外算子性能标准为:
a、客户shape: 小算子级联耗时/mc2算子耗时>1.x倍(各算子标准不同)
b、泛化shape: 相比小算子级联不劣化

**3、算子入图精度测试：**

```python
# 生成cpu真值的output文件
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene cpu
# 通过三方比对进行mc2精度判断
python3 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene graph
```

【注意】：

1、此脚本在生成数据时，支持多进程并发，可以提高数据生成效率；

2、在验证图模式时，**xlsx表格中必须有graph_type字段**，其中1表示静态shape图模式，2表示动态shape图模式，0/空值/无graph_type字段则表示单算子模式

**4、ACLNN直调场景测试：**
```python
# 生成cpu真值的output文件
python3.8 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene cpu
# 生成小算子级联结果，并保存输入bin
python3.8 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene aclnn_golden
# 调用可执行文件（allreduce_aarch）生成mc2在npu上计算结果，保存输出bin并比较
python3.8 mc2_test.py --xlsx excel/aclnnAllGatherMatmul.xlsx --sheet level0 --num 1 --scene aclnn
```
【注意】：

1、环境信息不同，需在cpp目录下重新编译对应算子的可执行文件
2、allgather场景采用双千分之五比对方式，无需生成cpu

**5、910C 跨server场景（alltoall_allgather_bmm & bmm_reducesatter_alltoall）测试：**
```python
# 分别在各server上设置HCCL环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=${本机IP}
# 分布式训练场景下，HCCL会使用Host服务器的部分端口进行集群信息收集，需要操作系统预留该部分端口
sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015
# 执行mc2用例并进行精度比较
# 主节点命令：
python3.8 mc2_test.py --xlsx excel/aclnnAlltoAllAllGatherBatchMatMul.xlsx --sheet level0 --num 1 --scene npu --world_size 8 --server_num 2 --server_index 0 --dev_num 4 --master_ip ${本机IP}
# 子节点命令：
python3.8 mc2_test.py --xlsx test_case/aclnnAlltoAllAllGatherBatchMatMul.xlsx --sheet level0 --num 1 --scene npu --world_size 8 --server_num 2 --server_index 1 --dev_num 4 --master_ip ${主节点IP}
## 其中：
# tp: tp_size
# ep: ep_size
# server_num: 通信节点数量
# server_index: 本节点序号
# dev_num: 每个节点上使用的device数量
# master_ip: 主节点IP
# master_port: 端口号
```
**6、MC2精度2.0说明**
```python
# 生成cpu真值的output文件
python3.8 mc2_test.py --xlsx excel/aclnnMatmulReduceScatter.xlsx --sheet level0 --scene one_hccl_cpu --num 1 2
# 生成小算子级联结果/mc2在npu计算结果，并保存输入bin
图模式
python3.8 mc2_test.py --xlsx excel/aclnnMatmulReduceScatter.xlsx --sheet level0 --scene one_hccl_mc2_save_graph --num 1 2
单算子模式
python3.8 mc2_test.py --xlsx excel/aclnnMatmulReduceScatter.xlsx --sheet level0 --scene one_hccl_mc2_save_npu --num 1 2

# 调用脚本生成精度比较结果
python3 script/op_precision_stat.py --data_path=bin_path
#生成详细直方图
python3 script/trans_graph.py --case_file=test_result/MC2.xlsx
#生成最终详细报表
python3 script/anayls.py

结果文件会在result文件夹生成MC2_result.xlsx，结果列显示PASS或者WARNING均为通过。
```
**7 david结果上传看板指导**
```
1:跑任务前请清空历史结果文件，aclnnAllGatherMatmul_result.csv，case_name.csv等，当前无法判断文件生成时间
2：上传命令
python3 upload_result.py op_name(如aclnnAllGatherMatmul，dispath等算子op_name为moe,eplb等算子为eplb)
FUZZ 场景
python3 upload_result.py op_name loop(op_name 为fuzz_xx 如fuzz_quantreducescatter loop为fuzz个数如51227 )
如果fuzz场景有部分轮次失败 上传失败方法
python3 upload_result.py op_name loop fail_num(op_name 为fuzz_xx 如fuzz_quantreducescatter loop为fuzz个数如51227 fail_num为失败个数比如200 )
如果想收集所有用例结果中包含未生成精度结果的用例，需要提前准备一个case_name.csv，里面存放用例名，采用mc2_test框架，会自动生成，其它文件需要自行准备.

3:查看结果
http://100.102.207.234:8008/op_alpha/?%E7%BB%84%E4%BB%B6_query=mc2_test
```

### 2、后续场景扩展

新增算子类，以aclnnMatmulAllReduce为例

（1）新建aclnnMatmulAllReduce.py

（2）新增aclnnMatmulAllReduce 图模式的类 MM_ALLREDUCE_GRAPH_Model

（3）新增aclnnMatmulAllReduce 单算子的类 MM_ALLREDUCE

（4）根据算子特点，重写以下函数

```
# 必选
get_npu         NPU场景，MC2算子
get_hccl_mm     NPU场景，小算子级联（部分算子依赖get_tmp_tensor）
get_cpu         CPU场景，小算子级联
get_graph       NPU场景，图模式
在 op_class/__init__.py 中 导入 类名 from . import aclnnMatmulAllReduce

# 可选
get_tmp_tensor  算子依赖临时tensor的生成
compare         精度比对方式，默认是三方比对；若子类有其他比对方式，可重写compare实现逻辑
get_tmp_tensor_memory  针对有tmp_tensor场景，为了方便性能测试（获取内存数据）
```


# CPP 部分说明：
（1）cpp目录下的 make.sh 会传入一个入参 SOC_VERSON 默认不传，编译的是obp目录下的代码；
（2）如果传入 david， 会把 SOC_VERSON 传入MakeFile，MakeFile 中会定义 CXXFLAGS += -DDAVID_FAST_MODEL，代码中的编译选项会生效。会编译代码中所有 #ifdef  DAVID_FAST_MODEL 范  围内的代码，主要用于区别仿真环境无法在host侧使用rtMalloc 和rtFree函数，回片后需要将此选项去掉；另外新增数据类型加了 CXXFLAGS += -DDAVID  定义，代码中David 上的新增数据类型用 #ifdef  DAVID 包括，如果不加，OBP上不支持的数据类型编译不过。
（3）如果回片将 CXXFLAGS += -DDAVID_FAST_MODEL去掉后重新编译
示例:  bash ./make.sh  生成OBP的可执行文件
      bash ./make.sh david 生成 David 的可执行文件

# Dispatch/Combine/CombineAddRmsNorm算子测试
（1）测试脚本路径：./script/dispatch_combine_test
（2）测试脚本介绍：参考wiki https://wiki.huawei.com/domains/120174/wiki/227727/WIKI202505136808360