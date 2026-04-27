# MoeDistributeDispatchV2 (<<<>>> Kernel直调版本)

## 简介

本算子为 MoeDistributeDispatchV2 算子的 Kernel 直调实现，采用 <<<>>> 调用方式直接启动 Ascend C Kernel。

关于 Dispatch 算子的核心功能、计算公式及通信原理，详见 [aclnnMoeDistributeDispatchV3](../../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)。

环境部署、安装步骤等信息参考 [README.md](../../../../README.md)。

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |

## 约束说明

想要完整使能功能，需要满足以下版本要求：
- gcc 9.4.0+
- python 3.9+
- torch>=2.7.1
- cann 9.0.0+

该算子支持的平台为 ascend910_93，默认构建环境为 ascend910b，在 ascend910_93 构建环境下构建，请先执行命令：
```sh
export NPU_ARCH=ascend910_93    
```

## <<<>>> Kernel直调调用流程

### 整体调用链路

```
Python层 (MoeDistributeBuffer)
    │
    ├── 初始化时调用 updateContext 获取通信上下文
    │
    └── 调用 npu_moe_distribute_dispatch_v2 方法
            │
            ↓
        torch.ops.ascend_ops.MoeDistributeDispatchV2
            │
            ↓
        C++ Extension Layer (npu_moe_distribute_dispatch_v2)
            │
            ├── 输入校验 (ValidateMoeDistributeDispatchV2Input)
            │
            ├── 输出Tensor Shape计算与创建
            │
            ├── Tiling数据构造 (calculate_tilingdata)
            │
            ├── 获取NPU Stream
            │
            └── 调用 moe_distribute_dispatch_v2_api
                    │
                    ├── Tensor地址转换
                    │
                    ├── TilingKey计算 (calculate_tilingkey)
                    │
                    ├── Profiling集成
                    │
                    └── 调用 moe_distribute_dispatch_v2_entry
                            │
                            ├── <<<>>> Kernel Launch
                            │       │
                            │       ├── moe_distribute_dispatch_v2_generic
                            │       │
                            │       └── SK模式 (moe_distribute_dispatch_v2_generic_sk)
                            │
                            └── TilingKey分支选择 (moe_dispatch_switch)
                                    │
                                    ↓
                                Ascend C Kernel执行
```

### 目录结构

```
moe_distribute_dispatch_v2/ascend910_93/
├── moe_distribute_dispatch_v2_torch.cpp      # C++ Extension主实现
├── moe_distribute_dispatch_v2_torch.h        # Tensor地址获取辅助函数
├── moe_distribute_dispatch_v2_torch_validate.cpp  # 输入参数校验
├── moe_distribute_dispatch_v2_torch_validate.h
├── moe_distribute_dispatch_v2_entry.cpp      # <<<>>>入口函数
├── moe_distribute_dispatch_v2_entry.h
├── op_kernel/                                # Kernel实现
└── CMakeLists.txt                            # 构建配置
```

## 调用流程详解

### 1. Python层 - MoeDistributeBuffer

MoeDistributeBuffer 类封装了算子调用的完整逻辑，为用户提供简洁的接口。

#### 初始化逻辑

MoeDistributeBuffer 初始化时执行以下操作：

**步骤1：获取分布式信息** - 通过 torch.distributed API 获取当前进程的 rank_id 和 world_size并存入成员变量

**步骤2：获取HCCL通信域名称** - 通过 group._get_backend 获取底层HCCL通信域的名称字符串，用于后续C++层的通信上下文创建

**步骤3：创建通信上下文** - 调用 torch.ops.ascend_ops.updateContext，传入通信域名称和world_size，该接口在C++层通过HCCL API获取各卡的Window地址，打包成 mc2_context Tensor存入成员变量供Kernel调用时使用，同时返回 HCCL Buffer Size

#### HCCL Buffer Size计算

get_ccl_buffer_size 静态方法根据配置参数计算所需的最小HCCL Buffer大小：

- 根据comm_alg类型（fullmesh_v1/fullmesh_v2）选择不同的数据对齐策略
- 计算单个Token的Dispatch和Combine数据占用
- 根据world_size、专家数、TopK等参数计算总Buffer需求
- 返回以MB为单位的Buffer大小，用户可根据返回的Buffersize设置 HCCL_BUFFSIZE 环境变量或自行设置大于计算所得大小的Buffersize

#### Dispatch调用逻辑

npu_moe_distribute_dispatch_v2 方法将用户参数与缓存的上下文信息合并，调用 torch.ops.ascend_ops.MoeDistributeDispatchV2：

- **传入缓存的上下文**：mc2_context、group_name、world_size、rank_id、ccl_buffer_size
- **传入用户参数**：x、expert_ids、moe_expert_num、quant_mode、comm_alg等
- **返回输出**：expand_x、dynamic_scales、expand_idx（assist_info）、expert_token_nums、ep_recv_counts、expand_scales

### 2. C++ Extension层

#### Torch接口注册

通过 TORCH_LIBRARY_FRAGMENT 注册算子签名，定义输入输出参数类型，通过 TORCH_LIBRARY_IMPL 进行绑定。

#### 主函数

主函数 npu_moe_distribute_dispatch_v2 执行以下流程：

**阶段1：输入校验**

将参数打包到 MoeDistributeDispatchV2ValidateParams 结构体，调用 ValidateMoeDistributeDispatchV2Input 执行校验（详见下节）。校验完成后，从validate_params获取计算后的维度信息（bs、h、k）。

**阶段2：输出Tensor Shape计算**

根据rank类型（共享专家卡或MoE专家卡）计算输出Shape：

- 共享专家卡：local_moe_expert_num=1，输出行数a为max_bs * max_shared_group_num
- MoE专家卡：local_moe_expert_num为moe_expert_num除以MoE专家卡数，输出行数a为global_bs * min(local_moe_expert_num, k)

**阶段3：输出Tensor创建**

根据计算出的Shape和配置创建输出Tensor：
- expand_x：行数为a，列数为h，数据类型根据量化模式选择（非量化保持原类型，量化为INT8）
- dynamic_scales：行数为a，数据类型为Float
- assist_info_forcombine：长度为max(bs*k, a*128)，数据类型为Int（存储三元组）
- expert_token_nums：长度为local_moe_expert_num，数据类型为Long
- ep_recv_counts：长度为ep_world_size * local_moe_expert_num，数据类型为Int
- expand_scales：行数为a，数据类型为Float
- new_workspace：16MB大小的Int Tensor，用于核间同步

**阶段4：Tiling数据构造**

调用 calculate_tilingdata 从输入参数和Tensor信息构造Tiling数据：直接将传入的配置参数（ep_world_size、ep_rank_id、moe_expert_num、quant_mode、global_bs等）赋值到结构体对应字段；从输入Tensor获取维度信息（bs、h、k）；从平台API获取AIV核数和UB大小；根据可选Tensor的存在情况判断Mask类型和smoothScale配置。

**阶段5：获取Stream并异步调用**

获取当前NPU Stream，构造lambda函数调用 moe_distribute_dispatch_v2_api，通过 OpCommand::RunOpApiV2 异步执行。

#### 输入校验流程

ValidateMoeDistributeDispatchV2Input 按以下顺序执行校验：

| 步骤 | 校验内容 | 约束 |
|-----|---------|------|
| 1 | EP域参数 | epWorldSize ∈ [2, 768]，epRankId ∈ [0, epWorldSize) |
| 2 | 共享专家参数 | sharedExpertNum和sharedExpertRankNum的组合必须合法 |
| 3 | MoE专家参数 | moeExpertNum ∈ (0, 1024]，且可被MoE专家卡数整除 |
| 4 | 量化模式 | 仅支持0（非量化）和2（PerToken动态量化） |
| 5 | 公共参数 | expertTokenNumsType ∈ {0,1}，特殊专家数量>=0 |
| 6 | Tensor维度 | x必须2D，expert_ids必须2D |
| 7 | 参数范围 | h ∈ [1024, 8192]，bs ∈ (0, 512]，k ∈ (0, 16]（fullmesh_v2上限12） |
| 8 | global_bs约束 | 若非0，必须等于bs*ep_world_size |
| 9 | 数据类型 | x支持BF16/FP16，expert_ids必须Int32，scales必须Float |

校验过程中计算中间参数：localMoeExpertNum、zeroComputeExpertNum等，并判断Mask类型（1D或2D）。

### 3. API层

API层函数 moe_distribute_dispatch_v2_api 负责将Tensor转换为<<<>>>调用所需的地址参数，集成Profiling，并调用下层Entry函数。

#### Tensor地址转换

通过 get_first_tensor_address 函数获取各Tensor的设备地址指针：

- 对于必选Tensor（x、expert_ids、mc2_context、输出Tensor）：直接获取data_ptr
- 对于可选Tensor（scales、x_active_mask、expert_scales、performance_info）：先判断has_value，若存在则获取地址，不存在则传nullptr

#### TilingKey计算

calculate_tilingkey 根据运行配置计算5位十进制TilingKey：

| 基础值 | 10000 |
|-------|-------|
| quantMode | 加到个位（0或2） |
| xType为BF16 | 加10（到十位） |
| scales存在 | 加100（到百位） |
| comm_alg为fullmesh_v2 | 加1000（到千位） |

#### Profiling集成

使用 INIT_ACL_PROF_TENSOR_INFO 宏初始化Profiling信息，调用 aclprofRangePushEx 和 aclprofRangePop 包裹<<<>>>调用，记录算子执行信息（Tensor类型、Shape、核数等）。

### 4. Entry层

Entry层包含<<<>>>调用入口和SK适配，定义在 moe_distribute_dispatch_v2_entry.cpp 和 moe_distribute_dispatch_v2_entry.h 文件中。

#### <<<>>>调用入口

moe_distribute_dispatch_v2_entry 函数使用 <<<>>> 语法启动 moe_distribute_dispatch_v2_generic Kernel，传入TilingKey、Block数量（AIV核数）、Stream、各Tensor地址和TilingData。

#### SK适配

通过 MoeDistributeDispatchV2SkArgs 结构体（定义在entry.h）封装参数，moe_distribute_dispatch_v2_generic_sk 函数支持4种split分割模式，通过 SK_BIND 宏绑定到主Kernel，提高多核并行效率。

### 5. Kernel层

#### TilingKey分支选择

Kernel入口 moe_dispatch_switch 通过switch语句根据TilingKey选择对应的模板实例化：

- **10000系列**：调用 moe_distribute_dispatch_v2 模板（fullmesh_v1通信算法）
- **11000系列**：调用 moe_distribute_dispatch_v2_full_mesh 模板（fullmesh_v2通信算法）

模板参数（输入类型、输出类型、量化模式、smoothScale存在与否）由TilingKey的各位值隐含确定。

#### Kernel总体功能

moe_distribute_dispatch_v2_generic Kernel基于HCCL Window机制实现MoE分发功能：根据expert_ids将Token数据发送到各目标卡的Window，等待所有卡发送完成后从Window读取数据写入输出Tensor。支持PerToken动态量化、Token活跃Mask、Expert活跃Mask等特性。

关于Kernel的详细算法原理、通信流程、数据排布等，详见 [aclnnMoeDistributeDispatchV3](../../../../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)。

## 与aclnnMoeDistributeDispatchV3的区别

### 参数简化

| 参数 | aclnn版本 | <<<>>>版本 |
|-----|----------|-----------|
| epRankId, epWorldSize, groupEp | 用户传入 | MoeDistributeBuffer自动获取 |
| tp域参数 | 支持 | 不支持 |
| totalWinsizeEp大小 | 用户自行配置环境变量 | 可调用get_ccl_buffer_size计算或用户自行配置 |

### 功能限制

| 功能 | aclnn版本 | <<<>>>版本 |
|-----|----------|-----------|
| TP域通信 | 支持 | 不支持 |
| Layered Dispatch | 支持 | 不支持 |
| expertShardType非0 | 支持 | 仅支持0 |
| 静态量化/PerGroup量化 | 支持(部分平台) | 不支持 |

### 量化模式支持

仅支持：非量化(0)、PerToken动态量化(2)

### commAlg支持

支持：""（默认fullmesh_v1）、fullmesh_v1、fullmesh_v2

## 调用示例

完整示例见 tests/moe_distribute_combine_v2/test_combine.py。

基本流程：

1. 初始化分布式环境，创建EP通信域
2. 调用 MoeDistributeBuffer.get_ccl_buffer_size 计算Buffer大小，设置HCCL_BUFFSIZE环境变量
3. 创建 MoeDistributeBuffer 实例
4. 调用 npu_moe_distribute_dispatch_v2 执行分发
5. 调用 npu_moe_distribute_combine_v2 执行组合（使用Dispatch输出）

## 构建说明

构建前设置环境变量：export NPU_ARCH=ascend910_93

通过CMake构建，输出共享库，由PyTorch Extension加载。