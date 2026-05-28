# aclnn开发接口列表

无论是CANN内置算子或自定义算子，均可通过aclnn API直调算子，无需提供IR（Intermediate Representation）定义。

本章提供了**实现aclnn API**所需的**框架基础能力接口**，比如算子执行器（opExecutor）处理、数据类型/格式/shape等，具体参见下表罗列的接口、常用宏和常用类等。

**头文件说明**：调用本章接口时请按实际情况include依赖的头文件，头文件路径为\$\{INSTALL\_DIR\}/include目录。其中\$\{INSTALL\_DIR\}请替换为CANN软件安装后的文件存储路径。若安装Ascend-cann-toolkit软件包，以root安装举例，安装后文件存储路径为`/usr/local/Ascend/cann`。

**表 1**  框架能力接口列表

| 接口分类 | 说明 | 所属头文件 |
| --- | --- | --- |
| [bfloat16](bfloat16.md) | 详细介绍了bfloat16数据类型在CPU侧的实现类。 | aclnn/opdev/bfloat16.h |
| [common_types](common_types.md) | 详细介绍了aclTensor、aclScalar等基础的aclnn数据结构。 | aclnn/opdev/common_types.h |
| [data_type_utils](data_type_utils.md) | 提供了DataType相关的基础接口，例如获取指定DataType的size等。 | aclnn/opdev/data_type_utils.h |
| fast_vector | 详细介绍了FastVector数据类型，该类型为aclnn中实现的高效vector数据结构。<br>**说明：该头文件定义的接口均为预留接口，开发者无需关注。** | aclnn/opdev/fast_vector.h |
| [format_utils](format_utils.md) | 提供了Format相关的基础接口。 | aclnn/opdev/format_utils.h |
| [fp16_t](fp16_t.md) | 详细介绍了float16数据类型在CPU侧的实现类。 | aclnn/opdev/fp16_t.h |
| [framework_op](framework_op.md) | 详细介绍了框架对外提供的从host侧到device侧拷贝能力。 | aclnn/opdev/framework_op.h |
| make_op_executor | 对外提供初始化aclOpExecutor的宏声明。<br>**说明：该头文件定义的接口均为预留接口，开发者无需关注。** | aclnn/opdev/make_op_executor.h |
| [object](object.md) | 详细介绍了aclnn中aclTensor等基础数据结构的基类Object类，用于重载实现new、delete方法。 | aclnn/opdev/object.h |
| [op_arg_def](op_arg_def.md) | 详细介绍了OpArgContext类，并对外提供OP_INPUT等宏声明。 | aclnn/opdev/op_arg_def.h |
| [op_cache](op_cache.md) | 详细介绍了OpExecCache及相关类，用于完成aclnn缓存，提升运行性能。 | aclnn/opdev/op_cache.h |
| [op_cache_container](op_cache_container.md) | 详细介绍了带LRU淘汰机制的aclnn缓存容器。 | aclnn/opdev/op_cache_container.h |
| [op_config](op_config.md) | 提供了算子运行时相关的配置信息，如确定性计算开关等。 | aclnn/opdev/op_config.h |
| [op_def](op_def.md) | 定义基础枚举及常量，例如精度模式OpImplMode等。 | aclnn/opdev/op_def.h |
| [op_dfx](op_dfx.md) | 详细介绍了DfxGuard类，用于接口打印及上报profiling。 | aclnn/opdev/op_dfx.h |
| [aclnn返回码](aclnn返回码.md) | 定义了aclnn错误码。 | aclnn/opdev/op_errno.h |
| [op_executor](op_executor.md) | 详细介绍了aclOpExecutor类。 | aclnn/opdev/op_executor.h |
| [op_log](op_log.md) | 定义aclnn中日志打印宏。 | aclnn/opdev/op_log.h |
| [platform](platform.md) | 详细介绍了PlatformInfo类，用于存放SOC平台信息。 | aclnn/opdev/platform.h |
| [pool_allocator](pool_allocator.md) | 详细介绍了PoolAllocator类，用于实现aclnn内部的CPU内存池。 | aclnn/opdev/pool_allocator.h |
| [shape_utils](shape_utils.md) | 提供了shape相关的基础操作，例如shape打印等。 | aclnn/opdev/shape_utils.h |
| [small_vector](small_vector.md) | 详细介绍了SmallVector类，该类为aclnn中实现的高效vector数据结构，主要针对已知数据量较小的场景。 | aclnn/opdev/small_vector.h |
| [tensor_view_utils](tensor_view_utils.md) | 提供了对于View类的基础操作，例如判断aclTensor是否连续等。 | aclnn/opdev/tensor_view_utils.h |
| [data_type_utils](data_type_utils-6.md) | 提供了DataType相关的基础接口，例如判断指定DataType是否为整数类型等。 | aclnn/opdev/op_common/data_type_utils.h |
| [aicpu_args_handler](aicpu_args_handler.md) | 提供了AI CPU相关的组合计算任务的处理逻辑，例如拼接计算任务相关的参数等。 | aclnn/opdev/aicpu/aicpu_args_handler.h |
| [aicpu_ext_info_handle](aicpu_ext_info_handle.md) | 提供了AI CPU相关的计算任务拓展参数的处理逻辑，例如拼接解析拓展参数的接口。 | aclnn/opdev/aicpu/aicpu_ext_info_handle.h |
| [aicpu_task](aicpu_task.md) | 提供了AI CPU任务设置、下发等逻辑，例如设置调用哪个AI CPU算子，设置算子输入、输出等接口。 | aclnn/opdev/aicpu/aicpu_task.h |
| [aicpu_utils](aicpu_utils.md) | AI CPU任务需要的一些公共接口。 | aclnn/opdev/aicpu/aicpu_utils.h |

**表 2**  常用宏表

| 宏名称 | 说明 | 所属头文件 |
| --- | --- | --- |
| [DFX_IN](DFX_IN.md) | 在L2_DFX_PHASE_1中，用于打包所有的host侧API输入参数。 | aclnn/opdev/op_dfx.h |
| [DFX_OUT](DFX_OUT.md) | 在L2_DFX_PHASE_1中，用于打包所有的host侧API输出参数。 | aclnn/opdev/op_dfx.h |
| [L0_DFX](L0_DFX.md) | 必须在host侧API L0接口中使用，用于接口及L0接口入参打印。 | aclnn/opdev/op_dfx.h |
| [L2_DFX_PHASE_1](L2_DFX_PHASE_1.md) | 必须在一阶段接口最前方调用，用于接口及一阶段入参打印。 | aclnn/opdev/op_dfx.h |
| [L2_DFX_PHASE_2](L2_DFX_PHASE_2.md) | 必须在二阶段接口最前方调用，用于接口打印。 | aclnn/opdev/op_dfx.h |
| [OP_TYPE_REGISTER](OP_TYPE_REGISTER.md) | 必须在L0接口最开始处使用，用于注册L0算子。 | aclnn/opdev/op_dfx.h |
| [OP_ATTR](OP_ATTR.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子属性参数。 | aclnn/opdev/op_arg_def.h |
| [OP_EMPTY_ARG](OP_EMPTY_ARG.md) | ADD_TO_LAUNCHER_LIST_AICORE中，用于占位一个空的输入或输出。 | aclnn/opdev/op_arg_def.h |
| [OP_INPUT](OP_INPUT.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输入aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_MODE](OP_MODE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子运行选项，例如是否使能HF32。 | aclnn/opdev/op_arg_def.h |
| [OP_OUTPUT](OP_OUTPUT.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输出aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_OUTSHAPE](OP_OUTSHAPE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，针对第三类算子，设置存放输出shape的aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_OPTION](OP_OPTION.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子指定的精度模式。 | aclnn/opdev/op_arg_def.h |
| [OP_WORKSPACE](OP_WORKSPACE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子显式指定的workspace参数。 | aclnn/opdev/op_arg_def.h |
| [CREATE_EXECUTOR](CREATE_EXECUTOR.md) | 创建一个UniqueExecutor对象，该对象为aclOpExecutor的生成工厂类。 | aclnn/opdev/make_op_executor.h |
| [INFER_SHAPE](INFER_SHAPE.md) | 针对指定算子，运行其infershape函数，推导输出shape。 | aclnn/opdev/make_op_executor.h |
| [ADD_TO_LAUNCHER_LIST_AICORE](ADD_TO_LAUNCHER_LIST_AICORE.md) | 创建某个AICore算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。 | aclnn/opdev/make_op_executor.h |
| [OP_ATTR_NAMES](OP_ATTR_NAMES.md) | String类型的vector，打包AI CPU算子的字符类型属性。 | aclnn/opdev/aicpu/aicpu_task.h |
| [ADD_TO_LAUNCHER_LIST_AICPU](ADD_TO_LAUNCHER_LIST_AICPU.md) | 创建某个AI CPU算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。 | aclnn/opdev/aicpu/aicpu_task.h |

**表 3**  常用class和struct表

| class/struct名称 | 说明 | 所属头文件 |
| --- | --- | --- |
| aclOpExecutor | 用于表示算子执行器，记录整个host侧API运行信息的上下文结构，如L2接口执行过程中的计算图、L0算子launch子任务、workspace地址和大小等信息。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[op_executor](op_executor.md)。 | aclnn/opdev/op_executor.h |
| aclTensor | 用于表示一个张量对象，包括描述张量的shape、dtype、format、address等信息，数据可以放在host侧或device侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclScalar | 用于表示一个标量对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclTensorList | 用于表示一组aclTensor类型组成的列表对象。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclScalarList | 用于表示一组aclScalar类型组成的列表对象。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclBoolArray | 用于表示一个布尔类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclIntArray | 用于表示一个int64_t类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclFloatArray | 用于表示一个fp32类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclFp16Array | 用于表示一个fp16类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| aclBf16Array | 用于表示一个bf16类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types.md)。 | aclnn/opdev/common_type.h |
| SmallVector | 该类使用内部内存池实现vector容器，基础功能与C++标准库中std::vector容器相同，无需每次扩容都申请内存，避免影响性能。其定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[small_vector](small_vector.md)。 | aclnn/opdev/small_vector.h |
| OpExecMode | 用于表示算子运行模式的枚举类，定义参见[OpExecMode](OpExecMode.md)。 | aclnn/opdev/op_def.h |
| OpImplMode | 用于表示算子精度模式的枚举类，定义参见[OpImplMode](OpImplMode.md)。 | aclnn/opdev/op_def.h |

> **SmallVector类使用说明**：
>
> - op::FVector：本质是存储容量长度为8的SmallVector。
>
>   ```cpp
>   namespace op {
>   template<typename T, size_t N = 8>
>   using FVector = op::internal::SmallVector<T, N, op::internal::PoolAllocator<T>>;
>   }
>   ```
>
> - op:Strides：本质是存储容量长度为25的FVector，元素类型int64_t，存储stride信息。
>
>   ```cpp
>   namespace op {
>   constexpr uint64_t MAX_DIM_NUM = 25;
>   using Strides = FVector<int64_t, MAX_DIM_NUM>;
>   }
>   ```
>
> - op::ShapeVector：本质是存储容量长度为25的FVector，元素类型int64_t，存储shape信息。
>
>   ```cpp
>   namespace op {
>   constexpr uint64_t MAX_DIM_NUM = 25;
>   using ShapeVector = FVector<int64_t, MAX_DIM_NUM>;
>   }
>   ```
