## QuantAllReduce <<<>>>直调示例

# 文件目录
```
├── kernel                                  # 算子核函数目录
│    ├── quant_all_reduce_mte_one_shot.h    
│    ├── quant_all_reduce_tiling_data.h     
│    ├── mte_comm.h                         
│    ├── vec_comp.h                         
│    └── utils.h                         
├── quant_all_reduce.cpp                    # <<<>>>核函数入口
├── quant_all_reduce.h                      # <<<>>>核函数入口声明头文件
├── test_quant_all_reduce.cpp               # <<<>>>直调示例代码
├── run.sh                                  # 工程运行脚本
└── README.md                              
```

# 文件改动
新增接口：
quant_all_reduce_mte_one_shot.h 
外部创建mc2Context与tilingData，传入GM地址mc2Context，tilingGM
__aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);
mte_comm.h 
__aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context);

# 测试用例参数

当前示例硬编码了第一个 int8 用例的参数，直接在 `test_quant_all_reduce.cpp` 中定义：

```cpp
int64_t bs = 16;
int64_t hiddenSize = 8192;
int64_t hcclBufferSize = 200;
int64_t workSpaceSize = 256 * 1024 * 1024;
int run_type = 1;

// 数据类型
input_tensor_type = int8_t
scales_type = float32_t
output_type = float16_t
mxfp = 0
ranksize = 2
```

如需修改参数，直接编辑 `test_quant_all_reduce.cpp` 中的 `LaunchOneThreadQuantAllReduce` 函数。

# 工程运行

1. 设置环境变量
```shell
source /usr/local/Ascend/cann/set_env.sh
```

2. 运行脚本
```shell
bash run.sh
```

3. 选项说明

```
-c | --cmake-rebuild     编译开关（默认：ON）
                        ON - 执行编译，OFF - 不执行编译

-v | --soc-version      芯片版本（默认：Ascend310P3）
                        ascend950pr_9599

-i | --install-path     CANN安装路径（可选，source环境后自动检测）

-b | --build-type       编译类型（默认：Debug）
                        Release / Debug

-p | --install-prefix   编译输出目录（默认：./out）

-r | --run-test         运行开关（默认：ON）
                        ON - 运行测试，OFF - 仅编译
```

4. 常用命令示例

```shell
# 编译并运行（默认）
bash run.sh

# 仅编译
bash run.sh -r OFF

# 指定芯片版本编译运行
bash run.sh -v ascend950pr_9599

# Release 模式编译运行
bash run.sh -b Release
```

# 输出结果

程序执行完成后，输出文件位于：
```
./golden/quantallreduce_output_npu_0.bin
./golden/quantallreduce_output_npu_1.bin
```