## MteAllGather <<<>>>直调示例

通过MTE通信拿到其它卡数据，仅完成AllGather功能（不做Dequant和ReduceSum），将各卡数据按rank顺序拼接到output GM。

# 文件目录
```
├── kernel                                  # 算子核函数目录
│    ├── mte_all_gather_mte_one_shot.h      # AllGather核心逻辑
│    ├── mte_all_gather_tiling_data.h       # Tiling数据定义
│    ├── mte_comm.h                         # MTE通信相关实现
│    └── utils.h                            # 工具函数
├── mte_all_gather.cpp                      # <<<>>>核函数入口
├── mte_all_gather.h                        # <<<>>>核函数入口声明头文件
├── test_mte_all_gather.cpp                 # <<<>>>直调示例代码
├── generate_golden.py                      # Golden数据生成（AllGather）
├── run.sh                                  # 工程运行脚本
└── README.md
```

# 核函数签名
```cpp
// <<<>>>调用函数
void mte_all_gather_demo(uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* output,
    uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);
```

# 数据布局
- 输入 x: `[bs, hiddenSize]` int8_t，每张卡各自的量化数据
- 输出 output: `[rankDim, bs, hiddenSize]` int8_t，按rank顺序拼接的AllGather结果
  - output[rank_i] = rank_i 卡上的 x 数据

# 测试用例参数

当前示例硬编码了参数，在 `test_mte_all_gather.cpp` 中定义：

```cpp
int64_t bs = 16;
int64_t hiddenSize = 8192;
int64_t hcclBufferSize = 200;
int64_t workSpaceSize = 256 * 1024 * 1024;

// 数据类型
input_tensor_type = int8_t
output_type = int8_t (与输入类型一致)
rankDim = 4
```

# Golden 数据

## 生成 Golden 数据

```shell
python3 generate_golden.py
```

生成的文件：
```
./golden/mteallgather_input_x_0.bin
./golden/mteallgather_input_x_1.bin
./golden/mteallgather_input_x_2.bin
./golden/mteallgather_input_x_3.bin
./golden/mteallgather_output_cpu_0.bin  (AllGather golden: [rankDim*bs, hiddenSize] int8)
```

# 工程运行

1. 设置环境变量
```shell
source /usr/local/Ascend/cann/set_env.sh
```

2. 运行脚本
```shell
bash run.sh
```

# 输出结果

程序执行完成后，输出文件位于：
```
./golden/mteallgather_output_npu_0.bin
./golden/mteallgather_output_npu_1.bin
./golden/mteallgather_output_npu_2.bin
./golden/mteallgather_output_npu_3.bin
```
