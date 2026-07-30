# APACE 系统测试（ST）

## 1. 概述

本目录是 APACE（Ascend PArallel Communication-compute Engine）通算融合算子的**系统测试（System Test, ST）**工程。

ST 的核心作用是**绕过框架（PyTorch/MindSpore 等），直接在 NPU 上调用 kernel 并验证精度**，用于在算子开发阶段快速定位问题：

- **直调验证**：Host 侧 `main.cpp` 通过 `fork` 多进程模拟多 rank，构建 HCCL 通信上下文后用 `<<<>>>` 语法直接 launch apace kernel，不依赖任何算子框架。
- **精度比对**：Python 脚本生成 MXFP8 输入数据与 CPU golden，NPU 输出回拷后用 `torch.allclose` 比对（`rtol=2%, atol=0.2`）。
- **回归保护**：kernel 修改后一键跑通 gen → build → run → verify，确保精度不退化。

当前已实现的 ST 用例：

| 用例 | 目录 | 对应 kernel | 通信模式 |
|:---|:---|:---|:---|
| QuantMatmulAllToAll | `quant_matmul_all_to_all/` | `MatmulAllToAllMxKernelE4M3E4M3`（MXFP8 量化 AllToAll+Matmul 融合） | AllToAll GET（拉结果） |
| AllToAllQuantMatmul | `all_to_all_quant_matmul/` | `AllToAllQuantMatmulKernelE4M3E4M3_Udma`（MXFP8 量化 AllToAll+Matmul 融合） | AllToAll PUT（推数据） |

两个 ST 用例的核心差异：

| | QuantMatmulAllToAll | AllToAllQuantMatmul |
|:---|:---|:---|
| 通信模式 | GET（从其他卡拉回计算结果） | PUT（将本地数据推到其他卡） |
| 切分轴 | N 轴切分（np = N/rankNum） | K 轴切分（Ka = K/rankNum） |
| 参数格式 | `m k n rankNum tileCnt tileSize tailSize localMatmul isPingPong` | `m k n rankNum [mode] [headMSize]` |
| target 名 | `apace_a2a_qmm_st` | `apace_a2a_qmm_udma_st` |
| 精度标准 | rtol=2%, atol=0.2 | rtol=1e-2, atol=1e-2 |
| CSV 列 | `m,k,n,rank_num,tile_cnt,local_matmul` | `m,k,n,rank_num,head_m_size` |

---

## 2. 目录结构

```
st/
├── README.md                          # 本文档
├── CMakeLists.txt                     # ST 顶层构建入口（bisheng 编译器 + ASC 模块 + ops-tensor 拉取）
├── utils/
│   ├── utils.h                        # 日志宏 / ACL 检查宏 / 文件读写工具
│   ├── comm_channel_builder.h         # HCCL channel 创建工具（内存注册 + 建链 + context 拷贝）
│   └── root_info_exchanger.h          # 基于 TCP 的 HcclRootInfo 交换工具（建链 + 广播 + barrier）
├── quant_matmul_all_to_all/           # QuantMatmulAllToAll ST 用例（GET 模式）
│   ├── CMakeLists.txt                 # 用例构建配置（kernel 模式编译 main.cpp）
│   ├── run.sh                         # 一键脚本: gen → build → run → verify
│   ├── README.md                      # 用例详细说明（数据流语义、已知阻塞、编译架构）
│   ├── src/
│   │   ├── main.cpp                   # Host launcher: fork 多 rank + HCCL + tiling + <<<>>> 直调
│   │   └── comm_channel_builder.h     # HCCL channel 创建工具（内存注册 + 建链）
│   └── scripts/
│       ├── gen_data.py                # 生成 MXFP8 输入 + CPU golden（可独立运行）
│       └── verify_result.py           # BF16 精度比对（可独立运行）
└── all_to_all_quant_matmul/           # AllToAllQuantMatmul ST 用例（PUT 模式）
    ├── CMakeLists.txt                 # 用例构建配置（kernel 模式编译 + -DASC_DEVKIT_MAJOR=9）
    ├── run.sh                         # 一键脚本: gen → build → run → verify
    ├── cases.csv                      # 测试用例参数表
    ├── src/
    │   ├── main.cpp                   # Host launcher: fork 多 rank + HCCL + tiling + <<<>>> 直调
    │   └── kernel_launcher.h          # 4 个 __global__ kernel launcher（从 udma_impl.h 抽取）
    └── scripts/
        ├── gen_data.py                # 生成 MXFP8 输入 + CPU golden（可独立运行）
        └── verify_result.py           # BF16 精度比对（可独立运行）
```

---

## 3. 环境依赖

| 依赖 | 说明 |
|:---|:---|
| **CANN** | 9.2.0，需设置 `ASCEND_HOME_PATH` 环境变量指向 toolkit 安装路径 |
| **bisheng 编译器** | `$ASCEND_HOME_PATH/<arch>-linux/ccec_compiler/bin/bisheng`，CMake 自动定位 |
| **ops-tensor** | 提供 blaze / tensor_api 头文件，首次 cmake 时自动从 `gitcode.com/cann/ops-tensor` clone 到 `.third_party/` |
| **Python 3** | 运行 gen_data.py / verify_result.py |
| **PyTorch + NumPy** | 数据生成与精度比对（CPU golden 计算） |
| **NPU 硬件** | Atlas A3（dav-3510 / Ascend950PR），本机需 2 卡或 4 卡 |

环境准备：

```bash
# 1. source CANN 环境（设置 ASCEND_HOME_PATH）
source <CANN_INSTALL_PATH>/set_env.sh

# 2. 确认 python3 + torch + numpy + en_dtypes + ml_dtypes 可用
python3 -c "import torch, numpy; from en_dtypes import float8_e8m0; from ml_dtypes import float8_e4m3fn; print('ok')"
```

---

## 4. 编译方式

### 4.1 一键脚本编译（推荐）

`run.sh` 封装了完整的 gen → build → run → verify 流程，编译是其中的第 2 步：

```bash
cd mc2/common/op_kernel/apace/tests/st/quant_matmul_all_to_all

# 完整流程（含编译 + 运行 + 比对）
bash run.sh

# 仅生成数据，不编译（用于不依赖 kernel 的场景）
bash run.sh --gen-only
```

编译产物路径：`st/build/quant_matmul_all_to_all/apace_a2a_qmm_st`

### 4.2 手动 CMake 编译

如需单独编译（不运行），可手动执行 cmake：

```bash
cd mc2/common/op_kernel/apace/tests/st

cmake -S . -B build
cmake --build build --target apace_a2a_qmm_st --parallel 4
```

### 4.3 编译架构说明

apace tiling 头文件传递包含 CANN `platform_ascendc.h`，在 `-xasc`（kernel 编译模式）下会引入 `__aicore__` / `__gm__` 等 kernel 端定义。因此 `main.cpp` 采用 **kernel 模式（`-xasc`）** 编译，直接包含 kernel 头文件并使用 `<<<>>>` 直调语法。

关键编译选项（见 `quant_matmul_all_to_all/CMakeLists.txt`）：

```cmake
# kernel 模式编译选项
set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-w" "-O3")

# 链接库：ACL 运行时 + HCCL + tiling_api + hccl_fwk（HCCL 内部符号）
set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++)
target_link_libraries(apace_a2a_qmm_st PRIVATE ${COMMON_LIBS} apace_blaze_api)
target_link_libraries(apace_a2a_qmm_st PRIVATE "-Wl,--no-as-needed" hccl_fwk "-Wl,--as-needed")
```

### 4.4 顶层 CMakeLists.txt 做了什么

`st/CMakeLists.txt` 完成以下工作（后续用例可直接 `add_subdirectory` 接入）：

1. 定位 bisheng 编译器并设为 `CMAKE_CXX_COMPILER`
2. 加载 ASC 编译器模块（`find_package(ASC REQUIRED)`）
3. 自动 clone ops-tensor 到 `.third_party/ops-tensor` 并初始化 tensor_api submodule
4. 创建 `apace_blaze_api` INTERFACE 库，提供 blaze / tensor_api 的 include 路径
5. `add_subdirectory(quant_matmul_all_to_all)` 接入具体用例

---

## 5. 使用方法

### 5.1 一键运行

```bash
cd mc2/common/op_kernel/apace/tests/st/quant_matmul_all_to_all

# 默认参数: M=2048 K=3584 N=4096 rankNum=2 tileCnt=4 localMatmul=1
bash run.sh

# 自定义参数: m k n rank_num tile_cnt local_matmul
bash run.sh 2048 3584 4096 4 4 1
```

### 5.2 分步运行

```bash
# 仅生成 CPU golden + 输入数据（不依赖 kernel 编译）
bash run.sh --gen-only 2048 3584 4096 2

# 跳过编译（复用已有二进制），仅运行 + 比对
bash run.sh --skip-build 2048 3584 4096 2 4 1

# 仅精度比对（需已有 output/{r}/npu_out.bin）
bash run.sh --verify-only 2048 4096 2
```

### 5.3 直接运行可执行文件

编译成功后，也可直接调用二进制（需手动准备 input/output 目录）：

```bash
cd build/quant_matmul_all_to_all

# 参数顺序: m k n rankNum tileCnt tileSize tailSize isLocalDelayed isPingPong
./apace_a2a_qmm_st 2048 3584 4096 2 4 512 0 1 0
```

参数说明：

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `m` | 矩阵 A 行数 | 2048 |
| `k` | 矩阵 A 列数（完整 K，不切分） | 3584 |
| `n` | 矩阵 B 列数（总 N，按 rankNum 切分） | 4096 |
| `rankNum` | rank 数量（支持 2 或 4） | 2 |
| `tileCnt` | M 轴规则 tile 数量 | 4 |
| `tileSize` | 每个 tile 的 M 行数 | M/tileCnt |
| `tailSize` | 尾部 tile 的 M 行数，0=无尾块 | 0 |
| `isLocalDelayed` | 是否延迟本地 batch 到末尾（0/1） | 1 |
| `isPingPong` | 是否使用 ping-pong shmem（0/1） | 0 |

### 5.2 AllToAllQuantMatmul（PUT 模式）

#### 一键运行

```bash
cd mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul

# 运行 CSV 全部 case
bash run.sh

# 运行 CSV 第 1 行
bash run.sh 1

# 运行 CSV 第 1~3 行
bash run.sh 1-3

# 命令行模式: m k n rank_num head_m_size
bash run.sh --cli 2048 3584 4096 2 512
```

#### 分步运行

```bash
# 仅生成 CPU golden + 输入数据
bash run.sh --gen-only 2048 3584 4096 2

# 跳过编译，仅运行 + 比对
bash run.sh --skip-build 2048 3584 4096 2 512

# 仅精度比对（需已有 output/{r}/npu_out.bin）
bash run.sh --verify-only 2048 4096 2
```

#### 直接运行可执行文件

```bash
cd build/all_to_all_quant_matmul

# 参数顺序: m k n rankNum [mode] [headMSize]
./apace_a2a_qmm_udma_st 2048 3584 4096 2 precision 512
```

| 参数 | 说明 | 默认值 |
|:---|:---|:---|
| `m` | 矩阵 A 行数 | 2048 |
| `k` | 矩阵 A 列数（总 K，按 rankNum 切分） | 3584 |
| `n` | 矩阵 B 列数 | 4096 |
| `rankNum` | rank 数量（支持 2 或 4） | 2 |
| `mode` | 运行模式：`precision`（精度验证）/ `perf`（性能测试） | precision |
| `headMSize` | M 轴长块大小，M % headMSize 的余数为尾块 | 512 |

#### 参数约束

- `K % rankNum == 0`（AllToAll 沿 K 轴切分）
- `CeilDiv(K/rankNum, 64) % 2 == 0`（scale 对齐要求）
- `K % 32 == 0`（MXFP8 量化对齐要求）

#### cases.csv

```
m,k,n,rank_num,head_m_size
2048,3584,4096,2,512
2048,3584,4096,4,512
4096,3584,4096,2,512
1024,3584,4096,2,512
4096,3584,4096,4,512
4040,8192,4096,4,512
2048,3584,4096,2,256
```

---

## 6. 数据生成与精度比对

`gen_data.py` 和 `verify_result.py` 可完全独立运行，不依赖 kernel 编译，适合在 kernel 未就绪时先验证数据流。

### 6.1 QuantMatmulAllToAll

#### 生成数据

```bash
cd quant_matmul_all_to_all
python3 scripts/gen_data.py 2048 3584 4096 2
# 参数: m k n rank_num
```

生成内容（写入 `input/{rank_id}/` 和 `output/{rank_id}/`）：

| 文件 | 说明 |
|:---|:---|
| `input/{r}/input_a.bin` | FP8 E4M3FN 矩阵 A `[M, K]` |
| `input/{r}/input_b.bin` | FP8 E4M3FN 矩阵 B `[K, N]`（DNExtLayoutPtn 列主序） |
| `input/{r}/input_scaleA.bin` | FP8 E8M0 scaleA `[M, ceil(K/64), 2]` |
| `input/{r}/input_scaleB.bin` | FP8 E8M0 scaleB `[ceil(K/64), 2, N]`（ScaleBDNLayoutPtn） |
| `output/{r}/cpu_output.bin` | CPU golden `[rankNum*M, N/rankNum]` BF16 |

#### 精度比对

```bash
python3 scripts/verify_result.py 2048 4096 2 ./output
# 参数: m n rank_num [base_dir]
```

精度标准：MXFP8 量化 matmul + BF16 输出，`rtol = 2%`，`atol = 0.2`（`torch.allclose`）。

### 6.2 AllToAllQuantMatmul

#### 生成数据

```bash
cd all_to_all_quant_matmul
python3 scripts/gen_data.py 2048 3584 4096 2
# 参数: m k n rank_num（K 沿 rankNum 切分，Ka = K/rankNum）
```

生成内容（写入 `input/{rank_id}/` 和 `output/{rank_id}/`）：

| 文件 | 说明 |
|:---|:---|
| `input/{r}/input_a.bin` | FP8 E4M3FN 矩阵 A `[M*rankNum, Ka]`（每 rank 持有完整 M 行） |
| `input/{r}/input_b.bin` | FP8 E4M3FN 矩阵 B `[N, K]`（所有 rank 相同，DNExtLayoutPtn 列主序） |
| `input/{r}/input_scaleA.bin` | FP8 E8M0 scaleA `[M*rankNum, ceil(Ka/64), 2]` |
| `input/{r}/input_scaleB.bin` | FP8 E8M0 scaleB `[N, K/32]`（ScaleBDNLayoutPtn 列主序） |
| `output/{r}/cpu_output.bin` | CPU golden `[M, N]` BF16 |

#### 精度比对

```bash
python3 scripts/verify_result.py 2048 4096 2 ./output
# 参数: m n rank_num [base_dir]
```

精度标准：MXFP8 量化 matmul + BF16 输出，`rtol = 1e-2`，`atol = 1e-2`（`torch.allclose`）。

---

## 7. 已知限制

无。两个 ST 用例均已跑通，7/7 case ALL PASS（详见各用例目录下的 `RUN_LOG.md`）。

`gen_data.py` / `verify_result.py` 可独立运行，不依赖 kernel 编译。
