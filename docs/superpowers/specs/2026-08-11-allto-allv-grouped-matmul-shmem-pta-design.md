# AllToAllV GroupedMatMul SHMEM PTA 验证桥设计

## 1. 背景

当前 Ascend950 `AlltoAllvGroupedMatMulV2` 已从 HCCL 接口迁移为 SHMEM/UDMA 接口。公开 V2 API 直接接收调用方初始化后的 `shmemContext`，并要求 `gmmX` 与 execution workspace 位于同一 SHMEM symmetric heap。现有 op-plugin PTA 接口仍调用 HCCL V1，依赖 HCCL 通信域字符串，不能覆盖新的 V2 ABI；纯 Python 也无法可靠地把 `aclshmem_malloc` 返回的地址包装为受支持的 `torch_npu.Tensor`。

因此新增一个只用于验证的轻量 C++ PyTorch bridge：Python 负责多进程编排、普通 NPU Tensor 构造和独立 CPU 标杆；bridge 负责 SHMEM 生命周期、symmetric allocation、ACL Tensor 描述符以及 V2 aclnn 调用。该方案以源算子的 SHMEM 初始化、内存和精度规则为准，不修改 op-plugin 的全局接口。

关联算子迁移设计为 `docs/superpowers/specs/2026-08-10-allto-allv-grouped-matmul-shmem-design.md`。

## 2. 目标与非目标

### 2.1 目标

- 在单机 Ascend950 环境上以一张卡一个进程的方式验证 SHMEM V2 接口，支持 1 到 8 个 rank，默认 2 个 rank。
- 提供可由 `torch.ops.load_library` 加载的独立 bridge，不依赖修改或重新发布 op-plugin。
- 由 bridge 内部将普通 BF16 NPU 输入复制到 symmetric heap，确保传给 V2 的 `gmmX` 和 workspace 满足 SHMEM 内存约束。
- 覆盖均匀路由与非均匀路由两个可重复 smoke case，并验证精确的收发计数、输出行序和 GroupedMatMul 结果。
- 提供独立构建脚本、运行脚本、逐 rank 日志、超时保护和明确的依赖检查。
- 在无 Ascend950 编译环境的本地工作区中完成 Python 语法、静态契约和 CMake 配置层面的检查；真实构建与多卡运行由目标环境执行。

### 2.2 非目标

- 不把 SHMEM V2 注册为面向最终用户的 torch_npu 正式算子。
- 不改变 op-plugin 的 HCCL `npu_alltoallv_gmm`，也不建立 HCCL process group。
- 不公开“把任意 SHMEM 地址包装为 torch Tensor”的接口。
- 不扩展当前源算子能力：仅 BF16、ND、连续 Tensor、TP=1、EP=SHMEM rank 数、EP 范围 1 到 8。
- 不在本次 smoke PTA 中替代现有 36 条静态精度用例和性能用例集合。

## 3. 文件与组件

新增目录：

```text
mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/
  CMakeLists.txt
  allto_allv_grouped_mat_mul_shmem_torch.cpp
  test_allto_allv_grouped_mat_mul_shmem.py
  build_pta.sh
  run_pta_test.sh
  README.md
```

各组件职责如下：

- `allto_allv_grouped_mat_mul_shmem_torch.cpp`：PyTorch bridge，独占 SHMEM runtime 状态并调用 V2 aclnn。
- `test_allto_allv_grouped_mat_mul_shmem.py`：多进程 PTA driver、确定性测试数据、CPU FP32 标杆和精度判定。
- `CMakeLists.txt`：只构建 bridge 动态库，不接入主仓 wheel 构建。
- `build_pta.sh`：检查环境变量和依赖，配置并构建动态库。
- `run_pta_test.sh`：设置超时并启动 Python driver，保留每个 rank 的阶段日志。
- `README.md`：记录目标环境的构建、运行、参数和故障定位方式。

bridge 使用独立 namespace 注册以下 torch operators：

```text
allto_allv_grouped_mat_mul_shmem_pta::init(
    int rank, int world_size, int heap_bytes, str ip_port) -> int

allto_allv_grouped_mat_mul_shmem_pta::run(
    Tensor x, Tensor weight, int[] send_counts, int[] recv_counts) -> Tensor

allto_allv_grouped_mat_mul_shmem_pta::barrier() -> int

allto_allv_grouped_mat_mul_shmem_pta::finalize() -> int
```

Python 必须先调用 `torch_npu.npu.set_device(rank)`，再调用 `init`。bridge 使用当前 NPU device 和当前 torch_npu stream，不创建第二条隐式计算 stream。

## 4. SHMEM 生命周期与资源所有权

每个 worker 进程只允许一个活动 SHMEM 上下文。状态机为：

```text
uninitialized -> initialized -> finalized
```

`init` 执行以下动作：

1. 校验 `world_size` 位于 `[1, 8]`、`rank` 位于 `[0, world_size)`、`heap_bytes` 为正数且 `ip_port` 非空。
2. 使用源示例相同的 `aclshmemx_init_attr_t`、`aclshmemx_uniqueid_t` 和 `ACLSHMEM_DATA_OP_UDMA` 初始化 SHMEM。
3. 保存 rank、world size、heap size 和 `shmemx_get_ffts_config()` 返回的 context 指针。
4. 对相同参数的重复初始化返回成功；对参数不一致的重复初始化立即报错。

`finalize` 在已初始化状态调用 `shmem_finalize` 并清空 bridge 状态；未初始化或已完成 finalize 时返回成功。Python worker 使用 `try/finally` 保证调用 `finalize`。

`x` 和 execution workspace 由 bridge 临时拥有，并按所有 rank 完全相同的顺序分配：

1. `aclshmem_malloc(x.nbytes())` 分配 symmetric `gmmX`。
2. `aclshmem_calloc(1, workspace_size)` 分配 symmetric workspace。
3. 算子完成并经过 SHMEM barrier 后，先释放 workspace，再释放 `gmmX`。

测试用例必须保证每个 rank 的 `x.nbytes()` 相同。V2 workspace 为固定的 `400 MiB + 4096 bytes`，因此第二次 symmetric allocation 的大小天然一致。`weight`、返回的 `y` 和 Python 输入 `x` 由 PyTorch 普通 NPU allocator 管理，不属于 symmetric heap。

所有 ACL Tensor、ACL IntArray 和 aclnn executor 元数据使用 RAII 或单一清理出口释放。bridge 不释放调用方的 Python Tensor，也不释放从 SHMEM runtime 借用的 context 指针。

## 5. `run` 数据流

`run` 按以下顺序工作：

1. 校验 runtime 已初始化，`x`/`weight` 均位于当前 NPU，dtype 为 BF16，storage format 为 ND，内存连续；`x` shape 为 `[M, K]`，`weight` shape 为 `[E, K, N]`。
2. 校验 `E > 0`、`E <= 64`，两个 counts 长度均为 `world_size * E`，元素非负，且 `sum(send_counts) == M`。
3. 计算 `recv_m = sum(recv_counts)`，用 PyTorch 普通 NPU allocator 创建连续 BF16 `y[recv_m, N]`。
4. 从 symmetric heap 分配与 `x` 等大的 `gmmX`，在当前 torch_npu stream 上执行 device-to-device asynchronous copy。后续 aclnn kernel 使用同一 stream，依靠 stream 顺序保证复制先完成。
5. 为 symmetric `gmmX`、普通 `weight` 和普通 `y` 创建非 owning `aclTensor` 描述符，并为 counts 创建 `aclIntArray`。
6. 将 `shmemx_get_ffts_config()` 保存的 context 直接传给 `aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize`。
7. 使用返回的大小从 symmetric heap 分配 workspace，然后在当前 stream 调用 `aclnnAlltoAllvGroupedMatMulV2`。
8. 同步当前 stream，使复制、通信、GMM 和输出写回全部完成。
9. 调用 SHMEM barrier；所有 rank 到达后，按一致顺序释放 workspace 与 symmetric `gmmX`。
10. 释放描述符和 executor 元数据，返回普通 NPU `y`。

bridge 的 `barrier` 仅暴露 SHMEM barrier，供 Python 在 case 边界对齐各 worker；它不建立 HCCL 或 torch.distributed 通信域。

## 6. 路由表、输入布局与标杆

Python 构造全局三维计数矩阵：

```text
C[source_rank][destination_rank][destination_local_expert]
```

对 rank `r`：

- `send_counts` 按 `[destination_rank][destination_local_expert]` 展平，值为 `C[r][dst][expert]`。
- `recv_counts` 按 `[source_rank][local_expert]` 展平，值为 `C[src][r][expert]`。
- rank `r` 的 `x` 行也按 `[destination_rank][destination_local_expert][token]` 排列，与 `send_counts` 前缀和完全一致。

测试数据不依赖 rank 间交换来生成标杆。所有 worker 使用相同 seed 和同一个全局 `C`，因此任一 worker 都能在 CPU 上重建所有 source rank 的输入和目标 rank 的权重。

目标 rank `d` 的标杆输出顺序与源 kernel 一致，为 local-expert-major、source-rank-minor：

```text
for expert in [0, E):
    for source in [0, world_size):
        rows = source_x[source] 中发往 (d, expert) 的连续切片
        chunk = rows.float() @ weight[d][expert].float()
        依次追加 chunk
```

所有 chunk 在 CPU FP32 中计算并拼接。实际 BF16 输出转为 FP32 后比较，同时打印 max absolute error、max relative error、mean relative error 和归一化均值偏差。

精度判定复用源算子 `verify_result.py` 的 BF16 规则：

- 当 `K < 2048` 时 `rtol = 2^-7`，否则 `rtol = 2^-6`。
- 每个元素必须满足 `abs(actual - golden) <= rtol * max(1, abs(golden))`。
- 归一化均值偏差 `abs(mean((actual - golden) / max(1, abs(golden))))` 不得超过 `2^-7`。
- NaN 或 Inf 直接判失败，不按零值替换。

## 7. Python 多进程编排

driver 使用 `torch.multiprocessing` 的 `spawn` context，一张卡启动一个独立进程，不调用 `torch.distributed.init_process_group`。默认参数为：

```text
world_size = 2
device_ids = 0,1
local_experts = 2
K = 32
N = 64
tokens_per_route = 2
heap_bytes = 1004 MiB
ip_port = 命令行参数或环境变量提供
```

不硬编码机器网卡地址。`--ip-port` 优先，其次读取 `SHMEM_IP_PORT`；两者都缺失时默认单机场景 `tcp://127.0.0.1:27101`。`--device-ids` 数量必须等于 `world_size`。

每个 worker 依次执行：设置 NPU device、加载 bridge、初始化 SHMEM、执行用例、case 间 barrier、上报结果、finally finalize。父进程收集每个 rank 的结构化结果和 traceback；只有全部 worker exit code 为 0 且全部 rank 精度通过时整体返回 0。

初始提供两个 case：

1. `balanced`：所有 `C[src][dst][expert]` 均为 `tokens_per_route`。
2. `uneven`：使用固定 seed 生成不同 route count，但保持每个 source rank 的总发送 token 数相同，允许部分 route 为零。这样每个 rank 的 symmetric `gmmX` allocation 大小保持一致，同时覆盖非均匀发送、非均匀接收和零 token route。

每个 rank 的输入和权重由 `(case_seed, rank)` 确定性生成，数值范围受控，便于重现精度问题。

## 8. 构建与加载

bridge 使用独立 CMake 工程，依赖：

- Python 对应的 PyTorch 和 torch_npu C++ headers/libraries；
- CANN AscendCL headers/libraries；
- SHMEM host headers 和 library；
- 已安装的 `libcust_opapi.so` 与包含当前 V2 ABI 的 `aclnn_allto_allv_grouped_mat_mul_v2.h`。

`build_pta.sh` 检查并使用以下输入：

- `ASCEND_HOME_PATH`：CANN 安装根目录；
- `SHMEM_HOME_PATH` 或 `SHMEM_HOME`：兼容源 SDK 与 HCCL 集成式 SHMEM SDK 布局；
- `CUST_OPAPI_LIBRARY`：已安装 `libcust_opapi.so` 的绝对路径；
- `PYTHON_BIN`：可选，默认 `python3`。

脚本从 Python 查询 torch/torch_npu 的 include 与 library 路径，输出：

```text
build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so
```

Python 使用 `torch.ops.load_library` 加载该动态库。构建脚本对缺失 header、library 或环境变量给出包含实际查找路径的错误，不静默回退到 HCCL V1。

## 9. 异常处理与防挂死

- C++ bridge 的参数错误、ACL/SHMEM/aclnn 非零返回码转换为带 API 名称、rank 和返回码的 `TORCH_CHECK` 异常。
- 进入 `run` 前完成可在 host 侧执行的全部检查，减少某个 rank 已进入 collective 而另一个 rank 提前失败的概率。
- 每个阶段打印 `rank/case/phase`：`init`、`allocate_x`、`copy_x`、`workspace_query`、`launch`、`stream_sync`、`barrier`、`verify`、`finalize`。
- collective 中途的单 rank 故障可能使其他 rank 阻塞，无法仅靠进程内异常安全恢复。`run_pta_test.sh` 必须使用系统 `timeout` 包裹整次测试，超时后返回非零并保留所有 rank 日志。
- stream 未同步完成前不得释放 descriptor、workspace、symmetric `gmmX` 或 Python Tensor。
- 只有已经成功完成 SHMEM 初始化的 worker 才调用 `shmem_finalize`。

## 10. 验收条件

### 10.1 离线可完成检查

- Python 文件通过 `python3 -m py_compile`。
- shell 文件通过 `bash -n`。
- CMake 能完成语法/依赖定位路径检查；无目标 SDK 时应以明确的缺失依赖错误退出。
- 现有 `check_allto_allv_grouped_mat_mul_shmem_static.py` 继续通过，并扩展检查 bridge schema、V2 调用、symmetric allocation 和不使用 HCCL process group 等关键契约。
- 文档、脚本和代码中不存在机器特定 IP、用户目录或测试设备数量硬编码。

### 10.2 Ascend950 环境验收

- bridge 能链接当前算子包的 `libcust_opapi.so` 并由 `torch.ops.load_library` 成功加载。
- 默认 2 卡运行中，`balanced` 和 `uneven` 两个 BF16 case 的所有 rank 均完成，返回的 `y` shape 为 `[sum(recv_counts), N]`。
- 输出顺序严格为 local-expert-major、source-rank-minor，全部元素及归一化均值偏差满足源算子 BF16 精度规则。
- 1 卡和可用的 4/8 卡拓扑可通过同一脚本切换；未实际执行的拓扑只记录为未验证，不声明通过。
- 任一 worker 参数错误、动态库加载失败、SHMEM 初始化失败、aclnn 返回错误、精度失败或超时均使父进程和运行脚本返回非零。

## 11. 已知环境边界

当前本地环境没有合适的 CANN/Ascend950 多卡编译与运行条件，因此本任务在本地不声称动态库已成功链接或多卡精度已通过。交付物必须把目标机上需要执行的构建和运行命令写完整，并尽可能通过离线静态检查提前发现 ABI、include、schema、资源释放和脚本编排问题。
