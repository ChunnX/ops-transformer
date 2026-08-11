# AlltoAllvGroupedMatMul

AlltoAllvGroupedMatMul 将非均匀 AlltoAllV 路由与本地专家 GroupedMatMul 融合执行。

## 产品与接口

| 产品 | 接口 | 通信后端 | 状态 |
|---|---|---|:---:|
| Ascend 950DT | `aclnnAlltoAllvGroupedMatMulV2` | CANN SHMEM/UDMA | 支持 |
| Atlas A3 训练/推理系列产品 | `aclnnAlltoAllvGroupedMatMul` | HCCL | 支持 |
| Atlas A2 及其他产品 | - | - | 不支持 |

Ascend 950 的 V2 接口已按源算子能力改造，并允许破坏旧 ABI：`shmemContext` 取代旧的 `group`/`commMode` 与扩展可选参数。V2 仅支持 BF16、TP=1、EP 1～8、非转置本地专家权重；A3 的 V1/HCCL 路径保持独立。

## Ascend 950 内存契约

- `gmmX` 与执行 workspace 必须由同一个已初始化 SHMEM 对称堆分配。
- `shmemContext` 传入 `shmemx_get_ffts_config()` 返回的设备配置地址。
- 远端通信使用 UDMA PUT，本 rank 自拷贝使用 MTE；kernel 使用两级 workspace stage 覆盖通信与计算。
- 详细 shape、counts 顺序、输出顺序及生命周期约束见 [V2 接口文档](./docs/aclnnAlltoAllvGroupedMatMulV2.md)。

## 示例

- Ascend 950 SHMEM V2：[test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/arch35/test_aclnn_allto_allv_grouped_mat_mul.cpp)
- Atlas A3 HCCL V1：[test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_grouped_mat_mul.cpp)

Ascend 950 编译支持两种 SHMEM SDK 布局：

- 源算子布局：设置 `SHMEM_HOME_PATH`，头文件位于 `${SHMEM_HOME_PATH}/shmem/include`。
- HCCL 工程集成布局：设置 `SHMEM_HOME`，头文件位于 `${SHMEM_HOME}/include`，库通常位于 `${SHMEM_HOME}/build/lib`。

运行示例时采用“一进程一 rank”，并在所有 rank 完成执行和流同步后再释放 SHMEM 资源。若 SHMEM 运行时复用 HCCL 网络环境，还需要为每台机器配置正确的 `HCCL_SOCKET_IFNAME`；`HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT` 和 SHMEM rendezvous 地址应由启动脚本按实际集群设置，禁止在仓库中硬编码 IP 或网卡名。
