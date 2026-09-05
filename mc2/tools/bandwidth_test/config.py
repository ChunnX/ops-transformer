#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

"""带宽测试的用户配置入口。运行时派生值由 bandwidth_test.py 统一计算。

修改下方各配置类的字段默认值，再运行同目录的 bandwidth_test.py。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ShapeConfig:
    bs: int = 2  # 单配置时每个 rank 发送的 token 数，须 >= 1
    data_size: int = 8192  # 单配置时每个 token 的元素数 H，须 >= 1
    max_bs: int = 2  # 与 bs 相等；总 token 数不受 URMA 单批提交容量限制
    # 多配置带宽测试的用例列表；字段含义同上。注释大小为 bs * H * 2 字节。
    test_configs: List[Dict[str, int]] = field(
        default_factory=lambda: [
            {"bs": 1, "data_size": 1024, "max_bs": 1},  # 2KB
            {"bs": 2, "data_size": 1024, "max_bs": 2},  # 4KB
            {"bs": 2, "data_size": 2048, "max_bs": 2},  # 8KB
            {"bs": 4, "data_size": 2048, "max_bs": 4},  # 16KB
            {"bs": 4, "data_size": 4096, "max_bs": 4},  # 32KB
            {"bs": 8, "data_size": 4096, "max_bs": 8},  # 64KB
            {"bs": 16, "data_size": 4096, "max_bs": 16},  # 128KB
            {"bs": 32, "data_size": 4096, "max_bs": 32},  # 256KB
            {"bs": 32, "data_size": 8192, "max_bs": 32},  # 512KB
            {"bs": 64, "data_size": 8192, "max_bs": 64},  # 1MB
            {"bs": 128, "data_size": 8192, "max_bs": 128},  # 2MB
            {"bs": 256, "data_size": 8192, "max_bs": 256},  # 4MB
            {"bs": 256, "data_size": 16384, "max_bs": 256},  # 8MB
            {"bs": 512, "data_size": 16384, "max_bs": 512},  # 16MB
            {"bs": 1024, "data_size": 16384, "max_bs": 1024},  # 32MB
            {"bs": 2048, "data_size": 16384, "max_bs": 2048},  # 64MB
            {"bs": 4096, "data_size": 16384, "max_bs": 4096},  # 128MB
            {"bs": 8192, "data_size": 8192, "max_bs": 8192},  # 128MB
        ]
    )
    x_dtype: str = "fp16"  # 算子输入类型：fp16 或 bf16，均为每元素 2 字节
    random_seed: int = 90  # 固定输入数值和目标 rank 随机排列的种子


@dataclass(frozen=True)
class ParallelConfig:
    server_num: int = 1  # 参与通信的服务器数量，world_size = server_num * rank_per_dev
    rank_per_dev: int = 8  # 每台服务器的 NPU 数量，用于 world_size 和本地设备编号取模
    is_cloud: bool = False  # True：从 VC_TASK_INDEX / VC_WORKER_HOSTS 解析服务器信息
    server_index: int = 0  # 非云环境的服务器编号，范围 [0, server_num)
    # 非云环境主节点 IP；None 沿用默认：单机 127.0.0.1，多机 71.20.45.123。
    master_ip: Optional[str] = None
    master_port: int = 50001  # HCCL 初始化使用的 TCP 端口，各参与进程须一致
    is_full_core: bool = True  # True：使用 32 个 AIC / 64 个 AIV；False：使用 partial_core_num
    partial_core_num: int = 16  # 非全核时的 AIC 数量，AIV 数量自动取其两倍
    aiv_num: Optional[int] = None  # 算子使用的 AIV 核数；None：取 AIC 数量的两倍，可单独覆盖


@dataclass(frozen=True)
class ModelConfig:
    graph_type: int = 3  # 0：单算子；3：aclgraph（当前编译路径仅支持全核）
    comm_alg: str = "mte"  # 通信算法：mte 或 urma（URMA batch，按目标卡分核）
    mode: Optional[int] = None  # None 沿用精度0/带宽1；3 为URMA直接Win打流，不支持精度校验
    # 带宽测试由调用方保证通信参数正确：有效 channel、rank ID、Win 容量及各 rank 配置一致。
    # 必须保证实际 batch_tokens 小于每条使用的 SQ 和 CQ 深度；0 选 tiling 最大值（最高 2048）。
    # kernel 不做参数校验、打印或自动缩小批量；误配会使测试结果无效。
    batch_tokens: int = 0  # 0 自动；正数由 tiling 按 UB 和 2048 上限校验，BS 不受单批上限限制
    static_kernel_path: str = "./static_kernel"  # aclgraph 静态内核构建目录，相对启动目录


@dataclass(frozen=True)
class RunConfig:
    is_precision_test: bool = False  # 是否对比 CPU golden；请开启单配置并关闭带宽测试
    is_bandwidth_test: bool = True  # 是否采集执行耗时并计算带宽；与精度测试分别运行
    is_single_test: bool = False  # True：使用 shape 的 bs/H/max_bs；False：使用 test_configs
    loop: int = 100  # 单算子性能采集总迭代数，也是图模式每次 forward 的算子重复次数
    repeat: int = 1  # 单算子 profiler 采集周期数，须 >= 1，且 loop // repeat > warmup
    warmup: int = 5  # 单算子 profiler 每个周期的预热迭代数，不计入正式采集
    graph_loop: int = 2  # 图模式 profiler 外层总迭代数，每次执行一次编译后的模型
    graph_repeat: int = 1  # 图模式采集周期数，须 >= 1，且 graph_loop // graph_repeat > graph_warmup
    graph_warmup: int = 1  # 图模式每个周期的预热迭代数，不计入正式采集
    performance_sample_count: int = 50  # 每份 op_summary 最后多少条算子记录参与带宽统计，须 > 0
    # 专用结果目录，相对启动目录；测试前清空此目录及其 _config_数字 同级目录。
    profiling_path: str = "./Profiling_Result_bandwidth_test"
    is_check_setting: bool = True  # 主流程开始时检查用例尺寸、服务器编号及图模式等已有约束


@dataclass(frozen=True)
class DiagnosticsConfig:
    precision_rtol: float = 1e-3  # 浮点输出与 golden 对比的相对误差容差
    precision_atol: float = 1e-3  # 精度对比的绝对误差容差；整数输出也使用此阈值
    precision_verbose: bool = True  # 是否打印参与精度对比的张量及详细误差信息
    is_save_tensor_to_bin: bool = False  # 是否保存调试张量；单用例保存输入输出，多用例保存输出
    tensor_output_path: str = "data"  # 调试 .bin 文件的保存目录，相对启动目录
    log_level: str = "INFO"  # 日志级别，如 DEBUG、INFO、WARNING、ERROR


@dataclass(frozen=True)
class BandwidthTestConfig:
    shape: ShapeConfig = field(default_factory=ShapeConfig)  # 输入尺寸、用例列表、数据类型与随机种子
    parallel: ParallelConfig = field(default_factory=ParallelConfig)  # 通信拓扑、主节点与核数
    model: ModelConfig = field(default_factory=ModelConfig)  # 算子算法与图编译方式
    run: RunConfig = field(default_factory=RunConfig)  # 测试模式与性能采集
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)  # 精度容差、张量落盘与日志


CONFIG = BandwidthTestConfig()  # 测试脚本统一读取的配置实例
