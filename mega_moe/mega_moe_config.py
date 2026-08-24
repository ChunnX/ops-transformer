"""MegaMoe 测试用例的用户配置入口。运行时派生值仍由 mega_moe.py 统一计算。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShapeConfig:
    # 沿用测试脚本中的常用名称：bs 为 token 数，h 为 hidden size，n 为中间层大小，k 为 topK。
    bs: int = 1  # 单卡 token 数
    h: int = 4096  # 每个 token 的 hidden size
    n: int = 1024  # FFN 中间层大小
    top_k: int = 6  # 每个 token 路由到的 MoE 专家数
    x_active_mask_dim: int = 0  # 支持 0、1、2 维
    is_variable_bs: bool = False  # 是否使能可变 BS


@dataclass(frozen=True)
class ParallelConfig:
    server_num: int = 1  # 参与测试的服务器数量
    rank_per_dev: int = 2  # 每个 host 使用的 die 数量
    tp_world_size: int = 1  # 1：纯 EP；2：Dispatch 做 TP AllGather，Combine 做 TP ReduceScatter
    local_moe_expert_num: int = 48  # 每个 rank 上的 MoE 专家数
    shared_expert_num: int = 0  # 每个 rank 上的共享专家数
    enable_shared_expert_multistream: bool = True  # 共享专家是否使用独立 stream
    ep_buffer_size: int = 6000  # EP 通信 buffer 大小，单位 MB
    tp_buffer_size: int = 1000  # TP 通信 buffer 大小，单位 MB
    is_full_core: bool = True  # False 时使用 partial_core_num
    partial_core_num: int = 18  # 非全核模式使用的 AIC 数量，AIV 数量为其两倍
    is_cloud: bool = False  # 云上环境设为 True，物理机环境设为 False
    server_index: int = 0  # 多机时各服务器使用不同且连续的 index
    master_ip: str = "127.0.0.1"  # 物理机多机测试使用的主节点 IP


@dataclass(frozen=True)
class QuantizationConfig:
    quant_mode: int = 4  # Dispatch 量化模式，当前 MegaMoe 路径使用 4（MXFP）
    combine_quant_mode: int = 0  # 0：不量化；3：E5M2；4：E4M3
    output_dtype: str = "bfloat16"  # Dispatch 输入与 Combine 输出类型：bfloat16 或 float16
    dispatch_quant_dtype: str = "float8_e4m3fn"  # Dispatch 输出量化类型
    swiglu_quant_dtype: str = "float8_e4m3fn"  # SwiGLU 输出量化类型
    weight_dtype: str = "float8_e4m3fn"  # GMM 权重类型
    a8w8_is_nz: bool = False  # A8W8 权重是否使用 NZ 格式
    topk_weights_type: bool = False  # False：关闭权重前移；True：开启权重前移


@dataclass(frozen=True)
class ModelConfig:
    graph_type: int = 0  # 0：单算子；1：静态图；2：动态图；3：ACL graph
    transpose_weight: int = 1  # 是否按转置后的权重布局生成输入
    is_interleaved: bool = False  # 是否使用交错排列的 gate/up 权重布局
    weight_is_tensor_list: int = 0  # 0：堆叠 Tensor；1：每个专家一个 Tensor
    clamp_limit: float = -1  # 小于 0 时不传 clamp 参数
    activation_type: str = "swiglu"  # swiglu、swiglustep、swigluoai 或 situglu
    swiglu_alpha: float = 1.702  # SwiGLU alpha 参数
    swiglu_beta: float = 1.0  # SwiGLU beta 参数
    situ_beta: float = 1.0  # SiTU beta 参数
    situ_linear_beta: float = 1.0  # SiTU linear beta 参数
    is_activation_params: int = 1  # 1：使用 activation_params 新接口；0：使用旧接口
    activation_direct_output: bool = False  # 输出并逐点对比 routed expert 的 GMM2 BF16 结果（Combine 前）
    activation_debug_max_print: int = 100  # 中间值对比最多打印的异常坐标数
    comm_alg: str = ""  # 指定通信算法，空字符串表示使用默认算法


@dataclass(frozen=True)
class RunConfig:
    is_performance: bool = False  # 是否执行性能测试
    functional_loop_count: int = 1  # 功能模式连续执行轮数
    performance_loop_count: int = 105  # 性能模式采集轮数
    golden_cache_enable: bool = True  # 是否缓存 CPU golden
    golden_cache_dir: str = "golden_cache"  # golden 缓存目录，相对 mega_moe.py
    golden_ffn_chunk_size: int = 16  # CPU golden 每批并行处理的专家数
    golden_batch_matmul: bool = False  # CPU golden 是否使用批量矩阵乘
    golden_batch_matmul_min_m: int = 128  # 启用批量矩阵乘的最小 M
    golden_log_interval: int = 512  # CPU golden 进度日志间隔
    golden_log_stats: bool = False  # 是否打印 CPU golden 统计信息
    golden_num_interop_threads: int = 1  # PyTorch CPU inter-op 线程数
    fast_random_weight: bool = True  # 是否使用快速随机权重生成路径
    is_topk_balance: bool = True  # 是否生成专家间均衡的 topK 路由
    is_x_random: bool = True  # 是否随机生成输入 x
    random_seed: int = 831  # 固定 x、expert_ids、expert_scales 等随机输入
    expert_scales_dtype: str = "float32"  # expert_scales 类型：float32 或 bfloat16
    print_compare_details: bool = False  # 是否打印逐项精度比较信息
    case_name: str = (
        "MegaMoeTestCase_Default"  # 用于区分不同测试用例的输出目录，默认值为 "MegaMoeTestCase"
    )


@dataclass(frozen=True)
class DiagnosticsConfig:
    kernel_timing_enable: bool = False  # 是否解析 kernel 输出中的简易计时数据
    kernel_timing_cycles_to_us: int = 1000  # Ascend 950/A5：1000 cycles = 1 us
    kernel_timing_output_dir: str = "timing_output"  # 简易计时输出目录，相对 mega_moe.py
    critical_path_trace_enable: bool = False  # 是否将关键路径 trace 原始字节按 rank 落盘
    critical_path_trace_output_dir: str = "critical_path_trace_output"  # trace 输出根目录


@dataclass(frozen=True)
class MegaMoeTestConfig:
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


CONFIG = MegaMoeTestConfig()
