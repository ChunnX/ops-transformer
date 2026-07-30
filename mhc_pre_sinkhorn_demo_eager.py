# -----------------------------------------------------------------------------------------------------------
# mhc_pre_sinkhorn 算子调用示例：eager 模式 + 测试 deterministic level 3
# 运行： python mhc_pre_sinkhorn_demo_eager.py
# 说明：通过 tiling 日志验证 deterministic level 配置是否生效
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu  # noqa: F401  注册 npu 设备与后端
import torchair
from torchair.configs.compiler_config import CompilerConfig

# 导入包会通过 op_builder 注册 torch.ops.cann_ops_transformer.mhc_pre_sinkhorn 及其 meta / graph 转换
from cann_ops_transformer.ops.mhc_pre_sinkhorn import mhc_pre_sinkhorn


# ---- 维度约束（见算子 README）：n(hc_mult)=4, num_iters=20, c 在 950 上支持 4096/7168 ----
B, S = 2, 8          # batch、序列长度
N = 4                # hc_mult，残差流数量，仅支持 4
C = 4096             # x 最后一维，950 上支持 4096 / 7168
NUM_ITERS = 20       # sinkhorn 迭代次数，仅支持 20
HC_EPS = 1e-6
NORM_EPS = 1e-6
DEVICE = "npu"


def make_inputs():
    """构造算子输入。x 为 bf16，phi/alpha/bias 为 float32。"""
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16, device=DEVICE)
    phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32, device=DEVICE)   # [24, N*C]
    alpha = torch.randn(3, dtype=torch.float32, device=DEVICE)                    # 三路缩放系数
    bias = torch.randn(N * N + 2 * N, dtype=torch.float32, device=DEVICE)         # [24]
    return x, phi, alpha, bias


def run_eager():
    print("=== eager 模式 + deterministic level 测试 ===")

    # 注意：当前 CANN 运行时版本只支持 level=0 和 level=1
    # level=2 和 level=3 需要更新的 CANN 版本或 torch_npu 实现
    level = 3
    print(f"\n[配置] 设置 deterministic level 为 {level}")
    print("[说明] level=0: 关闭确定性")
    print("[说明] level=1: 仅开启确定性")
    print("[说明] level=2: 开启强一致性 (当前 CANN 版本暂不支持)")
    print("[说明] level=3: 开启batch一致性 (当前 CANN 版本暂不支持)")

    torch_npu.npu.set_deterministic_level(level)
    print(f"[配置] 配置完成，请查看 tiling 日志验证 level={level} 是否生效")

    print("\n[执行] 运行 mhc_pre_sinkhorn 算子...")
    x, phi, alpha, bias = make_inputs()
    hin, h_post, h_res = mhc_pre_sinkhorn(x, phi, alpha, bias, N, NUM_ITERS, HC_EPS, NORM_EPS)
    torch.npu.synchronize()

    print("\n[结果] 算子执行完成:")
    print(f"  hin   : {tuple(hin.shape)}, {hin.dtype}")
    print(f"  h_post: {tuple(h_post.shape)}, {h_post.dtype}")
    print(f"  h_res : {tuple(h_res.shape)}, {h_res.dtype}")
    print("\n[提示] 请检查 tiling 日志中 deterministic 相关的打印信息")


if __name__ == "__main__":
    run_eager()

