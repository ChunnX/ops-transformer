import torch
import torch_npu
import numpy as np

torch_npu.npu.config.allow_internal_format=True

def check_npu_availability():
    """检查NPU设备可用性"""
    if torch_npu.npu.is_available():
        print(f"✓ NPU设备可用")
        print(f"✓ torch版本: {torch.__version__}")
        print(f"✓ torch_npu版本: {torch_npu.__version__}")
        return True
    else:
        print(f"✗ NPU设备不可用")
        return False


def gmm_a8w4_with_bias_example():
    """
    带偏置的A8W4使用示例

    场景：单输入张量，单权重张量，带偏置的分组计算

    注意：A8W4场景中，bias的N维度需要是 weight的N维度 * 8
    （因为INT32存储INT4时，N维度会扩展8倍）
    """
    print("\n" + "="*60)
    print("带偏置的A8W4使用示例")
    print("="*60)

    # 配置参数
    num_experts = 8
    total_tokens = 4866
    dim_k = 512
    dim_n = 8
    dim_n_bias = dim_n * 8  # bias的N维度需要扩展8倍
    quant_group_size = 256
    tokens_per_expert = total_tokens // num_experts

    try:
        # 准备输入数据
        x_int8 = torch.randint(low=-5, high=5,
                              size=(total_tokens, dim_k),
                              dtype=torch.int8).npu()

        weight_int32 = torch.randint(low=-5, high=5,
                                   size=(num_experts, dim_k, dim_n),
                                   dtype=torch.int32).npu()

        # 准备偏置（FLOAT32类型）
        # bias的N维度需要是 weight的N维度 * 8
        bias = torch.randn((num_experts, dim_n_bias), dtype=torch.float32).npu()

        # 准备量化参数
        group_list = torch.full((num_experts,), tokens_per_expert, dtype=torch.int64).npu()

        quant_group_num = dim_k // quant_group_size
        scale_np = np.random.normal(0, 0.01, (num_experts, quant_group_num, dim_n_bias)).astype(np.float32)
        scale_uint32 = (scale_np).astype(np.float16).astype(np.float32)
        scale_uint32.dtype = np.uint32
        scale_uint64 = np.zeros((num_experts, quant_group_num, dim_n_bias * 2), dtype=np.uint32)
        scale_uint64[..., ::2] = scale_uint32
        scale_uint64.dtype = np.int64
        scale = torch.from_numpy(scale_uint64).npu()

        per_token_scale = torch.rand((total_tokens, 1), dtype=torch.float32).npu()

        print(f"输入数据信息：")
        print(f"  x_int8形状: {x_int8.shape}, 数据类型: {x_int8.dtype}")
        print(f"  weight_int32形状: {weight_int32.shape}, 数据类型: {weight_int32.dtype}")
        print(f"  bias形状: {bias.shape}, 数据类型: {bias.dtype}")

        # 执行带偏置的A8W4分组矩阵乘法
        output = torch_npu.npu_grouped_matmul(
            [x_int8],
            [weight_int32],
            bias=[bias],
            scale=[scale],
            per_token_scale=[per_token_scale],
            group_list=group_list,
            group_type=0,
            split_item=3,
            group_list_type=1,
            output_dtype=torch.float16
        )

        print(f"\n执行结果：")
        print(f"  输出数量: {len(output)}")
        print(f"  输出形状: {output[0].shape}")  # 实际输出是原始dim_n

        return True

    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def gmm_a8w4_output_dtype_example():
    """
    A8W4不同输出数据类型示例

    场景：演示支持BFLOAT16和FLOAT16两种输出类型
    """
    print("\n" + "="*60)
    print("A8W4不同输出数据类型示例")
    print("="*60)

    # 配置参数
    num_experts = 8
    total_tokens = 4866
    dim_k = 512
    dim_n = 8
    quant_group_size = 256
    tokens_per_expert = total_tokens // num_experts

    # 输出数据类型列表
    output_dtypes = [torch.float16, torch.bfloat16]

    for output_dtype in output_dtypes:
        print(f"\n--- 测试输出类型: {output_dtype} ---")

        try:
            # 准备输入数据
            x_int8 = torch.randint(low=-5, high=5,
                                  size=(total_tokens, dim_k),
                                  dtype=torch.int8).npu()

            weight_int32 = torch.randint(low=-5, high=5,
                                       size=(num_experts, dim_k, dim_n),
                                       dtype=torch.int32).npu()

            # 准备量化参数
            group_list = torch.full((num_experts,), tokens_per_expert, dtype=torch.int64).npu()

            quant_group_num = dim_k // quant_group_size
            scale_np = np.random.normal(0, 0.01, (num_experts, quant_group_num, dim_n)).astype(np.float32)
            scale_uint32 = (scale_np).astype(np.float16).astype(np.float32)
            scale_uint32.dtype = np.uint32
            scale_uint64 = np.zeros((num_experts, quant_group_num, dim_n * 2), dtype=np.uint32)
            scale_uint64[..., ::2] = scale_uint32
            scale_uint64.dtype = np.int64
            scale = torch.from_numpy(scale_uint64).npu()

            per_token_scale = torch.rand((total_tokens, 1), dtype=torch.float32).npu()

            # 执行A8W4分组矩阵乘法
            output = torch_npu.npu_grouped_matmul(
                [x_int8],
                [weight_int32],
                scale=[scale],
                per_token_scale=[per_token_scale],
                group_list=group_list,
                group_type=0,
                split_item=3,
                group_list_type=1,
                output_dtype=output_dtype
            )

            print(f"  ✓ 执行成功")
            print(f"    输出形状: {output[0].shape}")
            print(f"    输出数据类型: {output[0].dtype}")
            print(f"    输出范围: [{output[0].min().item():.4f}, {output[0].max().item():.4f}]")

        except Exception as e:
            print(f"  ✗ 执行失败: {e}")

    return True

def performance_test():
    """
    性能测试示例
    """
    print("\n" + "="*60)
    print("A8W4性能测试")
    print("="*60)

    # 配置参数
    num_experts = 8
    total_tokens = 768
    dim_k = 7168
    dim_n = 4096
    quant_group_size = 256
    tokens_per_expert = total_tokens // num_experts
    iterations = 10

    try:
        # 准备输入数据
        x_int8 = torch.randint(low=-5, high=5,
                              size=(total_tokens, dim_k),
                              dtype=torch.int8).npu()

        weight_int32 = torch.randint(low=-5, high=5,
                                   size=(num_experts, dim_k, dim_n),
                                   dtype=torch.int32).npu()

        group_list = torch.full((num_experts,), tokens_per_expert, dtype=torch.int64).npu()

        quant_group_num = dim_k // quant_group_size
        scale_np = np.random.normal(0, 0.01, (num_experts, quant_group_num, dim_n)).astype(np.float32)
        scale_uint32 = (scale_np).astype(np.float16).astype(np.float32)
        scale_uint32.dtype = np.uint32
        scale_uint64 = np.zeros((num_experts, quant_group_num, dim_n * 2), dtype=np.uint32)
        scale_uint64[..., ::2] = scale_uint32
        scale_uint64.dtype = np.int64
        scale = torch.from_numpy(scale_uint64).npu()

        per_token_scale = torch.rand((total_tokens, 1), dtype=torch.float32).npu()

        # 预热运行
        print(f"预热运行...")
        for _ in range(3):
            output = torch_npu.npu_grouped_matmul(
                [x_int8],
                [weight_int32],
                scale=[scale],
                per_token_scale=[per_token_scale],
                group_list=group_list,
                group_type=0,
                split_item=3,
                group_list_type=1,
                output_dtype=torch.float16
            )
            torch_npu.npu.synchronize()

        # 性能测试
        print(f"开始性能测试...")
        import time
        start_time = time.time()

        for i in range(iterations):
            output = torch_npu.npu_grouped_matmul(
                [x_int8],
                [weight_int32],
                scale=[scale],
                per_token_scale=[per_token_scale],
                group_list=group_list,
                group_type=0,
                split_item=3,
                group_list_type=1,
                output_dtype=torch.float16
            )
            torch_npu.npu.synchronize()

            if (i + 1) % 5 == 0:
                print(f"  已完成 {i + 1}/{iterations} 次迭代")

        end_time = time.time()

        avg_time = (end_time - start_time) / iterations * 1000

        print(f"\n性能测试结果：")
        print(f"  总迭代次数: {iterations}")
        print(f"  总耗时: {end_time - start_time:.4f} 秒")
        print(f"  平均每次耗时: {avg_time:.2f} 毫秒")
        print(f"  吞吐量: {1000 / avg_time:.2f} 次/秒")

        return True

    except Exception as e:
        print(f"\n✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("GMM A8W4场景的torch调用示例")
    print("="*60)

    # 检查NPU可用性
    if not check_npu_availability():
        return

    # 执行不同示例
    examples = [
        ("带偏置示例", gmm_a8w4_with_bias_example),
        ("性能测试", performance_test)
    ]

    success_count = 0
    total_count = len(examples)

    for name, func in examples:
        print(f"\n\n=== 执行 {name} ===")
        if func():
            print(f"✓ {name} 执行成功")
            success_count += 1
        else:
            print(f"✗ {name} 执行失败")

    print(f"\n" + "="*60)
    print(f"执行结果统计：")
    print(f"  总示例数: {total_count}")
    print(f"  成功数: {success_count}")
    print(f"  失败数: {total_count - success_count}")
    print(f"  成功率: {success_count / total_count * 100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()

