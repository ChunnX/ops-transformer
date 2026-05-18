import torch
import os
import numpy as np

# 定义三个文件路径
paths = [
    "/home/c00953094/0421trans/transformer-readme/mlapo_atk/atk_output/all_aclnnMlaPreprocess_kv_cache_nc_dim0_2026-05-18-20-05-10-844096/output/cpu_benchmark/all_aclnnMlaPreprocess_kv_cache_nc_dim0/0/output_0.pt",
    "/home/c00953094/0421trans/transformer-readme/mlapo_atk/atk_output/all_aclnnMlaPreprocess_kv_cache_nc_dim0_2026-05-18-20-05-10-844096/output/npu_0/all_aclnnMlaPreprocess_kv_cache_nc_dim0/0/output_0.pt",
    "/home/c00953094/0421trans/transformer-readme/mlapo_atk/atk_output/all_aclnnMlaPreprocess_kv_cache_nc_dim0_2026-05-18-20-05-10-844096/output/pyaclnn_0/all_aclnnMlaPreprocess_kv_cache_nc_dim0/0/output_0.pt"
]

names = ["CPU", "NPU", "PyACLNN"]

# 加载所有张量
tensors = []
for i, path in enumerate(paths):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        exit(1)
    try:
        tensor = torch.load(path, map_location='cpu').double()  # 强制转为 fp64
        tensors.append(tensor)
        # print(f"[{names[i]}] Shape: {tensor.shape}, dtype: {tensor.dtype}, data:{tensor}")
        # print(f"[{names[i]}] Stats - min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, mean: {tensor.mean().item():.6f}")
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        exit(1)

# 比较差值：CPU vs GPU, CPU vs PyACLNN, GPU vs PyACLNN
pairs = [(0, 1), (0, 2), (1, 2)]
pair_names = [("CPU", "NPU"), ("CPU", "PyACLNN"), ("NPU", "PyACLNN")]

print("\n" + "="*60)
print("DIFFERENCE ANALYSIS")
print("="*60)

for (i, j), (name_i, name_j) in zip(pairs, pair_names):
    diff = tensors[i] - tensors[j]
    abs_diff = torch.abs(diff)
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    mse = (diff ** 2).mean().item()

    # 找到最大绝对误差的位置（展平后的索引）
    flat_index = abs_diff.argmax().item()
    # 转换为多维索引
    multi_index = np.unravel_index(flat_index, abs_diff.shape)
    
    # 获取两个张量在该位置的值
    val_i = tensors[i][multi_index].item()
    val_j = tensors[j][multi_index].item()

    print(f"\n{name_i} vs {name_j}:")
    print(f"  Max Abs Error     : {max_abs_err:.8f}")
    print(f"  Mean Abs Error    : {mean_abs_err:.8f}")
    print(f"  MSE               : {mse:.12f}")
    print(f"  Location (indices): {multi_index}")
    print(f"  Values            : {name_i}={val_i:.8f}, {name_j}={val_j:.8f}")