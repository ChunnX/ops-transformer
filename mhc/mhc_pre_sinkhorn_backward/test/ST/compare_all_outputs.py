import torch
import os
import glob


#base_path = "/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST/atk_output/all_aclnn_mhc_pre_sinkhorn_2026-04-14-11-00-25-420611/output"
atk_output_dir = "/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST/atk_output"

pattern = os.path.join(atk_output_dir, "all_aclnn_mhc_pre_sinkhorn_*")
all_folders = glob.glob(pattern)

if not all_folders:
    print(f"No folders found matching pattern: {pattern}")
    exit(1)

latest_folder = max(all_folders, key=os.path.getmtime)
print(f"Using latest folder: {latest_folder}")

base_path = os.path.join(latest_folder, "output")

cpu_dir = os.path.join(base_path, "cpu_0/all_aclnn_mhc_pre_sinkhorn/0")
npu_dir = os.path.join(base_path, "pyaclnn_0/all_aclnn_mhc_pre_sinkhorn/0")

cpu_output_file = "/data/hey/git/cpu_outputs.txt"
npu_output_file = "/data/hey/git/npu_outputs.txt"

output_files = ["output_0.pt", "output_1.pt", "output_2.pt", "output_3.pt", 
                "output_4.pt", "output_5.pt", "output_6.pt", "output_7.pt"]

with open(cpu_output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CPU Outputs\n")
    f.write("="*80 + "\n\n")
    
    for output_file in output_files:
        file_path = os.path.join(cpu_dir, output_file)
        if os.path.exists(file_path):
            # if output_file == "output_0.pt":
            #     torch.set_printoptions(threshold=float('inf'))
            tensor = torch.load(file_path, map_location='cpu')
            f.write(f"\n{'='*80}\n")
            f.write(f"{output_file}:\n")
            f.write(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}\n")
            f.write(f"Data:\n{tensor}\n")
            torch.set_printoptions(profile="default")
            print(f"Processed CPU {output_file}")

with open(npu_output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("NPU Outputs\n")
    f.write("="*80 + "\n\n")
    
    for output_file in output_files:
        file_path = os.path.join(npu_dir, output_file)
        if os.path.exists(file_path):
            # if output_file == "output_0.pt":
            #     torch.set_printoptions(threshold=float('inf'))
            tensor = torch.load(file_path, map_location='cpu')
            f.write(f"\n{'='*80}\n")
            f.write(f"{output_file}:\n")
            f.write(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}\n")
            f.write(f"Data:\n{tensor}\n")
            torch.set_printoptions(profile="default")
            print(f"Processed NPU {output_file}")

print(f"\nCPU outputs saved to: {cpu_output_file}")
print(f"NPU outputs saved to: {npu_output_file}")
