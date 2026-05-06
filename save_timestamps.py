import torch
import csv
import sys

def save_timestamps(y, output_file="timestamps.csv"):
    """
    保存时间戳到 CSV 文件
    y: 输出 tensor (torch.Tensor)
    output_file: 输出 CSV 文件路径
    """
    # 提取时间戳数据（3个核心，每个100个点）
    y_timestamps = y.cpu().view(torch.uint64).flatten()[:300].reshape(3, 100)
    
    # globalCoreIdx: 0=AIC, 1=AIV sub0, 2=AIV sub1
    core_names = ["AIC_core0", "AIV_sub0", "AIV_sub1"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ["core_name"] + [f"point_{i}" for i in range(100)]
        writer.writerow(header)
        
        # 写入每个核心的时间戳
        for i, name in enumerate(core_names):
            row = [name] + y_timestamps[i].tolist()
            writer.writerow(row)
    
    print(f"时间戳已保存到 {output_file}")
    print(f"格式: 每行一个核心，共 {len(core_names)} 行")
    print(f"列: core_name, point_0, point_1, ..., point_99")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python save_timestamps.py <input_tensor.pt> [output.csv]")
        print("示例: python save_timestamps.py y_tensor.pt timestamps.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "timestamps.csv"
    
    # 加载 tensor
    y = torch.load(input_file)
    save_timestamps(y, output_file)
