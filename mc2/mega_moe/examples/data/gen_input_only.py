import sys
import torch
import numpy as np
import os
torch.set_default_device('cpu')
m = int(sys.argv[1])
k = int(sys.argv[2])
n = int(sys.argv[3])
group = int(sys.argv[4])
topk = int(sys.argv[5])
rank = int(sys.argv[6])

trans_a = 0  # 固定为 false
trans_b = 1  # 固定为 true


def save_tensor(t, name):
    tnp = t.cpu().view(torch.int8).numpy()
    tnp.tofile(name)
    print("Saved:", name)
prefix = f"data_{m}_{k}_{n}_{group}_{topk}_{rank}"
os.makedirs(prefix, exist_ok=True)
for i in range(rank):
    rankdir = prefix+"/"+str(i)
    os.makedirs(rankdir, exist_ok=True)
    print("rank:", i)
    # gen input
    a, b = -0.01, 0.01  # 目标范围
    # 生成 [a, b] 范围内的均匀分布
    input_tokens = a + (b - a) * torch.rand(m, k, dtype=torch.bfloat16, device='cpu')
    expert_ids = torch.randint(0, group*rank, (m, topk), dtype=torch.int32, device='cpu')
    expert_weights = torch.rand(m, topk, dtype=torch.bfloat16, device='cpu')
    w1 = torch.randn((group, n, k), dtype=torch.float32, device='cpu').to(torch.float8_e4m3fn)
    w1_scale = torch.randn((group, n, k // 32), dtype=torch.float32, device='cpu').to(torch.float8_e4m3fn)
    w2 = torch.randn((group, k, n // 2), dtype=torch.float32, device='cpu').to(torch.float8_e4m3fn)
    w2_scale = torch.randn((group, k, n // 2 // 32), dtype=torch.float32, device='cpu').to(torch.float8_e4m3fn)
    save_tensor(input_tokens, f'{rankdir}/input_tokens.bin')
    save_tensor(expert_ids, f'{rankdir}/expert_ids.bin')
    save_tensor(expert_weights, f'{rankdir}/expert_weights.bin')
    save_tensor(w1, f'{rankdir}/w1.bin')
    save_tensor(w1_scale, f'{rankdir}/w1_scale.bin')
    save_tensor(w2, f'{rankdir}/w2.bin')
    save_tensor(w2_scale, f'{rankdir}/w2_scale.bin')
    