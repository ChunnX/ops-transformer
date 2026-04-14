
import torch
import torch_npu
from typing import Optional
print("$$$")
torch_npu.npu.set_device(7)
print("$$$")

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2
)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),    # 与prof.step()配套使用
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./testp1"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    experimental_config=experimental_config)

t=16384
e=256
x=torch.randn(t,e).npu().requires_grad_(True)
x = x - x.max(dim=-1, keepdim=True)
x = x - sumx
x = torch.exp(x)
sumx = x.sum(dim=-1, keepdim=True)

sinkhorn_out = x / sumx
v= torch.randn(t,e).npu()
prof.start()
sinkhorn_out.backward(v)
# v,k=torch.sort(x, dim=-1)
# # print(v)
# # v1=torch.gather(x,1,k)
# # print(v1)

# # v2 = x[k]
# # print(v2)

# a,i=torch.topk(x,2, dim=-1)
# # o1=torch.ones_like(a).npu()
# # # print(a,i)
# d= torch.gather(x,1,i)
# # print("d",d)
# routing_probs = torch.zeros_like(x).scatter(1, i, a)
# y=x * zx +x
prof.stop()
