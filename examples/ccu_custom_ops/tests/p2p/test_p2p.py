import torch, torch_npu, ascend_ops
WS, N = 4, 1024
bufs = []
for d in range(WS):
    torch.npu.set_device(d)
    bufs.append(torch.zeros(WS * N, dtype=torch.float32, device=f'npu:{d}'))
ptrs = [int(b.data_ptr()) for b in bufs]
for d in range(WS):
    torch.npu.set_device(d)
    a = torch.zeros(1, dtype=torch.int32, device=f'npu:{d}')
    torch.ops.ascend_ops.p2pEnable(a, WS)
ctxs, xs = [], []
for d in range(WS):
    torch.npu.set_device(d)
    a = torch.zeros(1, dtype=torch.int32, device=f'npu:{d}')
    ctxs.append(torch.ops.ascend_ops.buildCtxFromPtrs(a, ptrs, d, WS))
    torch.npu.synchronize()
    xs.append(torch.full((N,), float(d + 1), dtype=torch.float32, device=f'npu:{d}'))
for d in range(WS):
    torch.npu.set_device(d)
    torch.ops.ascend_ops.AllGatherCcuPut(xs[d], ctxs[d], d, WS)
for d in range(WS):
    torch.npu.set_device(d); torch.npu.synchronize()
ok = True
for d in range(WS):
    torch.npu.set_device(d)
    out = torch.ops.ascend_ops.AllGatherCcuGet(xs[d], ctxs[d], d, WS)
    torch.npu.synchronize()
    exp = torch.stack([torch.full((N,), float(k + 1)) for k in range(WS)]).to(out.device)
    good = torch.allclose(out, exp)
    print(f"device {d}: ok={good} out[:,0]={out[:,0].cpu().tolist()}", flush=True)
    ok = ok and good
print("P2P_ALLGATHER_PASS" if ok else "P2P_ALLGATHER_FAIL", flush=True)
