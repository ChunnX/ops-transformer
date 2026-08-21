import os, torch

P1 = "/home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/atk/atk_output/all_aclnnMlaPreprocess_2026-08-18-21-25-06-541377"
P2 = "/home/t00954448/code/ops-transformer_0811/attention/mla_preprocess/tests/atk/atk_output/all_aclnnMlaPreprocess_2026-08-18-21-27-00-558532"
sub = "all_aclnnMlaPreprocess/1393"

cpu = torch.load(f"{P1}/output/cpu_0/{sub}/result_cpu.bin", map_location="cpu", weights_only=False)
low = torch.load(f"{P1}/output/cpu_0/{sub}/result_cpu_lowprec_bfloat16.bin", map_location="cpu", weights_only=False)
npu = torch.load(f"{P2}/output/cpu_0/{sub}/result_npu.bin", map_location="cpu", weights_only=False)
aclnn = torch.load(f"{P1}/output/pyaclnn_0/{sub}/aclnn_tmp/result.bin", map_location="cpu", weights_only=False)
cm = torch.load(f"{P1}/output/cpu_0/{sub}/cacheMode.bin", map_location="cpu", weights_only=False)

OUT_NAMES = ["qOut", "kvCacheOut", "qRopeOut", "krCacheOut"]
MIN_ERR = 1e-7

def get_mare(g, a):
    g = g.to(torch.float32); return torch.max(torch.abs(a.to(torch.float32) - g) / (torch.abs(g) + MIN_ERR)).item()
def get_mere(g, a):
    g = g.to(torch.float32); return torch.mean(torch.abs(a.to(torch.float32) - g) / (torch.abs(g) + MIN_ERR)).item()
def get_rmse(g, a):
    g = g.to(torch.float32); return torch.sqrt(torch.mean(torch.pow(a.to(torch.float32) - g, 2))).item()
def get_eb(g, a):
    g = g.to(torch.float32); gn = torch.clamp(torch.abs(g), min=1); return torch.mean((a.to(torch.float32) - g) / gn).item()
def get_eb_th(dt):
    return {torch.bfloat16: 2**(-7), torch.float16: 2**(-10), torch.float32: 2**(-14), torch.int8: 2**(-7)}.get(dt, 0)
def get_err_th(dt):
    return {torch.bfloat16: 2**(-8), torch.float16: 2**(-11), torch.float32: 2**(-14), torch.int8: 2**(-8)}.get(dt, 0)
def pair_stats(a, b):
    af = a.to(torch.float32).flatten(); bf = b.to(torch.float32).flatten()
    d = (af - bf).abs(); r = d / (bf.abs() + 1e-7)
    return {"max_abs": d.max().item(), "mean_rel": r.mean().item(), "max_rel": r.max().item(), "numel": af.numel()}
def compare_cv(golden, ref, actual):
    eb_th = get_eb_th(actual.dtype); err_th = get_err_th(actual.dtype)
    ma = get_mare(golden, actual); mr = get_mare(golden, ref)
    mea = get_mere(golden, actual); mer = get_mere(golden, ref)
    ra = get_rmse(golden, actual); rr = get_rmse(golden, ref)
    mar = ma / max(mr, err_th, 1e-30); mer_ = mea / max(mer, err_th, 1e-30); rmr = ra / max(rr, err_th, 1e-30)
    eb = get_eb(ref, actual)
    p = (mar < 10) and (mer_ < 2) and (rmr < 2) and (abs(eb) < eb_th)
    return {"mare_rate": mar, "mere_rate": mer_, "rmse_rate": rmr, "EB": eb, "pass": p}

print(f"cacheMode={cm}  aclnn=available")

print(f"\n三方产物:")
for i in range(4):
    def info(t, l):
        return f"{l}={tuple(t.shape)}/{t.dtype}"
    print(f"  [{i}] {OUT_NAMES[i]:12s} {info(cpu[i],'cpu')}  {info(npu[i],'npu')}  {info(low[i],'low')}  {info(aclnn[i],'aclnn')}")

for label, ref, rn in [
    ("CPU高精度 vs NPU小算子拼接 vs aclnn融合算子", npu, "npu"),
    ("CPU高精度 vs CPU低精度 vs aclnn融合算子", low, "cpu_lp"),
]:
    print(f"\n{'='*90}\n {label}\n{'='*90}")
    overall = True
    print(f"{'output':12s} | {'mare_rate':>12s} {'mere_rate':>12s} {'rmse_rate':>12s} {'EB':>14s} | pass")
    print(f"{'-'*90}")
    for i in range(4):
        r = compare_cv(cpu[i].cpu(), ref[i].cpu(), aclnn[i].cpu())
        p = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]: overall = False
        print(f"{OUT_NAMES[i]:12s} | {r['mare_rate']:12.4e} {r['mere_rate']:12.4e} {r['rmse_rate']:12.4e} {r['EB']:14.4e} | {p}")
    print(f"{'-'*90}")
    print(f" 总结: {'PASS' if overall else 'FAIL'}")
    print(f"\n两两对比:")
    for i in range(4):
        sg = pair_stats(aclnn[i].cpu(), cpu[i].cpu())
        sr = pair_stats(ref[i].cpu(), cpu[i].cpu())
        sa = pair_stats(aclnn[i].cpu(), ref[i].cpu())
        print(f"  [{OUT_NAMES[i]:12s}] aclnn_vs_golden: max_abs={sg['max_abs']:.4e} mean_rel={sg['mean_rel']:.4e} | "
              f"{rn}_vs_golden: max_abs={sr['max_abs']:.4e} mean_rel={sr['mean_rel']:.4e} | "
              f"aclnn_vs_{rn}: max_abs={sa['max_abs']:.4e} mean_rel={sa['mean_rel']:.4e}")

print(f"\n{'#'*90}\n 最终总结\n{'#'*90}")
