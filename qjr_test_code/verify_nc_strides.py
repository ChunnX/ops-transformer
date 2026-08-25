#!/usr/bin/env python3
"""
校验所有非连续 case 的 K/K_scale stride 是否真的非连续。
依据布局特定约束公式（用户给出）：

KV (PA 布局):
  PA_BNBD [Bn,N2,Bs,D]   stride=(bnS,n2S,bsS,dS):  bnStride >= N2*n2Stride, n2Stride > D*Bs
  PA_NZ   [Bn,N2,D1,Bs,D0] stride=(bnS,n2S,d1S,bsS,d0S): 同 BNBD 形式 (n2Stride > D*Bs, bnStride >= N2*n2Stride)
  PA_BBH  [Bn,Bs,N2*D]   stride=(bnS,bsS,hS):      bnStride >= Bs*H  (H=N2*D)

Scale 非连续 (仅 mode4/5 有意义):
  mode4 [blockNum, bs]    stride=(bnS,bsS):        bnStride >= bs        (连续 bnStride=bs)
  mode5 [blockNum, N, bs] stride=(bnS,nS,bsS):     n2Stride >= Bs, bnStride >= N2*n2Stride

判断口径: 声明 axis0(bn)非连续 -> bnStride 应 > 连续下界; 声明 axis1(n2)非连续 -> n2Stride 应 > 连续下界。
用 > 严格大于判定非连续 (连续时 stride 恰等于下界)。
"""
import json, re, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES = json.load(open(os.path.join(SCRIPT_DIR, "cases.json")))
LOG_PATH = os.environ.get("NC_STRIDE_LOG", os.path.join(SCRIPT_DIR, "log/test_full_overview.log"))
LOG = open(LOG_PATH).read()

# 解析日志: 每个 case 的 K/K_scale stride + totalBlocks/N_kv/d/bs
actual = {}
cur = None
for line in LOG.split("\n"):
    m = re.match(r"\[RUN\]\s*(\S+):", line)
    if m:
        cur = m.group(1).rstrip(":")
        actual[cur] = {}
    m = re.search(r"k contig=(\S+)\s+stride=\(([^)]*)\)", line)
    if m and cur:
        actual[cur]["kcontig"] = m.group(1)
        actual[cur]["k"] = tuple(int(x) for x in m.group(2).split(",") if x.strip().isdigit())
    m = re.search(r"k_scale contig=(\S+)\s+stride=\(([^)]*)\)", line)
    if m and cur:
        actual[cur]["kscontig"] = m.group(1)
        nums = [x.strip() for x in m.group(2).split(",")]
        actual[cur]["ks"] = tuple(int(x) for x in nums if x.isdigit())
    m = re.search(r"totalBlocks=(\d+).*N_kv=(\d+).*d=(\d+)", line)
    if m and cur:
        actual[cur]["tb"] = int(m.group(1))
        actual[cur]["nkv"] = int(m.group(2))
        actual[cur]["d"] = int(m.group(3))

all_cases = {}
error_ids = set()
for c in CASES["normal_cases"]:
    all_cases[c["id"]] = c
for c in CASES["error_cases"]:
    all_cases[c["id"]] = c
    error_ids.add(c["id"])


def check_kv(cid, c, a):
    """返回 (bn非连续?, n2非连续?, 详情列表)"""
    kst = a.get("k")
    if not kst:
        return None, None, ["日志无K stride"]
    fmt = c["fmt"]
    nkv = a.get("nkv", c.get("N_kv", 2))
    bs = c.get("bs", 128)
    d = c.get("d", 128)
    # INT4 打包: 8个int4存1个int32, 存储D维 = 逻辑D/8
    if c["kv_dtype"] == "int4":
        d = d // 8
    detail = []
    if fmt == "BnNBsD":  # [Bn,N2,Bs,D] stride=(bnS,n2S,bsS,dS)
        if len(kst) != 4:
            return None, None, [f"BNBD stride维度{len(kst)}!=4"]
        bnS, n2S, bsS, dS = kst
        # 连续: dS=1, bsS=d, n2S=bs*d, bnS=nkv*bs*d
        n2_contig_val = d * bs
        bn_contig_val = nkv * d * bs
        n2_nc = n2S > n2_contig_val
        bn_nc = bnS > bn_contig_val
        detail.append(f"BNBD: n2S={n2S}(contig={n2_contig_val},>{'NC' if n2_nc else 'C'}) bnS={bnS}(contig={bn_contig_val},>{'NC' if bn_nc else 'C'})")
        return bn_nc, n2_nc, detail
    elif fmt == "BnBsH":  # [Bn,Bs,N2*D] stride=(bnS,bsS,hS)
        if len(kst) != 3:
            return None, None, [f"BBH stride维度{len(kst)}!=3"]
        bnS, bsS, hS = kst
        H = nkv * d
        bn_contig_val = bs * H
        bn_nc = bnS > bn_contig_val
        detail.append(f"BBH: bnS={bnS}(contig={bn_contig_val},>{'NC' if bn_nc else 'C'}) H={H}")
        return bn_nc, None, detail  # BBH 无独立 n2 轴
    elif fmt == "NZ":  # [Bn,N2,D1,Bs,D0] stride=(bnS,n2S,d1S,bsS,d0S)
        if len(kst) != 5:
            return None, None, [f"NZ stride维度{len(kst)}!=5"]
        bnS, n2S, d1S, bsS, d0S = kst
        D0 = 16
        D1 = d // 16
        # 连续: d0S=1, bsS=D0, d1S=bs*D0, n2S=D1*bs*D0, bnS=nkv*D1*bs*D0
        n2_contig_val = D1 * bs * D0
        bn_contig_val = nkv * D1 * bs * D0
        n2_nc = n2S > n2_contig_val
        bn_nc = bnS > bn_contig_val
        detail.append(f"NZ: n2S={n2S}(contig={n2_contig_val},>{'NC' if n2_nc else 'C'}) bnS={bnS}(contig={bn_contig_val},>{'NC' if bn_nc else 'C'})")
        return bn_nc, n2_nc, detail
    else:
        return None, None, [f"非PA布局{fmt}"]


def check_scale(cid, c, a):
    """返回 (bn非连续?, n2非连续?, 详情列表)"""
    ksst = a.get("ks")
    if not ksst:
        return None, None, ["日志无K_scale stride"]
    m = c["antiquant_mode_num"]
    nkv = a.get("nkv", c.get("N_kv", 2))
    bs = c.get("bs", 128)
    detail = []
    if m == 4:  # [blockNum, bs] stride=(bnS,bsS)
        if len(ksst) != 2:
            return None, None, [f"mode4 scale维度{len(ksst)}!=2"]
        bnS, bsS = ksst
        # 约束: bnStride >= bs (连续 bnS=bs)
        bn_nc = bnS > bs
        detail.append(f"mode4: bnS={bnS}(contig={bs},>{'NC' if bn_nc else 'C'})")
        return bn_nc, None, detail
    elif m == 5:  # [blockNum, N, bs] stride=(bnS,nS,bsS)
        if len(ksst) != 3:
            return None, None, [f"mode5 scale维度{len(ksst)}!=3"]
        bnS, nS, bsS = ksst
        # 约束: n2Stride >= Bs (连续 nS=bs), bnStride >= N2*n2Stride (连续 bnS=nkv*bs)
        n2_nc = nS > bs
        bn_nc = bnS > nkv * bs
        detail.append(f"mode5: nS={nS}(contig={bs},>{'NC' if n2_nc else 'C'}) bnS={bnS}(contig={nkv*bs},>{'NC' if bn_nc else 'C'})")
        return bn_nc, n2_nc, detail
    else:
        # mode0/1/2/3/6 scale 无 Bn 轴, 声明非连续应被 checker 拦(error case)或 scale nc=[] (normal)
        return None, None, [f"mode{m} scale无Bn轴(不支持scale非连续)"]


def main():
    print("=" * 120)
    print("FIA Antiquant PA 非连续 stride 校验 (KV + scale)")
    print(f"日志: {LOG_PATH}")
    print("=" * 120)
    problems = []
    for cid, c in all_cases.items():
        if cid not in actual:
            continue
        # error case (expect_pass=False) 测的是 checker 拦截, 不要求 stride 符合声明:
        #   - QS16/PA-QS8: reference 阶段就被拦, 没跑到非连续构造, 日志无 stride (预期)
        #   - SCALE-NC-ERR: 测 scale 非连续被拦, mode0/3 scale 无 Bn 轴本就不该非连续 (预期)
        # 故 error case 跳过 stride 校验, 只标注其拦截性质
        if cid in error_ids:
            print(f"\n[{cid}] ⊘ error case (测拦截, 不校验 stride)")
            print(f"  声明: KV nc={sorted(c.get('k_uncontiguous',[]))}  scale nc={sorted(c.get('k_antiquantScale_uncontiguous',[]))}")
            continue
        a = actual[cid]
        k_axes = set(c.get("k_uncontiguous", []))
        ks_axes = set(c.get("k_antiquantScale_uncontiguous", []))
        # 只校验有声明的
        has_k = bool(k_axes)
        has_ks = bool(ks_axes)
        if not has_k and not has_ks:
            continue
        kbn, kn2, kdet = check_kv(cid, c, a) if has_k else (None, None, ["未声明KV非连续"])
        ksbn, ksn2, ksdet = check_scale(cid, c, a) if has_ks else (None, None, ["未声明scale非连续"])

        # 核对: 声明 axis0 -> bn_nc True; axis1 -> n2_nc True
        k_ok = True
        if has_k:
            if 0 in k_axes and not kbn:
                k_ok = False
            if 1 in k_axes and kn2 is False:  # BBH kn2=None(无n2轴), 1轴声明在BBH是error case特殊
                k_ok = False
        ks_ok = True
        if has_ks:
            if 0 in ks_axes and not ksbn:
                ks_ok = False
            if 1 in ks_axes and ksn2 is False:
                ks_ok = False
        ok = k_ok and ks_ok
        if not ok:
            problems.append(cid)
        flag = "✓" if ok else "✗ 不符"
        print(f"\n[{cid}] {flag}")
        print(f"  声明: KV nc={sorted(k_axes)}  scale nc={sorted(ks_axes)}")
        if has_k:
            print(f"  KV:   {'; '.join(kdet)}  -> bn={'NC' if kbn else 'C'}, n2={'NC' if kn2 else ('C' if kn2 is False else 'N/A')}")
        if has_ks:
            print(f"  scale:{'; '.join(ksdet)}  -> bn={'NC' if ksbn else 'C'}, n2={'NC' if ksn2 else ('C' if ksn2 is False else 'N/A')}")

    print("\n" + "=" * 120)
    if problems:
        print(f"✗ 不符 case ({len(problems)}): {problems}")
    else:
        print("✓ 全部符合声明: 所有声明非连续的轴, stride 均大于连续下界")
    print("=" * 120)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
