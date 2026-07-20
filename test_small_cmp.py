import numpy as np

for run in range(3):
    sort = np.load(f'/tmp/sort_small_{run}.npy')
    radix = np.load(f'/tmp/radix_small_{run}.npy')
    
    k = sort.shape[2]
    exact = 0; partial = 0; none = 0; cm = 0
    overlaps = []
    for b in range(sort.shape[0]):
        s = set(sort[b,0][sort[b,0]!=-1].tolist())
        r = set(radix[b,0][radix[b,0]!=-1].tolist())
        ov = len(s & r)
        overlaps.append(ov)
        if s == r: exact += 1
        elif ov > 0: partial += 1
        else: none += 1
        if len(s) != len(r): cm += 1
    overlaps = np.array(overlaps)
    
    print(f"=== Run {run} (k={k}) ===")
    print(f"  Exact: {exact}/{sort.shape[0]}, Partial: {partial}, None: {none}, CountMismatch: {cm}")
    print(f"  Overlap: min={overlaps.min()}, max={overlaps.max()}, mean={overlaps.mean():.1f}, median={np.median(overlaps):.0f}")
    
    # By cuRealAcSeq
    for lo, hi in [(1,k), (k+1,2*k), (2*k+1,4*k), (4*k+1,1024)]:
        batches = [b for b in range(sort.shape[0]) if lo <= (b+1) <= hi]
        if not batches: continue
        ov = np.array([len(set(sort[b,0][sort[b,0]!=-1].tolist()) & set(radix[b,0][radix[b,0]!=-1].tolist())) for b in batches])
        ex = sum(1 for b in batches if set(sort[b,0][sort[b,0]!=-1].tolist()) == set(radix[b,0][radix[b,0]!=-1].tolist()))
        print(f"    [{lo:4d},{hi:4d}]: {len(batches):4d} batches, exact={ex:4d}, overlap mean={ov.mean():.1f}")
