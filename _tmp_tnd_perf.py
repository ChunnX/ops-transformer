"""Quantify old / new TND sparse-deter rounds vs mainline, k=32."""
from math import ceil

K = 32
CUBE = 128


def ceil_div(a, b):
    return 0 if b == 0 else (a + b - 1) // b


def outer_mn(s1, s2):
    if s1 <= 0 or s2 <= 0:
        return 0, 0
    return max(1, ceil_div(s1, CUBE)), max(1, ceil_div(s2, CUBE))


def lu_virt(m, n):
    if m < n:
        n = m
    return 2 * m - n + 1, n


def prefer_pool_old(k, vm, vn, pairs):
    cols = vn * pairs
    return pairs > 0 and cols > 0 and (cols < k or vm < min(k, vn))


def prefer_pool_new(k, vm, vn, pairs, group):
    if pairs <= 0 or vm <= 0 or vn <= 0:
        return True
    cols = vn * pairs
    return cols < k or vm < min(k, vn) or group <= 1 or pairs < k or vm * vn <= 8


def band_shape(s1, s2, pre, nxt):
    m, n = outer_mn(s1, s2)
    if m < 1 or n < 1:
        return 0, 0, 0, 0, 0
    p = min(m, ceil_div(pre, CUBE) + 1)
    q = min(n, ceil_div(nxt, CUBE) + 1)
    if p + q <= m:
        l1 = q - 1
        l2 = min(n - q + 1, m + 2 - p - q)
        l3 = max(0, min(p + n - m - 1, p + q - 2))
        actual_m = (p + q + l2 - 2) if l3 == 0 else m
        actual_n = l1 + l2 + l3
        kind = 43 if p >= actual_m else 4
        return actual_m, actual_n, p, q, kind
    actual_n = min(m - 1 + q, n)
    kind = 43 if p >= m else 4
    return m, actual_n, p, q, kind


def wide_cols(m, n, p, q):
    n_new = min(m - 1 + q, n)
    l1 = m - p
    l2 = p + q - m
    l3 = n_new - l1 - l2
    fold = max(0, l3 - p + 1)
    return n_new - fold


def narrow_rounds(k, n1, m, n, p, q):
    l1 = q - 1
    l2 = min(n - q + 1, m + 2 - p - q)
    l3 = max(0, min(p + n - m - 1, p + q - 2))
    overlap = 0
    n_new = l1 + l2 + l3
    if l3 != 0 and p <= l3:
        overlap = min(l3 - p + 1, l1)
    return (p + q - 1) * ceil_div((n_new - overlap) * n1 + overlap, k)


def single_causal_rounds(k, m, n):
    t = n // (2 * k)
    ell = n % (2 * k)
    rounds2k = (2 * m + 1) * t - 2 * k * t * t
    if ell == 0:
        return rounds2k
    if t == 0:
        if 2 * k < m + 1 and k < n:
            return max(0, 2 * m - 2 * k + 1)
        return m
    l = m - 2 * k * t
    return rounds2k + sum(max(0, l - s * k) for s in range(ceil_div(ell, k)))


def group_runs(shapes):
    runs = []
    i = 0
    while i < len(shapes):
        j = i + 1
        while j < len(shapes) and shapes[j] == shapes[i]:
            j += 1
        runs.append((shapes[i], j - i))
        i = j
    return runs


def is_band_case(sparse, pre, nxt, max_s1, max_s2):
    if sparse == 1:
        return False
    if sparse == 0 and pre >= max_s1 and -128 < nxt <= 0:
        return False
    if sparse == 0 and pre >= max_s1 and nxt >= max_s2:
        return False
    return True


def schedule(qs, ks, n1, sparse, pre, nxt, k=K, policy="new"):
    max_s1, max_s2 = max(qs), max(ks)
    is_band = is_band_case(sparse, pre, nxt, max_s1, max_s2)
    pair_one = (n1 // 2) if n1 % 2 == 0 else ((n1 - 1) // 2)
    shapes = []
    for s1, s2 in zip(qs, ks):
        if is_band:
            shapes.append(band_shape(s1, s2, pre, nxt))
        else:
            m, n = outer_mn(s1, s2)
            if m < n:
                n = m
            shapes.append((m, n, 0, 0, 2))

    r_line = r_pool_area = r_single_area = 0
    pool_m = pool_n = single_m = single_n = 0
    n_line = n_pool = 0
    for shape, group in group_runs(shapes):
        m, n, p, q, kind = shape
        total = n1 * group
        pairs = total // 2
        has_single = total % 2 == 1
        if m < 1 or n < 1:
            continue
        if not is_band:
            vm, vn = lu_virt(m, n)
            small = prefer_pool_new(k, vm, vn, pairs, group) if policy == "new" else prefer_pool_old(k, vm, vn, pairs)
            if policy == "new" and pairs <= 0 and has_single:
                r_single_area += m * n * group
                single_m, single_n = max(single_m, m), max(single_n, n)
                n_pool += group
            elif small:
                r_pool_area += vm * vn * group
                pool_m, pool_n = max(pool_m, vm), max(pool_n, vn)
                n_pool += group
                if n1 % 2 == 1:
                    r_single_area += m * n * group
                    single_m, single_n = max(single_m, m), max(single_n, n)
            else:
                r_line += ceil_div(vn * pairs, k) * vm
                if has_single:
                    r_line += single_causal_rounds(k, m, n)
                n_line += group
        elif kind == 43:
            small = prefer_pool_new(k, m, n, total, group) if policy == "new" else prefer_pool_old(k, m, n, total)
            if small:
                r_pool_area += m * n * group
                pool_m, pool_n = max(pool_m, m), max(pool_n, n)
                n_pool += group
            else:
                r_line += ceil_div(n * total, k) * m
                n_line += group
        elif p + q <= m:
            if policy == "new" and m * n <= 8:
                r_pool_area += m * n * group
                pool_m, pool_n = max(pool_m, m), max(pool_n, n)
                n_pool += group
            else:
                r_line += narrow_rounds(k, total, m, n, p, q)
                n_line += group
        else:
            dn = wide_cols(m, n, p, q)
            small = prefer_pool_new(k, m, dn, total, group) if policy == "new" else prefer_pool_old(k, m, dn, total)
            if small:
                r_pool_area += m * dn * group
                pool_m, pool_n = max(pool_m, m), max(pool_n, dn)
                n_pool += group
            else:
                r_line += ceil_div(dn * total, k) * m
                n_line += group

    copies = n1 if is_band else pair_one
    r_pool = max(ceil_div(r_pool_area * copies, k), max(pool_m, pool_n)) if r_pool_area > 0 and copies > 0 else 0
    r_single = max(ceil_div(r_single_area, k), max(single_m, single_n)) if r_single_area > 0 else 0
    return {
        "band": is_band,
        "r_line": r_line,
        "r_pool": r_pool,
        "r_single": r_single,
        "total": r_line + r_pool + r_single,
        "n_line": n_line,
        "n_pool": n_pool,
    }


def mainline(qs, ks, n1, sparse, pre, nxt, k=K):
    max_s1, max_s2 = max(qs), max(ks)
    is_band = is_band_case(sparse, pre, nxt, max_s1, max_s2)
    if is_band:
        area = 0
        mn_max = 0
        for s1, s2 in zip(qs, ks):
            m, n, p, q, kind = band_shape(s1, s2, pre, nxt)
            if m < 1 or n < 1:
                continue
            if p + q <= m:
                cells = min(m, n) * (p + q - 1)
                mn_max = max(mn_max, p + q - 1)
            else:
                n_new = min(m - 1 + q, n)
                if p + q <= n:
                    cells = m * (p + q - 1)
                    mn_max = max(mn_max, m, p + q - 1)
                else:
                    cells = m * n_new
                    mn_max = max(mn_max, m, n_new)
            area += cells
        n11, n12 = n1 // k, n1 % k
        return area * n11 + (max(ceil_div(area * n12, k), mn_max) if n12 else 0)

    pref0 = 0
    m0 = 0
    p1 = p2 = 0
    m1 = m2 = 0
    n12 = (n1 % k) % 2
    for s1, s2 in zip(qs, ks):
        m, n = outer_mn(s1, s2)
        if m < n:
            n = m
        if m < 1 or n < 1:
            continue
        pref0 += (2 * m - n + 1) * n
        m0 = max(m0, 2 * m - n + 1)
        if n12:
            p1 += (m - (n + 1) // 2 + 1) * (n // 2)
            p2 += (m - n // 2) * ((n + 1) // 2)
            m1 = max(m1, m - (n + 1) // 2 + 1)
            m2 = max(m2, m - n // 2)
    n11 = (n1 % k) // 2
    r = (pref0 // 2) * (n1 // k)
    if n11:
        r += max(ceil_div(pref0 * n11, k), m0)
    if n12:
        r += max(ceil_div(p1, k), m1) + max(ceil_div(p2, k), m2)
    return r


SEQ_001 = [506, 756, 543, 1710, 571, 1649, 1908]
SEQ_0017 = [
    2, 296, 11, 10, 2, 1, 2, 4, 9, 2, 2, 33, 12, 7, 1, 1, 2, 1, 4, 1, 1, 262, 2, 5, 2, 2, 1, 12, 8, 2, 1, 243,
    10, 7, 1, 1, 2, 18, 7, 2, 2, 97, 12, 10, 2, 1, 1, 8, 5, 2, 1, 240, 13, 8, 2, 2, 2, 1, 7, 2, 2, 269, 18, 5,
    1, 1, 1, 15, 1, 1, 1, 187, 7, 2, 2, 2, 2, 18, 9, 2, 1, 179, 3, 3, 1, 2, 2, 20, 3, 1, 1, 190, 3, 5, 1, 1, 1,
    20, 5, 2, 1, 243, 14, 2, 2, 1, 1, 14, 9, 2, 1, 130, 5, 3, 1, 1, 1, 12, 2, 2, 2, 240, 18, 2, 2, 1, 2, 10, 9,
    2, 1, 232, 3, 9, 2, 2, 2, 16, 3, 2, 2, 21, 19, 8, 2, 1, 2, 2, 5, 1, 2, 185, 8, 8, 1, 2, 1, 5, 1, 2, 2, 170,
    12, 4, 2, 2, 2, 17, 4, 1, 2, 66, 11, 7, 2, 2, 2, 4, 9, 2, 1, 260, 10, 4, 2, 2, 1, 8, 8, 1, 2, 269, 3, 8, 2,
    2, 1, 4, 7, 1, 1, 200, 17, 9, 1, 1, 1, 13, 6, 2, 1, 263, 6, 10, 2, 1, 1, 17, 7, 1, 1, 131, 14, 2, 2, 2, 2,
    3, 1, 1, 2, 5, 5, 1, 1, 1, 1, 1, 10, 2, 1, 237, 20, 1, 1, 2, 2, 16, 10, 2, 2, 144, 12, 10, 2, 1, 1, 4, 7, 1,
    1, 292, 8, 9, 1, 2, 1, 18, 7, 1, 1, 66, 13, 10, 2, 1, 2, 16, 8, 1, 2, 11, 7, 9, 1, 1, 2, 3, 8, 1, 2, 4, 17,
    7, 2, 2, 1, 11, 1, 2, 1, 70, 12, 6, 1, 1, 2, 10, 8, 1, 1, 20, 20, 6, 1, 2, 2, 17, 7, 2, 2, 291, 9, 7, 2, 2,
    2, 9, 7, 1, 1, 13, 17, 4, 1, 2, 2, 20, 3, 1, 2, 82, 19, 9, 1, 2, 1, 1, 4, 2, 1, 119, 1, 5, 1, 2, 1, 15, 5,
    1, 2, 246, 9, 3, 1, 2, 2, 10, 4, 2, 1, 124, 2, 1, 2, 1, 1, 12, 7, 1, 1, 174, 6, 10, 2, 1, 1, 17, 1, 1, 1,
    225, 11, 8, 1, 2, 1, 9, 1, 1, 2, 150, 7, 2, 2, 1, 1, 7, 9, 1, 1, 124, 19, 2, 2, 2, 1, 3, 4, 1, 2, 134, 17,
    1, 2, 2, 1, 16, 6, 1, 2, 22, 3, 4, 1, 2, 2, 17, 9, 2, 1, 68, 14, 6, 2, 1, 2, 2, 9, 1, 2, 249, 2, 6, 2, 2, 1,
    16, 9, 2, 1, 269, 8, 2, 2, 2, 2, 10, 2, 2, 2, 179, 17, 1, 1, 1, 2, 13, 4, 1, 2, 188, 1, 8, 1, 1, 1, 6, 8, 2,
    2, 23, 5, 10, 2, 1, 1, 20, 8, 1, 2, 56, 16, 3, 1, 1, 1, 12, 9, 1, 2, 19, 20, 10, 1, 2, 1, 5, 2, 1, 2, 177,
    7, 4, 1, 2, 1, 13, 7, 2, 2, 106, 13, 3, 1, 2, 2, 8, 3, 1, 2, 50, 2, 9, 1, 1, 2, 5, 7, 2, 2, 198, 18, 10, 2,
    1, 1, 12, 5, 2, 2, 114, 12, 2, 1, 2, 2, 4, 3, 2, 2, 168, 12, 1, 2, 2, 2, 9, 6, 1, 2, 102, 11, 4, 2, 1, 1,
    18, 10, 1, 2, 190, 18, 10, 1, 2, 1, 15, 1, 2, 1, 288, 6, 10, 1, 2, 1, 17, 2, 1, 2, 131, 17, 7, 1, 2, 2, 17,
    6, 2, 1, 190, 9, 9, 1, 2, 1, 13, 8, 2, 2, 172, 12, 9, 2, 2, 1, 10, 3, 2, 1, 59, 5, 10, 2, 2, 1, 12, 10, 1,
    1, 248, 17, 3, 1, 1, 2, 3, 7, 2, 2, 82, 16, 9, 2, 1, 2, 11, 2, 2, 1, 117, 2, 8, 1, 1, 2, 4, 7, 2, 2, 265,
    14, 10, 1, 1, 1, 20, 9, 2, 1, 221, 20, 3, 2, 1, 1, 7, 4, 1, 1, 18, 18, 6, 2, 2, 1, 5, 8, 1, 1, 293, 9, 4, 1,
    2, 1, 11, 9, 2, 1, 210, 18, 5, 2, 2, 1, 2, 8, 1, 1, 6, 10, 6, 2, 1, 1, 17, 8, 2, 1, 186, 7, 2, 1, 1, 2, 18,
    10, 1, 1, 179, 12, 9, 2, 1, 1, 7, 10, 1, 2, 39, 4, 3, 2, 1, 1, 15, 8, 1, 2, 254, 18, 6, 2, 1, 1, 9, 4, 1, 2,
    160, 16, 3, 1, 2, 2, 10,
]
SEQ_PSE = [
    26, 902, 383, 184, 1, 6, 26, 20, 33, 15, 32, 34, 10, 40, 20, 33, 23, 30, 35, 31, 22, 17, 34, 21, 28, 5, 38,
    14, 23, 31, 32, 3, 24, 30, 8, 4, 9, 491, 80, 10, 8, 18, 11, 22, 11, 28, 10, 3, 37, 4, 33, 36, 8, 31, 25, 22,
    28, 25, 14, 23, 19, 30, 33, 25, 12, 2, 27, 31, 10, 11, 19, 1750, 563, 197, 12, 26, 28, 23, 10, 30, 23, 4, 9,
    26, 4, 30, 39, 26, 7, 1, 28, 21, 14, 21, 39, 19, 12, 16, 10, 7, 19, 29, 40, 14, 6, 26, 435, 437, 155, 33, 35,
]
SEQ_FP32 = [48, 1466, 87, 228, 43, 113, 113, 84, 120, 155, 230, 1787, 194, 66, 140, 226, 87, 52, 284, 159]
SEQ_FP32_K = [54, 1646, 47, 40, 148, 74, 82, 28, 170, 76, 213, 947, 196, 10, 126, 234, 47, 126, 193, 119]

CASES = {
    "001": dict(qs=SEQ_001, ks=SEQ_001, n1=22, sparse=0, pre=512, nxt=0),
    "0017": dict(qs=SEQ_0017, ks=SEQ_0017, n1=30, sparse=0, pre=65536, nxt=0),
    "PSE": dict(qs=SEQ_PSE, ks=SEQ_PSE, n1=1, sparse=1, pre=65536, nxt=0),
    "FP32003": dict(qs=SEQ_FP32, ks=SEQ_FP32_K, n1=32, sparse=0, pre=1650, nxt=0),
    "fa_282": dict(qs=[2] * 27 + [1] * 255, ks=[2] * 27 + [1] * 255, n1=19, sparse=1, pre=1, nxt=0),
    "fa_196": dict(qs=[2] * 196, ks=[2] * 196, n1=5, sparse=1, pre=1143159147, nxt=0),
    "n1_22_causal": dict(qs=SEQ_001, ks=SEQ_001, n1=22, sparse=1, pre=65536, nxt=0),
}


def fmt(d):
    return f"{d['total']:5} (L{d['r_line']}+P{d['r_pool']}+S{d['r_single']} lineBat={d['n_line']} poolBat={d['n_pool']})"


if __name__ == "__main__":
    print(f"k={K}")
    for name, c in CASES.items():
        old = schedule(c["qs"], c["ks"], c["n1"], c["sparse"], c["pre"], c["nxt"], policy="old")
        new = schedule(c["qs"], c["ks"], c["n1"], c["sparse"], c["pre"], c["nxt"], policy="new")
        ml = mainline(c["qs"], c["ks"], c["n1"], c["sparse"], c["pre"], c["nxt"])
        ratio = (new["total"] / ml) if ml else float("inf")
        print(f"{name:14} band={str(new['band']):5} old={fmt(old)}")
        print(f"{'':14}          new={fmt(new)}  main={ml:5}  new/main={ratio:.2f}")
