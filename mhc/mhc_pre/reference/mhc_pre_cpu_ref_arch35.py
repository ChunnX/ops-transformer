"""
CPU reference (golden) implementation of the MhcPre operator, mirroring the
**arch35 (A5 / ascend950) kernel** (op_kernel/arch35/*).

arch35 computation flow (verified against mhc_pre_kernel_base.h / mhc_pre_split_bs.h):

  V0 (AIV0):
    xFp32   = Cast(x)                          # BF16/F16 -> FP32 (exact)
    xGamma  = gamma * xFp32                    # gamma applied here (if hasGamma); NO invRms yet
    sumSq   = sum_nD(xFp32^2)                  # NOTE: squares the ORIGINAL x (not gamma*x)
    invRms  = 1 / sqrt(sumSq * scaleMean + normEps)   # scaleMean = 1/(n*D)

  Cube (AIC):
    hMix    = xGamma . phi                     # matmul A = gamma*x (FP32), B = phi; -> hMixGm

  V1 (AIV1):
    normMix = hMix * invRms                    # invRms applied AFTER matmul (per row)
    H       = alpha * normMix + bias           # alpha per-section, bias per-column
      H_pre  = alpha_pre  * normMix[:, 0:n]      + bias[0:n]
      H_post = alpha_post * normMix[:, n:2n]     + bias[n:2n]
      H_res  = alpha_res  * normMix[:, 2n:]      + bias[2n:]   (only when alpha has 3 elems)
    h_pre   = sigmoid(H_pre)  + hcEps
    h_post  = 2 * sigmoid(H_post)
    h_res   = H_res            (pre-Sinkhorn; zeros when alpha has 2 elems)
    h_in    = round( sum_n x * h_pre )         # FP32 accumulate, cast back to BF16/F16 (rint)

Precision alignment with the arch35 NPU kernel:
  * all math in FP32 (cube matmul is FP32 x FP32 -> FP32 accumulate).
  * h_in rounded to input dtype (BF16 round-to-nearest-even / F16), matches CAST_RINT.
  * h_post / h_res / inv_rms / h_mix / h_pre are FP32 (no cast).
  * computation ORDER mirrors the kernel: gamma before matmul, invRms after matmul,
    sum(x^2) on the original (un-gamma'd) x.

Key difference vs the membase (A2/A3) reference: here hMix = (gamma*x).phi (gamma is
baked into the matmul), whereas the docs describe hMix = x.phi. When gamma is None
(inference) both coincide.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# dtype rounding helpers
# --------------------------------------------------------------------------- #
def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round a float32 array to bfloat16 (round-to-nearest-even), returned as float32."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).copy()
    bias = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)  # round-to-nearest-even
    u = (u + bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def to_f16(x: np.ndarray) -> np.ndarray:
    """Round a float32 array to float16 (round-to-nearest-even), returned as float32."""
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def round_fp(x: np.ndarray, dtype: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if dtype == "bf16":
        return to_bf16(x)
    if dtype == "f16":
        return to_f16(x)
    return x  # fp32


def sigmoid(x: np.ndarray) -> np.ndarray:
    # matches arch35: 1 / (1 + exp(-x)), FP32
    x = np.asarray(x, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))


# --------------------------------------------------------------------------- #
# arch35 reference
# --------------------------------------------------------------------------- #
def mhc_pre_cpu_arch35(x, phi, alpha, bias, gamma=None, norm_eps=1e-6, hc_eps=1e-6,
                       dtype="bf16"):
    """
    Args:
        x      : [T, n, D] or [B, S, n, D], BF16(as fp32)/F16/FP32
        phi    : [K, n*D], FP32  (K = n^2+2n when alpha has 3 elems, else 2n)
        alpha  : [3] or [2], FP32   (alpha_pre, alpha_post[, alpha_res])
        bias   : [K], FP32
        gamma  : [n, D], FP32, optional (RMSNorm scale; None => hasGamma=false)
        norm_eps, hc_eps : float
        dtype  : "bf16" | "f16" | "fp32"  (dtype of x and h_in)

    Returns:
        h_in   : [T, D],    rounded to `dtype`
        h_post : [T, n],    FP32
        h_res  : [T, n, n], FP32 (zeros if alpha has 2 elems)
        inv_rms: [T],       FP32
        h_mix  : [T, K],    FP32  ((gamma*x) . phi, the cube matmul output)
        h_pre  : [T, n],    FP32
    """
    x = np.asarray(x)
    phi = np.asarray(phi, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)

    # accept [B, S, n, D] -> [T, n, D]
    if x.ndim == 4:
        B, S, n, D = x.shape
        x = x.reshape(B * S, n, D)
    elif x.ndim == 3:
        T, n, D = x.shape
    else:
        raise ValueError(f"x must be [T,n,D] or [B,S,n,D], got shape {x.shape}")
    T = x.shape[0]

    x_fp32 = x.astype(np.float32)
    nD = n * D
    K = phi.shape[0]
    has_res = (alpha.size == 3)
    if has_res:
        assert K == n * n + 2 * n, f"phi rows {K} != n^2+2n ({n*n+2*n})"
    else:
        assert K == 2 * n, f"phi rows {K} != 2n ({2*n})"
    assert bias.size == K, f"bias size {bias.size} != K {K}"

    # ---- V0: RMSNorm (sum on ORIGINAL x^2) ----
    sum_sq = np.sum(x_fp32 * x_fp32, axis=(1, 2))                    # [T]
    scale_mean = np.float32(1.0 / (n * D))                           # arch35: scaleMean
    mean_sq = sum_sq * scale_mean                                    # sum * scaleMean
    inv_rms = np.float32(1.0) / np.sqrt(mean_sq + np.float32(norm_eps))  # [T]

    # ---- V0: xGamma = gamma * x (matmul A; NO invRms) ----
    x_gamma = x_fp32
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=np.float32)
        assert gamma.shape == (n, D), f"gamma shape {gamma.shape} != ({n},{D})"
        x_gamma = x_fp32 * gamma[None, :, :]

    # ---- Cube: hMix = (gamma*x) . phi ----
    xg_flat = x_gamma.reshape(T, nD)
    h_mix = xg_flat @ phi.T                                          # [T, K]  (FP32 matmul)

    # ---- V1: normMix = hMix * invRms (per row); H = alpha * normMix + bias ----
    norm_mix = h_mix * inv_rms[:, None]                              # [T, K]
    a_pre = float(alpha[0])
    a_post = float(alpha[1])
    pre_mix = norm_mix[:, 0:n]                                       # [T, n]
    post_mix = norm_mix[:, n:2 * n]                                  # [T, n]
    b_pre = bias[0:n]
    b_post = bias[n:2 * n]

    H_pre = a_pre * pre_mix + b_pre
    H_post = a_post * post_mix + b_post

    if has_res:
        a_res = float(alpha[2])
        res_mix = norm_mix[:, 2 * n:]                                # [T, n^2]
        b_res = bias[2 * n:]
        H_res = a_res * res_mix + b_res
        h_res = H_res.reshape(T, n, n).astype(np.float32)
    else:
        h_res = np.zeros((T, n, n), dtype=np.float32)

    # ---- activations ----
    h_pre = sigmoid(H_pre) + np.float32(hc_eps)                     # [T, n]
    h_post = np.float32(2.0) * sigmoid(H_post)                      # [T, n]

    # ---- h_in = round( sum_n x * h_pre )  (original x) ----
    h_in_fp32 = np.einsum("tnd,tn->td", x_fp32, h_pre)              # [T, D]
    h_in = round_fp(h_in_fp32, dtype)

    return h_in, h_post, h_res, inv_rms, h_mix, h_pre


# --------------------------------------------------------------------------- #
# self-test / demo
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(0)
    n, D = 4, 2560
    T = 8
    K = n * n + 2 * n  # alpha has 3 elems

    x = (rng.standard_normal((T, n, D)) * 0.5).astype(np.float32)
    x = to_bf16(x)  # mimic BF16 input
    phi = (rng.standard_normal((K, n * D)) * 0.02).astype(np.float32)
    alpha = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias = np.zeros(K, dtype=np.float32)
    gamma = (rng.standard_normal((n, D)) * 0.1).astype(np.float32)

    print(f"shapes: x={x.shape} phi={phi.shape} alpha={alpha.shape} bias={bias.shape}")

    # no gamma (inference; matches docs hMix = x.phi)
    hin, hpost, hres, invrms, hmix, hpre = mhc_pre_cpu_arch35(
        x, phi, alpha, bias, gamma=None, norm_eps=1e-6, hc_eps=1e-6, dtype="bf16")
    print(f"[no gamma] hin={hin.shape} h_post={hpost.shape} h_res={hres.shape} "
          f"inv_rms={invrms.shape} h_mix={hmix.shape} h_pre={hpre.shape}")
    print(f"  hin[0,:5]={hin[0,:5]}")
    print(f"  h_post[0,:4]={hpost[0,:4]}")
    print(f"  h_res[0,0,:4]={hres[0,0,:4]}")
    print(f"  inv_rms[:4]={invrms[:4]}")

    # with gamma (arch35 bakes gamma into hMix)
    hin2, hpost2, hres2, invrms2, hmix2, hpre2 = mhc_pre_cpu_arch35(
        x, phi, alpha, bias, gamma=gamma, norm_eps=1e-6, hc_eps=1e-6, dtype="bf16")
    print(f"[gamma]    h_mix diff (with vs without gamma) max={np.max(np.abs(hmix-hmix2)):.6f}")
    print(f"          (inv_rms identical since sum uses original x: "
          f"{np.allclose(invrms, invrms2)})")

    # alpha=2 case (h_res must be 0)
    alpha2 = np.array([1.0, 1.0], dtype=np.float32)
    K2 = 2 * n
    phi2 = (rng.standard_normal((K2, n * D)) * 0.02).astype(np.float32)
    bias2 = np.zeros(K2, dtype=np.float32)
    _, _, hres2c, _, _, _ = mhc_pre_cpu_arch35(x, phi2, alpha2, bias2, dtype="bf16")
    print(f"[alpha=2] h_res all zero: {np.allclose(hres2c, 0)}")


if __name__ == "__main__":
    _demo()
