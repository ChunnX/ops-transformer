"""MlaPreprocess 自定义 compare 逻辑 — 三方对比

三方对比模式 (与 ATK cv_fused_double_benchmark 一致):
  1. golden    = 高精度 CPU 输出 (由 spec.golden 返回)
  2. benchmark = 低精度 CPU 输出 (在 compare 中用 compare_context.input_tensors 重新计算)
  3. actual    = aclnn NPU 输出

判定逻辑 (移植自 ATK executor.py compare_cv):
  - mare_rate = MARE(golden, actual) / max(MARE(golden, benchmark), err_thd)
  - mere_rate = MERE(golden, actual) / max(MERE(golden, benchmark), err_thd)
  - rmse_rate = RMSE(golden, actual) / max(RMSE(golden, benchmark), err_thd)
  - EB = get_eb(benchmark, actual)
  - Pass: mare_rate < 10 AND mere_rate < 2 AND rmse_rate < 2 AND EB < eb_threshold

cacheMode 分支:
  - cacheMode 0: 只判 out[0,1]
  - cacheMode 1,3: 判 out[0,1,2,3]
  - cacheMode 2: out[0,1] 用 abs_diff<=1 (int8 量化), out[2,3] 用 compare_cv
"""

import importlib.util
import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_golden_module():
    path = os.path.join(_THIS_DIR, "golden.py")
    spec = importlib.util.spec_from_file_location("_mla_golden_for_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_golden_module()
cpu_mla_preprocess_lowprec = _golden_mod.cpu_mla_preprocess_lowprec
_restore_descale = _golden_mod._restore_descale
_restore_nz = _golden_mod._restore_nz

MIN_ERR = 1e-7


# ==================== 阈值定义 — 与 ATK executor.py 一致 ====================

def get_eb_threshold(dtype):
    if dtype == torch.bfloat16:
        return 2 ** (-7)
    if dtype == torch.float16:
        return 2 ** (-10)
    if dtype == torch.float32:
        return 2 ** (-14)
    return 2 ** (-14)


def get_err_threshold(dtype):
    if dtype == torch.bfloat16:
        return 2 ** (-8)
    if dtype == torch.float16:
        return 2 ** (-11)
    if dtype == torch.float32:
        return 2 ** (-14)
    return 2 ** (-14)


# ==================== 误差指标 — 移植自 ATK executor.py ====================

def get_mare(golden, actual):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.max(abs_error.flatten()).item()


def get_mere(golden, actual):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    return torch.mean(abs_error).item()


def get_rmse(golden, actual):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    return torch.sqrt(torch.mean(sqr_err)).item()


def get_eb(golden, actual):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    return torch.mean(actual_error / golden_nmax).item()


def compare_cv(golden, benchmark, actual):
    """三方对比精度判定 — 与 ATK executor.py compare_cv 完全一致

    Args:
        golden: 高精度 CPU golden 输出
        benchmark: 低精度 CPU 标杆输出
        actual: aclnn NPU 输出

    Returns:
        bool: 是否通过
    """
    orig_dtype = actual.dtype
    err_threshold = get_err_threshold(orig_dtype)
    eb_threshold = get_eb_threshold(orig_dtype)

    mare_actual = get_mare(golden, actual)
    mare_bench = get_mare(golden, benchmark)
    mare_rate = mare_actual / max(mare_bench, err_threshold)

    mere_actual = get_mere(golden, actual)
    mere_bench = get_mere(golden, benchmark)
    mere_rate = mere_actual / max(mere_bench, err_threshold)

    rmse_actual = get_rmse(golden, actual)
    rmse_bench = get_rmse(golden, benchmark)
    rmse_rate = rmse_actual / max(rmse_bench, err_threshold)

    eb = get_eb(benchmark, actual)

    result = (mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2) and (eb < eb_threshold)

    print(f"[COMPARE-MLA] mare_rate={mare_rate:.6f} (thd=10), "
          f"mere_rate={mere_rate:.6f} (thd=2), "
          f"rmse_rate={rmse_rate:.6f} (thd=2), "
          f"eb={eb:.8f} (thd={eb_threshold:.8f}) -> "
          f"{'PASS' if result else 'FAIL'}", flush=True)
    return result


# ==================== 标杆输出计算 — 用原 dtype 低精度重新计算 ====================

def _compute_benchmark_output(compare_context, golden_outputs):
    """通过 compare_context 获取输入数据, 用低精度 CPU 重新计算标杆输出

    Args:
        compare_context: CompareContext, 包含 input_tensors 和 attributes
        golden_outputs: 高精度 golden 输出 (用于确定 dtype)

    Returns:
        list: 低精度标杆输出 [q_out, kv_cache_out, q_rope_out, kr_cache_out]
    """
    if compare_context is None or not hasattr(compare_context, 'input_tensors'):
        print("[COMPARE-MLA] WARNING: compare_context is None, fallback to golden.to(dtype)", flush=True)
        orig_dtype = golden_outputs[0].dtype
        return [g.to(orig_dtype) for g in golden_outputs]

    input_tensors = compare_context.input_tensors
    if not isinstance(input_tensors, (tuple, list)):
        print("[COMPARE-MLA] WARNING: input_tensors is not list/tuple", flush=True)
        return golden_outputs

    attributes = compare_context.attributes if compare_context.attributes else {}

    # 从 input_tensors 还原原始输入 (需反转 customize_inputs 的修改)
    (input_t, gamma0, beta0, quantScale0, quantOffset0,
     wdqkv, descale0, bias0, gamma1, beta1,
     quantScale1, quantOffset1, wuq, descale1, bias1,
     gamma2, cos, sin, wuk, kvCache, kvCacheRope,
     slotmapping, ctkvScale, qNopeScale) = input_tensors[:24]

    # 输出占位 tensor
    qOutOut = input_tensors[24] if len(input_tensors) > 24 else None
    kvCacheOutOut = input_tensors[25] if len(input_tensors) > 25 else None
    qRopeOutOut = input_tensors[26] if len(input_tensors) > 26 else None
    krCacheOutOut = input_tensors[27] if len(input_tensors) > 27 else None

    # 还原 deScale (int64 → fp32)
    deScale0 = _restore_descale(descale0)
    deScale1 = _restore_descale(descale1)

    # 还原 wdqkv/wuq NZ → ND
    if isinstance(wdqkv, torch.Tensor):
        wdqkv_cpu = wdqkv.detach().cpu()
        if wdqkv_cpu.dim() == 3 and wdqkv_cpu.shape[0] != wdqkv_cpu.shape[1]:
            # NZ format — need original shape
            # wdqkv original shape: [wdqkv_rows, wdqkv_cols]
            # For w8a8: wdqkv is int8, align=32
            # We can infer from the weight matrix dimensions
            # wdqkv shape in JSON: [864, 2304] for case 0
            # NZ shape: [col_blocks, row_blocks*16, align]
            # For int8: align=32, so col_blocks = ceil(2304/32) = 72
            # row_blocks = ceil(864/16) = 54, so row_padded = 54*16 = 864
            # NZ shape: [72, 864, 32]
            # To reverse: need to know original [row, col]
            # We can compute: row = row_padded (if no padding), col = col_blocks * align
            # But we don't know if there was padding
            # For now, use the golden_mod's transdata_nz2nd
            transdata_nz2nd = _golden_mod.transdata_nz2nd
            col_blocks, row_padded, align = wdqkv_cpu.shape
            orig_row = row_padded
            orig_col = col_blocks * align
            wdqkv = transdata_nz2nd(wdqkv_cpu, (orig_row, orig_col), wdqkv_cpu.dtype)
        else:
            wdqkv = wdqkv_cpu

    if isinstance(wuq, torch.Tensor):
        wuq_cpu = wuq.detach().cpu()
        if wuq_cpu.dim() == 3 and wuq_cpu.shape[0] != wuq_cpu.shape[1]:
            transdata_nz2nd = _golden_mod.transdata_nz2nd
            col_blocks, row_padded, align = wuq_cpu.shape
            orig_row = row_padded
            orig_col = col_blocks * align
            wuq = transdata_nz2nd(wuq_cpu, (orig_row, orig_col), wuq_cpu.dtype)
        else:
            wuq = wuq_cpu

    # 获取属性参数
    cacheMode = attributes.get('cacheMode', 3)
    if isinstance(cacheMode, (list, tuple)):
        cacheMode = int(cacheMode[0])
    else:
        cacheMode = int(cacheMode)

    doRmsNorm = attributes.get('doRmsNorm', True)
    if isinstance(doRmsNorm, (list, tuple)):
        doRmsNorm = bool(doRmsNorm[0])
    else:
        doRmsNorm = bool(doRmsNorm)

    # 构建低精度 golden 调用参数
    kwargs = dict(
        wdqDim=attributes.get('wdqDim', 0),
        qRopeDim=attributes.get('qRopeDim', 0),
        kRopeDim=attributes.get('kRopeDim', 0),
        epsilon=attributes.get('epsilon', 1e-6),
        qRotaryCoeff=attributes.get('qRotaryCoeff', 2),
        kRotaryCoeff=attributes.get('kRotaryCoeff', 2),
        transposeWdq=attributes.get('transeposeWdq', True),
        transposeWuq=attributes.get('transeposeWuq', True),
        transposeWuk=attributes.get('transeposeWuk', True),
        cacheMode=cacheMode,
        quantMode=attributes.get('quantMode', 0),
        doRmsNorm=doRmsNorm,
        wdkvSplitCount=attributes.get('wdkvSplitCount', 1),
    )

    # 转为 CPU tensor
    _to_cpu = lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t

    print(f"[BENCHMARK-MLA] Computing low-precision benchmark (cacheMode={cacheMode})", flush=True)

    benchmark_outputs = cpu_mla_preprocess_lowprec(
        _to_cpu(input_t), _to_cpu(gamma0), _to_cpu(beta0),
        _to_cpu(quantScale0), _to_cpu(quantOffset0),
        wdqkv, _to_cpu(deScale0), _to_cpu(bias0),
        _to_cpu(gamma1), _to_cpu(beta1),
        _to_cpu(quantScale1), _to_cpu(quantOffset1),
        wuq, _to_cpu(deScale1), _to_cpu(bias1),
        _to_cpu(gamma2), _to_cpu(cos), _to_cpu(sin),
        _to_cpu(wuk), _to_cpu(kvCache), _to_cpu(kvCacheRope),
        _to_cpu(slotmapping), _to_cpu(ctkvScale), _to_cpu(qNopeScale),
        attributes.get('wdqDim', 0),
        attributes.get('qRopeDim', 0),
        attributes.get('kRopeDim', 0),
        attributes.get('epsilon', 1e-6),
        attributes.get('qRotaryCoeff', 2),
        attributes.get('kRotaryCoeff', 2),
        attributes.get('transeposeWdq', True),
        attributes.get('transeposeWuq', True),
        attributes.get('transeposeWuk', True),
        cacheMode,
        attributes.get('quantMode', 0),
        doRmsNorm,
        attributes.get('wdkvSplitCount', 1),
        _to_cpu(qOutOut), _to_cpu(kvCacheOutOut),
        _to_cpu(qRopeOutOut), _to_cpu(krCacheOutOut),
    )

    return benchmark_outputs


# ==================== compare 入口 ====================

def compare(*outputs, compare_context=None, **kwargs):
    """精度比较 — 三方对比 (高精度CPU / 低精度CPU / aclnn NPU)

    Args:
        *outputs: [actual0, actual1, actual2, actual3, golden0, golden1, golden2, golden3]
                  前半为 NPU 输出，后半为 golden
        compare_context: CompareContext (包含 input_tensors, attributes)

    Returns:
        list[dict]: 每个输出的比较结果
    """
    n_outputs = len(outputs) // 2
    results = []

    # 获取 cacheMode
    cacheMode = 3
    if compare_context is not None and compare_context.attributes:
        cm = compare_context.attributes.get('cacheMode', 3)
        if isinstance(cm, (list, tuple)):
            cacheMode = int(cm[0])
        else:
            cacheMode = int(cm)

    # 计算低精度标杆输出
    golden_list = [outputs[i + n_outputs] for i in range(n_outputs)]
    benchmark_outputs = _compute_benchmark_output(compare_context, golden_list)

    for i in range(n_outputs):
        actual = outputs[i]
        golden = outputs[i + n_outputs]
        benchmark = benchmark_outputs[i] if i < len(benchmark_outputs) else None

        if golden is None or actual is None:
            results.append({"pass": True, "precision": 100.0, "error_info": None})
            continue

        # 统一为 torch.Tensor
        if not isinstance(actual, torch.Tensor):
            actual = torch.from_numpy(actual)
        if not isinstance(golden, torch.Tensor):
            golden = torch.from_numpy(golden)
        if not isinstance(benchmark, torch.Tensor):
            benchmark = torch.from_numpy(benchmark)

        actual_cpu = actual.detach().cpu()
        golden_cpu = golden.detach().cpu()
        benchmark_cpu = benchmark.detach().cpu()

        print(f"[COMPARE-MLA] output[{i}] actual={tuple(actual_cpu.shape)} golden={tuple(golden_cpu.shape)} benchmark={tuple(benchmark_cpu.shape)}", flush=True)

        # cacheMode 分支判定
        if cacheMode == 0:
            # 只判 out[0] 和 out[1]
            if i not in (0, 1):
                results.append({"pass": True, "precision": 100.0, "error_info": None})
                continue
            pass_i = compare_cv(golden_cpu, benchmark_cpu, actual_cpu)

        elif cacheMode == 2:
            # out[0,1] 用 abs_diff <= 1 (int8 量化输出)
            if i in (0, 1):
                abs_diff = torch.max(torch.abs(benchmark_cpu.flatten().to(torch.float32) -
                                                actual_cpu.flatten().to(torch.float32))).item()
                pass_i = abs_diff <= 1
                print(f"[COMPARE-MLA] output[{i}] cacheMode=2 int8: max_abs_diff={abs_diff:.1f} -> "
                      f"{'PASS' if pass_i else 'FAIL'}", flush=True)
            else:
                # out[2,3] 用 compare_cv
                pass_i = compare_cv(golden_cpu, benchmark_cpu, actual_cpu)

        else:
            # cacheMode 1, 3: 判 out[0,1,2,3]
            pass_i = compare_cv(golden_cpu, benchmark_cpu, actual_cpu)

        results.append({
            "pass": pass_i,
            "precision": 100.0 if pass_i else 0.0,
            "error_info": None if pass_i else "precision mismatch",
        })

    return results
