#!/usr/bin/env python3
"""scatter_pa_kv_cache 路径测试：4 个场景，每个场景逐轮校验整块 cache 是否正确。

这 4 个场景各自会走到不同的 L0 路径，本次改的只影响前两个：
  1. norm_contiguous   两个 cache 都连续(ND)          -> ScatterPaKvCacheWithAllocOut
  2. pa_nz_contiguous  cache 是 PA_NZ(NZ 私有格式)    -> ScatterPaKvCacheWithAllocOut
  3. strided_cache     cache 首轴非连续(base[::2])    -> ProcessNonContiguous + 原接口(原地写)，回归项
  4. key_only          不传 value                     -> ScatterPaKvCacheWithAllocOut(只申请 key 的输出)

每个场景都检查两件事：
  * 整块 cache 和 CPU 参考完全一致，包括 48 个 slot 没覆盖到的位置（独立输出 buffer 的改法最容易在这里出错）
  * cache 张量地址前后不变
另外记录 host 耗时：第 1 轮 vs 第 2..N 轮中位数。**缓存是否真的恢复不能只看耗时**，请按脚本末尾打印的
关键字抓 DEBUG 日志确认。

跑法（NPU 机器，已 source CANN 环境和本次构建算子包的 set_env.bash）：
  python test_scatter_pa_kv_cache_paths.py                        # 4 个场景 x 20 轮
  python test_scatter_pa_kv_cache_paths.py --case strided_cache    # 只跑一个场景（可重复写多个）
  python test_scatter_pa_kv_cache_paths.py --self-check            # 不需要 NPU，先验证脚本自身逻辑
结果写到 --output 目录（默认 ./scatter_pa_kv_cache_paths）：results.json + summary.csv
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

# 固定 shape：8 个 block x 32 slot；48 个 token 跨 block，且故意留出没被覆盖的 slot
BLOCKS = 8
BLOCK_SIZE = 32
HEADS = 4
K_DIM = 64
V_DIM = 32
TOKENS = 48
SEED = 20261009


def make_slots(torch):
    """48 个 slot：跨 block、互不重复（步长 37 与 256 互质），并给 cache 留出未写入的位置。"""
    return torch.tensor([(i * 37 + 5) % (BLOCKS * BLOCK_SIZE) for i in range(TOKENS)], dtype=torch.int32)


def nz_pack(x):
    """逻辑 ND [b, s, h, d] -> PA_NZ 物理排布 [b, h*d/16, s, 16]"""
    b, s, h, d = x.shape
    return x.reshape(b, s, h * d // 16, 16).permute(0, 2, 1, 3).contiguous()


def nz_unpack(x, heads, dim):
    """PA_NZ 物理排布 -> 逻辑 ND，用于把 device 上的结果取回来比较"""
    b, _, s, _ = x.shape
    return x.permute(0, 2, 1, 3).contiguous().reshape(b, s, heads, dim)


def same_tensor(actual, expected):
    """返回 (是否逐元素完全相同, 最大绝对误差)"""
    a, e = actual.cpu().contiguous(), expected.cpu().contiguous()
    if a.shape != e.shape:
        return False, float('inf')
    max_err = float((a.float() - e.float()).abs().max().item()) if a.numel() else 0.0
    return bool((a == e).all().item()), max_err


# ---------------------------------------------------------------------------------------
# 4 个场景：每个函数只负责"造输入 + 说明怎么把结果取回来"，跑循环和校验由下面统一做
# ---------------------------------------------------------------------------------------
def case_norm_contiguous(torch, torch_npu, dtype, device):
    """两个 cache 都连续(ND)，会走 ScatterPaKvCacheWithAllocOut"""
    g = torch.Generator().manual_seed(SEED)
    key = torch.randn(BLOCKS, BLOCK_SIZE, HEADS, K_DIM, generator=g).to(dtype)
    value = torch.randn(BLOCKS, BLOCK_SIZE, HEADS, V_DIM, generator=g).to(dtype)
    return {
        'key_cache': key.to(device), 'value_cache': value.to(device),   # 传给算子的 device 张量
        'key_base': key, 'value_base': value,                            # CPU 基准，用来算期望值
        'slots': make_slots(torch), 'cache_mode': 'Norm', 'with_value': True, 'dtype': dtype,
        'read_key': lambda t: t.cpu(), 'read_value': lambda t: t.cpu(),
    }


def case_pa_nz_contiguous(torch, torch_npu, dtype, device):
    """cache 是 PA_NZ(NZ 私有格式)，私有格式在 IsContiguous 里按连续处理，一样走 WithAllocOut"""
    g = torch.Generator().manual_seed(SEED)
    key = torch.randn(BLOCKS, BLOCK_SIZE, HEADS, K_DIM, generator=g).to(dtype)
    value = torch.randn(BLOCKS, BLOCK_SIZE, HEADS, V_DIM, generator=g).to(dtype)
    if torch_npu is None:  # --self-check 没有 NPU，跳过 format_cast
        key_cache, value_cache = nz_pack(key), nz_pack(value)
    else:
        key_cache = torch_npu.npu_format_cast(nz_pack(key).to(device), 29)
        value_cache = torch_npu.npu_format_cast(nz_pack(value).to(device), 29)
    return {
        'key_cache': key_cache, 'value_cache': value_cache,
        'key_base': key, 'value_base': value,
        'slots': make_slots(torch), 'cache_mode': 'PA_NZ', 'with_value': True, 'dtype': dtype,
        'read_key': lambda t: nz_unpack(t.cpu(), HEADS, K_DIM),
        'read_value': lambda t: nz_unpack(t.cpu(), HEADS, V_DIM),
    }


def case_strided_cache(torch, torch_npu, dtype, device):
    """cache 首轴非连续(base[::2])，会走 ProcessNonContiguous + 原接口（原地写）"""
    g = torch.Generator().manual_seed(SEED)
    key_base = torch.randn(2 * BLOCKS, BLOCK_SIZE, HEADS, K_DIM, generator=g).to(dtype)
    value_base = torch.randn(2 * BLOCKS, BLOCK_SIZE, HEADS, V_DIM, generator=g).to(dtype)
    return {
        'key_cache': key_base.to(device)[::2], 'value_cache': value_base.to(device)[::2],
        'key_base': key_base[::2], 'value_base': value_base[::2],
        'slots': make_slots(torch), 'cache_mode': 'Norm', 'with_value': True, 'dtype': dtype,
        'read_key': lambda t: t.cpu(), 'read_value': lambda t: t.cpu(),
    }


def case_key_only(torch, torch_npu, dtype, device):
    """不传 value（只更新 key），新接口要正确跳过 value 输出 buffer 的申请"""
    g = torch.Generator().manual_seed(SEED)
    key = torch.randn(BLOCKS, BLOCK_SIZE, HEADS, K_DIM, generator=g).to(dtype)
    return {
        'key_cache': key.to(device), 'value_cache': None,
        'key_base': key, 'value_base': None,
        'slots': make_slots(torch), 'cache_mode': 'Norm', 'with_value': False, 'dtype': dtype,
        'read_key': lambda t: t.cpu(), 'read_value': None,
    }


CASE_BUILDERS = {
    'norm_contiguous': case_norm_contiguous,
    'pa_nz_contiguous': case_pa_nz_contiguous,
    'strided_cache': case_strided_cache,
    'key_only': case_key_only,
}


# ---------------------------------------------------------------------------------------
# 跑一个场景的 N 轮
# ---------------------------------------------------------------------------------------
def run_scatter_case(torch, torch_npu, case, device, iterations):
    api = (torch_npu.npu_scatter_pa_kv_cache if hasattr(torch_npu, 'npu_scatter_pa_kv_cache')
           else torch.ops.npu.npu_scatter_pa_kv_cache.default)
    slots = case['slots']
    blocks = (slots.long() // BLOCK_SIZE).cpu()
    offs = (slots.long() % BLOCK_SIZE).cpu()
    slots_dev = slots.to(device)
    key_cache, value_cache = case['key_cache'], case['value_cache']
    g = torch.Generator().manual_seed(SEED)

    rows = []
    for it in range(1, iterations + 1):
        # 每轮换一批 key/value，cache 本身不动，避免用上一轮的结果蒙对
        key = torch.randn(TOKENS, HEADS, K_DIM, generator=g).to(case['dtype'])
        value = torch.randn(TOKENS, HEADS, V_DIM, generator=g).to(case['dtype']) if case['with_value'] else None

        # 期望值：拿 CPU 基准按 slot 写入；没被 slot 命中的位置必须保持原值
        expect_key = case['key_base'].clone()
        expect_key[blocks, offs] = key
        expect_value = None
        if case['with_value']:
            expect_value = case['value_base'].clone()
            expect_value[blocks, offs] = value

        addrs_before = [t.data_ptr() for t in (key_cache, value_cache) if t is not None]
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        if case['with_value']:
            api(key.to(device), value.to(device), key_cache, value_cache, slots_dev,
                cache_mode=case['cache_mode'])
        else:
            api(key.to(device), None, key_cache, None, slots_dev, cache_mode=case['cache_mode'])
        t1 = time.perf_counter_ns()
        torch.npu.synchronize()
        t2 = time.perf_counter_ns()

        ok_key, err_key = same_tensor(case['read_key'](key_cache), expect_key)
        ok_value, err_value = True, 0.0
        if case['with_value']:
            ok_value, err_value = same_tensor(case['read_value'](value_cache), expect_value)
        ok_addr = addrs_before == [t.data_ptr() for t in (key_cache, value_cache) if t is not None]

        rows.append({'iteration': it, 'key_ok': ok_key, 'value_ok': ok_value, 'addr_ok': ok_addr,
                     'key_max_abs': err_key, 'value_max_abs': err_value,
                     'host_us': (t1 - t0) / 1000.0, 'sync_us': (t2 - t0) / 1000.0})
        print(f'  第 {it:02d}/{iterations} 轮  key={"一致" if ok_key else "不一致"}  '
              f'value={"一致" if ok_value else "-"}  地址未变={ok_addr}  '
              f'host={rows[-1]["host_us"]:.1f}us  最大误差=({err_key:g},{err_value:g})', flush=True)
        if not (ok_key and ok_value and ok_addr):
            break
    return rows


def summarize(rows):
    passed = sum(1 for r in rows if r['key_ok'] and r['value_ok'] and r['addr_ok'])
    later = [r['host_us'] for r in rows[1:]]
    return {
        'status': 'PASS' if rows and passed == len(rows) else 'FAIL',
        'passed': passed, 'iterations': len(rows),
        'host_first_us': rows[0]['host_us'] if rows else 0.0,
        'host_later_median_us': statistics.median(later) if later else 0.0,
        'rows': rows,
    }


# ---------------------------------------------------------------------------------------
# --self-check：不需要 NPU，先验证脚本自己的参考实现和输入构造没错
# ---------------------------------------------------------------------------------------
def self_check(torch, dtype):
    print('== self-check（不调用 NPU）: 校验参考实现、未命中 slot、每个场景的 stride 形态 ==')
    all_ok = True
    for name, build in CASE_BUILDERS.items():
        case = build(torch, None, dtype, 'cpu')
        slots = case['slots']
        blocks = (slots.long() // BLOCK_SIZE).cpu()
        offs = (slots.long() % BLOCK_SIZE).cpu()
        key_base = case['key_base']

        # 1) 脚本用的向量化写法 vs 最笨的逐 token 写法，两套实现互校
        key = torch.randn(TOKENS, HEADS, K_DIM)
        vectorized = key_base.clone()
        vectorized[blocks, offs] = key
        manual = key_base.clone()
        for t in range(TOKENS):
            manual[int(blocks[t]), int(offs[t])] = key[t]
        reference_ok = torch.equal(vectorized, manual)

        # 2) 没被 slot 命中的位置保持原值
        written = set(slots.tolist())
        untouched_ok = all(torch.equal(vectorized[b, o], key_base[b, o])
                           for b in range(key_base.shape[0]) for o in range(BLOCK_SIZE)
                           if b * BLOCK_SIZE + o not in written)

        # 3) stride 形态是否符合场景意图
        cache = case['key_cache']
        if name == 'strided_cache':
            intent_ok = (not cache.is_contiguous()) and cache.stride(0) == 2 * cache.shape[1] * cache.stride(1)
            intent = '首轴非连续(应走 ProcessNonContiguous)'
        elif name == 'pa_nz_contiguous':
            intent_ok = torch.equal(nz_unpack(nz_pack(key_base), HEADS, K_DIM), key_base)
            intent = 'NZ 排布可逆(私有格式按连续处理)'
        else:
            intent_ok = cache.is_contiguous()
            intent = '连续(应走 WithAllocOut)'
        all_ok = all_ok and reference_ok and untouched_ok and intent_ok
        print(f'  {name:18s} 参考实现一致={reference_ok}  未命中slot保持={untouched_ok}  '
              f'形态={intent} 符合={intent_ok}')
    print(f'== self-check {"OK" if all_ok else "FAILED"}')
    return 0 if all_ok else 1


LOG_HINT = """
== 缓存是否恢复，需要抓日志确认（本脚本只看得到数值和耗时）==
  开 DEBUG：export ASCEND_GLOBAL_LOG_LEVEL=0; export ASCEND_SLOG_PRINT_TO_STDOUT=1
  关键字（完整英文串见 ops-math contiguous.cpp:671 / 695，这里故意不写全，免得污染 grep）：
    * 出现 "same address"  -> ViewCopy 退化成自拷贝，会走 executor->AbandonCache(true)：改前应该出现，改后应为 0
    * 出现 "move align"    -> 落到了 memcpy 兜底分支（A2/A3 不该出现）
    * 出现 "no op cache in context"（opbase rts_arg.cpp）-> 说明这次调用根本没请求缓存，
      此时"缓存是否被废弃"没有意义；要先确认调用方(op-plugin)是否用了带缓存的 EXEC_NPU_CMD 宏
  如果日志确认不再 AbandonCache 但 host 耗时没改善，说明禁缓存是别的层（torch_npu/op-plugin）干的
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', action='append', choices=sorted(CASE_BUILDERS),
                   help='只跑指定场景，可重复；默认全跑')
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    p.add_argument('--label', default='unlabeled', help='写进结果文件，便于 before/after 对比')
    p.add_argument('--output', default='scatter_pa_kv_cache_paths')
    p.add_argument('--self-check', action='store_true', help='不需要 NPU，只验证脚本自身逻辑')
    a = p.parse_args()
    if a.iterations < 1:
        p.error('--iterations must be positive')
    return a


def main():
    args = parse_args()
    import torch
    dtype = getattr(torch, args.dtype)
    if args.self_check:
        return self_check(torch, dtype)

    import torch_npu
    if not torch.npu.is_available():
        raise SystemExit('没有可用的 NPU；想只验证脚本逻辑请加 --self-check')
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, 'set_compile_mode'):
        torch.npu.set_compile_mode(jit_compile=False)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'label': args.label, 'iterations_requested': args.iterations, 'dtype': args.dtype, 'device': args.device,
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'device_name': torch.npu.get_device_name(),
        'shape': {'blocks': BLOCKS, 'block_size': BLOCK_SIZE, 'heads': HEADS,
                  'k_dim': K_DIM, 'v_dim': V_DIM, 'tokens': TOKENS},
        'env': {k: os.environ.get(k) for k in ('ASCEND_CUSTOM_OPP_PATH', 'ASCEND_OPP_PATH',
                                               'ASCEND_GLOBAL_LOG_LEVEL', 'ACLNN_CACHE_LIMIT', 'TASK_QUEUE_ENABLE')},
    }

    results, failed = {}, []
    for name in (args.case or list(CASE_BUILDERS)):
        build = CASE_BUILDERS[name]
        print(f'\n===== 场景 {name}：{build.__doc__} =====')
        try:
            case = build(torch, torch_npu, dtype, args.device)
            results[name] = summarize(run_scatter_case(torch, torch_npu, case, args.device, args.iterations))
            if results[name]['status'] != 'PASS':
                failed.append(name)
        except TypeError as exc:  # 该 torch_npu 版本不支持这个形态（例如 value 传 None）
            results[name] = {'status': 'SKIP', 'error': f'{type(exc).__name__}: {exc}'}
            print(f'  跳过：{type(exc).__name__}: {exc}')
        except Exception as exc:  # 其它异常按失败算，避免把真问题藏成跳过
            results[name] = {'status': 'ERROR', 'error': f'{type(exc).__name__}: {exc}'}
            failed.append(name)
            print(f'  出错：{type(exc).__name__}: {exc}')

    # 写结果
    (outdir / 'results.json').write_text(
        json.dumps({'metadata': dict(metadata, results={k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                                                         for k, v in results.items()}),
                    'cases': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (outdir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['场景', '结论', '通过/总轮次', 'host第1轮us', 'host第2..N轮中位数us', '备注'])
        for name, r in results.items():
            writer.writerow([name, r['status'],
                             f"{r.get('passed', '')}/{r.get('iterations', '')}",
                             f"{r.get('host_first_us', 0):.1f}", f"{r.get('host_later_median_us', 0):.1f}",
                             r.get('error', '')])

    print('\n===== 汇总 =====')
    for name, r in results.items():
        if 'passed' not in r:
            print(f'  {name:18s} {r["status"]:5s} {r.get("error", "")}')
        else:
            print(f'  {name:18s} {r["status"]}  {r["passed"]}/{r["iterations"]}  '
                  f'host: 第1轮={r["host_first_us"]:.1f}us 第2..N轮中位数={r["host_later_median_us"]:.1f}us')
    print(LOG_HINT)
    print(f'结果目录：{outdir.resolve()}')
    if failed:
        print(f'未通过场景：{failed}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
