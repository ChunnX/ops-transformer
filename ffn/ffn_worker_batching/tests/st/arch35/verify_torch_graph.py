"""Ascend950 full-graph tests; reuse eager's real-pointer, byte-exact golden.

Requires the built Torch package and the matching custom OPP. Every compile uses
fullgraph=True so an eager graph break cannot masquerade as graph support.
Keep Context objects alive: the compiler cannot own allocations behind pointers.
"""
import argparse
import json
from pathlib import Path
import struct
import time

import torch
import torch_npu

from verify_torch_interface import Context, MASK, ops


def compile_call(c, need=1, sync=1):
    # Capture only public metadata. Never read ready flags or the cursor on host
    # while tracing. Each graph execution must run the device polling loop.
    dtype, shape, expert_num = c.dtype, c.shape, c.e

    def call(ctx):
        return ops.ffn_worker_batching(ctx, expert_num, shape, token_dtype=dtype,
                                      need_schedule=need, layer_num=1, sync_flag=sync, mx_output_uint8=True)

    return torch.compile(call, backend='npugraph_ex', fullgraph=True, dynamic=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reports = []

    def record(name):
        reports.append({'case': name, 'status': 'PASS'})
        print(f'PASS {name}', flush=True)

    for dtype in range(6):
        for need, sync in ((0, 0), (0, 1), (1, 0), (1, 1)):
            torch._dynamo.reset()
            c = Context(dtype)
            selected = [1, 3] if sync else list(range(c.a))
            mb = 2 if sync else 0
            c.prepare([(a, mb) for a in selected] if need else [])
            fn = compile_call(c, need, sync)
            c.check(fn(c.ctx_dev), need, selected, mb, mx_output_uint8=True)
            record(f'dtype_{dtype}_need_{need}_sync_{sync}')
        torch._dynamo.reset()
        c = Context(dtype)
        c.prepare([(1, 2)], masked=True)
        fn = compile_call(c)
        assert c.check(fn(c.ctx_dev), 1, [1], 2, mx_output_uint8=True) == 0
        record(f'dtype_{dtype}_all_masked')
        torch._dynamo.reset()
        c = Context(dtype, h=2, a=1, bs=1, k=1)
        c.prepare([(0, 0)])
        fn = compile_call(c)
        assert c.check(fn(c.ctx_dev), 1, [0], 0, mx_output_uint8=True) == 1
        record(f'dtype_{dtype}_singleton')
        torch._dynamo.reset()
        c = Context(dtype)
        c.prepare([(3, 2), (2, 0)], cursor=2)
        fn = compile_call(c)
        c.check(fn(c.ctx_dev), 1, [3], 2, mx_output_uint8=True)
        c.check(fn(c.ctx_dev), 1, [2], 0, mx_output_uint8=True)
        # Same slot, same addresses, new publication: old expert ids were cleared.
        c.desc[3, 2, 0] = 1
        c.desc_dev[3, 2, 0].fill_(1)
        assert c.check(fn(c.ctx_dev), 1, [3], 2, mx_output_uint8=True) == 0
        record(f'dtype_{dtype}_persistent_replay')

    # Two consumers in one graph must observe successive cursor/descriptor state.
    # Returning neither first-call output nor the context must still retain its
    # side effect. Test both forms to catch dead-code elimination and reordering.
    for discard_first in (False, True):
        torch._dynamo.reset()
        c = Context(0)
        c.prepare([(3, 2), (2, 0)], cursor=2)

        def two_calls(ctx):
            first = ops.ffn_worker_batching(ctx, 8, [4, 3, 3, 66], need_schedule=1, sync_flag=1)
            second = ops.ffn_worker_batching(ctx, 8, [4, 3, 3, 66], need_schedule=1, sync_flag=1)
            return second if discard_first else (first, second)

        fn = torch.compile(two_calls, backend='npugraph_ex', fullgraph=True, dynamic=False)
        result = fn(c.ctx_dev)
        if discard_first:
            c.desc[3, 2, 0] = 0
            c.desc[3, 2, 2:] = MASK
            struct.pack_into('<Q', c.ctx, 416, 0)
            c.check(result, 1, [2], 0, mx_output_uint8=True)
        else:
            c.check(result[0], 1, [3], 2, check_state=False)
            c.check(result[1], 1, [2], 0, mx_output_uint8=True)
        record(f'two_calls_discard_first_{discard_first}')

    # Verify replay still polls live external flags. Warm the graph with valid
    # work first; then reset context contents without changing any device address.
    for sync in (0, 1):
        torch._dynamo.reset()
        c = Context(0)
        c.prepare([(a, 0) for a in range(c.a)])
        fn = compile_call(c, sync=sync)
        c.check(fn(c.ctx_dev), 1, list(range(c.a)), 0)
        c.prepare([(0, 0)] if sync == 0 else [])
        consumer, producer = torch.npu.Stream(), torch.npu.Stream()
        with torch.npu.stream(consumer):
            result = fn(c.ctx_dev)
        time.sleep(0.2)
        assert not consumer.query(), 'graph returned before publication'
        publish = [1, 2, 3] if sync == 0 else [2]
        with torch.npu.stream(producer):
            for a in publish:
                c.desc_dev[a, 0, 0].fill_(1)
                c.desc[a, 0, 0] = 1
        producer.synchronize()
        with torch.npu.stream(consumer):
            c.check(result, 1, list(range(c.a)) if sync == 0 else [2], 0)
        record(f'delayed_publication_sync_{sync}')

    report = {'backend': 'npugraph_ex', 'fullgraph': True, 'dynamic': False,
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'device': torch.npu.get_device_name(0), 'supported_token_dtypes': list(range(6)), 'mx_output_uint8': True, 'cases': reports,
              'passed': len(reports)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
