

"""Torch bridge tests using real NPU pointers, padded rows and byte-exact golden.

Run with the custom OPP and built Python package on PYTHONPATH. Set
FFN_TORCH_PACKAGE for a selectively packaged cann_ops_transformer vendor.
"""
import importlib
import json
import os
import struct
import time

import numpy as np
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode

ops = importlib.import_module(os.getenv('FFN_TORCH_PACKAGE', 'cann_ops_transformer'))
MASK = np.iinfo(np.int32).max
REPORTS = []
DTYPES = [torch.float16, torch.bfloat16, torch.int8, torch.float8_e5m2,
          torch.float8_e4m3fn, torch.float4_e2m1fn_x2]


def record(name):
    REPORTS.append({'case': name, 'status': 'PASS'})


class Context:
    """Own every tensor referenced by ScheduleContext until stream completion."""
    def __init__(self, dtype, h=66, a=4, bs=3, k=3):
        self.dtype, self.h = dtype, h
        self.a, self.m, self.bs, self.k, self.e = a, 3, bs, k, 8
        self.row_bytes = h * 2 if dtype < 2 else h // 2 if dtype == 5 else h
        self.scale_bytes = 4 if dtype == 2 else (h + 31) // 32 if dtype >= 3 else 0
        # Exercise actual HS stride independently of valid payload length.
        # This catches a wrapper/kernel that assumes all token rows are packed.
        self.hs = self.row_bytes + self.scale_bytes + 16
        shape = (a, self.m, self.bs, self.k, self.hs)
        self.data = np.random.default_rng(27).integers(0, 256, shape, dtype=np.uint8)
        self.data_dev = torch.from_numpy(self.data).npu()
        self.desc = np.zeros((a, self.m, 2 + self.bs * self.k), np.int32)
        self.desc[:, :, 1] = 7  # nonzero layer catches accidental header overwrite
        self.desc[:, :, 2:] = np.arange(a * self.m * self.bs * self.k).reshape(a, self.m, -1) % self.e
        if self.bs * self.k > 1:
            self.desc[:, :, -1] = MASK
        self.desc_dev = torch.from_numpy(self.desc.copy()).npu()
        self.si = np.arange(a - 1, -1, -1, dtype=np.int32)
        self.mi = np.arange(a, dtype=np.int32) % self.m
        self.si_dev, self.mi_dev = torch.from_numpy(self.si).npu(), torch.from_numpy(self.mi).npu()
        self.norm_ids = self.desc[self.si, self.mi, 2:].copy()
        self.norm_dev = torch.from_numpy(self.norm_ids).npu()
        self.ctx = np.zeros(1024, np.int8)
        for off, val in [(0, a), (4, self.m), (8, self.bs), (12, self.k), (16, self.e), (20, self.hs), (576, a)]:
            struct.pack_into('<I', self.ctx, off, val)
        for off, tensor in [(384, self.desc_dev), (400, self.data_dev), (528, self.si_dev),
                            (544, self.mi_dev), (560, self.norm_dev)]:
            struct.pack_into('<Q', self.ctx, off, tensor.data_ptr())
            struct.pack_into('<Q', self.ctx, off + 8, tensor.numel() * tensor.element_size())
        self.ctx_dev = torch.from_numpy(self.ctx.copy()).npu()

    @property
    def shape(self):
        return [self.a, self.bs, self.k, self.h]

    def prepare(self, ready, cursor=0, masked=False):
        self.desc[:, :, 0] = 0
        if masked:
            self.desc[:, :, 2:] = MASK
        for a, m in ready:
            self.desc[a, m, 0] = 1
        self.desc_dev.copy_(torch.from_numpy(self.desc))
        struct.pack_into('<Q', self.ctx, 416, cursor)
        self.ctx_dev.copy_(torch.from_numpy(self.ctx))
        torch.npu.synchronize()

    def run(self, need, sync):
        return ops.ffn_worker_batching(self.ctx_dev, self.e, self.shape,
                                      token_dtype=self.dtype, need_schedule=need, layer_num=1, sync_flag=sync)

    def check(self, result, need, selected=(), mb=0, *, check_state=True, mx_output_uint8=False):
        torch.npu.current_stream().synchronize()
        y = self.a * self.bs * self.k
        expected_shapes = [(y, self.h // 2 if self.dtype == 5 else self.h), (self.e, 2)]
        expected_shapes += [(y,)] * 4
        expected_shapes += [(y, (self.h + 31) // 32) if self.dtype >= 3 else (y,), (1,)]
        expected_dtypes = [DTYPES[self.dtype], torch.int64] + [torch.int32] * 4
        expected_dtypes += [torch.float8_e8m0fnu if self.dtype >= 3 else torch.float32, torch.int64]
        if mx_output_uint8 and self.dtype >= 3:
            expected_dtypes[6] = torch.uint8
            if self.dtype == 5:
                expected_dtypes[0] = torch.uint8
        for t, shape, dtype in zip(result, expected_shapes, expected_dtypes):
            assert tuple(t.shape) == shape and t.dtype == dtype and t.device == self.ctx_dev.device
        if self.dtype == 5 and y == 1 and self.h == 2:
            assert result[0].stride() == (2, 1)
        rows = []
        for a in (selected if need else range(self.a)):
            ids = self.desc[a, mb, 2:] if need else self.norm_ids[a]
            sa, sm = (a, mb) if need else (int(self.si[a]), int(self.mi[a]))
            for pos, e in enumerate(ids):
                if e < 1000000:
                    rows.append((int(e), sa, sm, pos // self.k, pos % self.k))
        rows.sort(key=lambda x: x[0])
        assert result[7].item() == len(rows)
        out_bytes = result[0].view(torch.uint8).cpu().numpy().reshape(y, self.row_bytes)
        if self.dtype >= 2:
            scales = result[6].view(torch.uint8).cpu().numpy().reshape(y, self.scale_bytes)
        for i, (_, a, m, b, k) in enumerate(rows):
            np.testing.assert_array_equal(out_bytes[i], self.data[a, m, b, k, :self.row_bytes])
            if self.dtype >= 2:
                np.testing.assert_array_equal(scales[i], self.data[a, m, b, k, self.row_bytes:self.row_bytes + self.scale_bytes])
        for col, out in zip(range(1, 5), result[2:6]):
            np.testing.assert_array_equal(out.cpu().numpy()[:len(rows)], [row[col] for row in rows])
        group = np.zeros((self.e, 2), np.int64)
        for i, e in enumerate(sorted({r[0] for r in rows})):
            group[i] = [e, sum(r[0] == e for r in rows)]
        np.testing.assert_array_equal(result[1].cpu().numpy(), group)
        if need:
            for a in selected:
                self.desc[a, mb, 0] = 0
                self.desc[a, mb, 2:] = MASK
            struct.pack_into('<Q', self.ctx, 416, (mb + 1) % self.m)
        # A graph may consume two batches before returning either result. Advance
        # the golden state after the first output, then check device state after
        # the second output; eager and single-call tests always check immediately.
        if check_state:
            np.testing.assert_array_equal(self.desc_dev.cpu().numpy(), self.desc)
            np.testing.assert_array_equal(self.ctx_dev.cpu().numpy(), self.ctx)
            np.testing.assert_array_equal(self.data_dev.cpu().numpy(), self.data)
        return len(rows)


def check_meta_and_errors():
    for device in ('meta', 'npu'):
        mode = FakeTensorMode() if device == 'npu' else None
        if mode:
            mode.__enter__()
        try:
            ctx = torch.empty(1024, dtype=torch.int8, device=device)
            for dtype in range(6):
                result = ops.ffn_worker_batching(ctx, 8, [4, 3, 3, 66], token_dtype=dtype, need_schedule=1, sync_flag=1)
                assert len(result) == 8 and result[0].dtype == DTYPES[dtype]
                assert tuple(result[0].shape) == (36, 33 if dtype == 5 else 66)
                assert tuple(result[6].shape) == ((36, 3) if dtype >= 3 else (36,))
                assert all(t.device == ctx.device for t in result)
                record(f'{device}_metadata_dtype_{dtype}')
                raw = ops.ffn_worker_batching(ctx, 8, [4, 3, 3, 66], token_dtype=dtype,
                                             mx_output_uint8=True)
                for i, (native, transported) in enumerate(zip(result, raw)):
                    expected = torch.uint8 if (i == 6 and dtype >= 3) or (i == 0 and dtype == 5) else native.dtype
                    assert transported.dtype == expected
                    assert transported.shape == native.shape and transported.stride() == native.stride()
                record(f'{device}_uint8_metadata_dtype_{dtype}')
        finally:
            if mode:
                mode.__exit__(None, None, None)
    ctx = torch.empty(1024, dtype=torch.int8, device='meta')
    invalid = [('token_dtype', {'token_dtype': 6}), ('sync_flag', {'sync_flag': 2}),
               ('need_schedule', {'need_schedule': -1}), ('layer_num', {'layer_num': 9})]
    for name, kwargs in invalid:
        try:
            ops.ffn_worker_batching(ctx, 8, [4, 3, 3, 66], **kwargs)
        except RuntimeError:
            record('reject_' + name)
        else:
            raise AssertionError(name)
    for name, tensor, shape, kwargs in [
        ('fp4_odd_h', ctx, [4, 3, 3, 65], {'token_dtype': 5}),
        ('short_context', ctx[:100], [4, 3, 3, 66], {}),
        ('strided_context', torch.empty(2048, dtype=torch.int8, device='meta')[::2], [4, 3, 3, 66], {}),
        ('context_dtype', ctx.float(), [4, 3, 3, 66], {}),
        ('shape_length', ctx, [4, 3, 3], {}),
    ]:
        try:
            ops.ffn_worker_batching(tensor, 8, shape, **kwargs)
        except RuntimeError:
            record('reject_' + name)
        else:
            raise AssertionError(name)
    schema = torch.ops.cann_ops_transformer.ffn_worker_batching.default._schema
    assert schema.arguments[0].alias_info.is_write
    record('schema_context_mutation')


def main():
    check_meta_and_errors()
    for dtype in range(6):
        # Both smallest packed FP4 row and a tail scale group; HS includes padding.
        for h in (2, 66):
            for need, sync in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                c = Context(dtype, h)
                mb = 2 if sync else 0
                selected = [1, 3] if sync else list(range(c.a))
                ready = [(a, mb) for a in selected] if need else []
                c.prepare(ready)
                c.check(c.run(need, sync), need, selected, mb)
                record(f'dtype_{dtype}_h_{h}_need_{need}_sync_{sync}')
        c = Context(dtype)
        c.prepare([(1, 2)], masked=True)
        assert c.check(c.run(1, 1), 1, [1], 2) == 0
        record(f'dtype_{dtype}_all_masked')
    for dtype in range(6):
        c = Context(dtype, h=2, a=1, bs=1, k=1)
        c.prepare([(0, 0)])
        assert c.check(c.run(1, 1), 1, [0], 0) == 1
        if dtype == 5:
            meta = ops.ffn_worker_batching(torch.empty(1024, dtype=torch.int8, device='meta'),
                                          c.e, c.shape, token_dtype=dtype)
            assert meta[0].stride() == (2, 1)
        record(f'singleton_dtype_{dtype}')
    # Persistent context: mb=2 is selected first, mb=0 must remain untouched.
    c = Context(5)
    c.prepare([(3, 2), (2, 0)], cursor=2)
    c.check(c.run(1, 1), 1, [3], 2)
    c.check(c.run(1, 1), 1, [2], 0)
    # Reuse a cleaned slot by publishing only its flag; stale ids must not revive.
    c.desc[3, 2, 0] = 1
    c.desc_dev[3, 2, 0].fill_(1)
    assert c.check(c.run(1, 1), 1, [3], 2) == 0
    record('persistent_fp4_context_and_reuse')
    # A non-default stream must own allocation, workspace and launch ordering.
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        c = Context(3)
        c.prepare([(3, 1)])
        c.check(c.run(1, 1), 1, [3], 1)
    record('nondefault_stream')
    # JIT was warmed above. Start with insufficient flags and publish later on
    # another stream, without reading outputs or synchronizing the consumer.
    for sync in (0, 1):
        c = Context(0)
        c.prepare([(0, 0)] if sync == 0 else [])
        consumer, producer = torch.npu.Stream(), torch.npu.Stream()
        with torch.npu.stream(consumer):
            result = c.run(1, sync)
        time.sleep(0.2)
        assert not consumer.query(), 'consumer returned before flags were published'
        publish = [1, 2, 3] if sync == 0 else [2]
        with torch.npu.stream(producer):
            for a in publish:
                c.desc_dev[a, 0, 0].fill_(1)
                c.desc[a, 0, 0] = 1
        producer.synchronize()
        with torch.npu.stream(consumer):
            c.check(result, 1, list(range(c.a)) if sync == 0 else [2], 0)
        record(f'delayed_publication_sync_{sync}')
    print(json.dumps(REPORTS, indent=2))


if __name__ == '__main__':
    main()
