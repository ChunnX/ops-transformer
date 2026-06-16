#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 torch.save 导出的 .pt 读取 metadata tensor，写入 raw int32 .bin 供 C++ 读取。

纯 CPU 流程：不 import torch_npu，不占用 NPU 设备。
即使 .pt 里存的是 NPU tensor，也会通过 map_location='cpu' 落到 host 再写 bin。
"""

import sys


def _load_obj(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_tensor(obj):
    import torch
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("metadata", "meta", "data", "result"):
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]
        for v in obj.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"无法在 .pt 中找到 Tensor，实际类型: {type(obj)}")


def _to_host_int32(tensor):
    import torch
    arr = tensor.contiguous().reshape(-1).detach().cpu().to(dtype=torch.int32)
    if arr.device.type != "cpu":
        raise RuntimeError(f"metadata 未在 CPU 上，当前 device={arr.device}")
    return arr


def main():
    if len(sys.argv) != 3:
        print("用法: load_metadata_pt.py <metadata.pt> <out.bin>", file=sys.stderr)
        sys.exit(1)
    pt_path, bin_path = sys.argv[1], sys.argv[2]
    obj = _load_obj(pt_path)
    arr = _to_host_int32(_extract_tensor(obj))
    with open(bin_path, "wb") as f:
        f.write(arr.numpy().tobytes())
    print(f"elems={arr.numel()} dtype=int32 device=cpu -> {bin_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[load_metadata_pt.py] 失败: {exc}", file=sys.stderr)
        sys.exit(2)
