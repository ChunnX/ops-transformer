import torch


def get_inputs(param, device=None):
    b = int(param["B"])
    h1 = int(param["H1"])
    h3 = int(param["H3"])
    w = int(param["W"])
    ll = int(param["L"])
    h2 = int(param["H2"])
    r = int(param["R"])
    dtype = torch.float16

    y = torch.randn(b, h3, device=device, dtype=dtype)
    x = torch.randn(b, h1, device=device, dtype=dtype)
    weight_b = torch.randn(w, ll, h2, r, device=device, dtype=dtype)
    weight_a = torch.randn(w, ll, r, h1, device=device, dtype=dtype)
    indices = torch.randint(0, w, (b,), device=device, dtype=torch.int32)
    return (y, x, weight_b, indices, weight_a)


def get_init_inputs(param, device=None):
    return (
        int(param["layer_idx"]),
        float(param["scale"]),
        int(param["y_offset"]),
        int(param["y_slice_size"]),
    )


def custom_check_precision(param, ref_output, test_output):
    df = (ref_output.float() - test_output.float()).abs().flatten()
    rff = ref_output.float().abs().flatten()
    rd = df / (rff + 1e-2)
    ok = df.max() <= 0.2
    return (1 if bool(ok) else 0), df, rd
