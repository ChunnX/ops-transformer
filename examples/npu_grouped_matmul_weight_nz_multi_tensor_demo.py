import torch
import torch_npu


def make_weight_nz_fp8(weight_nd):
    weight_nz = torch_npu.npu_format_cast(weight_nd.npu(), 29, torch.float8_e4m3fn)
    return weight_nz.transpose(-1, -2)


def main():
    torch.manual_seed(0)
    device = "npu"

    # Single x, multi weight, single output. Bias is intentionally not used.
    m0 = 8
    m1 = 8
    m = m0 + m1
    k = 256
    n = 512
    group_size = 32

    split_item = 3
    group_type = 0
    group_list_type = 1

    x = torch.ones((m, k), dtype=torch.float8_e4m3fn, device=device)
    group_list = torch.tensor([m0, m1], dtype=torch.int64, device=device)

    # Each expert has one independent weight tensor.
    weight0_nd = torch.ones((n, k), dtype=torch.float32)
    weight1_nd = torch.ones((n, k), dtype=torch.float32)
    weight = [
        make_weight_nz_fp8(weight0_nd),
        make_weight_nz_fp8(weight1_nd),
    ]

    # MX weight scale. Keep it multi tensor, matching the weight list.
    # group_size is 32 logically; scale storage uses K // 64 with a tail axis of 2.
    weight_scale0 = torch.randint(
        low=124, high=130, size=(n, k // (group_size * 2), 2), dtype=torch.uint8
    ).transpose(0, 1).contiguous().npu()
    weight_scale1 = torch.randint(
        low=124, high=130, size=(n, k // (group_size * 2), 2), dtype=torch.uint8
    ).transpose(0, 1).contiguous().npu()
    weight_scale = [weight_scale0, weight_scale1]

    # MX x scale. For single x, pass one tensor covering all M rows.
    per_token_scale = [
        torch.randint(
            low=124, high=130, size=(m, k // (group_size * 2), 2), dtype=torch.uint8
        ).npu()
    ]

    output = torch_npu.npu_grouped_matmul(
        [x],
        weight,
        bias=None,
        scale=weight_scale,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=per_token_scale,
        group_list=group_list,
        split_item=split_item,
        group_type=group_type,
        group_list_type=group_list_type,
        output_dtype=torch.float16,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu,
    )

    print("npu_grouped_matmul output num:", len(output))
    print("output[0].shape:", tuple(output[0].shape))
    print("output[0].dtype:", output[0].dtype)
    print("weight formats:", [torch_npu.get_npu_format(w) for w in weight])


if __name__ == "__main__":
    main()
