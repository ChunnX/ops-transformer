import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, atten_mask):
        # query/key/value shape: BSND, transpose to BNSD for matmul
        scale = 1.0 / math.sqrt(128)
        q = query.transpose(1, 2)  # BSND -> BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        qk = torch.matmul(q, k.transpose(2, 3)).mul(scale)
        qk = qk + atten_mask * (-10000.0)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1)
        attention_out = torch.matmul(softmax_res, v)
        return attention_out.transpose(1, 2)  # BNSD -> BSND

    def custom_op_exec(self, query, key, value, sparse_params):
        scale = 1.0 / math.sqrt(128)
        sparse_mode, pre_tokens, next_tokens = sparse_params
        seq_len = query.shape[1]
        atten_mask = None

        if pre_tokens < seq_len or next_tokens < seq_len:
            atten_mask = torch.tril(
        torch.ones((2048, 2048), dtype=torch.uint8, device=query.device), diagonal=-1)
            # if sparse_mode == 2:
            #     atten_mask = torch.triu(
            #         torch.ones(128, 128, dtype=torch.bool, device=query.device), diagonal=-1)
            # else:
            #     shape = [1, 1, seq_len, seq_len]
            #     mask = torch.ones(shape, dtype=torch.bool, device=query.device)
            #     atten_mask_u = torch.triu(mask, diagonal=next_tokens + 1)
            #     atten_mask_l = torch.tril(mask, diagonal=-pre_tokens - 1)
            #     atten_mask = atten_mask_u | atten_mask_l
            # atten_mask = atten_mask.transpose(-1, -2).contiguous()

        return torch_npu.npu_fusion_attention_v2(
            query, key, value, head_num=8, input_layout="BSND", scale=scale, sparse_mode=sparse_params[0],
            atten_mask=atten_mask, pre_tokens=pre_tokens, next_tokens=next_tokens)

    # sparse_params = [sparse_mode, pre_tokens, next_tokens]
    def get_atten_mask(self, sparse_mode=0, pre_tokens=65536, next_tokens=65536):
        atten_masks = []
        shape = [1, 1, 4096, 4096]
        if sparse_mode == 0:
            atten_mask_u = np.triu(np.ones(shape), k=next_tokens + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-pre_tokens - 1)
            atten_masks = atten_mask_u + atten_mask_l

        elif sparse_mode == 1:
            atten_masks = np.zeros(shape)
            pre_tokens = 65536
            next_tokens = 65536

        elif sparse_mode == 2:
            atten_masks = np.triu(np.ones(shape), k=1)

        elif sparse_mode == 3:
            atten_masks = np.triu(np.ones(shape), k=1)

        elif sparse_mode == 4:
            atten_mask_u = np.triu(np.ones(shape), k=next_tokens + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-pre_tokens - 1)
            atten_masks = atten_mask_u + atten_mask_l

        atten_mask = torch.tensor(atten_masks).to(torch.float16)
        return atten_mask

    def check_result(self, query, key, value, sparse_params):
        atten_mask = self.get_atten_mask(sparse_params[0], sparse_params[1], sparse_params[2])
        output = self.supported_op_exec(query.float(), key.float(), value.float(), atten_mask)
        fa_result = self.custom_op_exec(query.npu(), key.npu(), value.npu(), sparse_params)
        self.assertRtolEqual(output.half(), fa_result[0].cpu(), prec=0.01, prec16=0.01)

    def test_npu_flash_attention(self, device="npu"):
        # BSND layout: [B, S, N, D]
        query = torch.randn(8, 4096, 8, 128, dtype=torch.float16)
        key = torch.randn(8, 4096, 8, 128, dtype=torch.float16)
        value = torch.randn(8, 4096, 8, 128, dtype=torch.float16)

        # sparse_params: [sparse_mode, pre_tokens, next_tokens]
        sparse_params_list = [
            [2, 65536, 0],
            [0, 65536, 65536],
        ]

        for sparse_params in sparse_params_list:
            self.check_result(query, key, value, sparse_params)

if __name__ == "__main__":
    run_tests()
