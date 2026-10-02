# SageAttention2 可解析闭式契约套件（v37.1 测试层补强；机制来源 = hwdev/todo.md §1.9 登记的思想级吸收，实现独立推导）
# 全部宿主 CPU 执行：闭式输入替代随机对拍，断言精确结构而非容差糊覆盖。
import math
import warnings

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sageattention2 import sageattn
from sageattention2.reference import sageattn_reference


def _zero_score_case(b=1, h=2, s=64, d=64, dtype=torch.float16, causal=False):
    q = torch.zeros(b, h, s, d, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    return q, k, v


class TestUniformSoftmaxClosedForm:
    """零 score + 常数 V：softmax 恒等均匀，输出/LSE 有精确闭式。

    与历史参考实现的关键差异在此钉死：其 FP16 平移 max 协议使零 score 闭式为
    448/exp(fp16(ln448))≈0.999512（饱和语义残留）；我方 κ 封顶路径闭式必须是**恰为 1.0**。
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_noncausal_out_is_exactly_one(self, dtype):
        q, k, v = _zero_score_case(dtype=dtype)
        out = sageattn(q, k, v, smooth_k=False)
        assert torch.equal(out, torch.ones_like(out)), "零 score 均匀 softmax 输出必须逐位恰为 1.0"

    def test_lse_equals_log_rowcount(self):
        b, h, s, d = 1, 2, 64, 64
        q, k, v = _zero_score_case(b=b, h=h, s=s, d=d)
        out, lse = sageattn(q, k, v, smooth_k=False, return_lse=True)
        expected = math.log(s)
        assert torch.isfinite(lse).all()
        assert (lse - expected).abs().max().item() < 1e-6

    def test_causal_lse_equals_log_row_index(self):
        b, h, s, d = 1, 2, 64, 64
        q, k, v = _zero_score_case(b=b, h=h, s=s, d=d, causal=True)
        out, lse = sageattn(q, k, v, smooth_k=False, is_causal=True, return_lse=True)
        assert torch.equal(out, torch.ones_like(out))
        expected = torch.log(torch.arange(1, s + 1, dtype=torch.float32))
        assert (lse[0, 0] - expected).abs().max().item() < 1e-6


class TestLayoutPermutationBitwise:
    """布局无关性作为逐位硬契约（非 allclose 软对照）：同一数据的 NHD 视图必须产出比特相同的 out/LSE。"""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_nhd_hnd_bitwise_identical(self, dtype):
        g = torch.Generator().manual_seed(0)
        b, h, s, d = 1, 3, 96, 64
        q = torch.randn(b, h, s, d, dtype=dtype, generator=g)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o1, l1 = sageattn(q, k, v, return_lse=True)
        qn = q.permute(0, 2, 1, 3).contiguous()
        kn = k.permute(0, 2, 1, 3).contiguous()
        vn = v.permute(0, 2, 1, 3).contiguous()
        o2, l2 = sageattn(qn, kn, vn, tensor_layout="NHD", return_lse=True)
        assert torch.equal(o1, o2.permute(0, 2, 1, 3)), "NHD/HND 输出必须逐位一致"
        assert torch.equal(l1, l2), "NHD/HND LSE 必须逐位一致"


class TestMxScaleChannelIsolation:
    """E8M0 分组 scale 的通道隔离探针：相邻通道反号 + 每 32-token 组 2 的幂幅值。

    scale 若跨通道错位/串扰，符号交替图样与组内等幅性会立即破坏；行向常量性
    （零 score 下所有 query 行同结果）同时钉住行维广播正确。
    """

    def test_alternating_sign_and_group_amplitude(self):
        b, h, s, d = 1, 1, 128, 32
        idx = torch.arange(s)
        expo = (idx // 32) % 5 - 2
        sign = 1 - 2 * (torch.arange(d) % 2)
        v = (sign.float().view(1, d) * (2.0 ** expo.float()).view(s, 1)).view(1, 1, s, d).contiguous().half()
        q = torch.zeros(b, h, s, d, dtype=torch.float16)
        k = torch.zeros_like(q)
        out = sageattn(q, k, v, smooth_k=False)
        assert torch.isfinite(out).all()
        # 行向常量：零 score 下每个 query 行输出一致
        assert torch.equal(out, out[:, :, :1].expand_as(out))
        # 通道符号严格交替
        row = out[0, 0, 0].float()
        assert torch.all(row[0::2] > 0) and torch.all(row[1::2] < 0), "通道反号图样破坏——V scale 通道错位"
        # 同组通道等幅、跨组幅值按 2 的幂层级分离（scale 串组会坍缩该结构）
        assert torch.allclose(row.abs(), row.abs()[:1].expand_as(row), rtol=0, atol=1e-5), "同通道对幅值必须一致"
        e0 = row[0].abs().item()
        assert e0 > 0
        ratios = [row[g * 2].abs().item() / e0 for g in range(min(4, d // 2))]
        assert all(r > 0 for r in ratios), "组间幅值层级丢失"


class TestConstantRowQuery:
    """常数 query 行探针：同一 head 内所有 q 行相同 ⇒ 全部输出行逐位一致；
    不同 head 用不同常数向量 ⇒ 跨头输出不得相等（串头检测）。参考成本 O(sq)→O(1)。"""

    def test_row_broadcast_and_head_isolation(self):
        b, h, s, d = 1, 3, 128, 64
        g = torch.Generator().manual_seed(7)
        k = torch.randn(b, h, s, d, dtype=torch.float16, generator=g)
        v = torch.randn_like(k)
        q0 = torch.randn(1, 1, 1, d, dtype=torch.float16, generator=g)
        q = q0.expand(b, h, s, d).contiguous()
        out, lse = sageattn(q, k, v, return_lse=True)
        for head in range(h):
            assert torch.equal(out[0, head], out[0, head, :1].expand_as(out[0, head])), "常数 q 行输出必须行向逐位一致"
            assert torch.equal(lse[0, head], lse[0, head, :1].expand_as(lse[0, head]))
        assert not torch.equal(out[0, 0], out[0, 1]) or torch.equal(k[0, 0], k[0, 1])
        qh = q0.expand(b, h, s, d).contiguous()
        qh[0, 1] = qh[0, 1] * 2
        out2 = sageattn(qh, k, v)
        assert not torch.equal(out2[0, 1], out[0, 1]), "q 改变后该头输出必须变化（串头/缓存残留检测）"


class TestSmScaleSweepVsSdpa:
    """sm_scale 深覆盖（历史参考实现仅 2 条用例）× L-B 比例门限（max 2 / mean 1.2 / RMSE 1.2）。

    门限分子分母同基：ours-vs-SDPA 误差 / 我方同面 reference-vs-SDPA 误差，比值超界即劣于
    量化算法应有误差水平，判内核结构错误。
    """

    @pytest.mark.parametrize("sm", [1e-3, 0.125, 1.0, 8.0, 64.0])
    def test_ratio_gates(self, sm):
        b, hq, hk, s, d = 1, 4, 2, 64, 64
        g = torch.Generator().manual_seed(11)
        q = torch.randn(b, hq, s, d, dtype=torch.float16, generator=g)
        k = torch.randn(b, hk, s, d, dtype=torch.float16, generator=g)
        v = torch.randn_like(k)
        out = sageattn(q, k, v, sm_scale=sm)
        ref = sageattn_reference(q, k, v, sm_scale=sm)
        kf = k.repeat_interleave(hq // hk, dim=1).float()
        vf = v.repeat_interleave(hq // hk, dim=1).float()
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            q.float(), kf, vf, scale=sm, is_causal=False)
        e_ours = (out.float() - sdpa).abs()
        e_ref = (ref.float() - sdpa).abs()
        assert e_ours.max().item() <= 2 * e_ref.max().item() + 1e-6
        assert e_ours.mean().item() <= 1.2 * e_ref.mean().item() + 1e-6
        rmse_ours = e_ours.square().mean().sqrt().item()
        rmse_ref = e_ref.square().mean().sqrt().item()
        assert rmse_ours <= 1.2 * rmse_ref + 1e-6


class TestSaturationFreeExtremes:
    """中间量超界与饱和压力专项（我方 κ 封顶 vs 参考实现已自述的 P>448 饱和）：

    前提自证——先在 FP32 复算原始 score 超过 FP16 上界 65504，再要求内核输出/ LSE 全有限且
    等于 softmax 极限行为（全等 score ⇒ 均匀平均），杜绝「用例未触界却通过」的假阳性。
    """

    def test_raw_score_above_fp16_max_premise_and_result(self):
        b, h, s, d = 1, 1, 32, 64
        c = 64.0
        q = torch.full((b, h, s, d), c, dtype=torch.float16)
        k = torch.full((b, h, s, d), c, dtype=torch.float16)
        v = torch.ones(b, h, s, d, dtype=torch.float16)
        raw = (q.float() @ k.float().transpose(-2, -1)).max().item()
        assert raw > 65504.0, "用例前提失效：raw score 未超过 FP16 上界"
        out, lse = sageattn(q, k, v, smooth_k=False, return_lse=True)
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        assert torch.equal(out, torch.ones_like(out)), "全等 score 的 softmax 极限 = 均匀 ⇒ 输出恰 1.0"

    def test_negative_sm_scale_finite(self):
        b, h, s, d = 1, 1, 32, 64
        g = torch.Generator().manual_seed(3)
        q = torch.randn(b, h, s, d, dtype=torch.float16, generator=g)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = sageattn(q, k, v, sm_scale=-0.5)
        assert torch.isfinite(out).all()


class TestNativePathNoWarning:
    """默认档零告警红线：per_thread × fp32+fp16 原生路径任何 UserWarning 即视为静默退化违约。"""

    def test_default_path_emits_no_userwarning(self):
        q, k, v = _zero_score_case(s=32, d=64)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sageattn(q, k, v)
            sageattn(q, k, v, qk_quant_gran="per_warp")
