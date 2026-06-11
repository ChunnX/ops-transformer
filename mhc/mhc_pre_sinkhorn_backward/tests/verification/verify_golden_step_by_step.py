#!/usr/bin/env python3
"""
MhcPreSinkhornBackward Golden逐步验证脚本
用途：
    1. 逐步验证每个辅助函数的行为
    2. 逐步验证主函数10个步骤
    3. 追踪Shape变化和数值范围
    4. 可视化中间结果
    5. 帮助理解算法细节
使用方式：
    python verify_golden_step_by_step.py --mode=interactive  # 交互式逐步验证
    python verify_golden_step_by_step.py --mode=auto        # 自动全部验证
    python verify_golden_step_by_step.py --mode=single --step=3  # 单步验证
"""
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple
import sys
sys.path.append('/mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/assets')
try:
    from golden_v2 import (
        SigmoidGrad, RMSNormGrad, ExpGrad, SinkhornGrad,
        mhc_pre_sinkhorn_backward_golden
    )
except ImportError as e:
    print(f"⚠️ 无法导入golden_v2模块: {e}")
    print("请确保golden_v2.py路径正确")
    sys.exit(1)
class GoldenStepVerifier:
    """Golden逐步验证器"""
    def __init__(self, B=2, S=8, n=4, C=64, iters=3, hc_eps=1e-6):
        """初始化测试数据"""
        self.B = B
        self.S = S
        self.n = n
        self.C = C
        self.iters = iters
        self.hc_eps = hc_eps
        print("=" * 80)
        print("MhcPreSinkhornBackward Golden 逐步验证器")
        print("=" * 80)
        print(f"\n【测试配置】")
        print(f"  Batch size (B): {B}")
        print(f"  Sequence length (S): {S}")
        print(f"  Hidden dim n: {n}")
        print(f"  Feature dim C: {C}")
        print(f"  Sinkhorn iterations: {iters}")
        print(f"  hc_eps: {hc_eps}")
        self._prepare_test_data()
    def _prepare_test_data(self):
        """准备测试数据"""
        print(f"\n【准备测试数据】")
        # 输入数据
        self.x = torch.randn(self.B, self.S, self.n, self.C, dtype=torch.float32)
        self.phi = torch.randn(self.n * self.n + 2 * self.n, self.n * self.C, dtype=torch.float32)  # [N²+2N, N*D]官方约束
        self.alpha = torch.randn(3, dtype=torch.float32)
        self.bias = torch.randn(self.n * self.n + 2 * self.n, dtype=torch.float32)
        # 正向输出（模拟）
        self.h_pre = torch.randn(self.B, self.S, self.n, dtype=torch.float32)
        self.hc_before_norm = torch.randn(self.B, self.S, self.n * self.n + 2 * self.n, dtype=torch.float32)
        self.inv_rms = torch.randn(self.B, self.S, 1, dtype=torch.float32).abs() + 0.1  # 避免过小
        # Sinkhorn正向保存数据
        self.sum_out = torch.randn(self.iters, 2, self.B, self.S, self.n, dtype=torch.float32).abs() + 0.1
        self.norm_out = torch.randn(self.iters, 2, self.B, self.S, self.n, self.n, dtype=torch.float32).abs() + 0.1
        # 梯度输入
        self.grad_hin = torch.randn(self.B, self.S, self.C, dtype=torch.float32)
        self.grad_h_post = torch.randn(self.B, self.S, self.n, dtype=torch.float32)
        self.grad_h_res = torch.randn(self.B, self.S, self.n, self.n, dtype=torch.float32)
        print(f"  ✅ x: {self.x.shape}, range: [{self.x.min():.3f}, {self.x.max():.3f}]")
        print(f"  ✅ phi: {self.phi.shape}, range: [{self.phi.min():.3f}, {self.phi.max():.3f}]")
        print(f"  ✅ alpha: {self.alpha.shape}, values: {self.alpha.tolist()}")
        print(f"  ✅ bias: {self.bias.shape}")
        print(f"  ✅ grad_hin: {self.grad_hin.shape}, range: [{self.grad_hin.min():.3f}, {self.grad_hin.max():.3f}]")
        print(f"  ✅ grad_h_post: {self.grad_h_post.shape}")
        print(f"  ✅ grad_h_res: {self.grad_h_res.shape}")
    def print_separator(self, title: str):
        """打印分隔符"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    def print_shape_info(self, name: str, tensor: torch.Tensor, description: str = ""):
        """打印Shape信息"""
        print(f"\n【{name}】")
        print(f"  Shape: {tuple(tensor.shape)}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        print(f"  Mean: {tensor.mean().item():.6f}")
        print(f"  Std: {tensor.std().item():.6f}")
        if description:
            print(f"  说明: {description}")
        # 对于小张量，打印具体值
        if tensor.numel() <= 20:
            print(f"  值: {tensor.tolist()}")
    def verify_step_by_step(self, mode='interactive'):
        """逐步验证"""
        if mode == 'interactive':
            self._interactive_verify()
        elif mode == 'auto':
            self._auto_verify()
        elif mode == 'single':
            self._single_step_verify()
    def _interactive_verify(self):
        """交互式验证"""
        self.print_separator("交互式逐步验证模式")
        print("\n提示: 按Enter继续下一步，输入'skip'跳过当前步骤，输入'quit'退出")
        steps = [
            ("辅助函数", self._verify_helper_functions),
            ("步骤1: 输入聚合梯度", self._verify_step1),
            ("步骤2: 正向中间变量", self._verify_step2),
            ("步骤3: 三路激活梯度", self._verify_step3),
            ("步骤4: bias梯度归约", self._verify_step4),
            ("步骤5: alpha梯度归约", self._verify_step5),
            ("步骤6: 线性变换梯度", self._verify_step6),
            ("步骤7: 乘法梯度", self._verify_step7),
            ("步骤8: RMSNorm倒数梯度", self._verify_step8),
            ("步骤9: Matmul梯度", self._verify_step9),
            ("步骤10: 汇总梯度", self._verify_step10),
            ("完整验证", self._verify_full_function),
        ]
        for step_name, step_func in steps:
            self.print_separator(step_name)
            try:
                step_func()
            except Exception as e:
                print(f"❌ 验证失败: {e}")
                import traceback
                traceback.print_exc()
            # 用户交互
            user_input = input("\n按Enter继续，或输入命令: ").strip().lower()
            if user_input == 'quit':
                print("\n退出验证")
                break
            elif user_input == 'skip':
                print("\n跳过当前步骤")
                continue
    def _auto_verify(self):
        """自动全部验证"""
        self.print_separator("自动验证模式")
        steps = [
            ("辅助函数验证", self._verify_helper_functions),
            ("步骤1", self._verify_step1),
            ("步骤2", self._verify_step2),
            ("步骤3", self._verify_step3),
            ("步骤4", self._verify_step4),
            ("步骤5", self._verify_step5),
            ("步骤6", self._verify_step6),
            ("步骤7", self._verify_step7),
            ("步骤8", self._verify_step8),
            ("步骤9", self._verify_step9),
            ("步骤10", self._verify_step10),
            ("完整验证", self._verify_full_function),
        ]
        for step_name, step_func in steps:
            print(f"\n▶ {step_name}...")
            try:
                step_func()
                print(f"✅ {step_name}验证通过")
            except Exception as e:
                print(f"❌ {step_name}验证失败: {e}")
                import traceback
                traceback.print_exc()
    def _single_step_verify(self):
        """单步验证（需要--step参数）"""
        # 这个方法会被main函数中调用
        pass
    # ==================== 辅助函数验证 ====================
    def _verify_helper_functions(self):
        """验证辅助函数"""
        print("\n▶ 验证SigmoidGrad...")
        self._verify_sigmoid_grad()
        print("\n▶ 验证RMSNormGrad...")
        self._verify_rmsnorm_grad()
        print("\n▶ 验证ExpGrad...")
        self._verify_exp_grad()
        print("\n▶ 验证SinkhornGrad...")
        self._verify_sinkhorn_grad()
    def _verify_sigmoid_grad(self):
        """验证SigmoidGrad"""
        print("\n【SigmoidGrad验证】")
        # 测试数据
        z = torch.randn(self.B, self.S, self.n, dtype=torch.float32)
        dy = torch.randn(self.B, self.S, self.n, dtype=torch.float32)
        self.print_shape_info("输入z", z, "激活前的线性输出")
        self.print_shape_info("输入dy", dy, "损失对sigmoid输出的梯度")
        # 验证h_pre支路（修正hc_eps）
        print("\n▶ 验证h_pre支路（is_pre=True, hc_eps修正）...")
        grad_z_pre = SigmoidGrad(z, dy, is_pre=True, hc_eps=self.hc_eps)
        self.print_shape_info("输出grad_z_pre", grad_z_pre, "修正后的sigmoid梯度")
        # 验证公式
        y = torch.sigmoid(z)
        sigma_pre = y - self.hc_eps
        expected_grad = dy * sigma_pre * (1 - y)
        print(f"\n▶ 公式验证:")
        print(f"  sigmoid(z): [{y.min():.3f}, {y.max():.3f}]")
        print(f"  sigma修正 (y - eps): [{sigma_pre.min():.3f}, {sigma_pre.max():.3f}]")
        print(f"  梯度公式: grad = dy * sigma * (1 - sigmoid(z))")
        # 验证数值一致性
        diff = torch.abs(grad_z_pre - expected_grad).max()
        print(f"  ✅ 数值一致性验证: max_diff = {diff:.6e} (< 1e-5)")
        # 验证h_post支路（不修正）
        print("\n▶ 验证h_post支路（is_pre=False, 不修正）...")
        grad_z_post = SigmoidGrad(z, dy, is_pre=False)
        self.print_shape_info("输出grad_z_post", grad_z_post, "标准sigmoid梯度")
        # 验证公式
        sigma_post = y
        expected_grad_post = dy * sigma_post * (1 - y)
        diff_post = torch.abs(grad_z_post - expected_grad_post).max()
        print(f"  ✅ 数值一致性验证: max_diff = {diff_post:.6e} (< 1e-5)")
    def _verify_rmsnorm_grad(self):
        """验证RMSNormGrad"""
        print("\n【RMSNormGrad验证】")
        # 测试数据（flatten后的2D）
        BS = self.B * self.S
        NC = self.n * self.C
        x_flat = torch.randn(BS, NC, dtype=torch.float32)
        inv_rms = torch.randn(BS, 1, dtype=torch.float32).abs() + 0.1
        grad_inv_rms = torch.randn(BS, 1, dtype=torch.float32)
        self.print_shape_info("输入x_flat", x_flat, "flatten后的2D特征 [B*S, n*C]")
        self.print_shape_info("输入inv_rms", inv_rms, "RMS倒数 [B*S, 1]")
        self.print_shape_info("输入grad_inv_rms", grad_inv_rms, "对inv_rms的梯度 [B*S, 1]")
        # 计算梯度
        grad_x = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
        self.print_shape_info("输出grad_x", grad_x, "对x的梯度 [B*S, n*C]")
        # 验证公式
        print("\n▶ 公式验证:")
        print(f"  公式: grad_x = -(inv_rms^3) * x * grad_inv_rms / NC")
        print(f"  NC = n*C = {self.n * self.C}")
        expected_grad = - (inv_rms ** 3) * x_flat * grad_inv_rms / float(NC)
        diff = torch.abs(grad_x - expected_grad).max()
        print(f"  ✅ 数值一致性验证: max_diff = {diff:.6e} (< 1e-5)")
        # 分析梯度特性
        print("\n▶ 梯度特性分析:")
        print(f"  grad_x范围: [{grad_x.min():.6e}, {grad_x.max():.6e}]")
        print(f"  grad_x均值: {grad_x.mean():.6e}")
        print(f"  说明: RMSNorm反向梯度通常较小（inv_rms^3衰减）")
    def _verify_exp_grad(self):
        """验证ExpGrad"""
        print("\n【ExpGrad验证】")
        # 测试数据（对应z_res）
        z_res = torch.randn(self.B, self.S, self.n, self.n, dtype=torch.float32)
        sk_grad = torch.randn(self.B, self.S, self.n, self.n, dtype=torch.float32)
        self.print_shape_info("输入z_res", z_res, "激活前的输入（对应h_res支路）")
        self.print_shape_info("输入sk_grad", sk_grad, "Sinkhorn反向的梯度输出")
        # 计算梯度
        grad_z_res = ExpGrad(z_res, sk_grad)
        self.print_shape_info("输出grad_z_res", grad_z_res, "Exp反向梯度")
        # 验证公式
        print("\n▶ 公式验证:")
        print(f"  数值稳定exp: y = exp(x - x_max)")
        print(f"  梯度公式: grad = y * y_grad - is_max * Σ(y * y_grad)")
        # 手动计算验证
        n = z_res.size(-1)
        x_max, idx = z_res.max(dim=-1, keepdim=True)
        y = (z_res - x_max).exp()
        sum_all = (y * sk_grad).sum(dim=-1, keepdim=True)
        # is_max矩阵
        arange_idx = torch.arange(n, device=z_res.device)[None, :]
        is_max = (idx == arange_idx)
        expected_grad = y * sk_grad - is_max * sum_all
        diff = torch.abs(grad_z_res - expected_grad).max()
        print(f"  ✅ 数值一致性验证: max_diff = {diff:.6e} (< 1e-5)")
        # 分析数值稳定性
        print("\n▶ 数值稳定性分析:")
        print(f"  x_max: [{x_max.min():.3f}, {x_max.max():.3f}]")
        print(f"  exp(x - x_max): [{y.min():.6e}, {y.max():.6e}]")
        print(f"  说明: x - x_max确保exp不溢出")
    def _verify_sinkhorn_grad(self):
        """验证SinkhornGrad"""
        print("\n【SinkhornGrad验证】")
        self.print_shape_info("输入grad_h_res", self.grad_h_res, "对h_res的梯度 [B, S, n, n]")
        self.print_shape_info("输入sum_out", self.sum_out, f"正向保存的sum [iters={self.iters}, 2, B, S, n]")
        self.print_shape_info("输入norm_out", self.norm_out, f"正向保存的norm [iters={self.iters}, 2, B, S, n, n]")
        # 计算梯度
        x_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        self.print_shape_info("输出x_grad", x_grad, "Sinkhorn反向后的梯度 [B, S, n, n]")
        print("\n▶ Sinkhorn反向算法说明:")
        print(f"  1. 从iters={self.iters}-1逆向迭代到0")
        print(f"  2. 每步先反向列归一化，再反向行归一化")
        print(f"  3. 使用正向保存的sum_out和norm_out")
        print("\n▶ 梯度变化追踪:")
        print(f"  输入grad_h_res范围: [{self.grad_h_res.min():.3f}, {self.grad_h_res.max():.3f}]")
        print(f"  输出x_grad范围: [{x_grad.min():.6e}, {x_grad.max():.6e}]")
        print(f"  说明: Sinkhorn反向会逐步衰减梯度（归一化影响）")
    # ==================== 主函数10步骤验证 ====================
    def _verify_step1(self):
        """验证步骤1: 输入聚合梯度"""
        print("\n【步骤1: 输入聚合梯度】")
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        self.print_shape_info("输入grad_hin", grad_hin_fp32, "[B, S, C]")
        self.print_shape_info("输入h_pre", self.h_pre, "[B, S, n]")
        self.print_shape_info("输入x", x_fp32, "[B, S, n, C]")
        # 计算
        grad_x_from_hin = grad_hin_fp32[..., None, :] * self.h_pre[..., None]
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        self.print_shape_info("输出grad_x_from_hin", grad_x_from_hin, 
                              "Broadcasting: [B, S, C] * [B, S, n] → [B, S, n, C]")
        self.print_shape_info("输出grad_h_pre", grad_h_pre,
                              "归约: [B, S, n, C] → [B, S, n] (sum over C)")
        print("\n▶ 公式解析:")
        print(f"  grad_x_from_hin = grad_hin[..., None, :] * h_pre[..., None]")
        print(f"    Broadcasting: grad_hin[B,S,C] * h_pre[B,S,n] → grad_hin[B,S,C,1] * h_pre[B,S,1,n]")
        print(f"    实际: grad_hin[B,S,C] expands to [B,S,1,C], h_pre[B,S,n] expands to [B,S,n,1]")
        print(f"    结果: [B,S,n,C]")
        print(f"\n  grad_h_pre = (grad_hin[..., None, :] * x).sum(-1)")
        print(f"    中间: grad_hin[B,S,1,C] * x[B,S,n,C] → [B,S,n,C]")
        print(f"    归约: sum(dim=C) → [B,S,n]")
    def _verify_step2(self):
        """验证步骤2: 正向中间变量"""
        print("\n【步骤2: 计算正向中间变量】")
        self.print_shape_info("输入hc_before_norm", self.hc_before_norm, "[B, S, n*n + 2*n]")
        self.print_shape_info("输入inv_rms", self.inv_rms, "[B, S, 1]")
        self.print_shape_info("输入alpha", self.alpha, "[3] = [pre, post, res]")
        self.print_shape_info("输入bias", self.bias, "[n*n + 2*n]")
        # 计算
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        self.print_shape_info("输出norm_out_forward", norm_out_forward, 
                              "Broadcasting: [B,S,n*n+2*n] * [B,S,1] → [B,S,n*n+2*n]")
        self.print_shape_info("输出z_pre", z_pre, "[B, S, n] - h_pre支路的线性输出")
        self.print_shape_info("输出z_post", z_post, "[B, S, n] - h_post支路的线性输出")
        self.print_shape_info("输出z_res", z_res, "[B, S, n, n] - h_res支路的线性输出")
        print("\n▶ 公式解析:")
        print(f"  norm_out = hc_before_norm * inv_rms")
        print(f"  z_pre = norm_out[:n] * alpha[0] + bias[:n]")
        print(f"  z_post = norm_out[n:2n] * alpha[1] + bias[n:2n]")
        print(f"  z_res = norm_out[2n:] * alpha[2] + bias[2n:] (reshape为n*n)")
    def _verify_step3(self):
        """验证步骤3: 三路激活梯度"""
        print("\n【步骤3: 三路激活梯度】")
        # 重新计算步骤1-2的中间变量
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        self.print_shape_info("输入z_pre", z_pre, "[B, S, n] - h_pre线性输出")
        self.print_shape_info("输入grad_h_pre", grad_h_pre, "[B, S, n] - 对h_pre的梯度")
        self.print_shape_info("输入z_post", z_post, "[B, S, n] - h_post线性输出")
        self.print_shape_info("输入grad_h_post", self.grad_h_post, "[B, S, n] - 对h_post的梯度")
        self.print_shape_info("输入z_res", z_res, "[B, S, n, n] - h_res线性输出")
        self.print_shape_info("输入grad_h_res", self.grad_h_res, "[B, S, n, n] - 对h_res的梯度")
        # 计算三路激活梯度
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        self.print_shape_info("输出grad_z_pre", grad_z_pre, 
                              "SigmoidGrad修正后: [B, S, n]")
        self.print_shape_info("输出grad_z_post", grad_z_post,
                              "2 * SigmoidGrad: [B, S, n] (系数2来自正向)")
        self.print_shape_info("输出sk_grad", sk_grad,
                              "Sinkhorn反向结果: [B, S, n, n]")
        self.print_shape_info("输出grad_z_res", grad_z_res,
                              "ExpGrad + flatten: [B, S, n*n]")
        print("\n▶ 三路激活说明:")
        print(f"  h_pre支路: Sigmoid修正（is_pre=True, hc_eps={self.hc_eps})")
        print(f"  h_post支路: 2 * Sigmoid（正向系数2影响反向梯度）")
        print(f"  h_res支路: Sinkhorn反向 + Exp反向（softmax第一步）")
    def _verify_step4(self):
        """验证步骤4: bias梯度归约"""
        print("\n【步骤4: bias梯度归约】")
        # 重新计算步骤3
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        self.print_shape_info("输入grad_z_pre", grad_z_pre, "[B, S, n]")
        self.print_shape_info("输入grad_z_post", grad_z_post, "[B, S, n]")
        self.print_shape_info("输入grad_z_res", grad_z_res, "[B, S, n*n]")
        # 归约计算
        grad_bias = torch.cat([
            grad_z_pre.sum((0, 1)),
            grad_z_post.sum((0, 1)),
            grad_z_res.sum((0, 1))
        ])
        self.print_shape_info("中间grad_z_pre.sum", grad_z_pre.sum((0, 1)),
                              "归约: [B,S,n] → [n] (sum over B,S)")
        self.print_shape_info("中间grad_z_post.sum", grad_z_post.sum((0, 1)),
                              "归约: [B,S,n] → [n]")
        self.print_shape_info("中间grad_z_res.sum", grad_z_res.sum((0, 1)),
                              "归约: [B,S,n*n] → [n*n]")
        self.print_shape_info("输出grad_bias", grad_bias,
                              "Concat: [n] + [n] + [n*n] → [n*n + 2*n]")
        print("\n▶ 归约说明:")
        print(f"  全局归约: sum over (B, S) → 所有batch和sequence位置累加")
        print(f"  三段concat: pre支路[n] + post支路[n] + res支路[n*n]")
        print(f"  总长度: {self.n} + {self.n} + {self.n * self.n} = {self.n * self.n + 2 * self.n}")
    def _verify_step5(self):
        """验证步骤5: alpha梯度归约"""
        print("\n【步骤5: alpha梯度归约】")
        # 重新计算
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        self.print_shape_info("输入norm_out_forward", norm_out_forward, "[B, S, n*n + 2*n]")
        self.print_shape_info("输入grad_z_pre", grad_z_pre, "[B, S, n]")
        self.print_shape_info("输入grad_z_post", grad_z_post, "[B, S, n]")
        self.print_shape_info("输入grad_z_res", grad_z_res, "[B, S, n*n]")
        # 计算
        grad_alpha_pre = (norm_out_forward[..., :self.n] * grad_z_pre).sum()
        grad_alpha_post = (norm_out_forward[..., self.n:2*self.n] * grad_z_post).sum()
        grad_alpha_res = (norm_out_forward[..., 2*self.n:] * grad_z_res).sum()
        grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res], dtype=torch.float32)
        self.print_shape_info("中间grad_alpha_pre", torch.tensor([grad_alpha_pre.item()]),
                              "scalar: (norm[:n] * grad_z_pre).sum() → 全局归约")
        self.print_shape_info("中间grad_alpha_post", torch.tensor([grad_alpha_post.item()]),
                              "scalar: (norm[n:2n] * grad_z_post).sum()")
        self.print_shape_info("中间grad_alpha_res", torch.tensor([grad_alpha_res.item()]),
                              "scalar: (norm[2n:] * grad_z_res).sum()")
        self.print_shape_info("输出grad_alpha", grad_alpha, "[3] = [pre, post, res]")
        print("\n▶ 归约说明:")
        print(f"  全局scalar归约: sum所有元素 → 一个浮点数")
        print(f"  三个scalar组成: grad_alpha = [pre_scalar, post_scalar, res_scalar]")
        print(f"  alpha是3维向量，对应pre/post/res三个分量")
    def _verify_step6(self):
        """验证步骤6: 线性变换梯度"""
        print("\n【步骤6: 线性变换梯度】")
        # 重新计算步骤3
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        self.print_shape_info("输入grad_z_pre", grad_z_pre, "[B, S, n]")
        self.print_shape_info("输入grad_z_post", grad_z_post, "[B, S, n]")
        self.print_shape_info("输入grad_z_res", grad_z_res, "[B, S, n*n]")
        self.print_shape_info("输入alpha", self.alpha, "[3] = [pre, post, res]")
        # 计算
        grad_norm_out_pre = grad_z_pre * self.alpha[0]
        grad_norm_out_post = grad_z_post * self.alpha[1]
        grad_norm_out_res = grad_z_res * self.alpha[2]
        grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
        self.print_shape_info("输出grad_norm_out_pre", grad_norm_out_pre,
                              "[B,S,n] = grad_z_pre * alpha[0]")
        self.print_shape_info("输出grad_norm_out_post", grad_norm_out_post,
                              "[B,S,n] = grad_z_post * alpha[1]")
        self.print_shape_info("输出grad_norm_out_res", grad_norm_out_res,
                              "[B,S,n*n] = grad_z_res * alpha[2]")
        self.print_shape_info("输出grad_norm_out", grad_norm_out,
                              "Concat: [B,S,n] + [B,S,n] + [B,S,n*n] → [B,S,n*n+2*n]")
        print("\n▶ 线性反向说明:")
        print(f"  线性正向: z = norm_out * alpha + bias")
        print(f"  线性反向: grad_norm_out = grad_z * alpha")
        print(f"  bias反向已在步骤4完成（全局归约）")
    def _verify_step7(self):
        """验证步骤7: 乘法梯度"""
        print("\n【步骤7: 乘法梯度】")
        # 重新计算步骤6
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        grad_norm_out_pre = grad_z_pre * self.alpha[0]
        grad_norm_out_post = grad_z_post * self.alpha[1]
        grad_norm_out_res = grad_z_res * self.alpha[2]
        grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
        self.print_shape_info("输入grad_norm_out", grad_norm_out, "[B, S, n*n + 2*n]")
        self.print_shape_info("输入inv_rms", self.inv_rms, "[B, S, 1]")
        self.print_shape_info("输入hc_before_norm", self.hc_before_norm, "[B, S, n*n + 2*n]")
        # 计算
        grad_hc_before_norm = grad_norm_out * self.inv_rms
        grad_inv_rms = (self.hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
        self.print_shape_info("输出grad_hc_before_norm", grad_hc_before_norm,
                              "Broadcasting: [B,S,n*n+2*n] * [B,S,1] → [B,S,n*n+2*n]")
        self.print_shape_info("输出grad_inv_rms", grad_inv_rms,
                              "归约: [B,S,n*n+2*n] → [B,S,1] (sum over n*n+2*n)")
        print("\n▶ 乘法反向说明:")
        print(f"  正向乘法: norm_out = hc_before_norm * inv_rms")
        print(f"  反向分支1: grad_hc_before_norm = grad_norm_out * inv_rms")
        print(f"  反向分支2: grad_inv_rms = sum(hc_before_norm * grad_norm_out)")
    def _verify_step8(self):
        """验证步骤8: RMSNorm倒数梯度"""
        print("\n【步骤8: RMSNorm倒数梯度】")
        # 重新计算步骤7
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        grad_norm_out_pre = grad_z_pre * self.alpha[0]
        grad_norm_out_post = grad_z_post * self.alpha[1]
        grad_norm_out_res = grad_z_res * self.alpha[2]
        grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
        grad_inv_rms = (self.hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
        self.print_shape_info("输入x", x_fp32, "[B, S, n, C]")
        self.print_shape_info("输入inv_rms", self.inv_rms, "[B, S, 1]")
        self.print_shape_info("输入grad_inv_rms", grad_inv_rms, "[B, S, 1]")
        # flatten和计算
        x_flat = x_fp32.flatten(0, 1).flatten(1)  # [B*S, n*C]
        inv_rms_flat = self.inv_rms.flatten(0, 1)  # [B, S, 1] → [B*S, 1]
        grad_inv_rms_flat = grad_inv_rms.flatten(0, 1)  # [B, S, 1] → [B*S, 1]
        grad_x_from_rms = RMSNormGrad(x_flat, inv_rms_flat, grad_inv_rms_flat)
        self.print_shape_info("中间x_flat", x_flat,
                              "flatten(0,2): [B,S,n,C] → [B*S,n*C]")
        self.print_shape_info("中间inv_rms_flat", inv_rms_flat,
                              "flatten(0,1): [B,S,1] → [B*S,1]")
        self.print_shape_info("中间grad_inv_rms_flat", grad_inv_rms_flat,
                              "flatten(0,1): [B,S,1] → [B*S,1]")
        self.print_shape_info("输出grad_x_from_rms", grad_x_from_rms,
                              "[B*S, n*C] - RMSNorm倒数梯度")
        # reshape回原形状
        grad_x_from_rms_reshaped = grad_x_from_rms.view(x_fp32.shape)
        self.print_shape_info("输出grad_x_from_rms (reshape)", grad_x_from_rms_reshaped,
                              "reshape: [B*S,n*C] → [B,S,n,C]")
        print("\n▶ RMSNorm倒数梯度说明:")
        print(f"  公式: grad_x = -(inv_rms^3) * x * grad_inv_rms / NC")
        print(f"  NC = n*C = {self.n * self.C}")
        print(f"  说明: RMSNorm反向梯度通常较小（inv_rms^3衰减）")
    def _verify_step9(self):
        """验证步骤9: Matmul梯度"""
        print("\n【步骤9: 矩阵乘法梯度】")
        # 重新计算步骤7
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        grad_norm_out_pre = grad_z_pre * self.alpha[0]
        grad_norm_out_post = grad_z_post * self.alpha[1]
        grad_norm_out_res = grad_z_res * self.alpha[2]
        grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
        grad_hc_before_norm = grad_norm_out * self.inv_rms
        self.print_shape_info("输入grad_hc_before_norm", grad_hc_before_norm, "[B, S, n*n+2*n]")
        self.print_shape_info("输入phi", self.phi, "[n*C, n*n+2*n] - 全局weight矩阵")
        self.print_shape_info("输入x", x_fp32, "[B, S, n, C]")
        # Flatten操作
        x_flat = x_fp32.flatten(0, 1).flatten(1)  # [B*S, n*C]
        grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)  # [T, N²+2N] = [16, 24]
        # 对x的梯度（根据官方约束）
        grad_x_from_matmul_flat = grad_hc_before_norm_flat @ self.phi  # [16,24] @ [24,256] = [16,256]
        grad_x_from_matmul = grad_x_from_matmul_flat.view(self.B, self.S, self.n, self.C)
        # 对phi的梯度（根据官方约束）
        grad_phi = grad_hc_before_norm_flat.T @ x_flat  # [24,16] @ [16,256] = [24,256]
        self.print_shape_info("mm2输入1", grad_hc_before_norm_flat,
                              "flatten: [B,S,n*n+2*n] → [B*S,n*n+2*n]")
        self.print_shape_info("mm2输入2", x_flat,
                              "flatten: [B,S,n,C] → [B*S,n*C]")
        self.print_shape_info("mm2输出grad_phi", grad_phi,
                              "[N²+2N, N*D] = [24, 256] - 对phi的梯度（全局归约）")
        print("\n▶ Matmul梯度说明:")
        print(f"  mm1: grad_x_from_matmul = grad_hc_before_norm @ phi")
        print(f"    Shape: [B,S,n*C] @ [B,S,n,n] → [B,S,n,C]")
        print(f"    每个位置独立计算，无全局归约")
        print(f"\n  mm2: grad_phi = grad_hc_before_norm.T @ x")
        print(f"    Flatten: [B,S,n*C] → [B*S,n*C]")
        print(f"    Shape: [n*C,B*S].T @ [B*S,n*C] → [n*C,n*C]")
        print(f"    全局归约: 所有B*S位置累加")
    def _verify_step10(self):
        """验证步骤10: 汇总梯度"""
        print("\n【步骤10: 汇总所有梯度】")
        grad_hin_fp32 = self.grad_hin.float()
        x_fp32 = self.x.float()
        grad_x_from_hin = grad_hin_fp32[..., None, :] * self.h_pre[..., None]
        grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
        norm_out_forward = self.hc_before_norm * self.inv_rms
        z_pre = norm_out_forward[..., :self.n] * self.alpha[0] + self.bias[:self.n]
        z_post = norm_out_forward[..., self.n:2*self.n] * self.alpha[1] + self.bias[self.n:2*self.n]
        z_res = (norm_out_forward[..., 2*self.n:] * self.alpha[2] + self.bias[2*self.n:]).reshape(
            self.B, self.S, self.n, self.n)
        grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=self.hc_eps)
        grad_z_post = 2 * SigmoidGrad(z_post, self.grad_h_post, is_pre=False)
        sk_grad = SinkhornGrad(self.grad_h_res, self.sum_out, self.norm_out)
        grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
        grad_norm_out_pre = grad_z_pre * self.alpha[0]
        grad_norm_out_post = grad_z_post * self.alpha[1]
        grad_norm_out_res = grad_z_res * self.alpha[2]
        grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
        grad_inv_rms = (self.hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
        x_flat_for_rms = x_fp32.flatten(0, 1).flatten(1)
        inv_rms_flat = self.inv_rms.flatten(0, 1)
        grad_inv_rms_flat = grad_inv_rms.flatten(0, 1)
        grad_x_from_rms = RMSNormGrad(x_flat_for_rms, inv_rms_flat, grad_inv_rms_flat)
        grad_hc_before_norm = grad_norm_out * self.inv_rms
        x_flat_for_matmul = x_fp32.flatten(0, 1).flatten(1)  # [T, N*D]
        grad_hc_flat = grad_hc_before_norm.flatten(0, 1)  # [T, N²+2N]
        grad_x_from_matmul_flat = grad_hc_flat @ self.phi  # [16,24] @ [24,256]
        grad_x_from_matmul = grad_x_from_matmul_flat.view(self.B, self.S, self.n, self.C)
        self.print_shape_info("输入grad_x_from_hin", grad_x_from_hin, "[B, S, n, C] - 来自步骤1")
        self.print_shape_info("输入grad_x_from_rms", grad_x_from_rms.view(x_fp32.shape),
                              "[B, S, n, C] - 来自步骤8")
        self.print_shape_info("输入grad_x_from_matmul", grad_x_from_matmul,
                              "[B, S, n, C] - 来自步骤9")
        grad_x = (
            grad_x_from_hin.view(x_fp32.shape)
            + grad_x_from_rms.view(x_fp32.shape)
            + grad_x_from_matmul.view(x_fp32.shape)
        ).to(x_fp32.dtype)
        self.print_shape_info("输出grad_x", grad_x,
                              "三项累加: [B,S,n,C] + [B,S,n,C] + [B,S,n,C] → [B,S,n,C]")
        print("\n▶ 梯度汇总说明:")
        print(f"  grad_x有三个来源:")
        print(f"    1. grad_x_from_hin: 来自h_in支路（输入聚合）")
        print(f"    2. grad_x_from_rms: 来自RMSNorm倒数梯度")
        print(f"    3. grad_x_from_matmul: 来自矩阵乘法反向")
        print(f"  三项累加得到最终grad_x")
        print("\n▶ 梯度贡献分析:")
        print(f"  grad_x_from_hin范围: [{grad_x_from_hin.min():.6e}, {grad_x_from_hin.max():.6e}]")
        print(f"  grad_x_from_rms范围: [{grad_x_from_rms.min():.6e}, {grad_x_from_rms.max():.6e}]")
        print(f"  grad_x_from_matmul范围: [{grad_x_from_matmul.min():.6e}, {grad_x_from_matmul.max():.6e}]")
        print(f"  最终grad_x范围: [{grad_x.min():.6e}, {grad_x.max():.6e}]")
    def _verify_full_function(self):
        """验证完整函数"""
        print("\n【完整函数验证】")
        # 调用golden函数
        grad_x, grad_phi, grad_alpha, grad_bias = mhc_pre_sinkhorn_backward_golden(
            self.grad_hin, self.grad_h_post, self.grad_h_res,
            self.x, self.phi, self.alpha, self.bias,
            self.h_pre, self.hc_before_norm, self.inv_rms,
            self.sum_out, self.norm_out,
            self.hc_eps
        )
        self.print_shape_info("输出grad_x", grad_x, "[B, S, n, C] - 对x的梯度")
        self.print_shape_info("输出grad_phi", grad_phi, "[n*C, n*C] - 对phi的梯度")
        self.print_shape_info("输出grad_alpha", grad_alpha, "[3] - 对alpha的梯度")
        self.print_shape_info("输出grad_bias", grad_bias, "[n*n + 2*n] - 对bias的梯度")
        print("\n✅ Golden函数完整验证成功!")
        print("\n【最终输出汇总】")
        print(f"  grad_x: shape={tuple(grad_x.shape)}, dtype={grad_x.dtype}")
        print(f"  grad_phi: shape={tuple(grad_phi.shape)}, dtype={grad_phi.dtype}")
        print(f"  grad_alpha: shape={tuple(grad_alpha.shape)}, dtype={grad_alpha.dtype}")
        print(f"  grad_bias: shape={tuple(grad_bias.shape)}, dtype={grad_bias.dtype}")
def main():
    parser = argparse.ArgumentParser(description='Golden逐步验证脚本')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'auto', 'single'],
                        help='验证模式: interactive(交互式), auto(自动), single(单步)')
    parser.add_argument('--step', type=int, default=1,
                        help='单步验证的步骤编号（1-10）')
    parser.add_argument('--B', type=int, default=2, help='Batch size')
    parser.add_argument('--S', type=int, default=8, help='Sequence length')
    parser.add_argument('--n', type=int, default=4, help='Hidden dim n')
    parser.add_argument('--C', type=int, default=64, help='Feature dim C')
    parser.add_argument('--iters', type=int, default=3, help='Sinkhorn iterations')
    args = parser.parse_args()
    verifier = GoldenStepVerifier(
        B=args.B, S=args.S, n=args.n, C=args.C, iters=args.iters
    )
    if args.mode == 'interactive':
        verifier.verify_step_by_step(mode='interactive')
    elif args.mode == 'auto':
        verifier.verify_step_by_step(mode='auto')
    elif args.mode == 'single':
        verifier.print_separator(f"单步验证: 步骤{args.step}")
        step_methods = {
            1: verifier._verify_step1,
            2: verifier._verify_step2,
            3: verifier._verify_step3,
            4: verifier._verify_step4,
            5: verifier._verify_step5,
            6: verifier._verify_step6,
            7: verifier._verify_step7,
            8: verifier._verify_step8,
            9: verifier._verify_step9,
            10: verifier._verify_step10,
        }
        if args.step in step_methods:
            step_methods[args.step]()
        else:
            print(f"❌ 无效步骤编号: {args.step}，应在1-10范围内")
if __name__ == "__main__":
    main()
# cd /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/verification
# # 交互式逐步验证（按Enter继续，可skip/quit）
# python verify_golden_step_by_step.py --mode=interactive
# # 自动验证所有步骤（无需交互）
# python verify_golden_step_by_step.py --mode=auto
# # 验证指定步骤（1-10）
# python verify_golden_step_by_step.py --mode=single --step=3
# # 自定义测试数据形状
# python verify_golden_step_by_step.py --mode=interactive --B=4 --S=16 --n=8 --C=128 --iters=5