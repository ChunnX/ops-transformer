#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import datetime
import torch

RATIO_THRESHOLD = 0.005
FAIL_RATIO_LIMIT = 0.005


def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                     str(sys._getframe().f_back.f_lineno).zfill(4), data))


class ConfigFmk:
    golden_inf_switch = False
    inf_clip_switch = False
    nan_toZero_switch = False


def get_pt_dtype(type_str):
    type_dict = {
        'fp32': torch.float32, 'fp16': torch.float16, 'fp64': torch.float64,
        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64,
        'uint8': torch.uint8, 'bool': torch.bool, 'complex64': torch.complex64,
        'complex128': torch.complex128, 'bf16': torch.bfloat16, 'uint1': torch.uint8,
        'float8_e4m3fn': torch.float32, 'float4_e2m1': torch.uint8, 'fp8_e8m0': torch.uint8,
        'complex32': torch.complex32
    }
    return type_dict.get(type_str, torch.float32)


class Result:
    def __init__(self, result_name, total_big_num, total_big_ratio, diff_big_max, diff_big_avg, diff_big_sum,
                 err_w1_num, err_w1_ratio, err_k1_num, err_k1_ratio, err_k5_num, err_k5_ratio, err_h1_num, err_h1_ratio,
                 total_small_num, total_small_ratio, err_small_num, err_small_ratio,
                 diff_rmse, rst_eb, diff_eb, bm_cmp_std,
                 num_total_nan=0, err_total_nan=0, num_total_inf=0, err_total_inf=0, num_total_ninf=0,
                 err_total_ninf=0,
                 diff_big_ratio_max=None, diff_big_ratio_avg=None, diff_big_ratio_rmse=None):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.err_w1_num = err_w1_num
        self.err_w1_ratio = err_w1_ratio
        self.err_k1_num = err_k1_num
        self.err_k1_ratio = err_k1_ratio
        self.err_k5_num = err_k5_num
        self.err_k5_ratio = err_k5_ratio
        self.err_h1_num = err_h1_num
        self.err_h1_ratio = err_h1_ratio
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.bm_cmp_std = bm_cmp_std
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf
        self.diff_big_ratio_max = diff_big_ratio_max
        self.diff_big_ratio_avg = diff_big_ratio_avg
        self.diff_big_ratio_rmse = diff_big_ratio_rmse

    def print_result(self):
        print_log(f"正在打印结果：{self.result_name}")
        print_log(f" 大值总数：{self.total_big_num}")
        print_log(f" 大值占比：{self.total_big_ratio:.2%}")
        print_log(f" 大值最大绝对误差：{self.diff_big_max:.8f}")
        print_log(f" 大值平均绝对误差：{self.diff_big_avg:.8f}")
        print_log(f" 大值绝对误差总和：{self.diff_big_sum:.2f}")
        print_log(f" 大值最大相对误差：{self.diff_big_ratio_max:.8f}")
        print_log(f" 大值平均相对误差：{self.diff_big_ratio_avg:.8f}")
        print_log(f" 大值相对误差均方根（RMSE）：{self.diff_big_ratio_rmse:.8f}")
        print_log(f" 大值万分之1误差数：{self.err_w1_num}，占比{self.err_w1_ratio:.2%}")
        print_log(f" 大值千分之1误差数：{self.err_k1_num}，占比{self.err_k1_ratio:.2%}")
        print_log(f" 大值千分之5误差数：{self.err_k5_num}，占比{self.err_k5_ratio:.2%}")
        print_log(f" 大值百分之1误差数：{self.err_h1_num}，占比{self.err_h1_ratio:.2%}")
        print_log(f" 小值总数：{self.total_small_num}")
        print_log(f" 小值占比：{self.total_small_ratio:.2%}")
        print_log(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print_log(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print_log(f" 均衡性偏差计数：{self.rst_eb}")
        print_log(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0) or \
                (self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0) or True:
            print_log(f" golden nan总数：{self.num_total_nan}")
            print_log(f" nan误差数：{self.err_total_nan}")
            print_log(f" golden inf总数：{self.num_total_inf}")
            print_log(f" inf误差数：{self.err_total_inf}")
            print_log(f" golden -inf总数：{self.num_total_ninf}")
            print_log(f" -inf误差数：{self.err_total_ninf}")

    def check_result_debug(self, benchmark, new=False, output_dtype='fp16'):
        if new:
            lo_bound = 0.0
            if output_dtype == 'fp16':
                lo_bound = 2 ** (-11)
            if output_dtype == 'bf16':
                lo_bound = 2 ** (-8)
            benchmark.diff_big_ratio_max = max(benchmark.diff_big_ratio_max, lo_bound)
            benchmark.diff_big_ratio_avg = max(benchmark.diff_big_ratio_avg, lo_bound)
            benchmark.diff_rmse = max(benchmark.diff_rmse, lo_bound)
        reason_str = ''
        if self.diff_big_ratio_max > benchmark.diff_big_ratio_max * self.bm_cmp_std['max_re_rtol']:
            reason_str += ' diff_big_ratio_max error/'
        elif self.diff_big_ratio_max > benchmark.diff_big_ratio_max:
            reason_str += ' diff_big_ratio_max warning/'
        if self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg * self.bm_cmp_std['avg_re_rtol']:
            reason_str += ' diff_big_ratio_avg error/'
        elif self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg:
            reason_str += ' diff_big_ratio_avg warning/'
        if not new:
            if self.diff_big_sum > benchmark.diff_big_sum * self.bm_cmp_std['avg_re_rtol']:
                reason_str += ' diff_big_sum error/'
            elif self.diff_big_sum > benchmark.diff_big_sum:
                reason_str += ' diff_big_sum warning/'
            if self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse * self.bm_cmp_std['rmse_rtol']:
                reason_str += ' diff_big_ratio_rmse error/'
            elif self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse:
                reason_str += ' diff_big_ratio_rmse warning/'
        if self.err_small_num > benchmark.err_small_num * self.bm_cmp_std['avg_re_rtol']:
            reason_str += ' err_small_num error/'
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += ' err_small_num warning/'
        if self.diff_rmse > benchmark.diff_rmse * self.bm_cmp_std['rmse_rtol']:
            reason_str += ' diff_rmse error/'
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += ' diff_rmse warning/'
        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += ' err_total_nan error/'
        elif self.err_total_nan > 0:
            reason_str += ' err_total_nan warning/'
        if self.err_total_inf > benchmark.err_total_inf or self.err_total_ninf > benchmark.err_total_ninf:
            reason_str += ' err_total_inf error/'
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += ' err_total_inf warning'
        return reason_str

    def check_result(self, benchmark, new=False, output_dtype='fp16'):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        reason_str = self.check_result_debug(benchmark, new, output_dtype)
        if 'error' in reason_str:
            print(self.result_name + ' compare result: error')
            return 'Failed', reason_str
        elif 'warning' in reason_str:
            print(self.result_name + ' compare result: warning')
            return 'warning', reason_str
        else:
            print(self.result_name + ' compare result: ok')
        return 'Pass', ''


def checkResultNew(params, value, golden, name, output_dtype):
    cfg_fk = ConfigFmk()
    debug_switch = False
    red = params['red_range'][output_dtype]
    red_list_str = red.split("/")
    red_list = [float(i) for i in red_list_str]
    bm_cmp_std = params['bm_cmp_std'][output_dtype]

    print(f"info：开始计算 {name} 精度。")
    if value.shape == golden.shape:
        if cfg_fk.golden_inf_switch:
            golden[golden > torch.finfo(value.dtype).max] = torch.inf
            golden[golden < torch.finfo(value.dtype).min] = -torch.inf

        if cfg_fk.inf_clip_switch:
            golden[torch.isinf(golden)] = torch.finfo(value.dtype).max
            value[torch.isinf(value)] = torch.finfo(value.dtype).max
        if cfg_fk.nan_toZero_switch:
            golden[torch.isnan(golden)] = 0
            value[torch.isnan(value)] = 0

        mask_golden_is_nan = torch.isnan(golden)
        mask_value_is_nan = torch.isnan(value)
        num_total_nan = torch.sum(mask_golden_is_nan)
        err_total_nan = torch.sum(mask_golden_is_nan.logical_xor(mask_value_is_nan))

        mask_golden_is_inf = torch.isinf(golden) & (golden > 0)
        mask_value_is_inf = torch.isinf(value) & (value > 0)
        num_total_inf = torch.sum(mask_golden_is_inf)
        err_total_inf = torch.sum(mask_golden_is_inf.logical_xor(mask_value_is_inf))

        mask_golden_is_ninf = torch.isinf(golden) & (golden < 0)
        mask_value_is_ninf = torch.isinf(value) & (value < 0)
        num_total_ninf = torch.sum(mask_golden_is_ninf)
        err_total_ninf = torch.sum(mask_golden_is_ninf.logical_xor(mask_value_is_ninf))

        if debug_switch:
            print(f" inf/nan总数：{num_total_nan + num_total_inf + num_total_ninf}")
            print(f" inf/nan误差数：{err_total_nan + err_total_inf + err_total_ninf}")

        golden[torch.isinf(golden)] = 1
        value[torch.isinf(value)] = 1
        golden[torch.isnan(golden)] = 1
        value[torch.isnan(value)] = 1

        total_big_num = torch.sum(golden.abs() >= bm_cmp_std['small_value'])
        total_big_ratio = total_big_num / golden.numel()

        value_big = value.clone()
        value_big[golden.abs() < bm_cmp_std['small_value']] = 1
        golden_big = golden.clone()
        golden_big[golden.abs() < bm_cmp_std['small_value']] = 1

        diff_big = torch.abs(value_big.sub(golden_big))
        diff_big_max = diff_big.max()
        diff_big_sum = diff_big.sum()
        diff_big_avg = diff_big_sum / total_big_num
        diff_big_rmse = torch.sqrt(torch.mean(torch.square(diff_big)))

        diff_big_ratio = diff_big / golden_big.abs()
        diff_big_ratio_max = diff_big_ratio.max()
        diff_big_ratio_avg = diff_big_ratio.sum() / total_big_num
        diff_big_ratio_rmse = torch.sqrt(torch.mean(torch.square(diff_big_ratio)))

        err_w1_num = torch.sum(diff_big_ratio > red_list[0])
        err_w1_ratio = err_w1_num / total_big_num
        err_k1_num = torch.sum(diff_big_ratio > red_list[1])
        err_k1_ratio = err_k1_num / total_big_num
        err_k5_num = torch.sum(diff_big_ratio > red_list[2])
        err_k5_ratio = err_k5_num / total_big_num
        err_h1_num = torch.sum(diff_big_ratio > red_list[3])
        err_h1_ratio = err_h1_num / total_big_num

        total_small_num = torch.sum(golden.abs() < bm_cmp_std['small_value'])
        total_small_ratio = total_small_num / golden.numel()

        value_small = value.clone()
        value_small[golden.abs() > bm_cmp_std['small_value']] = 1
        golden_small = golden.clone()
        golden_small[golden.abs() > bm_cmp_std['small_value']] = 1

        diff_small = torch.abs(value_small.sub(golden_small))
        err_small_num = torch.sum(diff_small > bm_cmp_std['small_value_atol'])
        err_small_ratio = err_small_num / total_small_num

        diff = torch.abs(value.sub(golden))
        diff_rmse = torch.sqrt(torch.mean(torch.square(diff)))

        eb_bigger = torch.sum(value > golden)
        eb_smaller = torch.sum(value < golden)
        rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
        diff_eb = torch.sum(value.sub(golden))

        return Result(name,
                      total_big_num.item(), total_big_ratio.item(),
                      diff_big_max.item(), diff_big_avg.item(), diff_big_sum.item(),
                      err_w1_num.item(), err_w1_ratio.item(),
                      err_k1_num.item(), err_k1_ratio.item(),
                      err_k5_num.item(), err_k5_ratio.item(),
                      err_h1_num.item(), err_h1_ratio.item(),
                      total_small_num.item(), total_small_ratio.item(),
                      err_small_num.item(), err_small_ratio.item(),
                      diff_rmse.item(), rst_eb.item(), diff_eb.item(),
                      bm_cmp_std,
                      num_total_nan.item(), err_total_nan.item(),
                      num_total_inf.item(), err_total_inf.item(),
                      num_total_ninf.item(), err_total_ninf.item(),
                      diff_big_ratio_max=diff_big_ratio_max.item(),
                      diff_big_ratio_avg=diff_big_ratio_avg.item(),
                      diff_big_ratio_rmse=diff_big_ratio_rmse.item())
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: value {value.shape}")
        print_log(f"debug: golden {golden.shape}")
        raise ValueError(f"Shape mismatch: value {value.shape}, golden {golden.shape}")


def data_compare_benchmark_new(params, npu_output, cpu_output, cpu_golden, output_dtype, i):
    cfg_fk = ConfigFmk()
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    cpu_golden = cpu_golden.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size and real_data.size == cpu_golden.size:
        print_log('The npu_output is [],and it is same as bm_output/cpu_golden, the result of data_compare is "Pass"')
        return "Pass", 100.0, 0
    max_error = ''
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, '', max_error

    if real_data.size != cpu_golden.size:
        print_log('Error,the size of npu output[%s] and golden[%s] is not equal.' % (real_data.size, cpu_golden.size))
        return result, '', max_error

    def to_target(t, dtype_str):
        if dtype_str in ("float8_e5m2", "float8_e4m3fn", "hifloat8"):
            qDtype = get_pt_dtype(params["dtype_input"][0])
            return t.to(qDtype)
        elif dtype_str == "bfloat16":
            return t.to(torch.bfloat16)
        elif dtype_str == "fp16":
            return t.to(torch.float16)
        else:
            return t

    npu_res = to_target(torch.from_numpy(real_data).detach(), output_dtype)
    benchmark_res = to_target(torch.from_numpy(data_compe).detach(), output_dtype)
    golden = torch.from_numpy(cpu_golden).detach()

    rst_npu = checkResultNew(params, npu_res, golden, params["case_name"], output_dtype)
    rst_npu.print_result()

    golden = torch.from_numpy(cpu_golden).detach()
    rst_gpu = checkResultNew(params, benchmark_res, golden, params["case_name"], output_dtype)
    rst_gpu.print_result()

    str1, str2 = rst_npu.check_result(rst_gpu, True, output_dtype)

    data = [params['op_name'], params['case_name'], f"{rst_npu.total_big_num}", f"{rst_npu.total_big_ratio:.2%}",
            f"{rst_npu.err_w1_ratio:.2%}", f"{rst_npu.err_k1_ratio:.2%}", f"{rst_npu.err_k5_ratio:.2%}",
            f"{rst_npu.err_h1_ratio:.2%}",
            f"{rst_npu.diff_big_max:.8f}", f"{rst_npu.diff_big_avg:.8f}", f"{rst_npu.diff_big_sum:.2f}",
            f"{rst_npu.diff_big_ratio_max:.8f}", f"{rst_npu.diff_big_ratio_avg:.8f}",
            f"{rst_npu.diff_big_ratio_rmse:.8f}",
            f"{rst_npu.total_small_num}", f"{rst_npu.total_small_ratio:.2%}", f"{rst_npu.err_small_num}",
            f"{rst_npu.err_small_ratio:.2%}", f"{rst_npu.num_total_nan:.2f}",
            f"{rst_npu.err_total_nan:.2f}", f"{rst_npu.num_total_inf:.2f}", f"{rst_npu.err_total_inf:.2f}",
            f"{rst_npu.num_total_ninf:.2f}", f"{rst_npu.err_total_ninf:.2f}",
            f"{rst_npu.diff_rmse:.8f}", f"{rst_npu.rst_eb}", f"{rst_npu.diff_eb:.8f}",
            f"{rst_gpu.total_big_num}",
            f"{rst_gpu.total_big_ratio:.2%}",
            f"{rst_gpu.err_w1_ratio:.2%}", f"{rst_gpu.err_k1_ratio:.2%}", f"{rst_gpu.err_k5_ratio:.2%}",
            f"{rst_gpu.err_h1_ratio:.2%}",
            f"{rst_gpu.diff_big_max:.8f}", f"{rst_gpu.diff_big_avg:.8f}", f"{rst_gpu.diff_big_sum:.2f}",
            f"{rst_gpu.diff_big_ratio_max:.8f}", f"{rst_gpu.diff_big_ratio_avg:.8f}",
            f"{rst_gpu.diff_big_ratio_rmse:.8f}",
            f"{rst_gpu.total_small_num}", f"{rst_gpu.total_small_ratio:.2%}", f"{rst_gpu.err_small_num}",
            f"{rst_gpu.err_small_ratio:.2%}", f"{rst_gpu.num_total_nan:.2f}",
            f"{rst_gpu.err_total_nan:.2f}", f"{rst_gpu.num_total_inf:.2f}", f"{rst_gpu.err_total_inf:.2f}",
            f"{rst_gpu.num_total_ninf:.2f}", f"{rst_gpu.err_total_ninf:.2f}", f"{rst_gpu.diff_rmse:.8f}",
            f"{rst_gpu.rst_eb}", f"{rst_gpu.diff_eb:.8f}", f"{str1}", f"{str2}",
            f"output_{i}"]

    return str1, str2, data


def check_result(test_name, expect, result, except_label="CPU", comp_label="NPU", verbose_diff=False):
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}  ({except_label} vs {comp_label})")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(f"│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(result.shape)}")
        print(f"└{SEP}┘")
        return False
    ef   = expect.float()
    rf   = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    rel_err = diff / (ref_abs + 1e-6)
    max_abs   = diff.max().item()
    mean_abs  = diff.mean().item()
    max_rel   = rel_err.max().item()
    mean_rel  = rel_err.mean().item()
    threshold = torch.max(ref_abs.mul(RATIO_THRESHOLD), torch.full_like(diff, 0.000025))
    fail_mask = diff > threshold
    fail_cnt  = int(fail_mask.sum().item())
    total     = expect.numel()
    fail_ratio = fail_cnt / total
    passed    = fail_ratio <= FAIL_RATIO_LIMIT
    print(f"│  Shape       : {tuple(expect.shape)}")
    print(f"│  MaxAbsErr   : {max_abs:.8f}")
    print(f"│  MeanAbsErr  : {mean_abs:.8f}")
    print(f"│  MaxRelErr   : {max_rel:.8f}")
    print(f"│  MeanRelErr  : {mean_rel:.8f}")
    print(f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio*100:.4f}%)")
    print(f"│  Threshold   : elemRelErr≤{RATIO_THRESHOLD*100:.2f}%  failRatio≤{FAIL_RATIO_LIMIT*100:.2f}%")
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}")
    if fail_cnt > 0:
        print(f"├{SEP}┤")
        if verbose_diff:
            all_idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            print(f"│  全部 {len(all_idxs)} 个超阈値元素 (relErr > {RATIO_THRESHOLD * 100:.2f}%):")
            print(f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}")
            for i in all_idxs:
                print(f"│  {i:>10}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>12.6f}")
        else:
            idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            print(f"│  前{len(idxs)}个不通过元素:")
            print(f"│  {'idx':>8}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>10}")
            for i in idxs:
                print(f"│  {i:>8}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>10.6f}")
    print(f"└{SEP}┘")
    stats = {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "fail_cnt": fail_cnt,
        "total": total,
        "fail_ratio": fail_ratio,
    }
    return stats


def analyze_fail_distribution(test_name, expect, result, dump_dir="./dump_output", **kwargs):
    ef = expect.float()
    rf = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    threshold = torch.max(ref_abs.mul(RATIO_THRESHOLD), torch.full_like(diff, 0.000025))
    fail_mask = diff > threshold

    shape = expect.shape
    ndim = len(shape)
    total_fail = int(fail_mask.sum().item())
    total_elem = expect.numel()

    SEP = "━" * 80
    lines = []
    lines.append(f"\n{SEP}")
    lines.append(f"  失败元素分布分析: {test_name}")
    lines.append(f"  Shape: {tuple(shape)}  FailElems: {total_fail}/{total_elem} ({total_fail/total_elem*100:.4f}%)")
    lines.append(SEP)

    if ndim == 4:
        B, N, S, D = shape
        lines.append(f"\n  [按 Batch 维度统计] (共 {B} 个 batch)")
        lines.append(f"  {'Batch':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}  {'MeanAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")
        batch_stats = []
        for bi in range(B):
            b_fail = fail_mask[bi]
            b_diff = diff[bi]
            cnt = int(b_fail.sum().item())
            tot = N * S * D
            ratio = cnt / tot
            max_err = b_diff.max().item() if cnt > 0 else 0.0
            mean_err = b_diff[b_fail].mean().item() if cnt > 0 else 0.0
            batch_stats.append((bi, cnt, tot, ratio, max_err, mean_err))
            lines.append(f"  {bi:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}  {mean_err:>12.8f}")

        lines.append(f"\n  [按 Head 维度统计] (共 {N} 个 head，显示 Top-10 失败率最高)")
        lines.append(f"  {'Head':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
        head_stats = []
        for ni in range(N):
            h_fail = fail_mask[:, ni]
            h_diff = diff[:, ni]
            cnt = int(h_fail.sum().item())
            tot = B * S * D
            ratio = cnt / tot
            max_err = h_diff.max().item() if cnt > 0 else 0.0
            head_stats.append((ni, cnt, tot, ratio, max_err))
        head_stats.sort(key=lambda x: x[3], reverse=True)
        for ni, cnt, tot, ratio, max_err in head_stats[:10]:
            lines.append(f"  {ni:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}")

        lines.append(f"\n  [按 Seq 位置统计] (共 {S} 个位置，显示 Top-10 失败率最高)")
        lines.append(f"  {'SeqPos':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}  {'MaxAbsErr':>12}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
        seq_stats = []
        for si in range(S):
            s_fail = fail_mask[:, :, si]
            s_diff = diff[:, :, si]
            cnt = int(s_fail.sum().item())
            tot = B * N * D
            ratio = cnt / tot
            max_err = s_diff.max().item() if cnt > 0 else 0.0
            seq_stats.append((si, cnt, tot, ratio, max_err))
        seq_stats.sort(key=lambda x: x[3], reverse=True)
        for si, cnt, tot, ratio, max_err in seq_stats[:10]:
            lines.append(f"  {si:>6}  {cnt:>10}  {tot:>10}  {ratio*100:>9.4f}%  {max_err:>12.8f}")

        seqused_kv = kwargs.get("seqused_kv", None)
        if seqused_kv is not None:
            lines.append(f"\n  [Batch vs seqused_kv 关联分析]")
            lines.append(f"  {'Batch':>6}  {'seqused_kv':>12}  {'FailRatio':>10}  {'说明':>20}")
            lines.append(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*20}")
            for bi, cnt, tot, ratio, max_err, mean_err in batch_stats:
                skv = seqused_kv[bi] if bi < len(seqused_kv) else "N/A"
                note = ""
                if ratio > 0.03:
                    note = "⚠ 超3%阈值"
                elif ratio > 0.01:
                    note = "△ 偏高"
                lines.append(f"  {bi:>6}  {skv:>12}  {ratio*100:>9.4f}%  {note:>20}")

        lines.append(f"\n  [错误位置连续性分析] (选取失败率最高的 3 个 batch，分析 D 维度连段)")
        top_batches = sorted(batch_stats, key=lambda x: x[3], reverse=True)[:3]
        for bi, cnt, tot, ratio, max_err, mean_err in top_batches:
            if cnt == 0:
                continue
            lines.append(f"\n  ── Batch {bi} (FailRatio={ratio*100:.4f}%) ──")
            b_fail = fail_mask[bi]
            hs_fail_cnt = b_fail.sum(dim=-1)
            max_hs_idx = int(hs_fail_cnt.argmax().item())
            max_h = max_hs_idx // S
            max_s = max_hs_idx % S
            d_fail_vec = b_fail[max_h, max_s]
            d_fail_cnt = int(d_fail_vec.sum().item())
            lines.append(f"    最密集位置: head={max_h}, seq={max_s}, 失败D元素={d_fail_cnt}/{D}")

            fail_positions = d_fail_vec.nonzero(as_tuple=False).squeeze(1).tolist()
            if len(fail_positions) == 0:
                lines.append(f"    无失败位置")
                continue

            segments = []
            seg_start = fail_positions[0]
            seg_end = fail_positions[0]
            for pos in fail_positions[1:]:
                if pos == seg_end + 1:
                    seg_end = pos
                else:
                    segments.append((seg_start, seg_end))
                    seg_start = pos
                    seg_end = pos
            segments.append((seg_start, seg_end))

            num_seg = len(segments)
            max_seg_len = max(e - s + 1 for s, e in segments)
            avg_seg_len = d_fail_cnt / num_seg

            if num_seg == 1 and d_fail_cnt == D:
                pattern = "全D失败（整行错误）"
            elif avg_seg_len >= 8 and num_seg <= d_fail_cnt / 4:
                pattern = "连段集中（可能与tiling/block边界相关）"
            elif num_seg >= d_fail_cnt * 0.8:
                pattern = "随机分散（典型精度累积误差）"
            else:
                pattern = "混合模式"

            lines.append(f"    连续段数: {num_seg}  最长段: {max_seg_len}  平均段长: {avg_seg_len:.1f}")
            lines.append(f"    模式判定: {pattern}")
            show_segs = segments[:8]
            seg_strs = [f"[{s}:{e}]({e-s+1})" for s, e in show_segs]
            lines.append(f"    前{len(show_segs)}段: {', '.join(seg_strs)}")
            if len(segments) > 8:
                lines.append(f"    ... 共 {num_seg} 段")

            lines.append(f"    该 head={max_h} 在 batch={bi} 中各 seq 位置失败率:")
            seq_fail_in_bh = b_fail[max_h].sum(dim=-1)
            chunk_size = max(1, S // 16)
            chunks = []
            for ci in range(0, S, chunk_size):
                c_end = min(ci + chunk_size, S)
                c_cnt = int(seq_fail_in_bh[ci:c_end].sum().item())
                c_tot = (c_end - ci) * D
                chunks.append(f"{c_cnt/c_tot*100:.1f}%")
            lines.append(f"    seq分段失败率(每{chunk_size}位置): [{', '.join(chunks)}]")

    elif ndim == 3:
        T, N, D = shape
        lines.append(f"\n  [按 Head 维度统计] (TND: T={T}, N={N}, D={D})")
        lines.append(f"  {'Head':>6}  {'FailCnt':>10}  {'Total':>10}  {'FailRatio':>10}")
        lines.append(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
        for ni in range(N):
            h_fail = fail_mask[:, ni]
            cnt = int(h_fail.sum().item())
            tot = T * D
            lines.append(f"  {ni:>6}  {cnt:>10}  {tot:>10}  {cnt/tot*100:>9.4f}%")
    else:
        lines.append(f"  [维度={ndim}，跳过分维统计]")

    lines.append(f"\n{SEP}")

    report = "\n".join(lines)
    print(report)

    dump_path = os.path.join(dump_dir, test_name)
    os.makedirs(dump_path, exist_ok=True)
    report_file = os.path.join(dump_path, "fail_analysis.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [分析报告已保存] {report_file}")

    return batch_stats if ndim == 4 else None
