import os
import argparse
import time
import numpy as np
import torch
import torch_npu
import queue
import multiprocessing

import openpyxl
from lightning_indexer_klloss_golden import LightningIndicesKLLoss

import datetime 
from npu_utils.npu_tools import *
global device
catch_refresh = True
cache_saving = False

def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(1, len(seqlens_list) + 1):
        cu[i] = cu[i - 1] + seqlens_list[i - 1]
    return cu


def check_cache(case):
    sparse_mode = case["sparse_mode"]
    dtype = case["dtype"].lower()
    DR = case["DR"]

    inpu_path = case.get("inpu_path", "NULL")
    print_log(inpu_path)
    # 检查golden数据是否齐全
    ck1 = os.path.exists(inpu_path + 'q.pt')
    ck2 = os.path.exists(inpu_path + 'k.pt')
    ck3 = os.path.exists(inpu_path + 'q_index.pt')
    ck4 = os.path.exists(inpu_path + 'k_index.pt')
    ck5 = os.path.exists(inpu_path + 'weights.pt')
    ck6 = os.path.exists(inpu_path + 'sparse_indices.pt')
    ck7 = os.path.exists(inpu_path + 'softmax_max.pt')
    ck8 = os.path.exists(inpu_path + 'softmax_sum.pt')
    ck9 = os.path.exists(inpu_path + 'q_rope.pt') if DR != 0 else True
    ck10 = os.path.exists(inpu_path + 'k_rope.pt') if DR != 0 else True

    ck11 = os.path.exists(inpu_path + 'd_query_index.pt')
    ck12 = os.path.exists(inpu_path + 'd_key_index.pt')
    ck13 = os.path.exists(inpu_path + 'd_weights.pt')
    ck14 = os.path.exists(inpu_path + 'loss.pt')

    catch_ok = ck1 and ck2 and ck3 and ck4 and ck5 and ck6 and ck7 and ck8 and ck9 and ck10 and ck11 and ck12 and ck13 and ck14
    print(f"ck1-q:{ck1},ck2-k:{ck2},ck3-q_index:{ck3},ck4-v_index:{ck4},ck5-weights:{ck5},ck6-sparse_indices:{ck6},"
          f"ck7-max:{ck7},ck8-sum:{ck8},ck9-q_rope:{ck9},ck10-k_rope:{ck10},ck11-d_query_index:{ck11},"
          f"ck12-d_key_index:{ck12},ck13-d_weights:{ck13},ck14-loss:{ck14}")
    return catch_ok


def gen_inputs(input_dtype, output_dtype, **kwargs):
    seqlens_list_array = np.array(case["seqlens_list_q"]) if case["seqlens_list_q"] is not None else None
    seqlens_list_kv_array = np.array(case["seqlens_list_kv"]) if case["seqlens_list_kv"] is not None else seqlens_list_array
    isTnd = kwargs.get("input_layout").upper() == 'TND'
    B = kwargs.get("B") if not isTnd else 1
    NQuery = kwargs.get("NQuery")
    NQueryIndex = kwargs.get("NQueryIndex")
    N2 = kwargs.get("N2")
    S1 = kwargs.get("S1")
    S2 = kwargs.get("S2")
    D = kwargs.get("D")
    DIndex = kwargs.get("DIndex")
    DR = kwargs.get("DR")
    dtype = kwargs.get("dtype")
    q = torch.randn(B, S1, NQuery, D, generator=torch.Generator().manual_seed(42), dtype=output_dtype, device=torch.device('cpu'))
    k = torch.randn(B, S2, N2, D, generator=torch.Generator().manual_seed(43), dtype=output_dtype, device=torch.device('cpu'))

    q_index = torch.randn(B, S1, NQueryIndex, DIndex, generator=torch.Generator().manual_seed(42), dtype=output_dtype, device=torch.device('cpu'))
    k_index = torch.randn(B, S2, N2, DIndex, generator=torch.Generator().manual_seed(43), dtype=output_dtype, device=torch.device('cpu'))
    if DR != 0:
        q_rope = torch.randn(B, S1, NQuery, DR, generator=torch.Generator().manual_seed(45), dtype=output_dtype, device=torch.device('cpu'))
        k_rope = torch.randn(B, S2, N2, DR, generator=torch.Generator().manual_seed(46), dtype=output_dtype, device=torch.device('cpu'))
        q_all = torch.cat((q, q_rope), -1)
        k_all = torch.cat((k, k_rope), -1)
    else:
        q_rope = None
        k_rope = None
        q_all = q
        k_all = k
    weights = torch.randn(B, S1, NQueryIndex, generator=torch.Generator().manual_seed(47), dtype=output_dtype, device=torch.device('cpu'))
    a = -0.05  # 最小值
    b = 0.05 # 最大值
    kk = 3.0  # 控制分布范围（3σ 覆盖绝大多数值）
    scale = (b - a) / (2 * kk)
    shift = (a + b) / 2
    weights = weights * scale + shift
    if isTnd:
        sparse_indices = torch.zeros(S1, N2, 2048).to(torch.int32)
        tIdx = 0
        for bIdx in range(kwargs.get("B")):
            for s1Idx in range(seqlens_list_array[bIdx]):
                s2RealSize = (int)((seqlens_list_kv_array[bIdx] - seqlens_list_array[bIdx]) + s1Idx + 1)
                if s2RealSize <= 0:
                    s2RealSize = seqlens_list_kv_array[bIdx];
                
                if s2RealSize > 2048:
                    s2RealLen = 2048
                else:
                    s2RealLen = s2RealSize
                #处理S2无效行场景，把对应的sparse indices置为-1
                # print(f"s2RealLen in batch {bIdx} s1 {s1Idx} is {s2RealLen}")
                sparse_indices[tIdx, :, 0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32))
                sparse_indices[tIdx, :, s2RealLen : 2048] = -1
                tIdx = tIdx + 1
        q_tnd = q.squeeze(dim=0)
        k_tnd = k.squeeze(dim=0)
        q_all = q_all.squeeze(dim=0)
        k_all = k_all.squeeze(dim=0)
        q_index_tnd = q_index.squeeze(dim=0)
        k_index_tnd = k_index.squeeze(dim=0)
        if q_rope is not None:
            q_rope_tnd = q_rope.squeeze(dim=0)
            k_rope_tnd = k_rope.squeeze(dim=0)
        else :
            q_rope_tnd = None
            k_rope_tnd = None
        weights_tnd = weights.squeeze(dim=0)
        return q_tnd, k_tnd, q_index_tnd, k_index_tnd, q_rope_tnd, k_rope_tnd, weights_tnd, sparse_indices, q_all, k_all
    else :
        sparse_indices = torch.zeros(B, S1, N2, 2048).to(torch.int32)
        for s1Idx in range(S1):
            s2RealSize = (int)(S2 - S1 + s1Idx + 1)
            if s2RealSize <= 0:
                s2RealSize = S2

            if s2RealSize > 2048:
                s2RealLen = 2048
            else:
                s2RealLen = s2RealSize
            sparse_indices[:, s1Idx, 0,  0 : s2RealLen] = (torch.randint(0, s2RealSize, (s2RealLen,)).to(torch.int32))
            sparse_indices[:, s1Idx, 0,  s2RealLen : 2048] = -1
        return q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, q_all, k_all

def lig_train_golden(q, k, sparse_indices, scale, input_layout, q_index, k_index, weights, q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen):
    op = LightningIndicesKLLoss(q, k, sparse_indices, scale, input_layout, q_index, k_index, weights, q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen)
    if case["input_layout"] == "TND":
        return op.run_unpad()
    else:
        return op.run()


def run(case, args, result_queue):
    testcase_name = case["testcase_name"]
    B = case["B"]
    NQuery = case["NQuery"]
    NQueryIndex = case["NQueryIndex"]
    N2 = case["N2"] if not (case["N2"] is None) else 1
    S1 = case["S1"]
    S2 = case["S2"] if not (case["S2"] is None) else S1
    seqlens_list_q = np.array(case["seqlens_list_q"])
    seqlens_list_kv = np.array(case["seqlens_list_kv"]) if case["seqlens_list_kv"] is not None else seqlens_list_q
    print(f"seqlens_list_q = {seqlens_list_q}")
    print(f"seqlens_list_kv = {seqlens_list_kv}")
    D = case["D"]
    DR = case["DR"]
    scale = 1 / (D ** 0.5)
    dtype = case["dtype"]  # 'fp16' & 'bf16' & 'fp32'
    sparse_mode = int(case['sparse_mode'] if (case["sparse_mode"] != '') else 0)
    input_layout = case["input_layout"]  # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    pre_tokens = case["pre_tokens"]
    next_tokens = case["next_tokens"]
    print(f"Test case {testcase_name}")
    if case["out_dtype"] in FP16:
        case["out_dtype"] = 0
    if case["out_dtype"] in BF16:
        case["out_dtype"] = 1
    out_dtype = case["out_dtype"]
    inpu_path = f"./auto_input/{testcase_name}/"
    case.update({"scale": scale})
    case.update({"inpu_path": inpu_path})
    input_dtype = torch.float
    pttype = torch.float
    if dtype in FP16:
        pttype = torch.float16
        input_dtype = torch.float16
    if dtype in BF16:
        pttype = torch.bfloat16
        input_dtype = torch.bfloat16
    if dtype in FP8_E5M2 or dtype in FP8_E4M3:
        input_dtype = torch.float8_e4m3fn if dtype in FP8_E4M3 else torch.float8_e5m2
        if out_dtype == 0:
            pttype = torch.float16
        elif out_dtype == 1:
            pttype = torch.bfloat16
        else:
            raise RuntimeError("[ERROR] Invalid out_dtype.")

    catch_ok = check_cache(case)
    # 读取golden缓存
    if catch_refresh:
        print_log("catch_refresh置位, 强制更新数据")
    elif not catch_ok:
        print_log("catch数据不存在或损坏, 重新生成数据")

    if input_layout == 'TND':
        actual_seq_qlen = tuple(get_cu_seqlens(seqlens_list_q)[1:].cpu().numpy().tolist())
        actual_seq_kvlen = tuple(get_cu_seqlens(seqlens_list_kv)[1:].cpu().numpy().tolist())
    else:
        actual_seq_qlen = None
        actual_seq_kvlen = None
    print(f"actual_seq_qlen = {actual_seq_qlen}")
    print(f"actual_seq_kvlen = {actual_seq_kvlen}")
    print(f"seqlens_list_q = {seqlens_list_q}")
    print(f"seqlens_list_kv = {seqlens_list_kv}")

    if catch_ok and not catch_refresh:
        print_log(f"catch命中, 正在读取标杆数据, path:{inpu_path}")
        q = torch.load(inpu_path + 'q.pt')
        k = torch.load(inpu_path + 'k.pt')
        q_index = torch.load(inpu_path + 'q_index.pt')
        k_index = torch.load(inpu_path + 'k_index.pt')
        weights = torch.load(inpu_path + 'weights.pt')
        sparse_indices = torch.load(inpu_path + 'sparse_indices.pt').to(torch.int32)
        softmax_max = torch.load(inpu_path + 'softmax_max.pt')
        softmax_sum = torch.load(inpu_path + 'softmax_sum.pt')
        if DR != 0:
            q_rope = torch.load(inpu_path + 'q_rope.pt') if os.path.exists(inpu_path + 'q_rope.pt') else None
            k_rope = torch.load(inpu_path + 'k_rope.pt') if os.path.exists(inpu_path + 'k_rope.pt') else None

        cpu_d_query_index = torch.load(inpu_path + 'd_query_index.pt')
        print(f"0000 cpu_d_query_index shape is {cpu_d_query_index.shape} dtype is {cpu_d_query_index.dtype}")

        cpu_d_key_index = torch.load(inpu_path + 'd_key_index.pt')
        cpu_d_weights = torch.load(inpu_path + 'd_weights.pt')
        cpu_loss = torch.load(inpu_path + 'loss.pt')
    else:
        print_log(f"正在生成标杆数据.")
        q, k, q_index, k_index, q_rope, k_rope, weights, sparse_indices, q_all, k_all = gen_inputs(input_dtype, pttype, **case)
        # 生成可选输入pse和atten_mask
        skip_invalid_row = False
        if sparse_mode == 3:
            skip_invalid_row = S1 > S2
        if sparse_mode == 4:
            skip_invalid_row = pre_tokens < 0 or next_tokens + S2 < S1
        if sparse_mode in [5, 6]:
            skip_invalid_row = True if 0 in prefix else False
        if sparse_mode == 0:
            skip_invalid_row = next_tokens < 0 or pre_tokens + S2 < S1
        case["is_skip_invalid_row"] = skip_invalid_row

        cpu_d_query_index, cpu_d_key_index, cpu_d_weights, cpu_loss, softmax_max, softmax_sum = \
            lig_train_golden(q_all, k_all, sparse_indices, scale, input_layout, q_index, k_index, weights, \
            q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen)
        if cache_saving:
            if not os.path.exists('./auto_input'):
                os.mkdir('./auto_input')
            if not os.path.exists(inpu_path):
                os.mkdir(inpu_path)
            torch.save(q.detach(), inpu_path + 'q.pt')
            torch.save(k.detach(), inpu_path + 'k.pt')

            torch.save(q_index.detach(), inpu_path + 'q_index.pt')
            torch.save(k_index.detach(), inpu_path + 'k_index.pt')
            if DR != 0:
                torch.save(q_rope.detach(), inpu_path + 'q_rope.pt')
                torch.save(k_rope.detach(), inpu_path + 'k_rope.pt')

            torch.save(weights.detach(), inpu_path + 'weights.pt')
            torch.save(sparse_indices.to(torch.int32).detach(), inpu_path + 'sparse_indices.pt')

            torch.save(softmax_max.detach(), inpu_path + 'softmax_max.pt')
            torch.save(softmax_sum.detach(), inpu_path + 'softmax_sum.pt')
            torch.save(cpu_d_query_index.detach(), inpu_path + 'd_query_index.pt')
            torch.save(cpu_d_key_index.detach(), inpu_path + 'd_key_index.pt')
            torch.save(cpu_d_weights.detach(), inpu_path + 'd_weights.pt')
            torch.save(cpu_loss.detach(), inpu_path + 'loss.pt')
            print(f"保存标杆数据成功")

    if args.run_mod != 'npu':
        print(f"run_mode is not npu, return.")
        return
    
    q1 = q.contiguous().to(device)
    k1 = k.contiguous().to(device)
    q_index1 = q_index.contiguous().to(device)
    k_index1 = k_index.contiguous().to(device)
    if DR != 0:
        q_rope = q_rope.contiguous().to(device)
        k_rope = k_rope.contiguous().to(device)
    else:
        q_rope = None
        k_rope = None

    
    # 采集内存消耗
    torch.npu.reset_peak_memory_stats()  # 重置显存统计信息
    start_memery = torch.npu.memory_allocated()
    print_log('start_memery:' + str(start_memery) + " B")
    print(f'======================== PTA eager BEGIN ========================')
    print(f"q1 shape is {q1.shape}")
    print(f"k1 shape is {k1.shape}")
    print(f"q_index1 shape is {q_index1.shape}")
    print(f"k_index1 shape is {k_index1.shape}")
    print(f"weights shape is {weights.shape}")
    print(f"sparse_indices shape is {sparse_indices.shape}")
    print(f"softmax_max shape is {softmax_max.shape}")
    print(f"softmax_sum shape is {softmax_sum.shape}")
    print(f"scale shape is {scale}")
    if q_rope is not None:
        print(f"q_rope shape is {q_rope.shape}")
        print(f"k_rope shape is {k_rope.shape}")
    print(f"actual_seq_qlen is {actual_seq_qlen}")
    print(f"actual_seq_qlen is {actual_seq_kvlen}")
    print(f"input_layout is {input_layout}")
    print(f"sparse_mode is {sparse_mode}")
    out_all = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss(
        q1, k1, q_index1, k_index1, weights.contiguous().to(device), sparse_indices.contiguous().to(device), softmax_max.contiguous().to(device), softmax_sum.contiguous().to(device), 
        q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen, scale, input_layout, sparse_mode, 65536, 65536, False
    )

    torch.npu.synchronize()
    fwd_memery = torch.npu.max_memory_allocated()
    print_log('fwd_memery:' + str(fwd_memery) + " B")
    allocated_memery_fwd = f"{((fwd_memery - start_memery) / 1024 ** 2):.2f}"  # 获取显存消耗情况，单位MB
    print_log('正向显存消耗:' + str(allocated_memery_fwd) + "MB")

    time20 = time.time()
    # time21 = time.time()
    # forward_e2e_time = (time21 - time20) * 1000
    # print_log(f"正向e2e耗时：{(time21 - time20):.2}s")
    npu_d_query_index = out_all[0].cpu()
    npu_d_key_index = out_all[1].cpu()
    npu_d_weights = out_all[2].cpu()
    npu_loss = out_all[3].cpu()

    rst0, max0, sum0 = checkResult(npu_d_query_index.float(), cpu_d_query_index, f"{testcase_name} d_query_index")
    rst1, max1, sum1 = checkResult(npu_d_key_index.float().contiguous().view(-1), cpu_d_key_index.contiguous().view(-1), f"{testcase_name} d_key_index")
    rst2, max2, sum2 = checkResult(npu_d_weights.float().contiguous().view(-1), cpu_d_weights.contiguous().view(-1), f"{testcase_name} d_weights")
    rst3, max3, sum3 = checkResult(npu_loss.float().contiguous().view(-1), cpu_loss.contiguous().view(-1), f"{testcase_name} loss")


    #### 测性能 ####
    for i in range(10):
        print('warmup i', i)
        out_all = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss(
            q1, k1, q_index1, k_index1, weights.contiguous().to(device), sparse_indices.contiguous().to(device), softmax_max.contiguous().to(device), softmax_sum.contiguous().to(device), 
            q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen, scale, input_layout, sparse_mode, 65536, 65536, False
        )

    torch.npu.synchronize()
    runtimes = 50
    start = time.time()
    for i in range(runtimes):
        print('speed i', i)
        out_all = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss(
            q1, k1, q_index1, k_index1, weights.contiguous().to(device), sparse_indices.contiguous().to(device), softmax_max.contiguous().to(device), softmax_sum.contiguous().to(device), 
            q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen, scale, input_layout, sparse_mode, 65536, 65536, False
        )
    torch.npu.synchronize()
    end = time.time()
    avg_time = (end - start) * 1000 / runtimes
    print('slig avg time(ms): ', avg_time)

    # args.prof = True
    # if args.prof:
    #     print_log('开始测试并启动profiler采集')
    #     prof_path = f"./prof_auto/{testcase_name}"
    #     # 删除历史性能文件
    #     os.system(f"rm -rf {prof_path}")
    #     experimental_config = torch_npu.profiler._ExperimentalConfig(
    #         profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    #         aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    #         l2_cache=False
    #     )
    #     with torch_npu.profiler.profile(
    #             activities=[
    #                 torch_npu.profiler.ProfilerActivity.CPU,
    #                 torch_npu.profiler.ProfilerActivity.NPU
    #             ],
    #             schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=5),
    #             on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_path),
    #             record_shapes=True,
    #             profile_memory=True,
    #             with_stack=True,
    #             # with_flops=True,
    #             experimental_config=experimental_config,
    #             with_modules=True,
    #     ) as prof:
    #         for i in range(10):
    #             torch.npu.manual_seed(2)
    #             out_all_prof = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss(
    #                 q1, k1, q_index1, k_index1, weights.contiguous().to(device), sparse_indices.contiguous().to(device), softmax_max.contiguous().to(device), softmax_sum.contiguous().to(device),
    #                 q_rope, k_rope, actual_seq_qlen, actual_seq_kvlen, scale, input_layout, sparse_mode, 65536, 65536, False
    #             )
 
    #             prof.step()

    #     # check profiling result
    #     if os.path.exists(prof_path):
    #         sub_folder = os.listdir(prof_path)[0]
    #     else:
    #         sub_folder = 'null'
    #         print_log('error：profiler采集失败！请检查是否有环境冲突')
    #     kernel_time_total = get_kernel_time_total(f"{prof_path}/{sub_folder}/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    #     print_log(f"profiler 总计时: {kernel_time_total:.6f} ms")

    #     kernel_time_forward = get_kernel_time_forward(f"{prof_path}/{sub_folder}/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    #     print_log(f"profiler 正向kernel耗时: {kernel_time_forward:.6f} ms")
    result_file = case["result_xlsx_name"]
    df = pd.read_excel(result_file, sheet_name=args.sheet)
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_out_pricision'] = rst0
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_max_pricision'] = rst1
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_sum_pricision'] = rst2

    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_out_err_max'] = max0
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_max_err_max'] = max1
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_sum_err_max'] = max2

    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_out_err_sum'] = sum0
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_max_err_sum'] = sum1
    df.loc[df['Testcase_Name'] == testcase_name, 'Actual_sum_err_sum'] = sum2
    # df.loc[df['Testcase_Name'] == testcase_name, 'Actual_e2e_time_forward'] = format(forward_e2e_time, '.6f')

    if args.prof:
        df.loc[df['Testcase_Name'] == testcase_name, 'Actual_Memory'] = str(allocated_memery_fwd)
    if args.prof or args.esl:
        df.loc[df['Testcase_Name'] == testcase_name, 'Actual_kernel_time_forward'] = format(kernel_time_forward, '.6f')
    if args.ckp:
        df.loc[(df['Testcase_Name'] == testcase_name) &
               (pd.to_numeric(df['Actual_out_pricision']) >= 0.95 * df['Expect_out_pricision']), 'Precison_result'] = 'Pass'
    df.to_excel(result_file, sheet_name=args.sheet, index=False)
    if result_queue != None:
        result_queue.put("Pass")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_id', type=str, default='all', help='case name')
    parser.add_argument('--device_id', type=int, default=11, help='device id')
    parser.add_argument('--case_file', type=str, default='SparseLightningIndexerKLLossTrain', help='case file')
    parser.add_argument('--version', type=str, default='910_92', help='device version')
    parser.add_argument('--run_mod', type=str, default='npu', help='device type: npu or cpu')
    parser.add_argument('--case_level', type=str, default='all', help='running specified level cases')
    parser.add_argument('--sheet', type=str, default='Sheet1', help='sheet name')
    parser.add_argument('--prof', type=int, default='0', help='enable prof')
    parser.add_argument('--esl', type=int, default='0', help='use esl model')
    parser.add_argument('--ckp', type=int, default='0', help='check precision')
    args = parser.parse_args()
    case_name = args.case_id
    device_id = args.device_id
    case_file = args.case_file
    version = args.version
    run_mod = args.run_mod
    case_level = args.case_level
    sheet = args.sheet
    if args.run_mod == 'npu':
        device = torch.device(f'npu:{args.device_id}')
    else:
        device = torch.device(f'cpu')
    torch_npu.npu.set_device(args.device_id)
    case_path = f"./case_xlsx/{case_file}.xlsx"
    if os.path.exists(case_path):
        data = openpyxl.load_workbook(case_path)
    else:
        print(f"ERROR: case file {case_path} does not exist.")
        exit()
    result_xlsx_name = case_file + f"_Result_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.xlsx"
    if os.path.exists(os.getcwd() + f"/result_xlsx"):
        os.system(f"mv SparseLightningIndexerLLossTrain*_Result_*.xlsx ./result_xlsx")
    else:
        os.mkdir(os.getcwd() + f"/result_xlsx")
        os.system(f"mv SparseLightningIndexerLLossTrain*.xlsx ./result_xlsx")
    os.system(f"cp {case_path} {result_xlsx_name}")
    table = data[sheet]
    nrows = table.max_row
    Testcase_Name_Col = get_col_index(table, 'Testcase_Name')
    Enable_Col = get_col_index(table, 'Enable')  # 'enable' & 'disable'
    case_level_Col = get_col_index(table, 'Level')
    B_Col = get_col_index(table, 'B')
    NQuery_Col = get_col_index(table, 'NQuery')
    NQueryIndex_Col = get_col_index(table, 'NQueryIndex')
    N2_Col = get_col_index(table, 'N2')
    S1_Col = get_col_index(table, 'S1')
    S2_Col = get_col_index(table, 'S2')
    seqlens_list_q_Col = get_col_index(table, 'seqlens_list_q')
    seqlens_list_kv_Col = get_col_index(table, 'seqlens_list_kv')
    D_Col = get_col_index(table, 'D')
    DIndex_Col = get_col_index(table, 'DIndex')
    DR_Col = get_col_index(table, 'DRope')
    Dtype_Col = get_col_index(table, 'Dtype')  # 'fp16' & 'bf16' & 'fp32'
    outDtype_Col = get_col_index(table, 'out_dtype')  # 0, 1
    Sparse_Col = get_col_index(table, 'sparse_mode')  # 0,1,2,3,4,5
    Input_layout_Col = get_col_index(table, 'Layout')  # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    pre_tockens_Col = get_col_index(table, 'pre_tokens')
    next_tockens_Col = get_col_index(table, 'next_tokens')
    # for i in range(2, nrows + 1):
    for i in range(2, 3):
        bValue = table.cell(i, B_Col).value
        if bValue is None:
            exit()
        case = {}
        case["testcase_name"] = table.cell(i, Testcase_Name_Col).value
        case["B"] = int(bValue)
        case["NQuery"] = int(table.cell(i, NQuery_Col).value)
        case["NQueryIndex"] = int(table.cell(i, NQueryIndex_Col).value)
        case["N2"] = 1 if table.cell(i, N2_Col).value is None else int(table.cell(i, N2_Col).value)
        case["S1"] = int(table.cell(i, S1_Col).value)
        case["S2"] = case["S1"] if table.cell(i, S2_Col).value is None else int(table.cell(i, S2_Col).value)
        case["seqlens_list_q"] = eval(table.cell(i, seqlens_list_q_Col).value) if table.cell(i, seqlens_list_q_Col).value is not None else None
        case["seqlens_list_kv"] = case["seqlens_list_q"] if table.cell(i, seqlens_list_kv_Col).value is None else eval(table.cell(i, seqlens_list_kv_Col).value)
        case["D"] = int(table.cell(i, D_Col).value)
        case["DIndex"] = int(table.cell(i, DIndex_Col).value)
        case["DR"] = 0 if table.cell(i, DR_Col).value is None else int(table.cell(i, DR_Col).value)
        case["dtype"] = table.cell(i, Dtype_Col).value
        case["out_dtype"] = 0 if table.cell(i, outDtype_Col).value is None else table.cell(i, outDtype_Col).value
        case["sparse_mode"] = int(table.cell(i, Sparse_Col).value)
        case["input_layout"] = table.cell(i, Input_layout_Col).value
        case["pre_tokens"] = int(table.cell(i, pre_tockens_Col).value) if table.cell(i, pre_tockens_Col).value is not None else 65536
        case["next_tokens"] = int(table.cell(i, next_tockens_Col).value) if table.cell(i, next_tockens_Col).value is not None else 65536
        case["case_level"] = str(table.cell(i, case_level_Col).value) if table.cell(i, case_level_Col).value is not None else "all"
        case["result_xlsx_name"] = result_xlsx_name

        enable = table.cell(i, Enable_Col).value
        # if (enable.lower() == 'enable' and case_name == 'all' and case_level == 'all') or case["testcase_name"] == case_name or (case['case_level'].lower() == case_level.lower() and enable.lower() == 'enable'):
        print_log(str(case))
        print_log(f"========================================Start run case {i - 1}/{nrows - 1},case_name:{case['testcase_name']}========================================")

        result = run(case, args, None)
            # result_queue = multiprocessing.Queue()
            # p = multiprocessing.Process(target=run, args=(case, args, result_queue))
            # p.start()
            # p.join(timeout=120000)
            # print_log('go out')
            # if p.is_alive():
            #     p.kill()
            #     print_log(f"error: {case} 执行超时，强制终止")
            #     pass_flag = False
            #     write_result(result_xlsx_name, args, case['testcase_name'], 'RTSYNC_TIMEOUT')
            # else:
            #     try:
            #         result = result_queue.get(timeout=1)
            #         if result == "Profiling_ERROR":
            #             write_result(result_xlsx_name, args, case['testcase_name'], 'Profiling_ERROR')
            #         else:
            #             write_result(result_xlsx_name, args, case['testcase_name'], 'PASS')
            #     except queue.Empty:
            #         result = "Failed"
            #         print_log(f"{case['testcase_name']} 异常退出，执行失败")
            #         write_result(result_xlsx_name, args, case['testcase_name'], 'AIC_ERROR')
            #     print(f"result: {result}")
            #     if result == "Failed":
            #         pass_flag = False

