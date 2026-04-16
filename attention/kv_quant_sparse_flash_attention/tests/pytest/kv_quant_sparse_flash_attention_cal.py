import time
import random
import numpy as np
import math
import torch

def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def qtensor_dim(q_shape, layoutQuery, numhead):
    if layoutQuery == "SH":  # SH格式
        return q_shape[0]
    elif layoutQuery == "BSH" or layoutQuery == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new


def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else:  # BNSD
        return q_shape[2]


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


def trans_input_dtype(input_dtype):
    if input_dtype == 'fp16':
        return 'float16'
    elif input_dtype == 'int8':
        return 'int8'
    elif input_dtype == 'uint8':
        return 'uint8'
    elif input_dtype == 'bf16':
        return 'bfloat16'
    elif input_dtype == 'bool':
        return 'bool'
    elif input_dtype == 'int32':
        return 'int32'
    elif input_dtype == 'fp32':
        return 'float32'
    elif input_dtype == 'int4':
        return 'int4'
    else:
        return input_dtype


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = torch.cat((tensor1, tensor2), dim=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


# 融合表格参数获取函数
def get_param_fus(torch_tensor_list, params):
    fa_param = {}
    fa_param['normal_flag'] = True
    # ===参数获取===
    # >> attr info
    fa_param['actualseqlengths'] = params['actualseqlengths']
    fa_param['actualseqlengthskv'] = params['actualseqlengthskv']
    fa_param['scaleValue'] = params['scaleValue']
    fa_param['layout_query'] = params['layout_query']
    fa_param['layout_kv'] = params['layout_kv']
    fa_param['sparsemode'] = params['sparsemode']
    fa_param['sparse_blocksize'] = params['sparse_blocksize']

    # >> q info
    fa_param['q_dtype'] = torch.bfloat16 if trans_input_dtype(params['dtype_input'][0]) == "bfloat16" else torch.float16
    fa_param['q_tensor'] = torch_tensor_list[0]

    # >> k info
    fa_param['k_dtype'] = trans_input_dtype(params['dtype_input'][1])
    fa_param['k_tensor'] = torch_tensor_list[1]

    # >> v info  MLA场景 KV要保持一致
    fa_param['v_dtype'] = trans_input_dtype(params['dtype_input'][2])
    fa_param['v_tensor'] = torch_tensor_list[2]

    # >> sparse_indices info
    fa_param['sparse_indices_dtype'] = trans_input_dtype(params['dtype_input'][3])
    fa_param['sparse_indices_tensor'] = torch_tensor_list[3]

    # >> block_table info
    fa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][4])
    fa_param['block_table'] = torch_tensor_list[4]

    # >> k cache info
    fa_param['k_cache_shape'] = params['shape_input'][5]
    fa_param['k_cache_dtype'] = trans_input_dtype(params['dtype_input'][5])

    # >> v cache info
    fa_param['v_cache_shape'] = params['shape_input'][6]
    fa_param['v_cache_dtype'] = trans_input_dtype(params['dtype_input'][6])

    # >> out info
    fa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])
    # ===flag_list判断===
    flag_list = str_to_bool_list(params['flaglist'])
    fa_param['pa_flag'] = True if fa_param['layout_kv'] in ["PA_BSND"] else False
    fa_param['tnd_flag'] = True if fa_param['layout_query'] in ["TND"] else False
    fa_param['kv_tnd_flag'] = True if fa_param['layout_kv'] in ["TND"] else False
    fa_param['return_softmax'] = params['return_softmax']
    if fa_param['return_softmax']==True:
        fa_param['softmax_max_shape'] = params['shape_output'][1]
        fa_param['softmax_max_dtype'] = trans_input_dtype(params['dtype_output'][1])
        fa_param['softmax_sum_shape'] = params['shape_output'][2]
        fa_param['softmax_sum_dtype'] = trans_input_dtype(params['dtype_output'][2])

    # >> batch_size info
    q_shape = list(fa_param['q_tensor'].shape)
    if fa_param['layout_query'] in ["TND"]:
        fa_param['b'] = len(fa_param['actualseqlengths'])
        fa_param['numHeads'] = q_shape[1]
    else:
        fa_param['b'] = q_shape[0]
        fa_param['numHeads'] = q_shape[2]

    fa_param['sparse_blockcount'] = fa_param['sparse_indices_tensor'].shape[-1]

    fa_param['blocksize'] = 1
    fa_param['numKeyValueHeads'] = fa_param['k_tensor'].shape[-2]
    if fa_param['layout_kv'] in ["PA_BSND"]:
        fa_param['blocksize'] = fa_param['k_cache_shape'][1]

    fa_param['rope_head_dim'] = params['rope_head_dim']

    # ===异常判断===
    if not flag_list[0] or not flag_list[1] or not flag_list[2]:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q/k/v/不输入场景！')
    if 0 in fa_param['q_tensor'].shape:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q为空场景！')
    if len(fa_param['k_tensor'].shape) == 0 or 0 in fa_param['k_tensor'].shape:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  k为空场景！')
    if len(fa_param['v_tensor'].shape) == 0 or 0 in fa_param['v_tensor'].shape:
        fa_param['normal_flag'] = False
        print('[WARNING]:异常-->  v为空场景！')

    return fa_param


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None, tensor_name=None):
    if layout in ["BSND", "PA_BSND"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(1, 2)
        return tensor, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s

            # print(f"[TEMP]b_index:{b_index} act_s:{act_s}, t_start:{t_start}, t_end:{t_end}")
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        # print(f"[TEMP]new_tensor:{new_tensor}")
        return new_tensor, [B, N, S, D]
    else:
        return tensor, shape


def trans_bnsd_to_input_layout(tensor, shape, layout, act_q=None):
    if layout in ["BSH", "BSH_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "BSND_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        return output
    else:
        return tensor


def gatherKV(k_tensor, v_tensor, sparse_indices_Indices, sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,curr_actualSeq, curr_actualSeq_q, sparse_mode):
    s2_sparse = list()
    if sparse_mode == 0:
        threshold = curr_actualSeq
    elif sparse_mode == 3:
        delta_s = curr_actualSeq - curr_actualSeq_q
        threshold = delta_s + s1Idx + 1
    validCount = min(sparse_blockcount, math.ceil(threshold / sparse_blocksize))
    for i in range(validCount):
        sparseIndicesId = sparse_indices_Indices[i]
        if sparseIndicesId == -1:
            break
        begin_Idx = sparseIndicesId * sparse_blocksize
        end_Idx = begin_Idx + sparse_blocksize if begin_Idx + sparse_blocksize <= curr_actualSeq else curr_actualSeq
        if begin_Idx >= threshold:
            continue
        if end_Idx <= threshold:
            s2_sparse.extend(np.arange(begin_Idx, end_Idx))
        else:
            s2_sparse.extend(np.arange(begin_Idx, threshold))

    emptyFlag = False
    if len(s2_sparse) == 0:
        k_sparse, v_sparse = [], []
        emptyFlag = True
    else:
        k_sparse, v_sparse = k_tensor[batch, n2Idx, s2_sparse, :], v_tensor[batch, n2Idx, s2_sparse, :]

    return emptyFlag, k_sparse, v_sparse, torch.tensor(s2_sparse)


def softmax(x):
    x = x.to(torch.float32)
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    
    return ans, x_max, x_sum


def _t_increattention_bnsd(fa_param):
    batch_size = fa_param['b']
    numheads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    scaleValue = fa_param['scaleValue']
    sparse_blocksize = fa_param['sparse_blocksize']
    sparse_blockcount = fa_param['sparse_blockcount']
    sparseIndicesIndices = fa_param['sparse_indices_bnsd_tensor']
    out_shape_bnsd = list(fa_param['q_bnsd_shape'])
    out_shape_bnsd[-1] = out_shape_bnsd[-1] - fa_param['rope_head_dim']
    sparse_mode = fa_param['sparsemode']
    g = numheads // numKeyValueHeads

    q_bnsd_tensor = fa_param['q_bnsd_tensor']
    k_bnsd_tensor = fa_param['k_bnsd_tensor']
    v_bnsd_tensor = fa_param['v_bnsd_tensor']
    y = np.zeros(out_shape_bnsd, dtype=np.float32)
    softmax_max = None
    softmax_sum = None
    if fa_param['return_softmax'] == True:
        softmax_max = np.zeros(fa_param['softmax_max_shape'], dtype=np.float32)
        softmax_sum = np.zeros(fa_param['softmax_sum_shape'], dtype=np.float32)

    for batch in range(batch_size):
        curr_actualSeq_q = actualSeqLengths_q[batch]
        curr_actualSeq = actualSeqLengths_kv[batch]
        qLastFrefill = 0 if batch == 0 else sum(actualSeqLengths_q[0: batch])

        for n2Idx in range(numKeyValueHeads):
            for s1Idx in range(curr_actualSeq_q):
                if s1Idx < curr_actualSeq_q - curr_actualSeq and sparse_mode != 0:
                    y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = np.zeros([g, out_shape_bnsd[-1]], dtype=np.float32)
                    continue
                q_curr = q_bnsd_tensor[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :]
                sparse_indices_Indices = sparseIndicesIndices[batch, n2Idx, s1Idx, :]
                emptyFlag, k_sparse, v_sparse, s2_index = gatherKV(k_bnsd_tensor, v_bnsd_tensor, sparse_indices_Indices,
                                                         sparse_blocksize, sparse_blockcount, batch, n2Idx, s1Idx,
                                                         curr_actualSeq, curr_actualSeq_q, sparse_mode)
                if emptyFlag:
                    continue
                bmm1Res = torch.matmul(q_curr, k_sparse.T)
                scaleRes = bmm1Res * scaleValue
                if sparse_mode == 3:
                    threshold = curr_actualSeq - curr_actualSeq_q + s1Idx + 1
                    mask_index = s2_index >= threshold
                    scaleRes[:, mask_index] = -1e12
                softmax_res, x_max, x_sum = softmax(scaleRes)
                if fa_param['q_dtype'] in (torch.float16, torch.bfloat16):
                    bmm2Res = torch.matmul(softmax_res.to(fa_param['q_dtype']).to(torch.float32), v_sparse.to(torch.float32))
                else:
                    bmm2Res = torch.matmul(softmax_res, v_sparse)
                y[batch, n2Idx * g: (n2Idx + 1) * g, s1Idx, :] = bmm2Res
                if fa_param['return_softmax'] :
                    if fa_param['layout_query'] == 'BSND':
                        softmax_max[batch, n2Idx, s1Idx, :] = x_max[:, 0]
                        softmax_sum[batch, n2Idx, s1Idx, :] = x_sum[:, 0]
                    else:

                        softmax_max[n2Idx, qLastFrefill + s1Idx, :] = x_max[:, 0]
                        softmax_sum[n2Idx, qLastFrefill + s1Idx, :] = x_sum[:, 0]

    return y, softmax_max, softmax_sum


def get_attention_mask_batch_num(npu_m_shape, q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "PA_BSND"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
        output = tensor.permute(1, 0, 2, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D))
        # output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s

            if act_s == 0:
                continue
            for n_index in range(N):
                slice_data = tensor[b_index, n_index, :act_s, :]
                if isinstance(slice_data, np.ndarray):
                    output[t_start:t_end, n_index, :] = torch.from_numpy(slice_data)
                else:
                    output[t_start:t_end, n_index, :] = slice_data
            t_start += act_s
        if layout == "TND_NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output
    else:
        return tensor

def deepseek_ds_pa_preprocessing(fa_param, params):
    k_np_dtype = fa_param['k_np_dtype']
    v_np_dtype = fa_param['v_np_dtype']
    k_ori_shape_list = fa_param['k_shape_old']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    numHeads = fa_param['numHeads']
    kv_layout = fa_param['layout_kv']
    actualSeqLengths_kv = fa_param['actualSeqLengths_kv']
    actualSeqLengths_q = fa_param['actualSeqLengths_q']
    # k_shape_bnsd_list = fa_param['k_shape_bnsd_list']
    blockSize = fa_param['blocksize']
    blockTableShape = list(fa_param['block_table'].shape)
    block_table = fa_param['block_table']
    layoutQuery = fa_param['layout_query']
    layout_kv = fa_param['layout_kv']
    k_cache = np.zeros(fa_param['k_cache_shape']).astype(k_np_dtype)
    v_cache = np.zeros(fa_param['v_cache_shape']).astype(v_np_dtype)
    k_rope_cache = np.zeros(fa_param['k_rope_cache_shape']).astype(k_np_dtype)
    k_dtype = fa_param['k_dtype']

    B = len(fa_param['actualSeqLengths_kv'])

    if len(fa_param['k_cache_shape']) != len(fa_param['k_rope_cache_shape']):
        print('[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！')
        # return torch.zeros(out_shape)
    if layout_kv in ["PA_BSND"]:  # len(fa_param['k_cache_shape']) == 3
        # import pdb; pdb.set_trace()
        k_tensor_bsnd = fa_param['k_tensor_old']
        if fa_param['k_tensor_old'].shape[1] != blockTableShape[1] * blockSize:
            k_tensor_old = fa_param['k_tensor_old']
            k_tensor_bsnd = np.zeros(
                (k_tensor_old.shape[0], blockTableShape[1] * blockSize, k_tensor_old.shape[2], k_tensor_old.shape[3]),
                dtype=k_np_dtype
            )
            k_tensor_bsnd[:, :k_tensor_old.shape[1], :, :] = k_tensor_old
        kvs = k_tensor_bsnd.shape[1]
        kvn = k_tensor_bsnd.shape[2]
        kvd = k_tensor_bsnd.shape[3]
        print(f"[INFO]k_tensor_bsnd: {k_tensor_bsnd.shape}")
        # k_tensor_bbnd[:, :kvs, :, :] = k_tensor_bsnd[:, :, :, :]

        # import pdb; pdb.set_trace()
        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(fa_param['k_cache_shape'][2]):
                        # print(f"b:{b}, block_i:{block_i}, kv_cache_blk_id:{kv_cache_blk_id}, blockSize:{blockSize}, block_offset:{block_offset}")
                        # print(f"shape0:{k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :].shape}, shape1:{k_tensor_bsnd[b,block_offset:(block_offset + blockSize), n:n + 1,:].shape}")
                        # import pdb; pdb.set_trace()
                        k_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_tensor_bsnd[b,
                                                                            block_offset:(
                                                                                    block_offset + blockSize), n:n + 1,
                                                                            :]

        k_rope_bsnd = fa_param['k_rope_tensor']
        if fa_param['k_rope_tensor'].shape[1] != blockTableShape[1] * blockSize:
            k_rope_tensor_old = fa_param['k_rope_tensor']
            k_rope_bsnd = np.zeros(
                (k_rope_tensor_old.shape[0], blockTableShape[1] * blockSize, k_rope_tensor_old.shape[2],
                 k_rope_tensor_old.shape[3]),
                dtype=k_np_dtype
            )
            k_rope_bsnd[:, :k_rope_tensor_old.shape[1], :, :] = k_rope_tensor_old
        kropes = k_rope_bsnd.shape[1]
        kropen = k_rope_bsnd.shape[2]
        kroped = k_rope_bsnd.shape[3]
        print(f"[INFO]k_rope_bsnd: {k_rope_bsnd.shape}")
        # k_rope_tensor_bbnd[:, :, :kropes, :] = k_rope_bsnd_tensor[:, :, :, :]

        # import pdb; pdb.set_trace()
        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(fa_param['k_rope_cache_shape'][2]):
                        k_rope_cache[kv_cache_blk_id, 0:blockSize, n:n + 1, :] = k_rope_bsnd[b,
                                                                                 block_offset:(
                                                                                         block_offset + blockSize),
                                                                                 n:n + 1,
                                                                                 :]
    else:
        print(f"[ERROR]Wrong input layout_kv: {fa_param['layout_kv']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 5
    v_cache_index = 6
    k_rope_cache_index = 9
    tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][1])
    tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][2])
    tools.modify_alcnn_input_file(ids=k_rope_cache_index, origin_index=[k_rope_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_rope_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][2])


def aclnn_op_func_cpu(torch_tensor_list, params):
    print("cpu执行中...")
    fa_param = get_param_fus(torch_tensor_list, params)
    if not fa_param['normal_flag']:
        return torch.zeros(fa_param['q_tensor'].shape)
    # ==========预处理=============
    # >> 1.参数预处理
    actualSeqLengths_q = fa_param['actualseqlengths']
    actualSeqLengths_kv = fa_param['actualseqlengthskv']
    layoutQuery = fa_param['layout_query']
    numHeads = fa_param['numHeads']
    numKeyValueHeads = fa_param['numKeyValueHeads']
    out_shape = list(fa_param['q_tensor'].shape)
    k_dtype = fa_param['k_dtype']
    v_dtype = fa_param['v_dtype']
    sparse_mode = fa_param['sparsemode']
    # fa_param['kv_layout'] = 'BNSD' if fa_param['tnd_flag'] and fa_param['pa_flag'] else layoutQuery
    # >> 3.将q、k、v、topkIndices转换为bnsd(TND时只有Q和TopK写成TND格式，其余均写成BSND格式)
    q_shape = list(fa_param['q_tensor'].shape)
    k_shape = list(fa_param['k_tensor'].shape)
    v_shape = list(fa_param['v_tensor'].shape)
    sparse_indices_shape = list(fa_param['sparse_indices_tensor'].shape)
    if fa_param['tnd_flag']:
        fa_param['actualSeqLengths_q'] = trans_tnd_actseq(actualSeqLengths_q)
        if fa_param['pa_flag']:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['q_tensor'], q_shape, layoutQuery,
                                                             numHeads, fa_param['actualSeqLengths_q'])
    else:
        q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['q_tensor'], q_shape, layoutQuery,
                                                             numHeads, actualSeqLengths_q)
    if fa_param['kv_tnd_flag']:
        fa_param['actualSeqLengths_kv'] = trans_tnd_actseq(actualSeqLengths_kv)
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['k_tensor'], k_shape,
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, fa_param['actualSeqLengths_kv'])
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['v_tensor'], v_shape,
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, fa_param['actualSeqLengths_kv'])
    else:
        k_bnsd_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['k_tensor'], k_shape,
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
        v_bnsd_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(fa_param['v_tensor'], v_shape,
                                                             fa_param['layout_kv'],
                                                             numKeyValueHeads, actualSeqLengths_kv)
    if fa_param['tnd_flag']:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['sparse_indices_tensor'], sparse_indices_shape, layoutQuery,
            numKeyValueHeads, fa_param['actualSeqLengths_q'], 'sparseIndices')
    else:
        sparse_indices_bnsd_tensor, sparse_indices_bnsd_shape = _n_trans_shape_to_bnsd(
            fa_param['sparse_indices_tensor'], sparse_indices_shape, layoutQuery,
            numKeyValueHeads, actualSeqLengths_q, 'sparseIndices')

    fa_param['q_bnsd_tensor'] = q_bnsd_tensor
    fa_param['q_bnsd_shape'] = q_bnsd_shape
    fa_param['k_bnsd_tensor'] = k_bnsd_tensor
    fa_param['k_bnsd_shape'] = k_bnsd_shape
    fa_param['v_bnsd_tensor'] = v_bnsd_tensor
    fa_param['v_bnsd_shape'] = v_bnsd_shape
    fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_bnsd_tensor
    fa_param['sparse_indices_bnsd_shape'] = sparse_indices_bnsd_shape
    fa_param['return_softmax'] = params.get('return_softmax',False)
    print(f"sparse_indices_bnsd_shape, {sparse_indices_bnsd_shape}")
    fa_param['ks_max'] = 0
    # >> 4.act seq处理
    if not fa_param['tnd_flag']:
        if len(actualSeqLengths_q) == 0:
            fa_param['actualSeqLengths_q'] = [q_bnsd_shape[2]] * q_bnsd_shape[0]
        elif len(actualSeqLengths_q) == 1:
            fa_param['actualSeqLengths_q'] = [actualSeqLengths_q[0]] * q_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_q'] = actualSeqLengths_q
    if not fa_param['kv_tnd_flag']:
        if len(actualSeqLengths_kv) == 0:
            fa_param['actualSeqLengths_kv'] = [k_bnsd_shape[2]] * k_bnsd_shape[0]
        elif len(actualSeqLengths_kv) == 1:
            fa_param['actualSeqLengths_kv'] = [actualSeqLengths_kv[0]] * k_bnsd_shape[0]
        else:
            fa_param['actualSeqLengths_kv'] = actualSeqLengths_kv

    # >> 5.生成sparseIndices
    if params["action_type"] != "bm_output_gold":
        seed = int(time.time())
        # random.seed(seed)
        sparse_indices_bnsd_shape = fa_param['sparse_indices_bnsd_shape']
        sparse_indices_tensor = fa_param['sparse_indices_bnsd_tensor']

        for b in range(sparse_indices_bnsd_shape[0]):
            act_seqlen_kv = fa_param['actualSeqLengths_kv'][b]
            act_seqlen_q = fa_param['actualSeqLengths_q'][b]

            for n in range(sparse_indices_bnsd_shape[1]):
                for s in range(sparse_indices_bnsd_shape[2]):
                    if fa_param['sparsemode'] == 0:
                        threshold = act_seqlen_kv
                    elif fa_param['sparsemode'] == 3:
                        threshold = act_seqlen_kv - act_seqlen_q + s + 1
                    if threshold <= 0:
                        sparse_indices_tensor[b, n, s, :] = [-1] * (fa_param['sparse_blockcount'])
                        continue
                    valid_blocks_max = math.ceil(max(0, threshold) / fa_param['sparse_blocksize'])
                    dtype = getattr(np, fa_param['sparse_indices_dtype'])
                    block_indices = torch.randperm(valid_blocks_max - 1).numpy().astype(dtype)
                    valid_blocks_topk = min(valid_blocks_max, fa_param['sparse_blockcount'])
                    sparse_indices_tensor[b, n, s, :valid_blocks_topk - 1] = block_indices[0: valid_blocks_topk - 1]
                    sparse_indices_tensor[b, n, s, valid_blocks_topk - 1:valid_blocks_topk] = valid_blocks_max - 1
                    sparse_indices_tensor[b, n, s, valid_blocks_topk:] = [-1] * (
                                fa_param['sparse_blockcount'] - valid_blocks_topk)

        # 打印信息：kv首地址，topk值
        if fa_param['tnd_flag']:
            sparse_indices_tensor_tnd = trans_bnsd_to_layout(sparse_indices_tensor, out_shape, layoutQuery,
                                                             fa_param['actualSeqLengths_q'])
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor',
                                          mode='rewrite',
                                          tensors=torch.tensor(sparse_indices_tensor_tnd, dtype=torch.int32),
                                          params=params,
                                          data_dtype=params['dtype_input'][3])

        else:
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor',
                                          mode='rewrite',
                                          tensors=sparse_indices_tensor.transpose(0, 2, 1, 3),
                                          params=params,
                                          data_dtype=params['dtype_input'][3])

        fa_param['sparse_indices_bnsd_tensor'] = sparse_indices_tensor
        if fa_param['pa_flag']:
            # 生成blocktable
            blockSize = fa_param['blocksize']
            blockTableShape = params['shape_input'][4]
            blockNum = fa_param['k_cache_shape'][0]
            if 0 in blockTableShape:
                print('[WARNING]:block_table为空场景，输出空tensor！')
                return torch.zeros(out_shape)
            blockNumPerBlock = []
            block_num_min = 0
            for actual_seq in actualSeqLengths_kv:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
                block_num_min += math.ceil(actual_seq / blockSize)
            if block_num_min > blockNum:
                print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
                exit(1)
            block_idx_list = np.arange(0, blockNum, 1)
            block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
            block_idx = 0
            block_table = [-1] * blockTableShape[1]
            block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
            block_table_batch_idx = 0
            for idx in blockNumPerBlock:
                for j in range(idx):
                    block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                    block_idx += 1
                block_table_batch_idx += 1
            fa_param['block_table'] = block_table
            # 将block_table 覆盖原有的input
            tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite',
                                          tensors=torch.tensor(block_table, dtype=torch.int32),
                                          params=params)
            k_np_dtype = tools.get_np_dtype(params['dtype_input'][1])
            v_np_dtype = tools.get_np_dtype(params['dtype_input'][2])
            fa_param['k_np_dtype'] = k_np_dtype
            fa_param['v_np_dtype'] = v_np_dtype
            print(f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]}")
            k_cache = np.zeros(fa_param['k_cache_shape']).astype(k_np_dtype)
            v_cache = np.zeros(fa_param['v_cache_shape']).astype(v_np_dtype)
            print('[INFO]:PA + Deepseek！')
            deepseek_ds_pa_preprocessing(fa_param, params)

    # ===计算入口===
    y_all, softmax_max, softmax_sum = _t_increattention_bnsd(fa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, layoutQuery, fa_param['actualSeqLengths_q'])
    if fa_param['return_softmax']:
        softmax_max = torch.from_numpy(softmax_max)
        softmax_sum = torch.from_numpy(softmax_sum)
        return [y_all, softmax_max, softmax_sum]
    else:
        return y_all


def get_operator(input_tensor_list, params, device, device_id):
    if device == "npu":
        torch.npu.set_device(f"npu:{device_id}")
    torch_tensor_list = list()
    for i in range(len(input_tensor_list)):
        input_tensor = input_tensor_list[i]

        if any("Tensor" in _key for _key in params.keys()):
            if device == "cpu" and input_tensor.dtype == torch.float16:
                input_tensor = input_tensor.float()
            torch_tensor_list.append(input_tensor.to(device))
        else:
            if device == "cpu":
                if input_tensor.dtype == np.float16 or str(input_tensor.dtype) == "bfloat16":
                    input_tensor = input_tensor.astype(np.float32)
                torch_tensor_list.append(input_tensor)
            if device in ["npu", "cuda"]:
                if str(input_tensor.dtype) == "uint64":
                    print(type(input_tensor), "torch不支持uint64！")
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.int64)).to(torch.int64)
                    torch_tensor_list.append(input_tensor_i.to(device))

                elif str(input_tensor.dtype) == "bfloat16":
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
                    torch_tensor_list.append(input_tensor_i.to(device))
                else:
                    input_tensor_i = torch.from_numpy(input_tensor)
                    torch_tensor_list.append(input_tensor_i.to(device))

    if device == "npu":
        output_data = training.aclnn_execute_npu(torch_tensor_list, params)
    elif device == "npu_off":
        training.aclnn_execute_npu_off(torch_tensor_list, params)
        return
    else:
        output_data = training.aclnn_execute_cpu(torch_tensor_list, params)

    if isinstance(output_data, list) or isinstance(output_data, tuple):
        output_value = []
        get_output_list(True, output_data, output_value)
        output_list = []
        for i in range(len(output_value)):
            tensor_tmp = output_value[i].to("cpu")
            if params["dtype_output"][i] == "bf16":
                tensor_tmp = tensor_tmp.float()
            tensor_tmp = tensor_tmp.detach().numpy()
            if device == "cpu" and tools.get_np_dtype(params["dtype_output"][0]) == np.float16:
                tensor_tmp = tensor_tmp.astype(np.float16)
            output_list.append(tensor_tmp)
        return output_list
    elif isinstance(output_data, bool):
        output_data = output_data
    else:
        output_data = output_data.to("cpu")
        if params["dtype_output"][0] == "bf16":
            output_data = output_data.float()
        output_data = output_data.detach().numpy()
        if device == "cpu" and params["dtype_output"][0] and tools.get_np_dtype(
                params["dtype_output"][0]) == np.float16:
            output_data = output_data.astype(np.float16)

    return output_data


# @set_timeout(120)
def aclnnSparseFlashAttention(action_type, params, case_path, op_name, soc_version, framework):
    case_name = params["case_name"]
    params["op_name"] = op_name
    params["case_path"] = case_path
    result, error_percent, max_error = "Pass", 100.0, 1.0
    if action_type in ("bm", "one"):
        tools.print_log("-----------------------Start to generate cpu pytorch atenIR golden--------")
        result, error_percent, max_error = training.run_pytorch_training_cpu(params)
    if action_type in ("bm_input",):
        tools.print_log("-----------------------Start to run bm_input-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_input(params)
    if action_type in ("bm_output",):
        tools.print_log("-----------------------Start to run bm_output-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output(params)
    if action_type in ("bm_gold",):
        tools.print_log("-----------------------Start to run bm_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_gold(params)
    if action_type in ("bm_output_gold",):
        tools.print_log("-----------------------Start to run bm_output_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output_gold(params)
    if action_type in ("npu", "one") and result == "Pass":
        tools.print_log("-----------------------Start to run aclnn online--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_online(params)
    if action_type in ("npu_off",) and result == "Pass":
        tools.print_log("-----------------------Start to run aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_multi_input(params)
    if action_type in ("npu_off_all",) and result == "Pass":
        if params['index'] == 0:
            tools.print_log("-----------------------Start to run aclnn offline all--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_all_multi_input(params)
    if action_type in ("compare",) and result == "Pass":
        tools.print_log("-----------------------Start to compare aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_compare_multi_input(params)
    return result, error_percent, max_error
