if (pfa_param['k_antiquant_mode']) == 6 and (pfa_param['query_quant_mode']) == 6 and (pfa_param['v_antiquant_mode']) == 8:
            ## MXFP8 全量化golden  Q/K:per-channel-group V:per-token-group
            print("====================开始MXFP8 GQA golden计算====================")
            # k_tensor = k_tensor_list[0]
            # v_tensor = v_tensor_list[0]
            deq_scale_q = dequantScale_query
            deq_scale_k = k_antiquant_scale
            deq_scale_v = v_antiquant_scale

            qDtype = q_tensor.dtype
            print("out_dtype is", out_dtype)
            print('running flash_attention_mxfp8')
            # torch.set_printoptions(profile="full", precision=4)
            torch.set_printoptions(sci_mode=False)

            if numKeyValueHeads != numHeads:
                print("------------------GQA场景, 广播KV的N------------------")
                k_tensor = broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype)
                v_tensor = broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_tensor.dtype)
                deq_scale_k = broadcastKV_sigle(numHeads, numKeyValueHeads, deq_scale_k, deq_scale_k.dtype)
                deq_scale_v = broadcastKV_sigle(numHeads, numKeyValueHeads, deq_scale_v, deq_scale_v.dtype)
                print(f"[INFO] q_tensor shape {q_tensor.shape} and dtype {q_tensor.dtype}")
                print(f"[INFO] k_tensor shape {k_tensor.shape} and dtype {k_tensor.dtype}")
                print(f"[INFO] v_tensor shape {v_tensor.shape} and dtype {v_tensor.dtype}")
                print(f"[INFO] deq_scale_q shape {deq_scale_q.shape} and dtype {deq_scale_q.dtype}")
                print(f"[INFO] deq_scale_k shape {deq_scale_k.shape} and dtype {deq_scale_k.dtype}")
                print(f"[INFO] deq_scale_v shape {deq_scale_v.shape} and dtype {deq_scale_v.dtype}")
                if len(flagList) >= 27 and flagList[26]:
                    kr_bnsd_fp16 = broadcastKV_sigle(numHeads, numKeyValueHeads, kr_bnsd_fp16, kr_bnsd_fp16.dtype)
                    print(f"[INFO] qr_bnsd_fp16.shape {qr_bnsd_fp16.shape} and dtype {qr_bnsd_fp16.dtype}")
                    print(f"[INFO] kr_bnsd_fp16.shape {kr_bnsd_fp16.shape} and dtype {kr_bnsd_fp16.dtype}")

            EPSILON = 1e-20
            Q_BLOCK_SIZE = 128
            KV_BLOCK_SIZE = 512  # 计算的基本块,C1V1C1V1C2V2 
            QK_SCALE_SIZE = 32   # QK scale生成的单位 per-token-group
            V_SCALE_SIZE = 32   # V scale生成的单位 per-channel-group
            K_BLOCK_SIZE = int(KV_BLOCK_SIZE/2)
            V_BLOCK_SIZE = KV_BLOCK_SIZE

            if qDtype == torch.float8_e5m2 or qDtype == torch.float8_e4m3fn:
                q_tensor = q_tensor.to(torch.float32)
                k_tensor = k_tensor.to(torch.float32)
                v_tensor = v_tensor.to(torch.float32)
            else:
                q_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(q_tensor.numpy()))
                k_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(k_tensor.numpy()))
                v_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(v_tensor.numpy()))
            b, n, s, d = q_tensor.shape
            if flagList[26] and flagList[27]: # nope rope分离传入
                qr_tensor = qr_bnsd_fp16
                kr_tensor = kr_bnsd_fp16
                dr = qr_tensor.shape[-1]
                d = d + dr
            dv = v_tensor.shape[-1]
            out = torch.zeros([b, n, s, dv], dtype=torch.float32)  # out
            o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]  # sum
            o_max = torch.ones(q_tensor.shape[:-1])[..., None] * torch.finfo(torch.float).min  # max
            Sq = q_tensor.shape[2]
            Skv = k_tensor.shape[2]
            Tq = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
            Tkv = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

            if inputLayout == "TND":  
                q_lens_t = torch.tensor(actualSeqLengths, dtype=torch.int32)
                k_lens_t = torch.tensor(actualSeqLengthsKV, dtype=torch.int32)
                q_lens_acl = q_lens_t.view(b, 1, 1, 1)  # [B, 1, 1, 1]
                k_lens_acl = k_lens_t.view(b, 1, 1, 1)  # [B, 1, 1, 1]
            else : 
                q_seq_len = q_bnsd_shape_org[2]
                k_seq_len = k_bnsd_shape_org[2]
                q_lens_t = torch.full((b,), q_seq_len, dtype=torch.int32)
                k_lens_t = torch.full((b,), k_seq_len, dtype=torch.int32)
                if actualSeqLengths != 0 :
                    q_lens_t = torch.tensor(actualSeqLengths, dtype=torch.int32)
                if actualSeqLengthsKV != 0 :
                    k_lens_t = torch.tensor(actualSeqLengthsKV, dtype=torch.int32)
                q_lens_acl = q_lens_t.view(b, 1, 1, 1)
                k_lens_acl = k_lens_t.view(b, 1, 1, 1)
            # print(f"q_lens_acl{q_lens_acl}")
            # print(f"k_lens_acl{k_lens_acl}")
            q_range = torch.arange(Sq).view(1, 1, -1, 1)  # (Sq, 1)
            k_range = torch.arange(Skv).view(1, 1, 1, -1) # (1, Skv)
            q_padding_mask = q_range >= q_lens_acl
            k_padding_mask = k_range >= k_lens_acl
            # Causal 条件: k_idx > q_idx
            delta = k_lens_acl - q_lens_acl
            causal_mask = k_range > (q_range + delta)  # [1, 1, Sq, Skv]
            if pfa_param['sparse' ]== 3:
                # causal_mask = k_range > q_range  # (Sq, Skv)
                mask_global = causal_mask | q_padding_mask | k_padding_mask
            else:
                # Sparse Mode 0: 仅应用 Padding Mask
                mask_global = q_padding_mask | k_padding_mask
            # print(f"[INFO] causal_mask_j shape: {mask_global.shape} \n{mask_global[:,:,:,:]}")

            Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
            K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
            V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
            o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
            s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
            m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))
            if len(flagList) >= 27 and flagList[26]:
                Qr_BLOCKS = list(torch.split(qr_tensor, Q_BLOCK_SIZE, dim=2))
                Kr_BLOCKS = list(torch.split(kr_tensor, K_BLOCK_SIZE, dim=2))

            for i in range(Tq):
                Qi = Q_BLOCKS[i]    # Q分块
                Sq_start = i * Q_BLOCK_SIZE
                Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
                deq_scale_q_i = deq_scale_q[:, :, Sq_start:Sq_end, :]  # (B, H, Q_BLOCK_SIZE, HEAD_DIM/32)
                # 扩展量化因子维度以匹配Q的最后一维（HEAD_DIM）
                # 方法：将(HEAD_DIM/32)维度重复32次，扩展到HEAD_DIM
                deq_scale_q_i_expanded = deq_scale_q_i.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # (B, H, Q_BLOCK_SIZE, HEAD_DIM)
                ln_p_scale = torch.log(p_qscale_np)
                for j in range(0, Tkv, 2):
                    oi = o_BLOCKS[i]    # out
                    si = s_BLOCKS[i]    # sum
                    mi = m_BLOCKS[i]    # max
                        
                    # 1. 获取当前 K 块并计算 S_ij
                    Kj = K_BLOCKS[j]
                    Sk_start = j * K_BLOCK_SIZE
                    Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
                    # 截取当前K分块对应的量化因子
                    deq_scale_k_j = deq_scale_k[:, :, Sk_start:Sk_end, :]
                    deq_scale_k_j_expanded = deq_scale_k_j.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # 扩展到HEAD_DIM
                    # print("start matmul1")
                    # print(f"[INFO] Qi {Qi.shape} deq_scale_q_i_expanded {deq_scale_q_i_expanded.shape} Kj {Kj.shape} deq_scale_k_j_expanded {deq_scale_k_j_expanded.shape}")
                    S_ij = torch.matmul(Qi.to(torch.float32)* deq_scale_q_i_expanded, (Kj.to(torch.float32)*deq_scale_k_j_expanded).permute(0, 1, 3, 2))
                    if len(flagList) >= 27 and flagList[26]:
                        Qri = Qr_BLOCKS[i]    # Q分块
                        Krj = Kr_BLOCKS[j]
                        S_ij += torch.matmul(Qri.to(torch.float32), Krj.to(torch.float32).permute(0, 1, 3, 2))
                    # print("end matmul1")
                    # print(f"=========C1-1: {S_ij}")

                    S_ij = S_ij * 1 / math.sqrt(d)
                    # if pfa_param['sparse'] == 3:
                    causal_mask_j = mask_global[:,:,Sq_start:Sq_end, Sk_start:Sk_end]
                        # print(f"[INFO] causal_mask_j shape: {causal_mask_j.shape} \n{causal_mask_j[:,:,:,:]}")

                    S_ij = S_ij.masked_fill(causal_mask_j, float('-inf'))
                        # print(f"[INFO] S_ij shape: {S_ij.shape} \n{S_ij[:,:,:,:]}")

                    # softmax计算（局部 max，暂不更新全局状态）
                    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
                    m_block_j = m_block_j - ln_p_scale
                    m_block_j = torch.max(mi, m_block_j)
                    P_ij_raw = torch.exp(S_ij - m_block_j)
                    # torch.set_printoptions(profile="full", precision=4)
                    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
                    # print(f"[INFO] v1-1 max slice: \n{m_block_j}") # [11.3137]
                    # print(f"[INFO] v1-1 exp slice: \n{P_ij_raw}") # 1
                    # print(f"[INFO] v1-1 sum slice: \n{s_block_j}") # 256

                    if qDtype == "float8_e5m2" or qDtype == torch.float8_e5m2:
                        P_ij_drop = P_ij_raw.to(torch.float8_e5m2)    # softmax计算后cast成FP8
                        P_ij_drop = P_ij_raw.to(torch.float32)
                    elif qDtype == torch.uint8:
                        P_ij_drop_hifp8 = trans_np_float_tensor_to_hifuint8(P_ij_raw.numpy())
                        P_ij_drop_np32 = trans_np_hifuint8_tensor_to_float32(P_ij_drop_hifp8)
                        P_ij_drop = torch.from_numpy(P_ij_drop_np32)
                    else:
                        P_ij_drop = P_ij_raw.to(torch.float8_e4m3fn)
                        P_ij_drop = P_ij_drop.to(torch.float32)

                    if j + 1 < Tkv:
                        # --- C1V1 round 2: KV block j+1 ---
                        Kj1 = K_BLOCKS[j + 1]
                        # print("[INFO][SWL] ==============Kj1:", Kj1.shape, Kj1)
                        Sk1_start = (j + 1) * K_BLOCK_SIZE
                        Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                        # 截取当前K分块对应的量化因子
                        deq_scale_k_j1 = deq_scale_k[:, :, Sk1_start:Sk1_end, :]
                        deq_scale_k_j1_expanded = deq_scale_k_j1.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # 扩展到HEAD_DIM
                        # print("start matmul1")
                        S_ij = torch.matmul(Qi.to(torch.float32)* deq_scale_q_i_expanded, (Kj1.to(torch.float32)*deq_scale_k_j1_expanded).permute(0, 1, 3, 2))
                        if len(flagList) >= 27 and flagList[26]:
                            Qri = Qr_BLOCKS[i]    # Q分块
                            Krj1 = Kr_BLOCKS[j+1]
                            S_ij += torch.matmul(Qri.to(torch.float32), Krj1.to(torch.float32).permute(0, 1, 3, 2))
                        # print(f"=========C1-2: {S_ij}")
                        # print("end matmul1")
                        S_ij1 = S_ij * 1 / math.sqrt(d)
                        # if pfa_param['sparse'] == 3:
                        causal_mask_j1 = mask_global[:,:,Sq_start:Sq_end, Sk1_start:Sk1_end]
                            # print(f"[INFO] causal_mask_j1 shape: {causal_mask_j1.shape} \n{causal_mask_j1[:,:,:,:]}")
                            # tensor_to_txt(S_ij1, "./fp8_input/S_ij1_bf.txt")
                        S_ij1 = S_ij1.masked_fill(causal_mask_j1, float('-inf'))
                            # tensor_to_txt(S_ij1, "./fp8_input/S_ij1_af.txt")
                            # print(f"[INFO] S_ij1 shape: {S_ij1.shape} \n{S_ij1[:,:,:,:]}")

                        # softmax计算（局部 max）
                        m_block_j1, _ = torch.max(S_ij1, dim=-1, keepdims=True)
                        m_block_j1 = m_block_j1 - ln_p_scale
                        m_block_j1 = torch.max(m_block_j, m_block_j1)
                        P_ij1_raw = torch.exp(S_ij1 - m_block_j1) 
                        s_block_j1 = torch.sum(P_ij1_raw, dim=-1, keepdims=True)
                        update_mul_j = torch.exp(m_block_j - m_block_j1)
                        update_mul_si = torch.exp(mi - m_block_j1)
                        # print(f"[INFO] m_block_j shape: {m_block_j.shape}\n{m_block_j}")
                        # print(f"[INFO] m_block_j1 shape: {m_block_j1.shape}\n{m_block_j1}")
                        # print(f"[INFO] v1-2 max slice: \n{m_block_j1}") # [11.3137]
                        # print(f"[INFO] v1-2 exp slice: \n{P_ij1_raw}") # 1
                        # print(f"[INFO] v1-2 sum slice: \n{s_block_j1}") # 256
                        
                        if qDtype == "float8_e5m2" or qDtype == torch.float8_e5m2:
                            P_ij1_drop = P_ij1_raw.to(torch.float8_e5m2)    # softmax计算后cast成FP8
                            P_ij1_drop = P_ij1_raw.to(torch.float32)
                        elif qDtype == torch.uint8:
                            P_ij1_drop_hifp8 = trans_np_float_tensor_to_hifuint8(P_ij1_raw.numpy())
                            P_ij1_drop_np32 = trans_np_hifuint8_tensor_to_float32(P_ij1_drop_hifp8)
                            P_ij1_drop = torch.from_numpy(P_ij1_drop_np32)
                        else:
                            P_ij1_drop = P_ij1_raw.to(torch.float8_e4m3fn)
                            P_ij1_drop = P_ij1_drop.to(torch.float32)

                            # --- C2V2 combined: 两块合并一次更新 ---
                        # 反量化V然后C2 
                        Vj  = V_BLOCKS[int(j/2)]
                        # print("[INFO][SWL] ==============Vj: ", Vj.shape)
                        Sv_start = int(j/2)* V_BLOCK_SIZE
                        Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                        v_scale_start = Sv_start // V_SCALE_SIZE
                        v_scale_end = (Sv_end - 1) // V_SCALE_SIZE + 1
                        deq_scale_v_j = deq_scale_v[:, :, v_scale_start:v_scale_end, :]
                        deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(V_SCALE_SIZE, dim=2)  # 扩展到V_BLOCK_SIZE
                        deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Vj.shape[2], :]
                        Vj_dequant = Vj.to(torch.float32) * deq_scale_v_j_expanded

                        # print(f"[INFO] C2 Vj \n{Vj}")
                        # print(f"[INFO] C2 deq_scale_v_j_expanded \n{deq_scale_v_j_expanded}")
                        # print(f"[INFO] C2 Vj_dequant \n{Vj_dequant}")

                        split_k = True
                        if not split_k:
                            P_ij_drop_concact = torch.cat((P_ij_drop, P_ij1_drop), dim=-1)                            
                            P_ij_Vj   = torch.matmul(P_ij_drop_concact,  Vj_dequant)
                        else: # 切k矩阵乘，方便定位
                            V_part1 = Vj_dequant[:, :, :Kj.shape[2], :]
                            V_part2 = Vj_dequant[:, :, Kj.shape[2] : Kj.shape[2] + Kj1.shape[2], :]
                            # print(f"[INFO] V_part1[:,:,256:,:] shape: {V_part1.shape} \n{V_part1}")
                            # print(f"[INFO] V_part2[:,:,256:,:] shape: {V_part2.shape} \n{V_part2}")
                            P_V_part1 = torch.matmul(P_ij_drop, V_part1)
                            # print(f"[INFO] P_V_part1 shape: {P_V_part1.shape} \n{P_V_part1[:,:,:,:]}")
                            P_V_part2 = torch.matmul(P_ij1_drop, V_part2)
                            # print(f"[INFO] P_V_part2 shape: {P_V_part2.shape} \n{P_V_part2[:,:,:,:]}")
                            P_ij_Vj = P_V_part1 + P_V_part2
                        
                        # print(f"P_ij_Vj shape: {P_ij_Vj.shape} \n{P_ij_Vj}")

                        #  V2
                        si_new = update_mul_si * si + s_block_j * update_mul_j + s_block_j1
                        # print(f"[INFO] update_mul_si: \n{update_mul_si}") # 0
                        # print(f"[INFO] si: \n{si}") # 0
                        # print(f"[INFO] s_block_j: \n{s_block_j}") # 256
                        # print(f"[INFO] update_mul_j: \n{update_mul_j}") # 1
                        # print(f"[INFO] s_block_j1: \n{s_block_j1}") # 256
                        # print(f"[INFO] row sum slice: \n{si_new}") # 512
                        o_BLOCKS[i] = (update_mul_si * oi * si + P_ij_Vj) / (si_new+ EPSILON)
                        s_BLOCKS[i] = si_new
                        m_BLOCKS[i] = m_block_j1

                    else:
                        # Tkv 为奇数时最后一块，正常处理

                        Vj = V_BLOCKS[int(j/2)]
                        Sv_start = j* K_BLOCK_SIZE
                        Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)

                        v_scale_start = Sv_start // V_SCALE_SIZE
                        v_scale_end = (Sv_end - 1) // V_SCALE_SIZE + 1
                        # print(f"[INFO] v_scale_start {v_scale_start} v_scale_end {v_scale_end} Sv_start {Sv_start} Sv_end {Sv_end}")

                        deq_scale_v_j = deq_scale_v[:, :, v_scale_start:v_scale_end, :]
                        deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(V_SCALE_SIZE, dim=2)  # 扩展到V_BLOCK_SIZE
                        deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Kj.shape[2], :]
                        Vj_expanded = Vj[:, :, :Kj.shape[2], :]
                        Vj_dequant = Vj_expanded.to(torch.float32) * deq_scale_v_j_expanded
                        # print("start matmul2")
                        P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant)
                        # print("end matmul2")
                        # 对bmm2结果反量化
                        # torch.set_printoptions(sci_mode=False)
                        update_mul_si = torch.exp(mi - m_block_j)
                        si_new = update_mul_si * si + s_block_j
                        o_BLOCKS[i] = (update_mul_si * oi * si + P_ij_Vj) / (si_new+ EPSILON)
                        s_BLOCKS[i] = si_new
                        m_BLOCKS[i] = m_block_j
            out = torch.cat(o_BLOCKS, dim=2)
            o_sum = torch.cat(s_BLOCKS, dim=2)
            o_max = torch.cat(m_BLOCKS, dim=2)
            y_all = out.numpy()
            # print(f"y_all slice : \n{y_all}")