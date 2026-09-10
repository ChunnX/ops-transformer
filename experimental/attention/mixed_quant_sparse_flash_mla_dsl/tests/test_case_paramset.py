# Query 长度示例（外层列表为候选值）：B=[3], cu_seqlens_q=[[0, 1, 4, 6]],
# seqused_q=[[1, 3, 2]]；可只传其中一项，两者都不传才使用 S1。
# 所有用例通过已构建安装的wheel调用torch.ops。
# kv_axis0_noncontiguous统一控制两侧，框架固定padding，默认False。
# topk_length由测试框架按各侧topk模式自动生成。

TEST_PARAMS = {'ori_cmp_sparse_8k_decode_b1': {'Testcase_Name': [None],
                                 'template_run_mode': ['ORI_CMP_SPARSE'],
                                 'B': [1],
                                 'S1': [1],
                                 'S2': [8192],
                                 'S2C': [4096],
                                 'K1': [128],
                                 'K2': [512],
                                 'ori_kv_topk_mode': ['fullK'],
                                 'cmp_kv_topk_mode': ['fullK'],
                                 'block_size1': [128],
                                 'block_size2': [128],
                                 'seed': [7],
                                 'kv_axis0_noncontiguous': [False]},
                         }

ENABLED_PARAMS = list(TEST_PARAMS.values())
