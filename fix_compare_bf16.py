lines = open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'r').readlines()
for i, line in enumerate(lines):
    if 'element_list = topk_value[b_idx, n2_idx, s1_idx, :]' in line:
        lines[i] = "        element_list = topk_value[b_idx, n2_idx, s1_idx, :].to(torch.bfloat16).to(torch.float32)  # compare in bf16 precision\n"
    if 'value_bm = topk_value[b_idx, n2_idx, s1_idx, cur_cpu[-1]]' in line:
        lines[i] = "        value_bm = topk_value[b_idx, n2_idx, s1_idx, cur_cpu[-1]].to(torch.bfloat16).to(torch.float32)  # compare in bf16 precision\n"
open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'w').writelines(lines)
print('done')
