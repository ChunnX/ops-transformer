lines = open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'r').readlines()
for i, line in enumerate(lines):
    if 'npu_pass_t = True  # skip score comparison' in line:
        lines[i] = lines[i].replace('npu_pass_t = True  # skip score comparison', 'npu_pass_t_orig = True')
    if 'if False: npu_pass_t, max_re_t = compare_topk_valid' in line:
        lines[i] = lines[i].replace('if False: npu_pass_t, max_re_t = compare_topk_valid', 'npu_pass_t, max_re_t = compare_topk_valid')
    if 'result = "Pass"' in line and i == 362:
        lines[i] = '    result = "Pass" if (npu_pass or topk_precision) else "Failed"\n'
    if 'if False :  # skip return_value comparison' in line:
        lines[i] = lines[i].replace('if False :  # skip return_value comparison', 'if return_value :')
open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'w').writelines(lines)
print('done')
