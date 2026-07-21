lines = open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'r').readlines()
for i, line in enumerate(lines):
    if '需要进行第二步比较的batch有' in line:
        lines[i] = line + "    # Print index diff for first 10 batches\n    for t_id in rows[:10]:\n        cpu_vals = sorted(cpu_reshape[t_id, :valid_lens[t_id]].tolist())\n        npu_vals = sorted(npu_reshape[t_id, :valid_lens[t_id]].tolist())\n        cpu_set = set(cpu_vals)\n        npu_set = set(npu_vals)\n        only_cpu = sorted(cpu_set - npu_set)\n        only_npu = sorted(npu_set - cpu_set)\n        print(f'  batch {t_id}: only_cpu={only_cpu[:5]}, only_npu={only_npu[:5]}, overlap={len(cpu_set&npu_set)}')\n"
open('/home/tjj/ops-transformer/attention/lightning_indexer/tests/pytest/result_compare_method.py', 'w').writelines(lines)
print('done')
