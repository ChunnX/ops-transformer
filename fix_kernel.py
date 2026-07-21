lines = open('/home/tjj/ops-transformer/attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel.h', 'r').readlines()
for i, line in enumerate(lines):
    if 'constInfo.radixTotalBatch = tilingData->radixTotalBatch; constInfo.radixTotalBatch = 0;' in line:
        lines[i] = line.replace('constInfo.radixTotalBatch = tilingData->radixTotalBatch; constInfo.radixTotalBatch = 0;',
                                'constInfo.radixTotalBatch = tilingData->radixTotalBatch;')
open('/home/tjj/ops-transformer/attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel.h', 'w').writelines(lines)
print('done')
