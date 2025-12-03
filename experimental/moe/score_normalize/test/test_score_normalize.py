import torch
import torch_npu
import npu_ops_transformer_ext
import logging

logging.basicConfig(level=logging.INFO)
TOTAL_CASE_NUM = 500
ROWS_CASE_NUM = 100
rows_cases = torch.randint(1, 16385, (ROWS_CASE_NUM, ))
ks = torch.randint(1, 17, (TOTAL_CASE_NUM // ROWS_CASE_NUM, ))
supported_dtypes = {torch.bfloat16}
for data_type in supported_dtypes:
    for rows in rows_cases:
        for k in ks:
            logging.info(f"DataType = <{data_type}>, rows = {rows}, k = {k}")
            x = torch.randn((rows, k)).to(data_type)
            logging.info(f"Input x: \n{x}")
            golden = (x / x.sum(-1).unsqueeze(-1)) * 2.5
            logging.info(f"cpu output: \n{golden}")
            x_npu = x.npu()
            torch.ops.npu_ops_transformer_ext.score_normalize(x=x_npu, rows=rows, k=k)
            npu_result = x_npu.cpu()
            logging.info(f"[OK] torch.ops.npu_ops_transformer_ext.score_normalize<{data_type}> successfully!")
            logging.info(f"npu output: \n{npu_result}")
            rotl, atol = torch.testing._comparison.get_tolerances(data_type, rtol=None, atol=None)
            logging.info(f"compare CPU Result vs NPU Result: {torch.allclose(golden, npu_result, rotl, atol)}\n\n")
