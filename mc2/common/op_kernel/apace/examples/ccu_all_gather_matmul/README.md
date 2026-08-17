# CCU AllGather + Matmul 融合算子（第二阶段）

2 卡 AllGather + Blaze Matmul 融合（FP16 A/B → BF16 C，M=256, K=256, N=256）。

## 编译

```bash
cd ~/code/ops-tensor && git submodule update --init --recursive include/tensor_api
source $ASCEND_HOME_PATH/set_env.sh
bash run.sh
```

## 验证

```python
import numpy as np
raw = np.fromfile("output/0/npu_out.bin", dtype=np.uint16)
c = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
assert np.all(c[:65536] == 256.0)   # rank0 行
assert np.all(c[65536:] == 512.0)   # rank1 行
```
