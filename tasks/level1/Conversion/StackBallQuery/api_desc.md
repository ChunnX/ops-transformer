# aclnnStackBallQuery

## 功能描述

### 算子功能

堆叠 Ball Query：对每个中心点，在其所属 batch 的点云 `xyz` 中，按遍历顺序收集平方距离小于 `max_radius²` 的点，最多 `sample_num` 个；输出为 **batch 内局部下标**（非全局堆叠下标）。参考 `ops-math-dev/conversion/stack_ball_query/README.md`。

### 计算公式

对中心 $(c_x,c_y,c_z)$ 与点 $(p_x,p_y,p_z)$，判定 $(p_x-c_x)^2+(p_y-c_y)^2+(p_z-c_z)^2 < \text{max\_radius}^2$ 是否成立，按点在 batch 内的顺序取前 `sample_num` 个满足条件的下标；不足则重复第一个命中下标；全无命中时首元素为 `-1`、其余为 `0`。

## 接口定义

### Python 接口

通过 PyBind 封装，编译替换后函数名为 `kernel_gen_ops.stack_ball_query`（模板中为 `custom_pybind_api` / `aclnnCustomOp`）。

```python
def stack_ball_query(
    xyz: Tensor,
    center_xyz: Tensor,
    xyz_batch_cnt: Tensor,
    center_xyz_batch_cnt: Tensor,
    max_radius: float,
    sample_num: int,
) -> Tensor:
    """
    参数:
        xyz: [3, N]，planar 存储（全体 x、再 y、再 z），与 NPU 上 dtype 一致（float32/float16）。
        center_xyz: [M, 3]，逐点 x,y,z。
        xyz_batch_cnt: 一维 int32，各 batch 的 xyz 点数。
        center_xyz_batch_cnt: 一维 int32，各 batch 的 center 点数。
        max_radius: 搜索半径（标量）。
        sample_num: 每中心最多记录的索引个数。

    返回:
        idx: 一维 int32，长度 M * sample_num。
    """
```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
N, M = 7, 4
xyz = torch.zeros(3, N, device=device, dtype=torch.float32)
center_xyz = torch.zeros(M, 3, device=device, dtype=torch.float32)
# 按业务填充坐标后：
xyz_batch_cnt = torch.tensor([4, 3], device=device, dtype=torch.int32)
center_xyz_batch_cnt = torch.tensor([2, 2], device=device, dtype=torch.int32)

idx = kernel_gen_ops.stack_ball_query(
    xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, 2.0, 4
)
```

## 约束与限制

- `xyz` 第二维为点数 $N$；`center_xyz` 第一维为中心数 $M$；batch 计数长度与 batch 数一致且与数据划分匹配。
- Kernel 分两支：`float`（tiling key 1）与 `half`（tiling key 2）；半精度场景注意与参考实现的数值边界。
- 构图张量须在正确 NPU device 上，dtype 满足 aclnn 白名单。
