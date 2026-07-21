# RotaryPositionEmbedding3D — Wan2.2 3D 空间 RoPE 算子

## 概述

本算子实现 **Wan2.2 风格的 3D 空间 RoPE**，对视频特征序列按帧/高/宽三轴独立编码位置信息。

与标准 1D RoPE 的核心区别：

- **不等长频带划分**：时间轴 $d_0 \approx D/2$，高度/宽度轴各 $D/4$（2:1:1 比例）
- **三维位置解码**：$l \rightarrow (t,i,j)$ 将一维序列位置解码为视频网格坐标
- **三轴独立旋转**：T/H/W 三个频带分别用各自坐标乘以独立频率表，再应用 half-mode 旋转

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 910B | √ |
| Atlas A2 训练系列产品 | √ |

## 算子规格

| 项目 | 规格 |
|------|------|
| 算子类型 | RotaryPositionEmbedding3D |
| 输入 x | FLOAT16 / FLOAT / BFLOAT16, ND, shape `[B, S, D]` |
| 输入 cos | FLOAT16 / FLOAT / BFLOAT16, ND, shape `[B, S, D]`（预计算 cos/sin 表） |
| 输出 y | FLOAT16 / FLOAT / BFLOAT16, ND, shape `[B, S, D]`（旋转结果） |
| 频带划分 | 2:1:1（Temporal:Height:Width），各频带为偶数 |
| 频率基数 | 10000.0 |
| position 解码 | `FactorVideoDims` 将 S 分解为 `T×H×W` |
| Block Dim | 8 |
| 支持 SoC | ascend910b, ascend910_93 |

## 数学公式

### 1. 频带划分

$$D_T \approx D/2,\quad D_H \approx D/4,\quad D_W \approx D/4$$

### 2. 频率衰减率

$$r_T = \beta^{-2/D_T},\quad r_H = \beta^{-2/D_H},\quad r_W = \beta^{-2/D_W}$$

其中 $\beta = 10000$。

### 3. Theta 表（等比序列）

$$\boldsymbol{\theta}_T[k] = r_T^{\,k},\quad
\boldsymbol{\theta}_H[k] = r_H^{\,k},\quad
\boldsymbol{\theta}_W[k] = r_W^{\,k}$$

### 4. 位置解码

$$t = \lfloor l/(H \cdot W)\rfloor,\quad
i = \lfloor (l \bmod H \cdot W)/W\rfloor,\quad
j = l \bmod W$$

### 5. Half-mode 旋转

$$\begin{aligned}
\mathbf{y}_l &= \mathbf{x}_l \odot \cos(\boldsymbol{\alpha}) - \mathbf{x}_r \odot \sin(\boldsymbol{\alpha}) \\
\mathbf{y}_r &= \mathbf{x}_l \odot \sin(\boldsymbol{\alpha}) + \mathbf{x}_r \odot \cos(\boldsymbol{\alpha})
\end{aligned}$$

其中 $\boldsymbol{\alpha}$ 按频带分别构造：
- T-band: $\alpha[k] = t \cdot \boldsymbol{\theta}_T[k]$
- H-band: $\alpha[k] = i \cdot \boldsymbol{\theta}_H[k]$
- W-band: $\alpha[k] = j \cdot \boldsymbol{\theta}_W[k]$

## 目录结构

```
rotary_position_embedding3d/
├── CMakeLists.txt
├── README.md
├── op_host/                          # Host 侧：算子定义、shape 推导、tiling
│   ├── CMakeLists.txt
│   ├── rotary_position_embedding3d_def.cpp
│   ├── rotary_position_embedding3d_infershape.cpp
│   └── rotary_position_embedding3d_tiling.cpp
├── op_kernel/                        # Device 侧：AscendC kernel
│   ├── rotary_position_embedding3d.cpp          # 模板入口
│   ├── rotary_position_embedding3d.h            # kernel 实现
│   ├── rotary_position_embedding3d_tiling_data.h
│   └── rotary_position_embedding3d_tiling_key.h
├── examples/
│   └── test_aclnn_rotary_position_embedding3d.cpp  # 测试示例
└── tests/
    └── ut/                            # 单元测试
        ├── op_host/                   # tiling & infershape UT
        └── op_kernel/                 # kernel UT（含数据生成脚本）
```

## 环境变量配置

```bash
source /ddhome/qkv_xpb/Ascend/cann-9.1.0-beta.1/set_env.sh
```

## 编译与运行

### 编译算子

```bash
cd /ddhome/qkv_xpb/work/ops-transformer
bash build.sh --pkg --experimental --soc=ascend910b --ops=rotary_position_embedding3d -j16
```

### 安装算子

```bash
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --noexec --extract=/tmp/rmk
cp -r /tmp/rmk/packages/vendors/custom_transformer/* /path/to/Ascend/cann-9.1.0-beta.1/opp/vendors/custom_transformer/
```

### 编译并运行测试

```bash
bash build.sh --run_example rotary_position_embedding3d eager cust
```

### 运行内核级 UT

```bash
bash build.sh -u --opkernel --ops=rotary_position_embedding3d
```

## 性能数据

（Ascend 910B, 1650 MHz, Block Dim=8）
| 输入配置 | NPU 纯执行时间（msprof） | test测试时间 |
|---------|------------------------|--------------|
| B=1, S=16, D=16 | ~3.7 μs | ~0.018 ms |
| B=2, S=128, D=512 | ~4.0 μs | ~0.038 ms |
| B=8, S=128, D=128 | ~4.2 μs | ~0.049 ms |

> NPU 纯执行时间使用 `msprof op` 采集，Task Duration 字段。
> Host 端时间包含 `GetWorkspaceSize` + kernel launch + stream sync 开销。

## 算子验证

测试包含三个方面：

1. **正确性验证**：9 组 shape（D=16~512, S=8~128, B=1~8），对比 CPU 参考实现，`eps=1e-4`
2. **3D 轴独立性验证**：固定 $(i,j)$ 改变 $t$ → T-band cos/sin 变化，H/W-band 不变
3. **随机输入验证**：使用固定种子的随机数据，确保与参考实现严格一致

所有测试通过后输出：
```
=== 9/9 tests PASSED, 0 FAILED ===
--- 3D Tests: 3/3 PASSED, 0 FAILED ---
```

## 参考实现

- Wan2.1/V2 Team. "Wan: Open and Advanced Large-Scale Video Generative Models." 2024.
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
