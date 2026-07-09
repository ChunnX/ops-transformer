# ATCOSS

## 项目定位

**ATCOSS**（**A**scend C **T**emplate For **C**ommunication-Compute **O**perator **S**ubroutine）是昇腾 NPU 平台 mc2 通算融合算子的底层模板库，为各类通算融合算子提供可复用的 block 层接口、kernel 层参考实现和tiling 算法，降低通算融合算子的开发成本。

**核心价值**：
- 降低开发门槛：提供分层接口与参考实现，开发者通过接口组合即可构建融合算子，无需从零实现通信与计算的协同编排
- 提升性能：通信与计算深度耦合，支持流水线重叠，充分发挥硬件算力
- 模块化复用：block层接口稳定可复用，kernel层参考实现方便快速移植

---

## 目录结构

```
atcoss/
├── kernel/               # 通算融合算子参考实现
├── block/
│   ├── blaze_ext/        #   Blaze 范式扩展
│   ├── aiv_comm/         #   AIV 通信接口
│   └── aiv_compute/      #   AIV 计算接口
├── tiling/               # tiling 算法
├── utils/                # 通用工具与常量
├── tests/                # 测试
├── docs/                 # 设计说明
└── README.md
```

---

## 模块说明

### kernel 算子层

基于block层接口的通算融合算子参考实现。kernel 层通过组合 `block` 层接口完成通信与计算的流水编排、AIV-AIC 协同调度、同步等，供具体算子直接复用或继承后定制。

### block 接口层

可复用的单核通信计算接口，按不同实现范式组织目录结构。

| 子目录 | 说明 |
|:---|:---|
| `blaze_ext` | 对 [ops-tensor](https://gitcode.com/cann/ops-tensor) Blaze 引擎在通算融合场景下的拓展实现。 |
| `aiv_comm` | AIV 核作为发起引擎的通信接口，提供跨卡数据搬移能力。 |
| `aiv_compute` | AIV 核实现向量计算接口（量化/反量化、类型转换、规约等），支撑通信前后的数据处理。 |

### tiling

智能切分算法，包括FusionTiling、性能建模、公式化切分等。

### utils

通用工具集合，包括通用常量和数据结构等。

---

## 与 mc2 算子的关系

`mc2/` 下的具体通算融合算子（如 `all_gather_matmul`、`matmul_all_reduce`、`matmul_allto_all` 等）可通过引用本库中的接口完成实现：

- 算子的 `op_host/op_tiling` 层可使用 `atcoss/tiling` 中的智能切分算法确定 tiling 参数。
- 算子的 `op_kernel` 层可基于 `atcoss/block` 接口组合构建融合 kernel，也可参考 `atcoss/kernel` 实现。

```
mc2/<op>/op_host    ──┐
                      └──▶ atcoss/tiling       (tiling 接口)
mc2/<op>/op_kernel  ──┐
                      ├──▶ atcoss/block        (block 接口，组合构建)
                      └──▶ atcoss/kernel       (算子框架，参考实现)
```
