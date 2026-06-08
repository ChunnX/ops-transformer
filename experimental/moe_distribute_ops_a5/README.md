# MoeDistributeDispatchV2 / MoeDistributeCombineV2 — A5 MTE Kernel 直调

基于 `mc2/moe_distribute_dispatch_v2` 和 `mc2/moe_distribute_combine_v2`，仅保留 A5（Ascend 950PR/950DT, dav-3510）MTE 通信模式的 `<<<>>>` 直调工程。

## 目录结构

```
moe_distribute_ops_a5/
├── CMakeLists.txt          # 编译配置，目标架构 dav-3510
├── dispatch.asc            # Dispatch 直调入口（__global__ + main）
├── combine.asc             # Combine 直调入口（__global__ + main）
├── include/                # 算子 kernel 实现头文件（扁平）
│   ├── moe_distribute_dispatch_v2.h      # Dispatch 主 kernel 类
│   ├── moe_distribute_dispatch_v2_full_mesh.h  # FullMesh 变体
│   ├── moe_distribute_combine_v2.h       # Combine 主 kernel 类
│   ├── *_tiling.h          # TilingData 结构体
│   ├── *_quant.h           # 量化辅助
│   ├── moe_distribute_*.h  # 常量 / base / elastic 等
│   └── mc2_*.h             # Mc2MoeContext / HcclOpParam 等公共依赖
└── build/                  # 编译产出目录
```

## 快速编译

```bash
mkdir -p build && cd build
ASC_CONFIG=$(dirname $(find $ASCEND_HOME_PATH -name "asc-config.cmake" -path "*/lib64/cmake/*" 2>/dev/null | head -1))
cmake -DCMAKE_PREFIX_PATH="$ASC_CONFIG" \
      -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j
./dispatch   # 单卡编译验证，仅打印信息
./combine
```

## 入口函数一览

### Dispatch（6 个入口）

| 入口函数 | XType | ExpandXType | 量化 | FullMesh |
|----------|-------|-------------|------|----------|
| `dispatch_a5_mte_nofullmesh` | 模板 | 模板 | 非量化 | 否 |
| `dispatch_a5_mte_fullmesh` | 模板 | 模板 | 非量化 | 是 |
| `dispatch_a5_mte_static_quant_nofullmesh` | 模板 | int8 | 静态量化 | 否 |
| `dispatch_a5_mte_pertoken_quant_nofullmesh` | 模板 | int8 | Pertoken 动态量化 | 否 |

### Combine（4 个入口）

| 入口函数 | ExpandXType | XType | 量化 | FullMesh |
|----------|-------------|-------|------|----------|
| `combine_a5_mte_nofullmesh` | 模板 | 模板 | 非量化 | 否 |
| `combine_a5_mte_fullmesh` | 模板 | 模板 | 非量化 | 是 |
| `combine_a5_mte_int8_quant_nofullmesh` | 模板 | 模板 | INT8 量化 | 否 |
| `combine_a5_mte_mxfp8_e5m2_quant_nofullmesh` | 模板 | 模板 | MXFP8 E5M2 | 否 |

## 调用方式

### 1. 准备 Tiling 参数

Dispatch 用 `MoeDistributeDispatchV2TilingData`，Combine 用 `MoeDistributeCombineV2TilingData`。关键字段：

```cpp
MoeDistributeDispatchV2TilingData td;
memset(&td, 0, sizeof(td));
td.moeDistributeDispatchV2Info.epRankId       = rankId;      // 本卡 EP rank
td.moeDistributeDispatchV2Info.epWorldSize    = worldSize;   // EP 域卡数
td.moeDistributeDispatchV2Info.moeExpertNum   = 8;           // MoE 专家数
td.moeDistributeDispatchV2Info.bs             = 32;          // batch sequence size
td.moeDistributeDispatchV2Info.k              = 4;           // topK
td.moeDistributeDispatchV2Info.h              = 4096;        // hidden size
td.moeDistributeDispatchV2Info.aivNum         = numCores;    // 使用 AIV 核心数
td.moeDistributeDispatchV2Info.sharedExpertNum    = 0;
td.moeDistributeDispatchV2Info.sharedExpertRankNum = 0;
td.moeDistributeDispatchV2Info.totalWinSizeEp     = winSize; // HCCL window 大小
```

完整字段参见 `include/moe_distribute_dispatch_v2_tiling.h`。

### 2. 启动 Kernel

```cpp
// 分配 Device 内存，初始化 HCCL，填充 tiling 后
dispatch_a5_mte_nofullmesh<half, half>
    <<<td.moeDistributeDispatchV2Info.aivNum, nullptr, stream>>>(
        xDevice, expertIdsDevice, scalesDevice, xActiveMaskDevice,
        elasticInfoDevice, performanceInfoDevice, expandXOutDevice,
        dynamicScalesOutDevice, assistInfoOutDevice, expertTokenNumsOutDevice,
        epSendCountsOutDevice, tpSendCountsOutDevice, workspaceDevice, td);
aclrtSynchronizeStream(stream);
```

## 多卡运行

Dispatch 和 Combine 是 MC2 通信融合算子，依赖 EP 域 AllToAllV 通信，需要多卡 HCCL 环境：

```bash
# 假设 8 卡
mpirun -np 8 --allow-run-as-root \
  ./dispatch
```

单卡可编译但无法完成有效通信（需要一个完整的 EP 通信域）。

使用前需完成：
1. HCCL 初始化（`hcclCommInitAll` 或 `hcclCommInitClusterInfo`）
2. HCCL buffer 分配（大小需满足 kernel 内 `totalWinSizeEp` 要求）
3. 设置环境变量 `HCCL_BUFFSIZE` 指定 buffer 大小

## 约束说明

- 仅支持 **A5 (dav-3510)**，不支持 A2/A3
- 仅支持 **MTE** 通信模式，不支持 CCU
- dispatch 和 combine 必须配套使用，且使用相同通信域
- 参数 `epWorldSize`、`moeExpertNum`、`sharedExpertRankNum`、`globalBs` 需各卡一致
- H 取值范围 [1024, 8192]，BS 取值范围 [1, 512]（A5 约束）
- 各卡 BS 不一致时需设置 `globalBs = max_bs * epWorldSize`

## 扩展进 PyTorch

如需通过 PyTorch 调用，参考 `examples/fast_kernel_launch_example/` 的 `TORCH_LIBRARY` 模式：

```cpp
TORCH_LIBRARY(ascend_ops, m) {
    m.def("dispatch(Tensor x, Tensor expert_ids, ...) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m) {
    m.impl("dispatch", &dispatch_a5_mte_nofullmesh<half, half>);
}
```
