# Broadcast NDDMA — PTA 直调最小工程（3D NDDMA 方案，DESIGN §9）

PyTorch 直调（PTA）最小闭环：ASC 编译 launch `.so` → pybind11 扩展 → Python 侧 `.npu()` 张量零拷贝驱动 kernel。
仅保留"编译 + 调起"必需件与一个 demo，性能采集/精度扫描等测试脚本已移除。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/broadcast_nddma_bench.h`
  （命名空间类 `BroadcastNddmaBenchNs::KernelBroadcastNDDMA<T, ubCapElems>` + `SCENE_AB/BA/BAB/ABA`），
  与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **`nddma_broadcast_launch.asc`**：thin benchmark kernel 入口 `BroadcastNDDMABenchKernel`
  （读 8 字段 attr GM → 按 dtype 派发命名空间类，scene 透传 `Init` → `Process`）+ 唯一 C-ABI
  `LaunchNddmaBroadcast5d`。blockNum 由入参控制（核数扫描），0 → 缺省 `min(dim0,56)`（`kVectorCoreCnt`）。

## 四场景（DESIGN §9.0）

`scene`：0=AB（dim4 广播）1=BA（dim3 广播）2=BAB（dim2+dim4 广播）3=ABA（≡BA，§9.1）。
输入 shape 随 scene（广播轴长度=1），输出恒 (dim0,dim1,dim2,dim3,dim4)。

## 构建 launch .so（需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target nddma_broadcast_launch -j
# 产物：pta/build/libnddma_broadcast_launch.so（pta_nddma.py 的 _BUILD 指向此处）
```

## PTA 直调 demo（需真实 NPU + torch_npu）

```bash
python3 pta_nddma.py    # SCENE_AB 场景 JIT 构建扩展 + 跑一条 case，bitwise 比对 broadcast_to
```

或在自己的脚本里：

```python
from pta_nddma import bcast_5d, SCENE_AB
y, golden = bcast_5d(56, 4, 32, 64, 16, SCENE_AB, is_mte3=1)   # 返回 (y, golden)
```

## 文件清单（名 → 角色）

| 文件 | 角色 |
|------|------|
| `nddma_broadcast_launch.asc` / `.h` | ASC launch kernel + C-ABI（compute 头在 `../op_kernel/arch35`） |
| `nddma_broadcast_ext.cpp` | pybind11 扩展（`nddma_broadcast_5d` + `SCENE_*` 常量） |
| `pta_nddma.py` | JIT 构建扩展 + Python API `bcast_5d(...)` / `nddma_broadcast_5d(...)` + `__main__` demo |
| `CMakeLists.txt` | standalone pta/ 工程（`find_package(ASC)` + `project()`） |
