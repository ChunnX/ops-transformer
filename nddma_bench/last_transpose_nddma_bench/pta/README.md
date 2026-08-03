# LastTranspose [0,1,3,2] NDDMA — PTA 直调最小工程

PyTorch 直调（PTA）最小闭环：ASC 编译 launch `.so` → pybind11 扩展 → Python 侧 `.npu()` 张量零拷贝驱动 kernel。
仅保留"编译 + 调起"必需件与一个 demo，性能采集/精度扫描等测试脚本已移除。

## 与 registry-invoke 主线的关系

- **compute 逻辑**：`../op_kernel/arch35/last_transpose_nddma_bench.h`
  （命名空间类 `LastTransposeNddmaBenchNs::KernelLastTransposeNDDMA<T, ubCapElems>`），
  与主线 registry-invoke 算子**共用同一份 compute 头**，不复制、不改写。
- **`last_transpose_launch.asc`**：thin benchmark kernel 入口 `LastTransposeNDDMABenchKernel`
  （读 attr GM → 按 dtype 派发命名空间类 → `Init`/`Process`）+ C-ABI（`LaunchLastTranspose` / `LaunchLastTransposeHost`）。

## 构建 launch .so（需 CANN + ASC 工具链）

```bash
cd pta
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
cmake --build build --target last_transpose_launch -j
# 产物：pta/build/liblast_transpose_launch.so（pta_nddma.py 的 _BUILD 指向此处）
```

## PTA 直调 demo（需真实 NPU + torch_npu）

```bash
python3 pta_nddma.py    # JIT 构建扩展 + 跑一条 case，bitwise 比对 x.transpose(-1,-2)
```

或在自己的脚本里：

```python
from pta_nddma import last_transpose
y = last_transpose(x.npu(), is_mte3=1)   # x: 4D (dim0,dim1,dim2,dim3)
```

## 文件清单（名 → 角色）

| 文件 | 角色 |
|------|------|
| `last_transpose_launch.asc` / `.h` | ASC launch kernel + C-ABI（compute 头在 `../op_kernel/arch35`） |
| `last_transpose_ext.cpp` | pybind11 扩展（`last_transpose` / `last_transpose_host`） |
| `pta_nddma.py` | JIT 构建扩展 + Python API `last_transpose(x, is_mte3)` + `__main__` demo |
| `CMakeLists.txt` | standalone pta/ 工程（`find_package(ASC)` + `project()`） |
