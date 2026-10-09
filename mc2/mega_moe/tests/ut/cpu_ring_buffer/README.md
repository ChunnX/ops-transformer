# Arch35 MTE 循环 buffer 的 CPU 检查

本目录验证无去重、非量化 Combine 路径的容量、地址映射和流水生命周期。无需 NPU，不加载 Torch/CANN 运行时。

## 流水生命周期模型

使用 Python 标准库即可运行：

```bash
python -B -m unittest discover -s mc2/mega_moe/tests/ut/cpu_ring_buffer -p 'test_*.py' -v
```

`test_ring_pipeline.py` 按 AIC、AIV1、AIV0 及各核独立推进操作，物理存储为每个阶段分别维护旧数据和全部未完成读取者。它不直接把 watermark 当作正确性结论：任何写入覆盖尚有读取者的旧数据都会失败。

模型覆盖：

- 256 行 group 的专家尾块补齐；逻辑 metadata 行号与循环物理行号保持分离。
- 启动 Dispatch wave 0、1，稳态 Dispatch wave `w+1`、GMM1/Activation wave `w`、滞后一轮的 GMM2/Combine wave `w-1`，以及无滞后路径和最后一轮排空。
- wave 预算 1、2、5、9，1～8 个核，多个 N tile 消费同一 group，没有分到任务的核，以及生产者优先、单核领先、随机执行顺序。
- 150 组固定种子的随机路由与执行顺序，以及 96 组结构化时序。
- 两个负例：两份 wave 容量不足以承受允许的角色偏移；三份容量在移除 watermark 约束后仍可能覆盖有效数据。

模型中的操作表示 GM 搬运已经完成。它不能验证设备事件、异步搬运排空或 cache 可见性；生产代码的完成发布必须在实际读取结束之后。

## 编译并运行生产布局公式

Windows 的备用检查使用 `validate_production_layout.py`。默认编译器是已安装的 ScopeCppSDK MSVC，完整工具链也可以通过 `--compiler` 指定：

```powershell
python -B mc2/mega_moe/tests/ut/cpu_ring_buffer/validate_production_layout.py
```

生成的源码、对象及可执行文件默认写入 checkout 外的 `D:\workspace\mega_moe_ring_cpu_validation\layout_*`，可用 `--output-dir` 改变输出位置。

检查直接编译生产 `mega_moe_workspace.h` 和 `mega_moe_ring_layout.h`；`stubs/` 仅补充固定宽度整数、未使用的 toolkit tiling include，以及 `CeilDiv/CeilAlign` 两个整数工具。生产布局代码不作替换。备用可执行文件不链接 CRT，另有返回 91 的负例确认测试退出码可被可靠取得。

覆盖范围为：

| 检查 | 范围 |
|---|---|
| 生产 WorkspaceLayout | 34,560 组参数：所有 GMM 模式、wave 预算 0/1/7/16、接收容量 1/128/65536、MTE/URMA、去重 0～3、量化开关、prefetch、共享专家 |
| 生产 padding 和取模 helper | 2,872,320 行映射，含完整 group 回绕 |
| 新增 host layout UT | 从正常 host UT 文件提取四个新增测试的原始源码，用轻量断言适配器编译执行 |
| Torch CCL workspace 上界 | 提取未修改的 `CalcCclWorkspaceSizeA5` 整数函数，对比 2,624,400 组实际 WorkspaceLayout，覆盖核数 1/28/36、四个共享专家、微小接收容量、去重和量化回退、URMA |

提取只用于该检查；正常 host UT 仍由原有 CMake/gtest harness 构建。此备用方式没有编译 Torch 扩展的其他部分，也没有执行 host tiling 框架。Torch 的浮点去重门控不是本检查的覆盖范围。

这些 CPU 检查不能替代 AscendC 编译和 NPU 的数值、通信、同步及性能验证。它们的作用是提前发现容量上界不足、尾块地址错误、布局回退变化及流水覆盖风险。
