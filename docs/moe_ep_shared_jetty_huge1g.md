# MoE D/C 共享通信窗口：仅分配 1 GiB 大页

## 范围与行为

修改 `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp` 的 `AllocateDeviceBuffer()`。
这里是我们自己申请、通过 `HcclCommMemReg` 注册给 HCCL 的共享 Jetty 窗口，不是 HCCL 内置 buffer，
也不改变普通 Channel、Engram 或其他算子的分配策略。

原实现优先尝试 `ACL_MEM_HUGE1G`，任一申请/映射/访问权限步骤失败后，回退到 `ACL_MEM_HUGE`。
现在删除回退，只允许 1 GiB 大页；失败时沿用已有资源清理并抛出初始化错误，不注册失败的地址。
该检查只发生在 host 初始化，不在 Dispatch/Combine 设备热路径新增分支、日志或断言。

物理申请、虚拟地址预留、映射与访问权限覆盖的大小仍为：

`physicalBytes = ceil(requiredBufferSize / 2^30) * 2^30`

不足 1 GiB 也会申请 1 GiB，略超过 1 GiB 会申请 2 GiB；原有对齐溢出检查保持不变。
`HcclCommMemReg`、清零、context 缓存容量与算子窗口仍使用原来的逻辑大小 `requiredBufferSize`，
不把向上取整的尾部变成额外可用空间，也不改变窗口排布和外部 ABI。

调用链：D/C 四个入口 → `GetMoeExecutionContext()` → `BuildMoeSharedJettyContext()` →
`RegisterDeviceBuffer()` → `AllocateDeviceBuffer()` → `AllocateHugePageBuffer(..., ACL_MEM_HUGE1G, ...)`。
成功注册后的生命周期、缓存复用、32 Jetty / SQ4096 / CQ64 和完成协议均保持不变。

`ACL_MEM_HUGE1G` 表示 1 GiB 粒度大页，`ACL_MEM_HUGE` 是 2 MiB 粒度，参见
[官方 aclrtMemAttr 定义](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1398.html)。
是否支持及能否成功分配仍以目标 CANN/设备的实际结果为准；本轮不增加降级分配。

## 部署与验证边界

- 修改在 Torch 扩展 host 代码中，需要重建并安装包含该文件的扩展/wheel；本轮未改算子 kernel。
- 使用新进程重新创建 communicator/context，不能期待已有进程中的旧注册窗口自动换页。
- 增大的是物理分配粒度，不是修改 `HCCL_BUFFSIZE`。
- 本地仅做差异、调用链、失败清理及对齐边界静态检查；没有执行 CANN 编译或远端/NPU 验证。
  静态检查不能证明目标机器成功获得了 1 GiB 大页，也不能保证实际性能增益。
- `git diff --check` 通过；源码检查确认只剩一次 `ACL_MEM_HUGE1G` 分配尝试，注册/清零逻辑大小未改。
  对齐检查覆盖 2 MiB、1 GiB、1 GiB+2 MiB、2 GiB 和 uint64 对齐溢出边界，均通过。
  除 `AllocateDeviceBuffer()` 外，该生产文件与 HEAD 逐字比较一致；修改函数为 20 个非空非注释行。
- 按规则调用 `deepseek_flash_reviewer` 独立审查，但服务返回当前 ChatGPT 账号不支持 `deepseek-flash`；
  重试一次仍失败，未替换模型。独立审查未完成，本轮结论仅基于主代理源码检查和上述静态验证。
