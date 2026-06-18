# dispatch_setup (CCU) —— 只做一件事: 往状态区写标志位

真正跑在 **CCU 单元** 上的 dispatch_setup, 用 0617 的新 CCU API。全程在 `experimental/mc2/dispatch_setup_ccu/` 下编译, 不依赖 hccl/build.sh。

## 它做什么
CCU kernel(`op_kernel_ccu/ccu_kernel.cc`)砍到只有一步: 用 `ccu::WriteVariableWithNotify(channel, 状态变量, XN_ID, CKE, mask)` 把状态值(标志位)写进对端状态区 + 随路通知。这就是 "setup 写状态位"。
启动走新 API: `HcclThreadAcquireWithStream` → `HcommCcuKernelRegisterStart/Register/RegisterEnd` → `HcommCcuKernelLaunch`(在 op_host/utils.cc + op_kernel_ccu/exec_op.cc)。

## 编译(在本目录, 一条命令)
```bash
source /home/0617/env/set_env.sh
cd /home/developer/ops-transformer/experimental/mc2/dispatch_setup_ccu
make                     # -> libdispatch_setup_ccu.so  ✅ A5 编译通过
```

## 运行(2-rank, 运行时态)
CCU 算子靠 HCCL 多 rank 通信, 运行需 2 个 device:
```bash
# 1. 把 .so 装到算子库(或让 testcase 直接链本目录的 .so)
# 2. 编 testcase(testcase/main.cc 调 HcclDispatchSetup 接口, 链 .so)
cd testcase && make
# 3. 2-rank 跑(本机 4 卡, 起 2 进程; 或上 A6 真机)
#    testcase 内部用 HcclGetRootInfo + HcclCommInitRootInfo 起 2 rank
make test
```
注: 运行时态按计划在 A6 真机验证; A5 上有 4 卡也可试 2-rank。跑 dispatch_setup 时只看 CCU 是否把状态位写对、teardown 是否轮询到。

## 与 AIV teardown 的衔接
- setup(本算子, CCU)在一条 stream 上把状态位写 1。
- teardown(AIV, 见 `../dispatch_sync_demo/`)在另一条 stream 上忙等轮询状态区, 见 1 往下走。
- 二者靠状态位握手、串行跑。AIV↔AIV 版已在 A5 编译+运行验证了这套握手机制本身。

## 文件
- `op_host/dispatch_setup.cc` / `utils.cc` —— host: 资源/channel/thread/注册/下发。
- `op_kernel_ccu/ccu_kernel.cc` —— CCU kernel: 只写状态位(`ccu::WriteVariableWithNotify`)。
- `op_kernel_ccu/exec_op.cc` —— 下发编排(`HcommCcuKernelLaunch`)。
- `Makefile` —— 独立构建 .so(experimental/mc2 下, 不依赖 hccl/build.sh)。
- `CMakeLists.txt` —— 官方 custom-op 工程构建配置(若走 hccl/build.sh 时用)。
