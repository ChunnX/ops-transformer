# Dispatch 同步修复与设备热路径清理

## 已有设备反馈与最终同步

用户报告：无诊断的 A+B 通过，仅 A 通过，仅 B 错误；随后仅 A 的单跑与 continue 均通过。
这些是清理前版本的用户实测反馈，不代表本次移除常驻打点后的版本已验证。

A 位于 DedupAndSend() 返回后、CalSendCntPerRank() 前，保留为无条件执行：

```cpp
SyncFunc<AscendC::HardEvent::V_S>();
SyncFunc<AscendC::HardEvent::S_V>();
```

该边界等待去重完成，再进入复用 UB 的统计流程。实验支持此边界的等待有效，
尚未通过生成指令证明具体缺失的底层依赖；不将经验定位扩大成已证明的硬件根因。

## 清理范围

- 删除 COUNT_CHECK、COUNT_SYNC_MASK 及所有实验配置、标量参考扫描、诊断日志与诊断 UB。
- 删除 B 点实验同步，A 点不再依赖宏。
- Dispatch、Combine 及两个 epilogue 删除异常 metadata 写入、每核运行位置记录及 writer 成员。
- 删除设备侧 context/rank/Jetty 数量重复检查、SQ/CQ 深度检查及其 printf/Trap。
- 删除通信返回值检查包装；WriteNbi、WriteWithNotifyNbi 和 BatchCommit 调用本身全部保留，
  直接调用，不保存或检查返回值，也不添加 (void) 强制转换。
- 删除只用于这些检查的成员、私有参数及两个已无调用者的公共目录 helper 头文件。

不改变公开 kernel 参数类型/顺序、tiling ABI、业务窗口偏移、32 Jetty、SQ4096/CQ64
配置或动态分核。保留 host 参数/资源创建检查；它们不在设备 token 发送热循环中。
诊断预留窗口与 tiling metadata 字段保留以维持布局兼容，但四个 kernel 不再更新诊断记录。

以下是功能代码，不能当作“检查”删除：
完成 flag 等待/清理、prepare-ready 协议、批次容量判断与提交、必要 DCCI 和同步、
空任务分支、有效路由筛选及 Combine epilogue 的 MaskCheck()（实际生成计算掩码）。

## 行为变化与风险

移除返回值检查后，通信错误不会在原检查点主动报错/Trap，可能表现为错误数据或后续等待不结束。
移除设备深度检查并不保证队列不会超限，也不改变队列实际配置。
不承诺检查一定导致分支预测失败，也不承诺本次清理带来确定加速，性能以设备实测为准。

常驻异常 writer 原本分配 UB、读写 GM 并插入同步；删除后 UB 布局和执行时序会变化。
Dispatch epilogue 两路径的业务输出 MTE3_S 完成同步保留，不依赖已删除的诊断写入。
清理版需要重新验证单跑、continue、cached 和不同 shape/rank；通过后再测性能。

## 本地检查

检查实际差异、通信调用是否保留、A 同步位置、诊断符号残留及 git diff --check。
独立执行逻辑、Clean Code、完整性和架构审查。没有执行远端操作、CANN 编译或 NPU 测试。
两个删除的 helper 可以从 Git 历史恢复。
