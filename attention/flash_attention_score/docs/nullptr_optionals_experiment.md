# FA 缺失可选输入 nullptr 透传实验

日期：2026-09-23。分支：`debug/fas-nullptr-optionals`。
基线：本次拉取的 upstream/master `4c310b724`。

## 目的

验证 FA 不再将缺失可选输入转换为 workspace-backed 占位 tensor 后，
是否能避免其与 workspace 起点同址，以及整网中的 Fixpipe/OOM trap。
本分支是诊断实验，尚不作为已验证的正式修复。

## 改动范围

仅修改 `op_api/flash_attention_score.cpp` 中 `l0op::FlashAttentionScore()`：

- realShift、dropMask、paddingMask、attenMask、dScaleQ/K/V、pScale 为 nullptr 时，
  不再调用 `AllocTensor`，原样进入 `INFER_SHAPE` 和 `ADD_TO_LAUNCHER_LIST_AICORE`。
- prefix、actualSeqQLen/KvLen、qStartIdx/kvStartIdx 数组未提供时，
  对应局部 tensor 指针保留 nullptr，不再创建占位对象。
- 实际提供的 tensor 原样保留；实际提供的数组继续使用原转换逻辑。
- `isMaxWorkspace` 下真实序列数组的 FakeArray 创建与销毁逻辑不变。
- 四个输出仍正常分配；输入参数顺序、属性、tiling 和完整 kernel 计算不改。
- queryRope、keyRope、sink 原先就未在这里进行缺省占位转换，保持不变。

不携带 `fix/fas-workspace-prefix-oom` 的 512 字节前缀，不修改 devkit、OOM 开关、
Fixpipe 参数或通信协议，也不携带 debug/oom 的跳过计算探针。
本 L0 函数由多个 FA 接口/架构共用，本实验未做仅 Ascend950 的条件限制。

## 运行和结果反馈

按用户要求只提供代码，未进行编译、UT 或设备执行，整网效果待用户验证。
改动位于 Host API 层，必须重新构建并加载对应 op-api 库；仅替换 kernel 不会生效。
保持故障环境、输入与 OOM 开启条件，使用原 BF16 BNSD `[1,1,6256,384]` 整网复现流程。

建议记录：

1. 实际加载的 op-api 路径、代码提交、CANN 包版本，确认无旧库/旧执行器缓存干扰。
2. 是否通过 GetWorkspaceSize、InferShape、tiling、launcher 和设备同步；失败发生在哪一阶段。
3. 若仍 AICore：完整 error string、校正 PC/符号偏移、kernel 名及参数地址 dump。
4. 原缺省可选输入是否仍为 workspace 地址；不要预设 nullptr 经框架打包后必定是零地址。
5. 若执行成功：整网输出是否正确，并进行数值精度对照；成功不等于已实锤 OOM 根因。

opbase 的 `KernelContextHolder::UpdateInputArg()` 有 nullptr 分支，将输入实例数设为 0，
但旁边仍有 RT2.0 中间空参数支持的警告注释。需要实际确认缺省输入展开、kernel 参数槽位、
kernel 匹配和缓存重放兼容性。若在 Host 阶段失败，应先分析空输入协议，不能解释为 Fixpipe 失败。

严格 A/B 应比较本分支与同一基线 `4c310b724`；此前 512 字节规避分支基于 `62b7607bf`，
不能把两个分支整网结果之差全部归因于占位处理。可将本分支代码改动单独应用到同一故障基线。
