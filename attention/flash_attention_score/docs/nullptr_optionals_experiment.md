# FA 缺失可选输入 nullptr 透传与验证记录

日期：2026-09-23。正式提交分支：`fix/fas-nullptr-optionals`。
原实验分支：`debug/fas-nullptr-optionals`，功能改动提交 `3bb366c44`。
基线：本次拉取的 upstream/master `4c310b724`。

## 目的

验证 FA 不再将缺失可选输入转换为 workspace-backed 占位 tensor 后，
是否能避免其与 workspace 起点同址，以及整网中的 Fixpipe/OOM trap。
本修改已完成下述本地功能对照，提交正式 PR 供检视；异常包整网修复效果尚待确认。

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
本 L0 函数由多个 FA 接口/架构共用，本修改未做仅 Ascend950 的条件限制。

## 本地验证结果

环境：CANN 9.1.0、Ascend950PR；同一上游基线 `4c310b724` 与修改版 `3bb366c44` 对照。

- 按 `build.sh -u --ops=flash_attention_score --soc=ascend950` 构建运行，
  Host 165 个测试通过，API 7 个测试通过；图测试目标无用例，不能算作图功能覆盖。
- 两版均按 `build.sh --opapi --ops=flash_attention_score --soc=ascend950` 构建真实 op-api 库，
  通过 dlopen 指定库路径调用 V4 接口，使用同一套本地已安装的 tiling/kernel。
- BF16 BNSD `[1,1,6256,384]`，可选输入缺省，keepProb=1，原故障 tiling key
  `2328361010304385840`：两版均通过 workspace 查询、launch 和设备同步。
- 固定随机 Q/K、常量 V=1：两版各运行 3 次，attention 输出相对 1 的最大误差均为 0。
- 固定种子随机 Q/K/V：两版各运行 3 次，回读的 BF16 attention 输出逐字节一致。
  每版累计 14413824 字节，SHA256 均为
  `d190438684efcb67ed6948767dbade5c817cec66b5d51fb4a63e7c56fb41f64f`。
- 日志确认缺失输入进入 `UpdateInputArg` 的 nullptr 分支。
- 验证中曾使用 UT 目录切换构建的另一配置，出现内部输出空 shape；
  按上述正式 op-api 模式重建后不复现，不将该中间结果归因于 nullptr 修改。

边界：设备对照仅覆盖上述无 mask/dropout 的 V4 场景，未逐项比较 softmax 辅助输出，
未覆盖所有可选功能组合、其他架构、最大 workspace 接口或全部缓存模式。
本地基线也可执行成功，且未重新编译 OOM kernel，因此结果证明的是本地功能兼容性，
不证明异常 CANN 包中的 OOM 问题已经修复，更不等同于整网通过。

## 运行和结果反馈

本地结果见上节，异常包整网效果仍待用户验证。
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
