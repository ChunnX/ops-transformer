# Flash Decode 开发问题总结

## 问题 1: vf_flash_decode.h 路径不可达
- **阶段**: Phase 2 编译
- **现象**: `fatal error: '../../attention_pioneer/op_kernel/common/arch35/vf/vf_flash_decode.h' file not found`
- **原因**: 构建系统将源码复制到 build 目录时，跨算子的相对路径 `../../attention_pioneer/...` 失效
- **修复**: 将 vf_flash_decode.h 复制到 QSFA 算子自己的 `op_kernel/vf/` 目录，修改 include 为 `"vf/vf_flash_decode.h"`
- **教训**: 算子编译时只能引用自身目录下的文件，跨算子引用需复制到本地

## 问题 2: MicroAPI::StoreDist 双重前缀
- **阶段**: Phase 2 编译
- **现象**: `error: no member named 'MicroAPI' in namespace 'AscendC::MicroAPI'`
- **原因**: vf_flash_decode.h 原始代码中已有 `MicroAPI::StoreDist::DIST_NORM_B32`，replace_all 将 `StoreDist::DIST_NORM_B32` 替换为 `MicroAPI::StoreDist::DIST_NORM_B32` 时，已有前缀的变成了 `MicroAPI::MicroAPI::StoreDist::DIST_NORM_B32`
- **修复**: 将所有 `MicroAPI::MicroAPI::StoreDist` 修正为 `MicroAPI::StoreDist`
- **教训**: 全局替换时需注意已有前缀的情况，避免双重限定

## 问题 3: DataCopyPad GM→UB 缺少 PadExtParams
- **阶段**: Phase 2 编译
- **现象**: `error: no matching function for call to 'DataCopyPad'`
- **原因**: `DataCopyPad` GM→UB 方向需要两个参数结构体 `(DataCopyExtParams, DataCopyPadExtParams<T>)`，而 UB→GM 方向只需要 `DataCopyExtParams`
- **修复**: 添加 `DataCopyPadExtParams<T> accumPadParams` 参数（isPad=false）
- **教训**: DataCopyPad 的 GM→UB 和 UB→GM 方向签名不同，GM→UB 必须提供 pad 参数

## 问题 4-8: Phase 1.8 预防性检查发现的数据类型 bug（编译前修复）
- **Bug 4**: GM tensor 类型错误 — softmaxFDMax/SumGm 声明为 `GlobalTensor<T>` 应为 `GlobalTensor<float>`
- **Bug 5**: Workspace 指针算术错误 — logSumExpSize 单位是 float 元素数，但指针是 T 类型
- **Bug 6**: 缺少 BroadcastMaxSum — softmaxMaxBuf 连续布局需转换为 32B 对齐布局才能写入 GM
- **Bug 7**: 缺少 subBlock 偏移 — 两个 subBlock 写入相同 GM 偏移导致数据覆盖
- **Bug 8**: CombineSplitKVRes 类型不匹配 — LSE 数据应为 float，accumOut 需 Cast T→float 后才能调用 ReduceFinalRes_VF<float>
