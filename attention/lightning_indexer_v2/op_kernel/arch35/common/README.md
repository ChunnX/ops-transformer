# LI v2 / QLI v2 arch35 common

本目录由 `lightning_indexer_v2` 与 `quant_lightning_indexer_v2` 单向引用。
公共 C++ 命名空间为 `LIV2Common`，不再增加 `Arch35` 层；公共 VF 使用
`liV2Vector1` 和 `liV2TopkCommon`。

## 文件职责

| 文件 | 公共内容 |
| --- | --- |
| `lightning_indexer_v2_base_arch35.h` | `LightningIndexerV2DataType`、公共 RunInfo/ConstInfo 字段、分核结构、工具和 UB 常量 |
| `lightning_indexer_v2_kernel_base_arch35.h` | CRTP 主循环、service 初始化、BN2 分解、actual length、mask 块数、metadata 装载与 decode 同步 |
| `lightning_indexer_v2_service_cube_base_arch35.h` | CRTP tensor 绑定、cube event 分配和释放 |
| `lightning_indexer_v2_service_vector_base_arch35.h` | CRTP workspace 绑定、公共 event、无效 index 清理、TND padding、LD 地址计算 |
| `vf/lightning_indexer_v2_vector1_base.h` | 从原 `arch35/vf/common` 原样迁移的 vector1 VF |
| `vf/lightning_indexer_v2_topk_base.h` | 根据临时 index 位宽参数化的 gather buffer 大小和循环偏移 |
| `vf/vf_topk_base_v2.h` | histogram cast/store、前三轮 target-bin/nextK、uint32 index offset |

## 保留边界

- 两侧对外 Init、kernel 类名、入口模板参数和 tiling/metadata 协议不变。
- 公共 CRTP 类不持有数据成员，不引入虚函数、函数指针或运行时算子分支。
  各算子的 tensor、buffer、tiling 状态继续由本地类持有。
- QLI 的 scale 计数由本地 loop hooks 更新；MX/FP4 tensor offset、load/cast 和
  Matmul 指令继续留在 QLI。LI 的 metadata=null fallback 保留。
- mask 计算保留两侧原有 cmpRatio 类型、带符号中间值及至少一个有效元素的规则。
  零长度提前返回时不新增对 QLI validS2Len 的赋值。
- float/bfloat16 value 输出及 TopK b32/b16 gather 指令仍有各自实现。
  TND padding 仅以负无穷编码的整数类型参数化，不改变原有事件顺序。
- ComputeMm1 外层循环和 ProcessLD 指令段暂不进一步上移；详设要求先取得设备
  编译、输出与性能对比结果，再推进这些阶段。
- 不创建包含两侧所有量化字段的 RunInfo/ConstInfo 超集。公共字段通过基类共享，
  QLI-only 字段留在扩展结构。设备编译仍需检查继承布局及寄存器占用。

## include 与打包

QLI 使用 `__has_include` 同时支持源码树和安装目录：源码树保留 `op_kernel`，
安装布局中两个算子的 kernel 目录相邻。现有 QLI host CMake 已声明 LI v2 依赖；
`cmake/func.cmake` 的递归收集及目录复制规则会包含新增 common 子目录，未新增反向依赖。
下面的 Python 检查验证两种目录映射，但不能替代实际打包及安装后的编译。

## 验证记录与命令

重构基线：`d3d2ef72493faca6b0c00eaceb94df90049796bf`。
在仓库根目录运行：

```text
python attention/lightning_indexer_v2/tests/validation/check_arch35_common_refactor.py -v
git diff --check
```

检查包括展开公共 helper 后与基线比较事件/VF 指令顺序、kernel 循环、metadata 边界
回退、mask 计算及 padding，并检查 include 映射、依赖方向和 alias 使用。
这些是源码级迁移检查，不是设备执行、数值精度或性能测试。

当前本地 Windows 环境未找到 AscendC 编译器，因此设备编译、STC、性能及二进制资源
对比尚未执行。在配置好 CANN 的 Linux 环境中，按项目 build.sh 支持的命令执行：

```bash
bash build.sh --opkernel --soc=ascend950 --ops=lightning_indexer_v2,quant_lightning_indexer_v2
bash build.sh --pkg --soc=ascend950 --ops=lightning_indexer_v2,quant_lightning_indexer_v2
```

随后按两侧 `tests/pytest/README.md` 运行既有 STC/paramset，覆盖 FP16/BF16、INT8、
FP8/HIFLOAT8、MXFP8/MXFP4、BSND/TND/PA、空序列、LD、returnValue、TopK 边界；
另需验证 arch22 编译。完成上述验证前，不将本轮变更视为设备验收通过。
