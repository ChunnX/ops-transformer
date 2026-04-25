# op_kernel 迁移指南

本文用于把 CANN 开源仓算子的 `op_kernel` 侧代码迁移到 NPUKernelBench 的 msopgen 单算子工程。原则是保留 910B/910_93 可用的 kernel 计算路径，只清理非目标架构、不可用依赖和工程入口差异。

## 1. 迁移前检查

先在原始算子 `op_kernel` 目录运行：

```bash
rg -n "__CCE_AICORE__|__NPU_ARCH__|TILING_KEY_IS|GetTilingKey|extern \"C\"|__global__|GM_ADDR|workspace|tiling|#include \"\\.\\./|#include \"\\.\\./\\.\\./" op_kernel
```

重点确认：

- kernel 入口函数名、参数顺序和数量。
- 是否存在多个架构分支、`arch35/`、`*_apt.cpp`、Kirin/310P 专用实现。
- host tiling 是否设置了 kernel 侧需要的 tiling key。
- 是否通过相对路径 include 其他算子、公共工具或私有头文件。
- workspace 参数是否存在，以及是否被 kernel 实际使用。

## 2. 目标架构裁剪

只保留目标平台可编译、可运行的路径：

| 原始内容 | 迁移规则 |
|---|---|
| `#if __CCE_AICORE__ == 200` | 删除整个 200 分支，包括 `arch35/` 和 `*Simt` 命名空间 |
| `#if __CCE_AICORE__ == 300` | 优先保留 220 路径，移除 300 专用路径 |
| `#if __NPU_ARCH__ == 3101` | 删除 310P 专用代码 |
| `*_apt.cpp` | 删除，通常是 regbase/910_95 芯片入口 |
| `arch35/` | 删除，通常是 regbase 专用实现 |

如果算子只有 `arch35/` 或 `*_apt.cpp` 主实现，不要把该路径硬迁到 910B。先在开源仓其他目录寻找 membase/910B 实现；找不到时报告阻塞。

## 3. kernel 入口签名

kernel 签名必须与 answer 的 pybind/autogen 调用链和 host 注册保持一致：

```cpp
extern "C" __global__ __aicore__ void my_kernel(
    GM_ADDR input1,
    GM_ADDR input2,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    ...
}
```

核对规则：

- 函数名与 `kernel_name`、OpDef、autogen 生成入口一致；必要时只做命名适配。
- 参数顺序保持为输入、输出、workspace、tiling；多输出按 OpDef 输出声明顺序排列。
- 不要删除未使用的 `workspace` 或 `tiling` 参数，除非原始 kernel ABI 本身没有该参数。
- question 骨架的 kernel 签名必须与 answer 完全一致。

## 4. TilingData 和 tiling key

kernel 侧读取 TilingData 的字段、类型和顺序是 ABI，不得为了清理而改动：

- `GET_TILING_DATA` 后访问的字段必须和 op_host `_tiling.h` 中 `TILING_DATA_FIELD_DEF` 顺序一致。
- kernel 中出现 `TILING_KEY_IS(...)`、`GetTilingKey()` 或按 key 分支时，op_host 必须设置相同的 `context->SetTilingKey(key)`。
- 删除架构分支后，保留的 tiling key 分支必须仍能被 host 选中；否则应在 host 侧修正 key 选择逻辑，而不是改 kernel 分支含义。

## 5. 依赖本地化

开源仓 kernel 常通过相对路径引用公共实现：

```cpp
#include "../../apply_rotary_pos_emb/op_kernel/rotary_common.h"
```

迁移规则：

- 将被 include 的必要头文件或实现文件复制到当前算子 `answer/0/op_kernel/` 下。
- 修改 include 路径指向本地副本。
- 若依赖文件继续 include 其他仓库相对路径，递归本地化到最小可编译集合。
- 不要通过新增全局 include 路径依赖开源仓根目录；单算子工程必须自包含。

公共 helper 如果只包含模板函数、常量或小工具，可保留原命名空间；如果与当前算子命名冲突，只做命名空间或 include guard 级别的工程适配，不改函数语义。

## 6. 宏和头文件清理

优先使用 CANN/msopgen 可用头文件：

- 保留 `#include "kernel_operator.h"`。
- 删除只服务非目标架构的私有头。
- 私有宏来自其他算子或公共目录时，优先本地化定义所在头文件。
- 不为绕过编译错误而空定义计算相关宏；如果宏控制计算路径，必须找到原始定义并迁移。

如果遇到版本不兼容 API，先确认本仓已迁移样例是否有替代写法；只有确认是工程 API 差异时才做等价替换。

## 7. 检查清单

- [ ] 只保留 910B/910_93 可用 kernel 路径。
- [ ] 已删除 `arch35/`、`*_apt.cpp`、Kirin、310P 专用代码。
- [ ] kernel 入口函数名和 GM_ADDR 参数顺序与 OpDef、pybind、question 骨架一致。
- [ ] workspace 和 tiling 参数保留情况与原始 ABI 一致。
- [ ] kernel 读取的 TilingData 字段、类型、顺序与 op_host 定义一致。
- [ ] kernel 使用 tiling key 时，op_host 设置了匹配的 key。
- [ ] 所有仓库相对 include 已本地化，answer 工程不依赖开源仓根目录。
- [ ] 未通过空宏、跳分支或改计算代码绕过编译/精度问题。
