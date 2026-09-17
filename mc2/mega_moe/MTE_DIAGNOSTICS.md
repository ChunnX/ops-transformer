# 950 MegaMoe 批跑 MTE 故障维测

用于定位 009 单跑通过、从 004 开始批跑失败的问题，目前未确认根因。
本分支在访问点直接调用 `printf`，格式字符串直接写在调用处，无字符串传参封装、Trap 或新增同步。

## 编译运行

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe
```

直接启用，无额外开关。安装本次构建的 Kernel 算子包，仅更新 Torch wheel 不会替换 Kernel。
先单跑 009，确认有 `[MM_DIAG_INIT_V2]`；再新建进程运行原失败批次，保存两次日志供对比。
保持原 Excel 和运行参数，在原命令中给 Python 增加 `-u`，并追加 `2>&1 | tee mega_moe_diag.log`。

打印范围为 MTE、`maxOutputSize=2310`、AIV `GetBlockIdx()==5`。
设备错误日志的 core 编号可能采用不同编号方式，需对照打印中的 core/sub。
Dispatch、Combine 每次 Kernel 调用各最多 64 条地址日志；Combine 另打印对应的原始路由。
达到上限后地址日志停止，阶段和 count 日志仍保留。printf 会扰动时序。

## 日志内容

- `MM_DIAG_INIT_V2`：当前 shape、容量、maxOutputSize、量化和预取配置。
- `MM_DIAG_LAYOUT` / `MM_DIAG_PEER`：context、workspace 及逻辑大小、分区与跨卡 window 地址。
- `MM_DIAG`：输入准备、输入同步返回、count 读取/解码、专家计算返回、输出同步及 unpermute 入口。
- `MM_DIAG_COUNT` / `MM_DIAG_EPOCH_WAIT`：count 槽原始值、期望 epoch；解码返回打印截断前总数和上限。
- `MM_DIAG_ADDR`：访问位置、peer、base、字节 offset、bytes 和逻辑分区 limit。
- `MM_DIAG_ROUTE`：Combine 原始 peer/token/topk 与 window 分区 offset。

`limit` 是 Kernel 推导的逻辑容量，不是 HCCL 实际注册容量的验证。
这些日志记录标量执行到的位置；异步 MTE 报错时，最后一条打印不能单独证明具体失败的 DataCopy。

```bash
rg -n 'MM_DIAG|invalid GM|cross-device|MTE|Case FAILED|Stack Trace|Reason:' mega_moe_diag.log > mega_moe_diag_excerpt.log
```

对照单跑和批跑的配置、count、地址及最后阶段，结合设备 plog 继续定位。
本地未安装 CANN/AscendC 工具链，尚未验证设备编译和 950 运行。
