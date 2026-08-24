# MhcPreBackward 极致性能优化任务

## 目标

对以下 Ascend 950 算子内核进行极致性能优化，同时始终保证精度正确：

`mhc/mhc_pre_backward/op_kernel/arch35/mhc_pre_backward_kernel.h`

每一项优化必须独立完成代码修改、提交、推送、远程拉取、编译安装、精度测试、性能测试和结果记录。以可复现的实测数据判断是否保留优化，不能仅根据理论分析认定存在收益。

## 优化顺序

严格一次只做一项，避免多个变量混在同一次实验中：

1. 优先优化 C 和 V 之间的数据搬运，使其不经过 GM，直接通过 UB 完成。
2. 在第一项完成并验证后，尝试矩阵乘法使用 HF32。
3. 前两项逐项完成后，再根据以下分析文档逐项实施其他优化：
   `D:\Works\260809mhc_pre_backward_a5\record\MhcPreBackward算子原有实现和潜在性能优化点分析.md`

每次实验应保持改动最小、目的单一，并与上一有效版本进行对比。若优化无收益、精度不通过或结果不稳定，应明确记录，不得写成有效优化；后续实验应基于最近一个精度正确且确认有收益的版本继续。

## 本地代码与 Git 流程

本地仓库：

`D:\Code\mhc_pre_backward_a5\ops-transformer`

每轮流程：

1. 修改目标内核，检查工作区，避免覆盖或提交用户的无关改动。
2. 对改动进行静态检查，确认改动只服务于本轮单一优化目标。
3. 创建能够说明优化内容的 Git commit，并记录完整提交 ID。
4. push 到当前工作分支。
5. 在远程 Docker 环境中拉取该提交，确认远程测试代码的 `HEAD` 与记录的提交 ID 一致。
6. 编译、安装、执行精度与性能测试。
7. 读取并汇总性能结果，计算相对上一有效基线的收益。
8. 更新修改记录和性能汇总文件；仅当有性能收益时下载对应 profiler CSV。

未经用户明确授权，不得覆盖、丢弃或回滚用户已有改动，不得使用 `git reset --hard` 等破坏性命令。

## 远程执行环境与边界

SSH 配置：

```text
Host A5-138-docker-21032
    HostName 29.246.144.138
    User root
    Port 21032
```

使用 zsh 连接远程 Docker。所有创建、修改、构建、安装、测试、日志和临时文件操作都必须限制在：

`/home/matmul/h00944173`

不得在该目录之外创建、修改或删除文件。唯一允许的目录外操作是读取并 `source` 用户指定的基础 CANN 环境脚本：

`/home/matmul/l00947498/CANN-0812/cann/set_env.sh`

执行任何远程写操作前，应确认当前路径和目标路径均位于 `/home/matmul/h00944173` 下。不得在宿主机或其他 Docker/用户目录中操作。

## 代理配置

代理地址为 `10.224.204.233:8080`，用户名为 `h00944173`。代理密码属于敏感凭据，不得写入本仓库、提交、日志或测试记录；应从安全环境变量中读取，并在 URL 中进行百分号编码。

示例（`<URL_ENCODED_PASSWORD>` 由安全环境注入）：

```zsh
git config --global http.proxy 'http://h00944173:<URL_ENCODED_PASSWORD>@10.224.204.233:8080'
git config --global https.proxy 'http://h00944173:<URL_ENCODED_PASSWORD>@10.224.204.233:8080'
export http_proxy='http://h00944173:<URL_ENCODED_PASSWORD>@10.224.204.233:8080'
export https_proxy='http://h00944173:<URL_ENCODED_PASSWORD>@10.224.204.233:8080'
export no_proxy='127.0.0.1,.huawei.com,localhost,local,.local'
```

注意：设置全局 Git 代理会写入 `/root/.gitconfig`，与“只在 `/home/matmul/h00944173` 下写入”的边界冲突。除非用户明确允许该例外，否则优先使用当前命令环境变量或仓库级 `git config --local`，且配置文件必须位于允许目录内。

## 远程编译与安装

进入 `/home/matmul/h00944173` 下对应的远程仓库，拉取并确认目标提交后执行：

```zsh
source /home/matmul/l00947498/CANN-0812/cann/set_env.sh
bash build.sh --pkg --soc=ascend950 --ops=mhc_pre_backward
./build/cann-ops-transformer-custom_linux-aarch64.run \
  --install-path='/home/matmul/h00944173/CANN/mhc_pre_backward_0824' \
  --quiet
source /home/matmul/h00944173/CANN/mhc_pre_backward_0824/vendors/custom_transformer/bin/set_env.bash
```

每轮编译必须检查命令退出状态和关键日志，确认本轮生成并安装的是目标 commit，不得使用旧产物冒充本轮结果。

## 精度与性能测试

测试目录：

```zsh
cd /home/matmul/h00944173/backward_mhc_pre/test
```

先执行精度测试：

```zsh
python precision_2_1_mhcpre_backward.py
```

只有精度测试通过后才将该版本作为有效性能候选。随后执行性能测试：

```zsh
msprof python perf_mhcpre_backward_fused.py
```

性能原始结果位于类似路径：

`/home/matmul/h00944173/backward_mhc_pre/test/PROF_000001_xx/mindstudio_profiler_output/op_summary_2026xx.csv`

每次测试都必须确认使用的是本轮新生成的 profiler 目录和 CSV，读取目标算子的耗时数据，不能误用历史 `PROF_*` 结果。比较时必须使用相同 shape、测试脚本、环境和统计口径。

## 性能计算与判定

基线及后续汇总文件：

`D:\Works\260809mhc_pre_backward_a5\record\pre反向性能.xlsx`

对每个 shape，只向汇总文件写入该 shape 的 10 次性能平均值；如果有效数据只有 9 次，则写入 9 次平均值，并在修改记录中注明样本数和排除原因。不要把每次原始耗时逐条写入汇总表。

性能收益统一按以下公式计算：

```text
性能收益 = 老版本平均耗时 / 新版本平均耗时 - 1
性能收益百分比 = (老版本平均耗时 / 新版本平均耗时 - 1) * 100%
```

其中“老版本”默认为本轮修改前最近一个精度正确且已确认有效的基线版本。正值表示提升，负值表示退化。对极小变化或波动较大的结果，应复测后再下结论。

## 结果记录

每轮实验均更新：

`D:\Works\260809mhc_pre_backward_a5\record\修改记录.md`

每条记录至少包括：

- 日期、实验序号和完整 Git commit ID。
- 本轮优化目标及其原因。
- 修改前后的文字说明。
- 关键代码实现对照，包含文件、函数/代码位置及必要的修改前后代码片段。
- 编译与安装是否成功。
- 精度测试命令、结果及是否通过。
- 性能测试命令、对应 profiler CSV 路径、shape、有效运行次数、旧平均耗时和新平均耗时。
- 按 `老 / 新 - 1` 计算的收益值和百分比。
- 明确结论：有收益、无明显收益、性能退化、精度失败或测试无效。
- 是否保留该修改，以及下一步计划。

如果确认存在性能收益，将本轮远程 `op_summary_*.csv` 下载到：

`D:\Works\260809mhc_pre_backward_a5\性能数据`

下载文件应使用包含实验序号、commit 短 ID 或优化名称的唯一文件名，避免覆盖历史数据。无收益、退化、精度失败或无效测试的 CSV 不要求下载，但仍必须在 `修改记录.md` 中如实记录远程路径和结论。

## 完成标准

一项优化只有同时满足以下条件才算完成：

1. 改动已形成独立 commit 并成功 push。
2. 远程环境拉取并确认了同一 commit。
3. 自定义算子包编译和安装成功。
4. 精度测试通过。
5. 完成 9 次或 10 次有效性能采样，并按统一口径计算平均值与收益。
6. `修改记录.md` 已完整记录本轮实验。
7. `pre反向性能.xlsx` 已写入对应 shape 的平均值。
8. 若有收益，对应 profiler CSV 已下载并妥善命名保存。

