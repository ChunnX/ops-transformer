# 个人 Fork 开发与上游同步工作流

本文面向需要在一台或多台服务器上长期编译、调试、开发及性能采集 `ops-transformer` 算子的开发者。目标是同时做到：

- 开发提交安全地推送到个人云端仓库；
- 持续跟进官方 `master` 分支；
- 不在官方同步分支中混入个人开发提交；
- 避免把编译产物、日志和大型性能数据提交到 Git；
- 让同一个开发分支能在多台服务器之间可靠交接。

本文使用以下仓库地址：

| 名称 | 地址 | 用途 |
|---|---|---|
| `origin` | `https://gitcode.com/xwqtju/ops-transformer.git` | 个人 Fork，保存并分享个人分支 |
| `upstream` | `https://gitcode.com/cann/ops-transformer.git` | 官方主仓，获取上游更新 |

## 1. 首次配置

### 1.1 新服务器克隆个人 Fork

```bash
git clone https://gitcode.com/xwqtju/ops-transformer.git
cd ops-transformer
git remote add upstream https://gitcode.com/cann/ops-transformer.git
```

禁止向 `upstream` 误推送，同时保留正常的 fetch 地址：

```bash
git remote set-url --push upstream DISABLED
```

验证配置：

```bash
git remote -v
```

预期结果类似：

```text
origin    https://gitcode.com/xwqtju/ops-transformer.git (fetch)
origin    https://gitcode.com/xwqtju/ops-transformer.git (push)
upstream  https://gitcode.com/cann/ops-transformer.git (fetch)
upstream  DISABLED (push)
```

### 1.2 已克隆官方仓库时转换远端

仅在当前仓库的 `origin` 仍指向官方仓库时执行：

```bash
git remote rename origin upstream
git remote add origin https://gitcode.com/xwqtju/ops-transformer.git
git remote set-url --push upstream DISABLED
git fetch --all --prune
```

转换前先运行 `git status`，未提交的本地修改不会因为重命名远端而消失，但仍建议先创建开发分支并提交或暂存重要修改。

### 1.3 配置身份和提交检查

```bash
git config user.name "<your-name>"
git config user.email "<your-email>"
```

项目支持 pre-commit。Python 3.9 及以上环境可执行：

```bash
pip3 install pre-commit
pre-commit install
```

详细检查项参见 [Pre-commit 配置指导书](pre-commit_guide.md)。

## 2. 分支模型

建议只赋予各分支一种职责：

```text
upstream/master ──官方最新代码
       │
       └── origin/master ──个人 Fork 中的官方镜像，不承载个人开发
                    │
                    ├── feat/<operator>-<topic>
                    ├── fix/<operator>-<topic>
                    ├── perf/<operator>-<topic>
                    └── debug/<operator>-<topic>
```

基本规则：

1. 不直接在 `master` 上开发。
2. 一个独立功能、修复或实验使用一个分支。
3. 个人分支优先用 `rebase` 跟进上游，保持历史清晰。
4. 多人共同使用且不能改写历史的分支，用 `merge` 跟进上游。
5. 编译产物和采集数据不作为源码提交。

## 3. 日常开发流程

### 3.1 更新本地及个人 Fork 的 master

先确认工作区干净：

```bash
git status
```

再同步官方分支：

```bash
git fetch upstream --prune
git switch master
git merge --ff-only upstream/master
git push origin master
```

`--ff-only` 能避免在 `master` 上意外生成合并提交。如果命令失败，说明本地 `master` 已经出现了额外提交；不要直接强制覆盖，先按本文“异常恢复”章节检查。

### 3.2 创建任务分支

从已经同步的 `master` 创建分支：

```bash
git switch master
git switch -c feat/<operator>-<topic>
git push -u origin feat/<operator>-<topic>
```

示例：

```bash
git switch -c perf/grouped-matmul-splitk
git push -u origin perf/grouped-matmul-splitk
```

### 3.3 提交和云端备份

先检查实际改动，避免使用不加区分的 `git add .`：

```bash
git status --short
git diff
git add <source-file> <test-file> <document-file>
git diff --cached
git commit -m "perf: optimize grouped matmul split-k"
git push
```

开发中的阶段性成果也应形成可理解的小提交并及时推送。不要依赖未提交工作区作为唯一副本。

### 3.4 开发分支跟进官方更新

个人独占分支推荐使用 rebase：

```bash
git fetch upstream --prune
git switch feat/<operator>-<topic>
git rebase upstream/master
git push --force-with-lease origin feat/<operator>-<topic>
```

`--force-with-lease` 会在远端分支出现未获取的新提交时拒绝覆盖，比 `--force` 安全。不要使用裸 `git push --force`。

发生冲突时：

```bash
git status
# 编辑冲突文件，删除冲突标记并保留正确内容
git add <resolved-file>
git rebase --continue
```

需要放弃本次 rebase 时：

```bash
git rebase --abort
```

多人共享分支使用 merge：

```bash
git fetch upstream --prune
git switch <shared-branch>
git merge upstream/master
git push origin <shared-branch>
```

## 4. 多台服务器协作

Git 提交是服务器之间唯一可靠的源码交接单位。离开服务器 A 前：

```bash
git status
git add <files>
git commit -m "wip: save reproducible operator experiment"
git push
```

在服务器 B 上继续：

```bash
git fetch origin --prune
git switch <branch>
git pull --ff-only
```

如果服务器 B 尚无该本地分支：

```bash
git fetch origin --prune
git switch --track origin/<branch>
```

注意事项：

- 同一个分支不要在两台服务器上同时产生未协调的提交。
- 确实需要并行开发时，分别创建分支，例如 `<topic>-server-a` 和 `<topic>-server-b`，完成后再合并。
- 不使用共享文件系统直接复制 `.git` 目录。
- 环境差异用脚本、容器定义或文档记录，不提交账号口令和服务器私有路径。

## 5. 编译、调试和性能采集

首先加载与源码版本匹配的 CANN 环境，默认安装示例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

源码与 CANN 版本配套关系应以项目 README 和 release-management 仓库为准。

### 5.1 编译指定算子

先查看当前分支实际支持的参数：

```bash
bash build.sh --help
```

自定义算子包示例：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<operator_name> -j16
```

芯片参数常用取值：

| 产品 | `--soc` |
|---|---|
| Atlas A2 | `ascend910b` |
| Atlas A3 | `ascend910_93` |
| Ascend 950 | `ascend950` |

编译结果默认位于 `build_out/`。完整说明参见 [源码构建](../install/compile.md) 和 [build 参数说明](../install/build.md)。

### 5.2 调试构建和单元测试

调试构建：

```bash
bash build.sh --debug --ops=<operator_name> --soc=<soc_version>
```

Kernel 单元测试：

```bash
bash build.sh -u --opkernel --ops=<operator_name> --soc=<soc_version>
```

只编译测试、不自动执行：

```bash
bash build.sh -u --opkernel --noexec --ops=<operator_name> --soc=<soc_version>
```

仓库中的算子样例可通过 `--run_example` 执行，具体模式和参数以对应算子的 `examples/` 及 [快速入门](../../QUICKSTART.md) 为准。

### 5.3 仿真和现场调试

仿真、CPU/NPU 孰化、日志及 Dump 的支持范围随芯片和 CANN 版本变化。开始前阅读：

- [CANN 仿真器调试](../debug/cann_sim.md)
- [算子调试与性能采集](../debug/op_debug_prof.md)

每次采集至少记录以下元数据，建议保存为小型文本或 JSON：

```text
Git commit ID
Git branch
CANN version
Driver/Firmware version
SoC model
Build command
Run command and input shape/dtype
Relevant environment variables
Profiler command and configuration
```

获取当前提交标识：

```bash
git rev-parse HEAD
git status --short --branch
```

只有可复现的小型配置、采集脚本、结果摘要和分析结论适合提交 Git。原始大文件放到对象存储或团队存储，并在文档中记录地址、校验值和对应 commit ID。

## 6. 不应提交的内容

提交前执行：

```bash
git status --short
git diff --cached --stat
```

通常不应提交：

- `build/`、`build_out/` 等编译目录；
- `.run`、`.so`、`.a`、`.o` 等产物；
- profiler 原始数据、Dump、日志、core 文件；
- 临时测试数据、缓存、IDE 配置；
- Token、SSH 私钥、密码和包含敏感信息的环境文件。

如果某类本地文件只对个人环境有意义，优先加入 `.git/info/exclude`，避免修改项目共享的 `.gitignore`：

```bash
printf '%s\n' '<local-only-pattern>' >> .git/info/exclude
```

必须版本化的大文件应先与维护者确认是否采用 Git LFS；不要直接把大批二进制采集结果推入普通 Git 历史。

## 7. 提交前自检

检查本分支相对上游的提交和改动：

```bash
git fetch upstream --prune
git log --oneline upstream/master..HEAD
git diff --stat upstream/master...HEAD
```

运行提交检查：

```bash
pre-commit run --all-files
```

根据所改算子执行最小必要编译、单元测试和样例，并在提交或合并请求说明中记录：

- 测试环境及 SoC；
- 完整构建命令；
- 完整测试命令；
- 结果摘要；
- 已知限制。

## 8. 异常恢复

### 8.1 master 被误提交

先创建备份分支，确保提交不会丢失：

```bash
git switch master
git branch backup/master-$(date +%Y%m%d-%H%M%S)
git log --oneline --decorate -10
```

把个人提交迁移到开发分支后，再恢复 `master`：

```bash
git switch -c feat/recovered-work
git push -u origin feat/recovered-work
git switch master
git fetch upstream
git reset --hard upstream/master
git push --force-with-lease origin master
```

`reset --hard` 会删除当前工作区未提交改动，因此只能在确认备份分支和工作区状态后执行。

### 8.2 错误提交了大文件或敏感信息

立即停止继续推送。删除最新提交中的普通大文件可使用：

```bash
git rm --cached <file>
git commit --amend
```

如果敏感信息已经推到云端，仅删除文件不够：应立即轮换密钥，再联系仓库管理员清理历史。改写已共享历史会影响所有协作者，不应擅自执行。

### 8.3 临时切换任务但不想提交

短时保存可以使用：

```bash
git stash push -u -m "temporary work before switching task"
git switch <other-branch>
```

恢复：

```bash
git stash list
git stash pop
```

stash 只存放在当前服务器，不会同步到云端；跨服务器交接仍应提交并推送分支。

## 9. 推荐的最短日常命令序列

开始新任务：

```bash
git fetch upstream --prune
git switch master
git merge --ff-only upstream/master
git push origin master
git switch -c feat/<operator>-<topic>
```

保存开发成果：

```bash
git add <files>
git diff --cached
git commit -m "feat: <summary>"
git push -u origin HEAD
```

提交合并请求前：

```bash
git fetch upstream --prune
git rebase upstream/master
pre-commit run --all-files
git push --force-with-lease origin HEAD
```
