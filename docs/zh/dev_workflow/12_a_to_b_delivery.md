# 12 服务器 A 开发、服务器 B 一键编译运行

## 最终链路

```text
服务器 A：AI 开发 + 任务文件 + commit + push
                       ↓ GitCode HTTPS
服务器 B：一条命令拉取精确 commit + 编译 + 安装 + 可选运行 + 日志归档
                       ↓ 手工复制
服务器 A：读取 B 的 summary/build/run 日志并继续修复
```

服务器 B 使用 Ascend950PR、`--soc=ascend950` 和 `/usr/local/Ascend/cann-9.2.0/set_env.sh`。B 不开发代码，也不需要 GitCode SSH 写权限。

## A 的交付内容

每个任务必须同时提交：

- 算子源码和测试；
- `scripts/dev_tasks/<task-name>/task.sh`；
- 可选的任务说明 `docs/zh/dev_tasks/<task-name>.md`。

从模板创建任务文件：

```bash
mkdir -p scripts/dev_tasks/<task-name>
cp scripts/dev_workflow/task_template.sh scripts/dev_tasks/<task-name>/task.sh
```

任务文件必须设置 `TASK_NAME`、`OPERATOR`、`SOC`、`CANN_ENV`、`TASK_HAS_RUNNER`，并定义：

- `task_build`：编译 `.run` 算子包；
- `task_select_package`：只输出本次安装包的绝对路径；
- `task_install`：安装到执行器给定的任务目录；
- `task_run`：仅当 `TASK_HAS_RUNNER=1` 时定义。

没有可运行 example 时设置 `TASK_HAS_RUNNER=0`。B 会在编译和安装成功后输出 `[NO_RUNNER]`，不会把缺少 runner 当成编译失败。

## A 提交和生成 B 的单行命令

```bash
git add <source-files> scripts/dev_tasks/<task-name>/task.sh
git commit -m "feat: <task summary>"
git push -u origin HEAD
git rev-parse HEAD
```

向 B 提供的命令固定为：

```bash
cd /home/x00961093/cann/ops-transformer && bash scripts/dev_workflow/server_b_task.sh --repo /home/x00961093/cann/ops-transformer --commit <完整40位commit> --task scripts/dev_tasks/<task-name>/task.sh
```

## B 自动完成的操作

1. 检查并用 `git stash push -u` 保存未提交和未跟踪文件。
2. 通过 HTTPS 执行 `git fetch origin --prune`。
3. 以 detached HEAD 切换到指定的完整 commit。
4. 验证任务限定为 Ascend950 和 CANN 9.2.0。
5. 执行任务定义的编译、安装和可选 runner。
6. 保存 commit、环境、NPU 信息、各阶段日志、SHA256 和退出码。
7. 生成不包含安装目录的 `.tar.gz` 小型诊断包。
8. 切回原分支或原 commit，并恢复原先 stash。
9. 只保留最近 5 次任务目录和诊断包。

若任务自身产生了 tracked 或 untracked 修改，执行器会把它们另存为一个 stash，再恢复用户原来的工作区。若恢复用户 stash 发生冲突，stash 会被保留并要求人工处理，不会自动覆盖。

## B 的结果目录

默认位置：

```text
/home/x00961093/cann/task-runs/run-<timestamp>-<commit-prefix>/
```

关键文件：

```text
metadata.txt
summary.log
build.log
build-exit-code.txt
install.log
install-exit-code.txt
run.log                    # 有 runner 时
run-exit-code.txt          # 有 runner 时
package-sha256.txt
final-exit-code.txt
```

命令结束时会打印结果目录、压缩包和查看 summary 的命令。

## B 手工返回错误

优先复制：

```bash
cat <result-dir>/summary.log
cat <result-dir>/metadata.txt
cat <result-dir>/build.log
cat <result-dir>/install.log
cat <result-dir>/run.log 2>/dev/null || true
```

如果日志较长，至少返回诊断压缩包路径、完整 commit、失败阶段、退出码和日志末尾：

```bash
tail -200 <result-dir>/<failed-phase>.log
```

## 安全边界

- B 只通过公开仓库 HTTPS 拉取，不配置 SSH 推送。
- 任务文件是 A 在指定 commit 中提交的可信 Bash 代码；B 只执行 A 明确给出的 commit。
- 执行器不运行 `git reset --hard`，不删除用户 stash。
- 自动清理仅限显式任务根目录中名称为 `run-*` 的旧任务，并保留最近 5 次。
