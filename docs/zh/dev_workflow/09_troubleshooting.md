# 09 常见问题

## 脚本存在，但使用 `$HOME/ops-transformer` 时提示不存在

`$HOME` 属于当前登录用户，不代表仓库所在目录。例如以 `root` 登录时 `$HOME` 为 `/root`，但仓库可能位于 `/home/<user>/cann/ops-transformer`。

先进入实际仓库，然后从 Git 自动解析路径：

```bash
cd "/home/<user>/cann/ops-transformer"
export OPS_REPO="$(git rev-parse --show-toplevel)"
export OPS_BASE="$(dirname "$OPS_REPO")"
export OPS_SCRIPT="$OPS_REPO/scripts/dev_workflow/remote_operator.sh"
export OPS_WORKSPACE="$OPS_BASE/ops-workspace"

"$OPS_SCRIPT" check-env --output-dir "$OPS_WORKSPACE/environment"
cat "$OPS_WORKSPACE/environment/remote_environment.txt"
```

若仓库位于 `/home/x00961093/cann/ops-transformer`，报告将位于：

```text
/home/x00961093/cann/ops-workspace/environment/remote_environment.txt
```

## `.run` 包无法在远端安装或运行

依次检查：

```bash
uname -m
npu-smi info
sha256sum -c SHA256SUMS
cat build-info.txt
```

常见原因：

- 本地与远端 CPU 架构不兼容。
- `--soc` 与远端 SoC 不匹配。
- 编译用 CANN 与远端 CANN ABI 不兼容。
- 上传中断或文件损坏。

解决方向：用与远端匹配的构建容器，或改在远端构建算子包。

## 安装后找不到算子或动态库

打开新 shell 并按顺序加载：

```bash
source /usr/local/Ascend/cann/set_env.sh
source "<custom-install>/set_env.bash"

echo "$ASCEND_CUSTOM_OPP_PATH"
echo "$LD_LIBRARY_PATH"
```

检查 runner 的动态库解析：

```bash
ldd <runner-path> | grep 'not found'
```

不要通过全局覆盖大量系统路径来掩盖缺失依赖，应确认缺失库属于 CANN、自定义算子还是 runner。

## 更新算子包后仍像在运行旧版

- 终止长期运行的 runner 进程。
- 打开新 shell。
- 加载新 commit 安装目录中的 `set_env.bash`。
- 输出当前 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`。
- 确认结果目录中的 `git-commit.txt` 是新 commit。

## runner 何时必须重新编译

如果下列任一项变更，重新编译：

- aclnn 接口签名或头文件。
- 输入输出数量、dtype 或属性。
- 链接库名称或 ABI。
- CANN 主要版本或 CPU 架构。

只改 Kernel 内部或 Tiling 且接口/ABI 不变时，通常可以重用 runner。

## 远端代码与算子包 commit 不一致

从产物元数据读取 commit，并切换纯运行工作区：

```bash
export OPS_COMMIT="$(cat "$OPS_ARTIFACT_DIR/git-commit.txt")"
cd "$OPS_REPO"
git fetch origin --prune
git switch --detach "$OPS_COMMIT"
test "$(git rev-parse HEAD)" = "$OPS_COMMIT"
```

如果 commit 不在远端 Fork，回到本地执行：

```bash
git push -u origin HEAD
```

## Git 推送要求用户名或失败

HTTPS 需要在当前机器配置 GitCode 凭据；SSH 需要将当前机器公钥添加到 GitCode 账号。不要把 Token 直接写入 remote URL、Markdown 或 shell 历史。

检查远端：

```bash
git remote -v
```

期望：

```text
origin    https://gitcode.com/xwqtju/ops-transformer.git
upstream  https://gitcode.com/cann/ops-transformer.git
```

## rebase 冲突

```bash
git status
# 修复冲突文件
git add <resolved-files>
git rebase --continue
```

放弃本次 rebase：

```bash
git rebase --abort
```

完成 rebase 后推送个人分支：

```bash
git push --force-with-lease origin HEAD
```

不要使用裸 `--force`。

## 上传或下载中断

rsync 支持增量续传，重新执行原命令即可。续传后必须再执行：

```bash
sha256sum -c SHA256SUMS
```

## 返回总入口

回到 [算子本地构建与远端运行操作手册](README.md)。
