# 03 发布与上传产物

## 目标

将 `.run` 包、Git commit、构建信息和 SHA256 校验值组织成一个独立发布目录，并上传到远端运行机。

## 参数

```bash
export OPS_REMOTE="<user>@<remote-host>"
export OPS_REMOTE_ROOT="/home/<user>/ops-workspace"
```

`OPS_REMOTE_ROOT` 必须是远端绝对路径。

## 创建产物目录

```bash
cd "$OPS_REPO"
export OPS_COMMIT="$(git rev-parse HEAD)"
export OPS_ARTIFACT_DIR="/tmp/ops-transformer-artifacts/$OPS_OPERATOR/$OPS_COMMIT"
mkdir -p "$OPS_ARTIFACT_DIR"
```

确认 `build_out` 中只有要发布的目标包，再复制明确的文件：

```bash
find build_out -maxdepth 1 -type f -name '*.run' -ls
cp "build_out/<exact-package-name>.run" "$OPS_ARTIFACT_DIR/"
```

不建议盲目使用 `cp build_out/*.run`，因为目录中可能残留旧版本或其他 SoC 产物。

## 记录元数据

```bash
git rev-parse HEAD > "$OPS_ARTIFACT_DIR/git-commit.txt"
git branch --show-current > "$OPS_ARTIFACT_DIR/git-branch.txt"
git status --short --branch > "$OPS_ARTIFACT_DIR/git-status.txt"
uname -m > "$OPS_ARTIFACT_DIR/build-architecture.txt"
```

创建 `build-info.txt`，至少写入：

```text
operator=<operator_name>
soc=<soc_version>
vendor=<vendor_name>
cann_version=<CANN version>
build_command=<complete build command>
```

生成校验值：

```bash
cd "$OPS_ARTIFACT_DIR"
sha256sum ./*.run > SHA256SUMS
```

检查最终内容：

```bash
find "$OPS_ARTIFACT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
cat "$OPS_ARTIFACT_DIR/git-commit.txt"
cat "$OPS_ARTIFACT_DIR/SHA256SUMS"
```

## 上传

先建立远端目录：

```bash
ssh "$OPS_REMOTE" \
  "mkdir -p '$OPS_REMOTE_ROOT/artifacts/$OPS_OPERATOR/$OPS_COMMIT'"
```

再使用 rsync 上传：

```bash
rsync -av --progress \
  "$OPS_ARTIFACT_DIR/" \
  "$OPS_REMOTE:$OPS_REMOTE_ROOT/artifacts/$OPS_OPERATOR/$OPS_COMMIT/"
```

rsync 不可用时可使用：

```bash
scp -r "$OPS_ARTIFACT_DIR" \
  "$OPS_REMOTE:$OPS_REMOTE_ROOT/artifacts/$OPS_OPERATOR/"
```

## 成功判据

远端能查到正确 commit 目录：

```bash
ssh "$OPS_REMOTE" \
  "find '$OPS_REMOTE_ROOT/artifacts/$OPS_OPERATOR/$OPS_COMMIT' -maxdepth 1 -type f -printf '%f\n'"
```

## 下一步

登录远端后执行 [04 远端环境检查](04_remote_environment.md)。
