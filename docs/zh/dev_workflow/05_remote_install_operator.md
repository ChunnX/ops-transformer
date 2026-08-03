# 05 远端安装算子包

## 目标

校验上传的 `.run` 包，安装到独立开发目录，并加载自定义算子环境。

## 校验产物

```bash
cd "$OPS_ARTIFACT_DIR"
sha256sum -c SHA256SUMS
cat git-commit.txt
cat build-info.txt
```

任何 SHA256 检查失败都应停止安装并重新上传。

## 查看安装参数

```bash
chmod +x ./*.run
./<exact-package-name>.run --help
```

当前包支持自定义路径时，优先安装到个人目录：

```bash
export OPS_INSTALL_ROOT="$HOME/cann-operator-dev/$OPS_OPERATOR/$OPS_COMMIT"
mkdir -p "$OPS_INSTALL_ROOT"
./<exact-package-name>.run --install-path="$OPS_INSTALL_ROOT"
```

如果安装包不支持 `--install-path`，以 `--help` 及 [源码构建](../install/compile.md) 中当前分支的安装指导为准。不要在未确认安装范围时覆盖系统 CANN。

## 加载算子环境

查找安装生成的环境脚本：

```bash
find "$OPS_INSTALL_ROOT" -name set_env.bash -print
```

加载系统 CANN 和自定义算子：

```bash
source "$OPS_CANN_ENV"
source "<path-to-custom-set_env.bash>"
```

检查路径：

```bash
echo "$ASCEND_CUSTOM_OPP_PATH"
echo "$LD_LIBRARY_PATH"
```

将实际环境脚本路径记录到：

```bash
printf '%s\n' "<path-to-custom-set_env.bash>" \
  > "$OPS_ARTIFACT_DIR/custom-env-path.txt"
```

## 切换新版算子包

新包使用新 commit 目录安装，不覆盖上一版：

```text
~/cann-operator-dev/<operator>/<old-commit>/
~/cann-operator-dev/<operator>/<new-commit>/
```

切换时打开新 shell，重新 source 新版环境脚本，避免长期进程继续占用旧动态库。

## 成功判据

- SHA256 校验全部通过。
- 算子安装目录与 `OPS_COMMIT` 一一对应。
- `ASCEND_CUSTOM_OPP_PATH` 或相关动态库路径包含本次安装位置。

## 下一步

首次部署或接口变更时，继续执行 [06 远端编译调用程序](06_remote_build_runner.md)。仅 Kernel 实现或 Tiling 策略变更且接口/ABI 不变时，可直接进入 [07 运行、调试与采集](07_remote_run_and_profile.md)。
