# vllm-ascend 容器环境搭建完整步骤

## 步骤一：宿主机清除代理

```bash
export http_proxy=""
export https_proxy=""
export no_proxy="localhost,127.0.0.1"
```

## 步骤二：基于镜像创建容器

```bash
docker run --name tyy_b120 -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /dev:/dev \
    -v /dev/nvme1n1p2:/dev/nvme1n1p2 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy= \
    -e https_proxy= \
    quay.io/ascend/vllm-ascend:v0.20.2rc1
```

> 注：原指南镜像 `ddfa90673e90` 不存在，选用了 `quay.io/ascend/vllm-ascend:v0.20.2rc1`

## 步骤三：进入容器

```bash
docker exec -it tyy_b120 bash
```

## 步骤四：配置代理与 pip 镜像源

```bash
# 配置代理（按需）
HOST_IP=xxx
export http_proxy="http://${HOST_IP}:3128"
export https_proxy=${http_proxy}
export ftp_proxy=${http_proxy}

# 清除 git 代理和 insteadOf 重写
git config --global http.sslVerify false
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global --unset url.https://gh-proxy.org/https://github.com/.insteadof
git config --global --unset url.https://mirror.ghproxy.com/https://github.com/.insteadof

# 配置 pip 镜像源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 步骤五：卸载镜像自带的 vllm 和 vllm-ascend

```bash
pip uninstall vllm vllm_ascend -y
```

## 步骤六：克隆代码仓库

```bash
cd /home
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/ader47/vllm-ascend.git
git clone https://gitcode.com/yrewzjsx/memcache.git
git clone https://gitcode.com/Ascend/memfabric_hybrid
```

> vllm-ascend 克隆超时，改用 `git clone --depth 1`，后台 nohup 克隆成功

## 步骤七：切换 vllm 到 v0.19.0

```bash
cd /home/vllm
git fetch --depth 1 origin tag v0.19.0
git checkout v0.19.0
```

## 步骤八：源码安装 vllm（可编辑模式）

```bash
cd /home/vllm
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
```

## 步骤九：源码安装 vllm-ascend（可编辑模式）

```bash
cd /home/vllm-ascend
pip install -v -e .
```

### 编译失败排查与修复（关键问题）

**问题1**：`TILINGKEY_PAR_COMPILE=1` 导致 opc 并行编译冲突 `Another process is using this dir`

**修复**：修改 `build.sh` 限制并行度 + 禁用 `TILINGKEY_PAR_COMPILE`

```bash
# 限制编译并行度为1（完全串行）
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend/csrc/build.sh

# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done

# 彻底清理 build 目录后重新安装
rm -rf /home/vllm-ascend/csrc/build
cd /home/vllm-ascend
pip install -v -e . > /tmp/install_vllm_ascend_j1.log 2>&1
```

## 步骤十：监控编译进度

```bash
# 检查安装是否完成
grep INSTALL_EXIT_CODE /tmp/install_vllm_ascend_j1.log

# 查看编译进度
grep -o '\[[0-9]*/533\]' /tmp/install_vllm_ascend_j1.log | tail -1

# 检查错误数
grep -c "Another process" /tmp/install_vllm_ascend_j1.log
grep -c "FAILED" /tmp/install_vllm_ascend_j1.log

# 查看最新日志
tail -3 /tmp/install_vllm_ascend_j1.log
```

## 环境信息

- 服务器：121.37.53.41 (SSH端口 10001)
- NPU：910B3 x 8
- 容器：tyy_b120
- 基础镜像：quay.io/ascend/vllm-ascend:v0.20.2rc1
- Python：3.11.15
- vllm：0.19.0+empty
- vllm-ascend：0.1.dev1 (源码可编辑安装)
