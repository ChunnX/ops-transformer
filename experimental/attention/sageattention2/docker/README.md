# 容器化毕昇编译与运行指南（独立仓库交付形态）

## 设计契约

- 本算子目录是独立 Git 仓库 `ops-transformer` 的一部分，验收方/CI 拉取后**不存在任何仓库外部宿主目录**；
- 卷挂载全部为**相对路径**且只指向仓库内部目录（见下"挂载关系"），不存在 `../../../../../` 之类的仓库外部引用；
- 唯一外部依赖 CATLASS 经 `init.sh` 拉取固定 commit 内聚到 `docker/third_party/catlass`（gitignore），随仓库根本体挂载进入容器；
- 所有编译（C++ 宿主单测、设备核、OPP vendor 树、测试动态库）统一由容器内 **BiSheng** 编译器承载，
  `run.sh` 各目标在宿主侧仅做容器包装，宿主**唯一依赖 Docker（含 compose 插件）**。

## 前置条件

- 宿主安装 Docker 与 Docker Compose 插件；
- 宿主与容器网络畅通（首次拉取基础镜像与依赖包），或本地已有基础镜像缓存。

## 挂载关系（全部相对路径，源均为仓库内部）

| 源（相对 `docker/`） | 容器内路径 | 用途 |
| --- | --- | --- |
| `../../../..`（ops-transformer 仓库根） | `/work/ops-transformer` | 仓库根本体实时同步；容器工作目录为其下 `experimental/attention/sageattention2` |
| `./cache` | `/root/.cache` | pip/编译缓存持久化（gitignore） |
| `./out` | `/work/out` | 仿真日志等散件产物出口（gitignore，不进仓） |

CATLASS 不经挂载：`init.sh` 拉到 `docker/third_party/catlass`，处于仓库根挂载之内，
容器内路径即 `/work/ops-transformer/experimental/attention/sageattention2/docker/third_party/catlass`，
由 `run.sh` 导出 `CATLASS_ROOT` 供全部编译消费（与 `cmake/SageAttnDeps.cmake` 解析链同源）。

## 核心动作

```bash
cd ops-transformer/experimental/attention/sageattention2

# 1) 一键初始化（幂等）：CATLASS 固定 commit → docker/third_party/、.env、基础镜像校验
bash docker/init.sh
#    可选：把 golden 数据集复制到仓库根 golden/（位级比对基线，pytest 门禁需要）
#    bash docker/init.sh --golden <golden 数据集路径>

# 2) 门禁（自动包装进容器，容器内统一 BiSheng）
bash run.sh host      # C++ 宿主单测（BiSheng 宿主模式编译）
bash run.sh kernel    # 设备核 BiSheng 交叉编译（dav-3510）
bash run.sh python    # tiling/quant/twin/gqa Python 门禁（容器内 torch）
bash run.sh opp       # OPP vendor 树 BiSheng 出包 + 本地部署
bash run.sh all       # 全链

# 3) 交互调试
./docker/sage exec
```

## 文件清单

- `docker/Dockerfile`：CANN 9.2.0-beta.1 950 devel 基础镜像 + BiSheng + torch 配套；
- `docker/docker-compose.yml`：门禁服务编排（构建/运行，全部相对挂载）；
- `docker/docker-compose.npu.yml`：Atlas 950 真机环境设备与驱动透传叠加（`docker/sage` 检测驱动自动启用）；
- `docker/sage`：容器生命周期唯一入口（`up` / `exec` / `run` / `stop` 四个动词）；
- `docker/init.sh`：一键初始化（目录、.env、CATLASS 固定 commit、基础镜像、可选 golden）；
- `docker/.env.example`：compose 覆盖项模板；
- `docker/third_party/`、`docker/cache/`、`docker/out/`：初始化产物与缓存出口（gitignore，不入库）。

## 环境变量覆盖

- `CANN_IMAGE`：CANN 基础镜像，默认 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel`（与验收 CANN 9.2.0-beta.1 对齐）；
- `TORCH_SPEC` / `TORCH_NPU_SPEC`：torch 配套（默认 `torch==2.12.0` / `torch-npu==2.12.0`，置空可跳过安装、沿用基础镜像自带 torch）；
- `TORCH_INDEX`：torch CPU 轮子源（默认阿里云）；
- `SAGE_IMAGE`：产出镜像 tag（默认 `sageattn2-op-dev:latest`）；
- `CATLASS_REPO` / `CATLASS_COMMIT`：init.sh 拉取来源与固定 commit；
- `GOLDEN_SRC`：golden 数据集来源（等价 `--golden`）。

## 质量准出门禁

- `bash run.sh kernel`：容器内 `bisheng --npu-arch=dav-3510 -std=c++17 -xasc` 成功编译
  `op_kernel/sageattention2.cpp` → `build/sageattention2_bisheng.o`，退出码 0；
- `bash run.sh host`：BiSheng 宿主模式编译并运行 `test_aclnn_host` / `test_op_host` / `test_tiling_struct` 全部自检，退出码 0；
- `bash run.sh python`：`test_tiling_struct.py` / `test_quant.py` / `test_kernel_twin.py` / `test_gqa_chunk.py` 全部通过（golden 25 桶位级判据不动）；
- `bash run.sh opp`：BiSheng 出包 `build/opp/vendors` 并部署 `build/opp_installed`，完整性自检通过，退出码 0。
