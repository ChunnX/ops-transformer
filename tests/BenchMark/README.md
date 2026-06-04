# ND2NZ

 - FA BenchMark 工程
 - 用于看护FA Common层接口性能，包括 MTE CUBE VEC
 - 支持 V121 V122 仿真运行

## 1、CANN环境
安装cann包（V122 V121）
- source /usr/local/Ascend/cann/set_env.sh

## 2、项目结构

### 2.1、目录结构
- common：同FA代码仓 common代码，被测试接口的实现
- esl：ESL model存放目录，目前支持 121 122 esl_model
- src：case/性能测试代码
- peon_v121_v122.py：带宽计算脚本
- run_test.py：运行脚本
- batch_com.sh：算子编译脚本
- batch_sim.sh：ESL仿真脚本
- batch_run.sh：批跑脚本

#### 2.2、[batch_run.sh](v122_WORK/batch_run.sh)

```text
查看帮助：

bash batch_run.sh --help

```

#### 2.3、[build_com.sh](v122_WORK/build_com.sh)

```text
查看帮助：

bash batch_com.sh --help

脚本入参：（默认使用脚本中的默认值）
    - CANN_PATH      - CANN 安装路径（默认：/home/00949060/Ascend/v122_b150_patch）
    - WORK_DIR       - 算子编译工作目录（默认：v122_WORK）
    - CCE_FILES      - 待编译的 CCE 文件（多个用空格分隔，默认：main.cce DoExpTest.cce）
    - BIN_NAME       - 编译输出可执行文件名（默认：test）
    - TYPE           - 算子类型：vec、cube（默认：vec）
    - AICORE_ARCH    - SOC VERSION（默认：920r1）
```

#### 2.5、[build_sim.sh](v122_WORK/build_sim.sh)

```text
查看帮助：

bash batch_sim.sh --help

脚本入参：（默认使用脚本中的默认值）
    - CANN_PATH      - CANN 安装路径（默认：/home/00949060/Ascend/v122_b150_patch）
    - ESL_DIR        - ESL 仿真运行目录（默认：v122_ESL）
    - BIN_NAME       - 编译输出可执行文件名（默认：test）
    - CSV_FILE       - case文件
    - CASE_NUM       - 要运行的case的Idx
```

## 3、执行方式

<ol>
<li>安装CANN包，source环境变量</li>
<li>执行batch_run.sh脚本</li>
</ol>

根目录下创建 esl 目录，传入 eslmodel (121 或者 122)

修改 batch_run.sh 配置：
CANN_INSTALL_PATH 改为cann包目录
WORK_DIR
KERNEL_TYPE     KERNEL 类型 cube vec 或者无 代表mix

```text
ESL仿真：bash batch_run.sh <仿真model版本> <输入csv>
示例：bash batch_run.sh Ascend910_9691 testcase_nValue.csv mte/Test_CopySingleMatrixNDToNZ
   
```