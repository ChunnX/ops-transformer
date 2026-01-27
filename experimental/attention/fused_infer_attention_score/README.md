# FA算子

##  概述
本样例的FA算子实现使用自定义算子工程，其kernel包含两个模板：GQA模板以及MLA模板。本工程默认使用GQA模板，示例通过msprof工具采集了模板的性能数据。
##  支持的AI处理器
| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 目录结构介绍
```
|-- experimental
|   |-- attention                            // 通过aclnn调用的方式调用fa算子
|   |   |-- CMakeLists.txt                   // 算子编译文件
|   |   |-- common                           // 算子公共层文件
|   |   |   |-- CMakeLists.txt
|   |   |   |-- op_host                      // 算子tiling侧公共文件
|   |   |   `-- op_kernel                    // 算子kernel侧公共文件
|   |   `-- fused_infer_attention_score       
|   |       |-- CMakeLists.txt
|   |       |-- README.md                    // 本demo使用说明
|   |       |-- op_host                      // FA算子tiling侧文件
|   |       |-- op_kernel                    // FA算子kernel侧入口文件
|   |       |-- run.sh                       // 一键式运行脚本
|   |       `-- tests                        // 测试文件

```

## 功能说明

- 算子功能：
本样例算子实现的是FA算子，不支持mask，其数学表达式为：
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">FusedInferAttentionScore</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">layout</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">query</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  <tr><td align="center">key</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  <tr><td align="center">value</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">atten_out</td><td align="center">bnsd、bsnd</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
  </table>

不支持任何高阶特性（mask、pse、pa、fd、actual_seq等）不支持入图
## 编译运行 
- 配置环境变量  
以命令行方式下载样例代码，master分支为例 
```bash
cd ${git_clone_path}/experimental/attention/fused_infer_attention_score
cd ${git_clone_path}/experimental/attention/common
```
根据当前环境上CANN开发套件包（toolkit包+ops包）的安装方式，选择对应配置环境变量的命令。  
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/cann
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/cann
    ```


- 编译与安装自定义算子包
```bash
# 切换到工程根目录
cd ${git_clone_path}  
# 编译样例算子run包
bash build.sh --pkg  --experimental --soc=ascend910b --ops=fused_infer_attention_score  
#安装自定义算子run包
./build_out/cann-ops-transformer-${vendor_name}-${arch}_linux.run
```

- 编译+执行aclnn接口样例，采集样例性能：
```bash
# 切换到fused_infer_attention_score目录
cd ${git_clone_path}/experimental/experimental/attention/fused_infer_attention_score/
# 编译+执行aclnn接口+采集性能数据
bash run.sh
# 切换aclnn用例性能数据目录(在/experimental/experimental/attention/fused_infer_attention_score目录下会生成output目录，里面存放了性能数据)
cd ${git_clone_path}/experimental/experimental/attention/fused_infer_attention_score/output
```

本demo提供了preload流水优化对比：通过修改**attention/common/op_kernel/arch32/fia_kernel_nonquant.h** 和 **/common/op_kernel/arch32/fia_kernel_nonquant_mla.h** 中的**PRELOAD_NUM_ACTUAL**变量，然后再次运行run.sh即可得到修改后的性能数据。该变量值为0时表示无preload，该变量值为2时表示开启2轮preload流水优化。
下表为preload开启与关闭的性能对比

<table>
  <tr>
    <td align="center">fa算子类型</td>
    <td align="center">输入shape</td>
    <td align="center">layout</td>
    <td align="center">dtype</td>
    <td align="center">无preload算子执行时间</td>
    <td align="center">preload2轮算子执行时间</td>
  </tr>

  <tr>
    <td align="center">GQA</td>
    <td align="center">q:(24,64,128,128)<br>k:(24，32，512，128)<br>v:(24，32，512，128)</td>
    <td align="center">bnsd</td>
    <td align="center">float16</td>
    <td align="center">646.76us</td>
    <td align="center">377.20us</td>
  </tr>

  <tr>
    <td align="center">MLA</td>
    <td align="center">q:(24,128,64,128)<br>k:(24，64，512，128)<br>v:(24，64，512，128)<br>rope: 64</td>
    <td align="center">bnsd</td>
    <td align="center">float16</td>
    <td align="center">828.30us</td>
    <td align="center">583.24us</td>
  </tr>
</table>



