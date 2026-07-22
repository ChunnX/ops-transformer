# Operator Debugging and Tuning

## Debugging and Positioning (AI Core Operators)

During operator execution, if operator execution failure, precision anomalies, or other issues occur, you can print information at various stages, such as Kernel intermediate results, for problem analysis and positioning.

### 1. Host-Side Log Acquisition Method

* **plog Acquisition**

   After program execution ends, you can view logs by default under "$HOME/ascendc/log". The host log file storage path is as follows:

   ```text
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   Enable the environment variable ASCEND_SLOG_PRINT_TO_STDOUT to display log output directly on screen (1: enable screen display, 0: disable screen display). The configuration sample is as follows:

   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   For log-related introduction, refer to [Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0001.html). For environment variable introduction, refer to [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

* **aclnn Exception Error Information Acquisition**
   
   Obtain exception information during aclnn interface invocation through the aclGetRecentErrMsg interface (refer to [acl API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0005.html)). The usage method is as follows:

   ```cpp
   printf(aclGetRecentErrMsg());
   ```

   The printed error information sample is as follows:

   ```text
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTennsor.self.
   ```

### 2. Kernel Debugging

Common debugging methods are as follows:

* **printf**

  This interface supports printing Scalar type data, such as integers, characters, Boolean, and so on. For detailed introduction, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html) in "Operator Debugging API > printf".
  
  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print the current core calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of a specified Tensor, and also supports printing custom additional information, such as the current line number. For detailed introduction, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html) in "Operator Debugging API > DumpTensor".
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

For complex scenario problem positioning, such as operator deadlock, GM/UB access out-of-bounds, and other scenarios, you can use **step-by-step debugging**. For specific operations, refer to the [msDebug](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/mskl_quick_start.md) operator debugging tool.

## Performance Tuning

### Method 1 (For Atlas A2/A3 Series Products)

During operator execution, if execution precision degradation, abnormal memory usage, or other issues occur, you can analyze operator runtime stage indicator data (such as throughput, memory usage, latency) through the [msProf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msopprof_quick_start.md) performance analysis tool to determine the problem root cause and optimize accordingly.

This chapter uses the `AddExample` custom operator as an example, mainly introducing the two commonly used methods in operator tuning: on-board performance data collection and pipeline diagram simulation. By collecting on-board runtime pipeline indicators to analyze operator Bound scenarios, understanding simulation pipeline diagrams facilitates optimizing operator internal pipelines.

1. Prerequisites.

   After completing operator development and compilation, assuming the aclnn interface invocation method is used, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/` directory.

2. Collect performance data.

   When you need to collect on-board runtime pipeline indicators for the operator, enter the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op ./test_aclnn_add_example
   ```
   
   The collection results are in the project `examples/add_example/examples/build/bin/OPPROF_*` directory. After collection is completed, the following information is printed:
   
    ```text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector             
    Task Duration(us): 97.861954 
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```

   Task Duration is the current operator Kernel latency, and Block Dim is the current operator execution core count.

   For detailed pipeline indicators of the operator, refer to the `ArithmeticUtilization` file under `OPPROF_*`, which contains the current pipeline proportion. For detailed introduction, refer to [msProf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msopprof_quick_start.md) in "Performance Data Files > msprof op > ArithmeticUtilization (cube and vector type instruction latency and proportion)" chapter.

3. Collect simulation pipeline diagram.
   
   Before using the msProf tool for operator simulation tuning, execute the following command to configure environment variables.

   ```bash
   export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH 
   ```

   Modify the above environment variables according to the actual CANN software package installation path and AI processor model.
   
   Then enter the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
   ```

   The collection results are in the project `$PWD/pipeline_auto/OPPROF_**` directory.
   The pipeline-related file path is `OPPROF**/simulator/visualize_data.bin`, which can be viewed using the [MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/latest/progressiveknowledge/index.html) tool.
   
### Method 2 (For Ascend 950PR)

During operator development, if execution precision degradation, abnormal memory usage, or other issues occur, you can analyze the operator instruction pipeline through the [CANN Simulator](./cann_sim.md) simulation tool to determine the problem root cause and optimize accordingly.

This chapter uses the `AddExample` custom operator as an example, mainly introducing the use of the simulation tool. How to perform precision and performance tuning through the simulation tool.

1. Prerequisites.

   After completing operator development and compilation, assuming the aclnn interface invocation method is used, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/` directory.

2. Execute simulation command to generate simulation data.

   ```text
   cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
   ```

   The simulation results are in the project `examples/add_example/examples/build/bin/cannsim_*` directory. The pipeline-related file is:

   ```text
   trace_core0.json
   ```

3. Enter "chrome://tracing" in the Chrome browser address bar, and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For specific parameter introduction, refer to the [Simulation Result Parsing](./cann_sim.md#simulation-result-parsing-description) chapter in CANN Simulator.
