# Operator debugging and optimization #

## Debugging and Locating (AI Core Operator) ##

If problems such as operator execution failure and precision exception occur during operator running, you can print information about each phase, such as kernel intermediate results, for fault analysis and location.

### 1. Obtaining Host Logs ###

 *  **Obtaining plogs**
    
    After the program execution is complete, you can view the host log file in $HOME/ascendc/log by default. The path for storing the host log file is as follows:
    
    ```
    $HOME/ascend/log/debug/plog/plog-pid_*.log
    ```
    
    Enable the environment variable ASCEND_SLOG_PRINT_TO_STDOUT to display logs on the screen. (1: enabled; 0: disabled). A configuration example is as follows:
    
    ```
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    ```
    
    For details about logs, see.[Log Reference](https://hiascend.com/document/redirect/CannCommunitylogref)	For details about environment variables, see.[Environment Variable Reference](https://hiascend.com/document/redirect/CannCommunityEnvRef)	.
 *  **Obtaining ACLnn Exception Error Information**
    
    Through the aclGetRecentErrMsg interface (see[ACL API (C)](https://hiascend.com/document/redirect/CannCommunityCppApi)	Obtain the exception information during the invoking of the aclnn interface. The usage method is as follows:
    
    ```
    printf(aclGetRecentErrMsg());
    ```
    
    The following is an example of the printed error information:
    
    ```
    [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTennsor.self.
    ```

### 2. Kernel debugging ###

Common debugging methods are as follows:

 *  **printf**
    
    This interface can print Scalar data, such as integer, character, and Boolean data. For details, see.[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi)	Operator Commissioning API > printf in.
    
    ```
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    tileNum_ = tilingData->tileNum;
    tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
    //Displays the block length calculated by the current core.
    AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
    ```
 *  **DumpTensor**
    
    This interface supports dump and specifying the content of a Tensor and printing customized additional information, such as the current line number. For details, see.[Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi)	Operator Commissioning API > DumpTensor in.
    
    ```
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    //Print zLocal Tensor information.
    DumpTensor(zLocal, 0, 128);
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    ```

For fault locating in complex scenarios, such as operator suspension and GM/UB out-of-bounds access, you can perform step-by-step debugging. For details, see.[msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)	Operator debugging tool.

## Performance Tuning ##

### Method 1 (for Atlas A2/A3 series products) ###

If the execution precision decreases and the memory usage is abnormal during operator running, you can run the following command:[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)	The performance analysis tool analyzes the indicator data in each running phase of the operator. (for example, throughput, memory usage, and time consumption) to determine the root cause of the problem and optimize the problem accordingly.

This chapter begins with`AddExample`This section uses custom operator as an example to describe two common methods for operator optimization: performance collection and pipeline diagram simulation. Analyze the operator Bound scenario by collecting various pipeline indicators when the operator is running on the board, and learn about the simulation pipeline diagram to optimize the internal pipeline of the operator.

1.  Prerequisites.
    
    After the operator development and compilation are complete, assume that the aclnn interface is used to invoke the operator, and the generated executable file (test_aclnn_add_example) is stored in the directory of this project.`examples/add_example/examples/build/bin/`.
2.  Collect performance data.
    
    To collect the serial indicators of the operator running on the board, go to the directory where the operator executable file is stored and run the following command:
    
    ```
    msprof op ./test_aclnn_add_example
    ```
    
    The collection result is in this project.`examples/add_example/examples/build/bin/OPPROF_*`Directory. After the collection is complete, the following information is printed:
    
    ```
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
    
    Task Duration indicates the kernel duration of the current operator, and Block Dim indicates the number of execution cores of the current operator.
    
    Pay attention to the detailed indicators of each operator.`OPPROF_*`Lower`ArithmeticUtilization`File, which contains the proportions of the current SNs. For details, see.[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)	For details, see section "Performance Data File > msprof op > ArithmeticUtilization (Time Consumption and Percentage of Cube and Vector Commands)" in.
3.  Collect the simulation flow chart.
    
    Before performing operator simulation optimization using the msProf tool, run the following commands to configure environment variables:
    
    ```
    export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    ```
    
    Modify the preceding environment variables based on the actual installation path of the CANN software package and AI processor model.
    
    Go to the directory where the operator executable file is stored and run the following command:
    
    ```
    msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
    ```
    
    The collection result is in this project.`$PWD/pipeline_auto/OPPROF_**`directory. The path of the SN-related files is`OPPROF**/simulator/visualize_data.bin`, you can use the[MindStudio Insight](https://www.hiascend.com/document/redirect/MindStudioInsight)	View the tool.

### Method 2 (for Ascend 950PR) ###

During operator development, if the execution precision decreases and the memory usage is abnormal, you can use the[CANN Simulator](./cann_sim_en.md)	The simulation tool analyzes the instruction pipeline of the operator, and then locates the root cause of the problem and optimizes the problem accordingly.

This chapter begins with`AddExample`The custom operator is used as an example to describe how to use the simulation tool. How to use the simulation tool to tune the precision and performance.

1.  Prerequisites.
    
    After the operator development and compilation are complete, if the aclnn interface is invoked, the generated operator executable file (test_aclnn_add_example) is stored in the directory of this project.`examples/add_example/examples/build/bin/`.
2.  Run the simulation command to generate the simulation data.
    
    ```
    cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
    ```
    
    Simulation results in this project`examples/add_example/examples/build/bin/cannsim_*`Directory. The SN-related files are as follows:
    
    ```
    trace_core0.json
    ```
3.  Enter chrome://tracing in the address box of the Chrome browser, and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For details about the parameters, see the CANN Simulator.[Simulation Result Analysis](./cann_sim_en.md#simulation-result-analysis-description)	Section.

