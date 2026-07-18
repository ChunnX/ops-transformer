# Brief Introduction #

CANN Simulator is a SoC-level simulation tool for operator development. It is used to analyze the precision and performance data (such as instruction execution) of AI tasks running on the AI simulator in each phase. This tool helps users perform in-depth performance optimization. In this way, R&D personnel can obtain the verification effect and performance feedback almost the same as that of real chips even if the chip resources are insufficient or cannot be obtained.

# Main Function #

The tool is binary compatible with running on board (The same kernel can be executed on both the emulation and AI processors.) The main uses are as follows:

 *  Precision simulation: Outputs bit-level precision results to help users verify the operator precision.
 *  Performance simulation: Outputs an instruction pipeline diagram, helping users locate operator performance bottlenecks.

# Preparation Before Use #

## Constraints ##

 *  Recommended environment configuration: 16-core CPU, 32 GB or more memory.
 *  Ensure that the running user has the read and write permissions for all paths described in this document.
 *  To ensure security and minimize permissions, you are advised to run this tool as a common user. Do not use a high-privilege account such as root.
 *  This tool depends on the CANN software package. Before using this tool, install the CANN software package without installing the driver and firmware. Run the source command to run the set_env.sh environment variable file of the CANN. To ensure security, do not modify the environment variables involved in set_env.sh after the source command is executed.
 *  Users must comply with the principle of least privilege. For example, other users cannot write the files input to the tool. In some functions that have stricter security requirements, group users must not write the files input to the tool.
 *  This tool is a development tool and is not recommended in the production environment.
 *  The emulation function of the tool supports only the single-card scenario. The multi-card scenario cannot be emulated. In the code, only 0 card can be set. If the visible card number is changed, the emulation will fail.
 *  The simulation environment supports only AI Core computing operators. Operators of the MC2 and HCCL types are not supported.
 *  Currently, the CANN Simulator is in the new version and supports only the Ascend 950PR chip. It is recommended that the simulator run with a 16-core CPU and 32 GB or more memory be configured.
 *  Currently, ARM environment simulation is not supported.

## Environment preparation ##

The CANN Simulator is integrated in the CANN toolkit package. For details, see.[Environment Deployment]()	Install the software package.

# Quick start #

The following is marked with[add_examples](../../../examples/add_example/)	This section describes operator simulation in detail.

## Compiling an Operator ##

 *  Reference[Operator Invoking](../invocation/quick_op_invocation_en.md)	Compile and install the add_example operator.

```
#Note: Go to the root directory of the project and run the following compilation command. The command is for reference only. For details, see the operator invoking description.
bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example
#Installing the custom operator Package
./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
```

 *  Reference[aclnn invoking](../invocation/op_invocation_en.md#aclnn-invoking)	Compile test_aclnn_add_example.cpp and compile the executable file test_aclnn_add_example.

## Run the emulation command. ##

```
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
```

The emulation tool execution log file is stored in the examples/add_example/examples/build/bin/cannsim_\* directory. The execution log file is

```
cannsim.log
```

The following information is displayed in the log file of the emulator:

```
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
```

## Querying the performance sequence number ##

Simulation performance pipeline files are stored in the \`examples/add_example/examples/build/bin/cannsim_\*/report directory of this project. The pipeline-related files are as follows:

```
trace_core0.json
```

Enter chrome://tracing in the address box of the Google Chrome browser and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For details about the parameters, see section "Simulation Result Parsing."

# Simulation Execution Description #

## Command Function ##

Execute the application in a simulated environment.

## Command Format ##

cannsim record \[options\] user_app --user_options

## Parameter Description ##

Table 1 Simulation execution parameters

| Parameters                               | Optional/Mandatory | Description                                                                                                                                                                                                                                           |
| ---------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -s or --soc_version \[options\] argument | Mandatory.         | Specifies the version of the target chip, for example, Ascend950.                                                                                                                                                                                     |
| -o or --output \[options\] argument      | Optional.          | Indicates the path of the generated file. The value can be an absolute path or a relative path. The user who runs the tool must have the read and write permissions. If you do not specify a path, data is saved in the current directory by default. |
| -g or --gen-report \[options\] argument  | Optional.          | Enables whether to perform automatic analysis after the simulation is complete and generate an analysis report. By default, the system does not automatically parse the data.                                                                         |
| user_app                                 | Mandatory.         | Executable file of the operator.                                                                                                                                                                                                                      |

\--user_options\|Optional\|Operational parameters of the operator executable file.\|

## Use Example ##

1.  Operator development and compilation are complete.
2.  Run the emulation command. For details, see the following example.
    
    ```
    #Method 1: Enable simulation and save the output to the ./output directory. /path/to/app is the operator program.
    $ cannsim record /path/to/app -o ./output -s Ascend950
    
    #Method 2: Enable simulation and generate a report for subsequent performance analysis.
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```
3.  After the command is executed, a folder named cannsim_\{timestamp\}_$\{user_app\} is generated in the default path or the specified output directory. The following is an example of the folder structure:
    
    ```
    ├─cannsim_{timestamp}_${user_app}
    ├── cannsim.log
    ```
4.  You can obtain operator execution results and compare precisions. The results are displayed in the cannsim.log file. The following is an example:
    
    The following output is only an example of the direct adjustment precision comparison result of a single Ascend C operator. The actual output prevails.
    
    ```
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```
5.  View the operator instruction pipeline diagram and parse the diagram based on the simulation result.

# Simulation Result Analysis Description #

## Command Function ##

Generates a visualized instruction pipeline diagram.

## Command Format ##

cannsim report \[options\]

## Parameter Description ##

Table 1 Parameters for parsing simulation results

| Parameters                           | Optional/Mandatory | Description                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -e or --export \[options\] argument  | Mandatory.         | Directory for storing original result files. The directory must be the result directory generated after simulation execution. The directory must be an absolute or relative path at the cannsim_\{timestamp\}_$\{user_app\} layer. The tool execution user must have the read and write permissions.                                                                              |
| -o or --output \[options\] argument  | Optional.          | Parsing result output directory, which can be set to an absolute or relative path. The execution user must have the read and write permissions. If no path is specified, data is saved in the current directory by default. If the generated result file has the same name as an existing file, the original file will be overwritten.                                            |
| -n or --core-id \[options\] argument | Optional.          | Specifies the ID of the core that generates the instruction sequence number. If the ID is not specified, the instruction sequence number of core 0 is generated by default. The configuration format is as follows: Generate the sequence number of all cores. Set this parameter to all. Specifies the core ID range, for example, '0-1'. Specifies the core ID, for example, 5. |

## Use Example ##

1.  Perform operator simulation by referring to Simulation Execution and compare the output examples to ensure that the corresponding results are correct.
2.  Run the simulation result parsing command. For details, see the following test cases.
    
    ```
    #A performance analysis report is generated in the current directory. (By default, only core 0 is analyzed.)
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} 
    
    #Generate performance analysis reports for cores 0, 1, 11, and 12 in the specified directory.
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} -o /path/to/report -n ‘0-1, 11-12’
    ```
3.  After the command is executed, a JSON sequence file is generated in the directory specified by output. The following is an example of the output:
    
    ```
    trace_core0.json
    trace_core1.json
    ...
    ```
4.  To view the simulation result, enter chrome://tracing in the address box of the Google Chrome browser, drag the generated instruction pipeline diagram file (trace.json) to the blank area, and enter the shortcut key on the keyboard. (W: zoom in, S: zoom out, A: shift left, D: shift right) You can view it.![指令流水图](../figures/指令流水图.png)	
    
    Table 2 Key fields

```
|字段名|字段含义|
| --- | --- |
|VECTOR|向量运算单元。|
|SCALAR|标量运算单元。|
|Cube|矩阵乘运算单元。|
|MTE1|数据搬运流水，数据搬运方向为：L1 ->{L0A/L0B, UBUF}。|
|MTE2|数据搬运流水，数据搬运方向为：{DDR/GM, L2} ->{L1, L0A/B, UBUF}。|
|MTE3|数据搬运流水，数据搬运方向为：UBUF -> {DDR/GM, L2, L1}、L1->{DDR/L2}。|
|FIXP|数据搬运流水，数据搬运方向为：FIXPIPE L0C -> OUT/L1。|
|FLOWCTRL|控制流指令。|
|ICACHELOAD|查看未命中的ICache。|
```

# Querying Help Information #

## Command function ##

Query the tool help information.

## Command Format ##

To query the tool help information, run the following command:

```
cannsim --help
```

To query the help information about the record subcommand of the tool, run the following command:

```
cannsim record --help
```

To query the help information about the report subcommand of the tool, run the following command:

```
cannsim report --help
```

## Parameter Description ##

None

## Use Example ##

1.  Log in to the server on the host side.
2.  Run the following command:
    
    ```
    cannsim --help
    ```

## Output Description ##

```
usage: cannsim [-h] {record,report} ...

Command-line tool for performance simulation analysis on Ascend hardware.

positional arguments:
  {record,report}  Available commands
    record         Run user application in AscendOps simulation environment
    report         Generate performance analysis reports

options:
  -h, --help       show this help message and exit
```

