# Operator Invoking Mode #

## Overview ##

Operators can be invoked in multiple ways. This chapter describes`AddExample`This section uses an operator invoking as an example to describe the operator invoking and execution process.

> Note: The aclnn interface mode is recommended. If this mode is not supported, the composition mode can be used to invoke the operator.

 *  API invoking by aclnn (recommended): The API mode of the aclnnXxx operator is used. (A set of C-based APIs, requiring no IR definition) Invoke the operator.
 *  Invoking in graph mode: Invokes an operator in Intermediate Representation (IR) mode.

## aclnn invoking ##

### Invoking Process ###

![原理图](../figures/aclnn调用.png)	

### Sample Code ###

`AddExample`The sample code for invoking the operator through the aclnn interface is as follows (for details, see)[test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp)	) is for reference only. The API calling process of other operators is similar. Replace it with the actual aclnn API. Before calling this API, set environment variables as prompted.

Note: To invoke other operators in this project, access the corresponding operators.`examples`In the test_aclnn_$\{op_name\}.cpp directory, $\{op_name\} indicates the operator name.

```
int main()
{
    //1. Invoke the ACL to initialize the device/stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    //2. Construct the input and output based on the API.
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    std::vector<int64_t> selfXShape = {32, 4, 4, 4};
    std::vector<float> selfXHostData(2048, 1);
    ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* selfY = nullptr;
    void* selfYDeviceAddr = nullptr;
    std::vector<int64_t> selfYShape = {32, 4, 4, 4};
    std::vector<float> selfYHostData(2048, 1);
    ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclDataType::ACL_FLOAT, &selfY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {32, 4, 4, 4};
    std::vector<float> outHostData(2048, 1);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //3. Invoke the CANN operator library API. Change the API name to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    //4. Invoke the first segment of aclnnAddExample API.
    ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    //5. Apply for device memory based on the workspaceSize calculated by the first interface.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    //6. Invoke the aclnnAddExample interface in the second segment.
    ret = aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);

    //7. (fixed) Wait for the end of the task synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    //8. Obtain the output value and copy the result in the memory on the device side to the host side. Modify the result based on the API definition.
    PrintOutResult(outShape, &outDeviceAddr);

    //9. Release the aclTensor, which needs to be modified based on the API definition.
    aclDestroyTensor(selfX);
    aclDestroyTensor(selfY);
    aclDestroyTensor(out);

    //10. Release device resources.
    aclrtFree(selfXDeviceAddr);
    aclrtFree(selfYDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    //11. Deinitialize the ACL.
    aclFinalize();
    return 0;
}
```

### Compilation and running ###

> Note: For implemented operators (non-custom operator) in a project, you can log in to the root directory.[build.sh](../../../build.sh)	Run the operator directly. For details, see.[Local authentication](./quick_op_invocation_en.md#local-authentication)	.

1.  Prerequisites. Please refer to this project[Compiling the source code](./quick_op_invocation_en.md#compiling-the-source-code)	Compile and deploy the target operator.
2.  Create the CMakeLists file.
    
    Create the CMakeLists file in the same directory as test_aclnn_$\{op_name\}.cpp, with`AddExample`The operator is used as an example. Modify the operator based on the site requirements.
    
    ```
    cmake_minimum_required(VERSION 3.14)
    #Setting the project name
    project(ACLNN_EXAMPLE)
    
    #Setting the C++ Compilation Standard
    add_compile_options(-std=c++11)
    
    #Set the compilation output directory to the bin folder in the current directory.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    
    #Setting compilation options for debug and release modes
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
    
    #Obtains the LD_LIBRARY_PATH environment variable.
    if(NOT DEFINED ENV{LD_LIBRARY_PATH})
       message(FATAL_ERROR "LD_LIBRARY_PATH environment variable is not set")
    endif()
    set(LD_LIB_PATH "$ENV{LD_LIBRARY_PATH}")
    
    #Split the path list and find the path that contains /vendors/ (required only for custom operator)  
    string(REPLACE ":" ";" LD_LIB_LIST "${LD_LIB_PATH}")
    set(TARGET_PATH "")
    foreach(path ${LD_LIB_LIST})
       #Matches paths that contain /vendors/ (location-insensitive).
       if(path MATCHES "/vendors/")
           set(TARGET_PATH "${path}")
           break()
       endif()
    endforeach()
    if(NOT TARGET_PATH)
       message(FATAL_ERROR "未在LD_LIBRARY_PATH中找到包含/vendors/的路径")
    endif()
    if(TARGET_PATH MATCHES "/vendors/([^/]+)")
       set(TARGET_SUBDIR "${CMAKE_MATCH_1}")
    else()
       message(FATAL_ERROR "在路径${TARGET_PATH}中未找到/vendors/的直接子目录")
    endif()
    
    #Add an executable file (replace it with the actual operator executable file) and specify the *.cpp file invoked by the operator.
    add_executable(test_aclnn_add_example              
                  test_aclnn_add_example.cpp)
    
    #ASCEND_PATH (Replace the CANN software package directory based on the site requirements.)
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")      
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    
    #Setting the header file path
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
       ${INCLUDE_BASE_DIR}
       ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/include    #Required only by custom operator
       #${INCLUDE_BASE_DIR}/aclnn # Required only by the built-in operator
    )
    
    #Link the required dynamic library.
    target_link_libraries(test_aclnn_add_example PRIVATE             #Replacing the Actual Operator Executable File
       ${ASCEND_PATH}/lib64/libascendcl.so
       ${ASCEND_PATH}/lib64/libnnopbase.so
       ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so   #Required only by custom operator
       #${ASCEND_PATH}/lib64/libopapi_transformer.so # Required only by built-in operators
    )
    
    #Install the target file to the bin directory.  
    install(TARGETS test_aclnn_add_example DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```
3.  Create the run.sh file.
    
    Create the run.sh file in the directory at the same level as test_aclnn_$\{op_name\}.cpp.`AddExample`The operator is used as an example. Modify the operator based on the site requirements.
    
    ```
    if [ -n "$ASCEND_INSTALL_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi
    
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    
    rm -rf build
    mkdir -p build 
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    cd bin
    ./test_aclnn_add_example            #Replace it with the actual executable name of the operator.
    ```
4.  Run the run.sh file. Run the following command in the path of the run.sh file:
    
    ```
    bash run.sh
    ```
    
    The current execution path is used by default.`/build/bin`The executable file test_aclnn_add_example is generated in the. The running result is as follows:
    
    ```
    mean result[2046] is 2.000000
    mean result[2047] is 2.000000
    ```

## Graph mode call ##

### Invoking Process ###

![原理图](../figures/IR调用.png)	

### Sample Code ###

`AddExample`The sample code for invoking an operator in graph mode is as follows. For details, see.[test_geir_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp)	) is for reference only. The invoking procedures of other operators are similar. Replace it with the actual operator prototype.

To invoke other operators in this project, access the corresponding operators.`examples`In the test_geir_$\{op_name\}.cpp directory, $\{op_name\} indicates the operator name.

```
int main() {
    //1. Create a graph object.
    Graph graph(graphName);

    //Figure 2 Initializing global compilation options
    Status ret = ge::GEInitialize(globalOptions);

    //3. Create an AddExample operator instance.
    auto add1 = op::AddExample("add1");

    //4. Define the graph input and output vectors.
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    //5. Prepare the input data.
    std::vector<int64_t> xShape = {32,4,4,4};
    //6. Process variable assignment in macro expansion mode.
    ADD_INPUT(1, x1, inDtype, xShape);
    ADD_INPUT(2, x2, inDtype, xShape);
    ADD_OUTPUT(1, y, inDtype, xShape);

    outputs.push_back(add1);

    //7. Set the input and output operators of the graph object.
    graph.SetInputs(inputs).SetOutputs(outputs);

    //8. Create a session object.
    ge::Session* session = new Session(buildOptions);

    //9. Adding a session
    ret = session->AddGraph(graphId, graph, graphOptions);

    //10. Running chart
    ret = session->RunGraph(graphId, input, output);

    //11. Release resources.
    GEFinalize();

    return 0;
}
```

### Compilation and running ###

> Note: For implemented operators (non-custom operator) in a project, you can log in to the root directory.[build.sh](../../../build.sh)	Run the operator directly. For details, see.[Local authentication](./quick_op_invocation_en.md#local-authentication)	.

1.  Prerequisites. Please refer to this project[Compiling the source code](./quick_op_invocation_en.md#compiling-the-source-code)	Compile and deploy the target operator.
2.  Create the CMakeLists file.
    
    Create the CMakeLists file in the same directory as test_geir_$\{op_name\}.cpp,`AddExample`The following is an example of the operator. Modify the operator based on the site requirements.
    
    ```
    cmake_minimum_required(VERSION 3.14)
    
    #Setting the project name
    project(GE_IR_EXAMPLE)
    
    if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
       get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
    elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    
    set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")
    
    message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")
    
    file(GLOB files CONFIGURE_DEPENDS
        test_geir_add_example.cpp         
    )
    
    #Add an executable file. (Replace it with the actual executable file of the operator.)
    add_executable(test_geir_add_example ${files})      
    
    find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
    find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
    find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")
    
    #Link the required dynamic library.
    target_link_libraries(test_geir_add_example PRIVATE      
        ${GRAPH_LIBRARY_DIR}
        ${GE_RUNNER_LIBRARY_DIR}
        ${GRAPH_BASE_LIBRARY_DIR}
    )
    
    #Setting the header file path
    target_include_directories(test_geir_add_example PRIVATE       
        ${FWK_INCLUDE_DIR}/graph/
        ${FWK_INCLUDE_DIR}/ge/
        ${ASCEND_PATH}/opp/built-in/op_proto/inc/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_PATH}/compiler/include
    )
    ```
3.  Create the run.sh script.
    
    Create the run.sh file in the directory at the same level as test_geir_$\{op_name\}.cpp.`AddExample`The operator is used as an example. Modify the operator based on the site requirements.
    
    ```
    if [ -n "$ASCEND_INSTALL_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi
    
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    
    rm -rf build
    mkdir -p build 
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    make
    ./test_geir_add_example                  #Replace it with the actual executable name of the operator.
    ```
4.  Run the run.sh script. Run the following command in the path of the run.sh file:
    
    ```
    bash run.sh
    ```
    
    By default, the current execution path is used.`/build/bin`The executable file test_geir_add_example is generated in the. The running result is as follows:
    
    ```
    INFO - [XIR]: Finalize ir graph session success
    ```

