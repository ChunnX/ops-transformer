# Project Directory

## Detailed Directory Level Introduction

> ### Some directories listed in this chapter are optional. Refer to actual deliverables. Especially **single operator directories**, deliverables vary in different scenarios, as described below
>
> - If the op_host directory is missing, it may be calling another operator's op_host implementation. Refer to that operator's op_api or op_graph directory source code for calling logic; or the Kernel may not have an Ascend C implementation yet. If needed, developers are welcome to refer to [Contribution Guide](../../../CONTRIBUTING_en.md) to contribute this operator.
> - If the op_kernel directory is missing, it may be calling another operator's op_kernel implementation. Refer to that operator's op_api or op_graph directory source code for calling logic; or the Kernel may not have an Ascend C implementation yet. If needed, developers are welcome to refer to [Contribution Guide](../../../CONTRIBUTING_en.md) to contribute this operator.
> - If the op_api directory is missing, this operator does not support aclnn invocation.
> - If the op_graph directory is missing, this operator does not support graph mode invocation.

```text
├── cmake                                               # Project engineering compilation directory
│   ├── aclnn_ops_transfomer.h.in                       # aclnn summary header file template
│   └── ...
├── common                                              # Project common header files and common code
│   ├── CMakeLists.txt
│   ├── inc                                             # Common header file directory
│   └── src                                             # Common code directory
├── experimental                                        # User custom operator storage directory
│   ├── attention                                       # Optional, user-developed attention type operator directory
│   │   └── CMakeLists.txt
│   └── ...                                        
│    
├── ${op_class}                                         # Operator category, such as attention, ffn, gmm type operators
│   ├${op_name}                                         # Operator project directory, ${op_name} represents operator name (lowercase underscore form)
│   │   ├── CMakeLists.txt                              # Operator cmakelist entry
│   │   ├── README.md                                   # Operator introduction document
│   │   ├── docs                                        # Operator document directory
│   │   │   └── aclnn${OpName}.md                       # Operator aclnn interface introduction document, ${OpName} represents operator name (upper camel case form)
│   │   ├── examples                                    # Operator invocation sample directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Operator aclnn invocation sample
│   │   │   └── test_geir_${op_name}.cpp                # Operator geir invocation sample
│   │   ├── op_graph                                    # Graph fusion-related implementation
│   │   │   ├── CMakeLists.txt                          # op_graph side cmakelist file
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file, implementing operator data type derivation
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition, used for graph optimization and fusion phase operator identification
│   │   │   └── fusion_pass                             # Operator fusion rule directory
│   │   ├── op_host                                     # Host-side implementation
│   │   │   ├── CMakeLists.txt                          # Host-side cmakelist file
│   │   │   ├── config                                  # Optional, binary configuration file, automatically generated if not configured
│   │   │   │   ├── ${soc_version}                      # Binary information of operator configured on NPU, ${soc_version} represents NPU model
│   │   │   │   │   ├── ${op_name}_binary.json          # Operator binary configuration file
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # Operator SimplifiedKey configuration information
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library, defining basic operator information such as name, input/output, data types
│   │   │   ├── ${op_name}_infershape.cpp               # Optional, InferShape implementation, deriving output shape based on operator shape. If not configured, output shape is the same as input shape
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # Optional, Tiling optimization for certain sub-scenarios, ${sub_case} represents sub-scenario. If this file does not exist, the operator has no specific Tiling strategy for the corresponding sub-scenario
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # Optional, header file for Tiling implementation in ${sub_case} sub-scenario
│   │   │   ├── ${op_name}_tiling.cpp                   # Optional, if this file does not exist, there is no Tiling implementation for the corresponding scenario (dividing tensors into multiple small blocks, distinguishing data types for parallel computation)
│   │   │   ├── ${op_name}_tiling.h                     # Optional, header file for Tiling implementation
│   │   │   └── op_api                                  # Optional, operator aclnn implementation file directory, automatically generated if not configured
│   │   │       ├── aclnn_${op_name}.cpp                # Operator aclnn interface implementation file
│   │   │       ├── aclnn_${op_name}.h                  # Operator aclnn interface implementation header file
│   │   │       ├── ${op_name}.cpp                      # Operator l0 interface implementation file
│   │   │       ├── ${op_name}.h                        # Operator l0 interface implementation header file
│   │   │       └── CMakeLists.txt
│   │   │── op_kernel                                   # AI Core operator Device-side Kernel implementation
│   │   │   ├── ${sub_case}                             # Optional, directory used by ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file, ${model} represents user-defined file name suffix, usually Tiling template name
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # Optional, TilingKey file, defining the Key of Tiling strategy, identifying different partitioning methods. If not configured, the operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}_tiling_data.h                # Optional, TilingData file, storing Tiling strategy related configuration information such as block size, parallelism. If not configured, the operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file, containing main function and scheduling logic
│   │   │   └── ${op_name}.h                            # Kernel implementation file, defining Kernel header file, including function declarations, structure definitions, logic implementation
│   │   └── tests                                       # Operator test case directory
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # Optional, UT test cases, develop corresponding cases according to actual situations
│   └── ...
├── docs                                                # Project related document directory
├── examples                                            # End-to-end operator development and invocation samples
│   ├── add_example                                     # AI Core operator sample directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file 
│   │   ├── examples                                    # Operator usage sample directory
│   │   ├── op_graph                                    # Operator graph composition related directory
│   │   ├── op_host                                     # Operator information library, Tiling, InferShape related implementation directory
│   │   ├── op_kernel                                   # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── CMakeLists.txt
│   └── README.md                                       # Project sample introduction document
├── scripts                                             # Script directory, containing custom operator, Kernel build related configuration files
├── tests                                               # Project-level test directory
├── CMakeLists.txt                                      # Project engineering cmakelist entry
├── CONTRIBUTING.md                                     # Project contribution guide file
├── LICENSE                                             # Project open-source license information
├── OAT.xml                                             # Configuration script, repository tool usage, for checking whether License is compliant
├── README.md                                           # Project engineering overview document
├── SECURITY.md                                         # Project security statement file
├── build.sh                                            # Project engineering compilation script
├── classify_rule.yaml                                  # Component division information
├── install_deps.sh                                     # Project dependency package installation script
├── requirements.txt                                    # Project third-party dependency packages
└── version.info                                        # Project version information
```
