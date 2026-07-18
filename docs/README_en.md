# Project Documentation #

## Content Description ##

The key directory structure is as follows:

```
├── context                            #Public directory, which stores documents such as basic concepts, project directory introduction, and build parameter description.
│   ├── dir_structure.md    
│   ├── build.md               
│   └── ...
├── debug                              #Operator debugging and optimization document directory
│   ├── op_debug_prof.md
│   ├── ...
├── develop                            #Operator Development Document Directory
│   ├── aicore_develop_guide.md
│   ├── ...
├── figures                            #Picture Catalog
├── invocation                         #Operator Invoking Document Directory (including aclnn invoking and graph mode invoking)
│   ├──op_invocation.md
│   ├── ...
├── op_api_list.md                     #Full Operator Interface List (aclnn)
├── op_list.md                         #Full operator list
└── README.md
```

## Document Description ##

The following table lists all project documents. Obtain the documents as required.

| Document                                                          | Description                                                                                                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Operator List](zh/op_list.md)	                                   | This section describes all operators involved in the project.                                                                                                                        |
| [ACLN list](zh/op_api_list.md)	                                   | This section describes all operator APIs contained in the project. You can use these APIs to directly adjust operators.                                                              |
| [Environment Deployment](zh/context/quick_install.md)	            | This section describes how to set up the basic environment, including how to obtain and install software packages and third-party dependencies in different scenarios.               |
| [Operator Invoking](zh/invocation/quick_op_invocation.md)	        | This section describes how to compile source code and execute operators, including operator package compilation, operator sample execution, and UT execution in different scenarios. |
| [Operator development](zh/develop/aicore_develop_guide.md)	       | This section describes how to develop new operators based on the project, including operator prototype definition, Tiling implementation, and Kernel implementation.                 |
| [Operator Invoking Mode](zh/invocation/op_invocation.md)	         | This section describes multiple operator invoking modes and invoking processes, such as aclnn invoking and graph mode invoking.                                                      |
| [Operator debugging and optimization](zh/debug/op_debug_prof.md)	 | This section describes common operator debugging and optimization methods.                                                                                                           |

## Appendixes ##

| Document                                            | Description                                                                                                                             |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Basic Operator Concepts](zh/context/基本概念.md)	      | This section describes basic concepts and terms related to the operator field, such as quantization/sparse, data type, and data format. |
| [Build parameter description](zh/context/build.md)	 | This section describes the functions and parameters of the build.sh script.                                                             |

