# Contribution Guide #

Developers are welcome to experience this project and contribute before contributing to the community. For details, see[cann-community](https://gitcode.com/cann/community)	Understand the Code of Conduct, sign the CLA agreement, and understand the contribution process of the source code repository.

When preparing local code and submitting a PR, developers need to pay attention to the following points:

1.  When submitting the PR, please fill in the business background, purpose, and solution of the PR according to the PR template.
2.  If the modification involves new features, interfaces, configuration parameters, or code process, the solution must be discussed in the Issue first to prevent your code from being rejected. If you are not sure whether this change can be classified as a "simple bug fix", you can also submit an issue for solution discussion.

Developers contribute in the following scenarios:

## 1. Contribution new operators ##

The operator development contribution process is as follows:

![算子开发贡献流程](./docs/zh/figures/算子开发贡献流程.png)	

If you want to design and implement new operators based on the NPU, please provide your ideas and design solutions in the Issue. The complete contribution process is as follows:

### 1. Create an issue requirement. ###

Newly created`Requirement|需求建议`Class Issue, and clarify the design scheme of the new additional operator. Generally, an issue contains the following information:

 *  **Background information**
 *  **Value/Function**
 *  **Design scheme**

Please comment on the submitted issue.`/assign @yourself`Claim the task.

### 2. Requirement review ###

The SIG team will assign the committer to review the issue and provide the modification suggestions. After the modification is complete, set @ in Issue to the committer.

if a request is accept,[SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md)	The appropriate operator classification path will be assigned to you (e.g.`experimental/attention`), please submit the contributed operator to`experimental`Indicates the operator category directory.

### 3. Submit the PR. ###

The following table lists the deliverables of the simplified ecosystem operator.

```
${op_class}                                          #Operator Category
├── ${op_name}                                       #Operator name
│   ├── ${op_name}.cpp                               #Kernel implementation file of the operator.
│   └── tests                                        
│   │   ├── test_${op_name}.py                       #Operator test file
│   ├── CMakeLists.txt                               #Operator compilation configuration file
│   ├── README.md                                    #Operator README Document
```

Requirements for PR inventory:

 *  Code deliverables: Operator Kernel implementation, operator test files, and development process reference must be provided.[fast_kernel_launch_example](examples/fast_kernel_launch_example/README.md)	.
 *  Document deliverables: The README document is mandatory. Other documents can be provided as required. Document Writing Template and Specifications[Document Contribution Guide](docs/CONTRIBUTING_DOCS.md)	.
 *  Compliance Check:
    
     *  Whether the code complies with the[C++ Programming Specifications](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)	"
     *  Check whether the code is compiled successfully.
     *  Check whether the Markdown document syntax complies with the specifications.
 *  Contribution Directory: Submit the contribution to the specified directory according to the comments of the SIG members.`experimental/${op_class}`For details, see the rules for storing existing operator files.
 *  PR submission: approved`git`Submit the PR of the target branch and check whether the PR title is clear and whether the PR description is standard. (Specify the change content and reason, and whether the change is associated with the corresponding issue.) 2. Whether the CLA is signed.

If you want to contribute project standard operators whose deliverables are richer than ecosystem operators, including Kernel and Tiling implementation, see.[Appendixes](#appendixes)	.

### 4. CI access control ###

Through the comments`compile`The command triggers the open source warehouse access control and modifies the open source warehouse access control based on the CI test result. Currently, the CI access control includes the following check items:

 *  Code compilation
 *  Static check (If the code check is incorrectly reported, submit it to the SIG members for shielding.)
 *  UT test
 *  Smoke test

After the access control is passed, click @ Assigned Committer in the associated issue.

### 5. Committer review ###

The committer will provide review comments after the review. Modify the committer based on the comments. After the review is complete, @ the assigned committer.

### 6. Integrate the maintainer. ###

Mark the committer after the committer passes the review.`/lgtm`Label. Maintainer will conduct a final review within 1 day and will label after confirming that there are no problems`/approve`Labels are incorporated into the PR.

## 2. Operator bug fixing ##

If you find some operator bugs in this project and want to fix them, you are welcome to create an issue for feedback and tracking.

You can follow the[Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务)	Guide New`Bug-Report|缺陷反馈`Class Issue describes the bug, and then assigns the issue to you by entering "/assign" or "/assign @yourself" in the comment box.

## 3. Operator optimization ##

If you want to enhance the generalization or optimize the performance of some operators in this project, you are welcome to contribute to the operator optimization.

You can follow the[Submit issue/handle issue task](https://gitcode.com/cann/community#提交Issue处理Issue任务)	Guide New`Requirement|需求建议`Class Issue describes the optimization points and provides your design scheme. Then enter "/assign" or "/assign @yourself" in the comment box to assign the issue to you for tracking optimization.

## 4. Document error correction ##

If you find errors in the operator document, create an issue to provide feedback and fix the errors. For details, see.[Document Contribution Guide](docs/CONTRIBUTING_DOCS.md)	.

You can follow the[Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务)	Guide New`Documentation|文档反馈`Class Issue indicates the problem with the corresponding document, and then assigns the issue to you by typing "/assign" or "/assign @yourself" in the comment box to correct the corresponding document description.

## 5. Help resolve others' issues. ##

If you have a solution to the problem encountered by others in the community, you are welcome to comment on the issue to help others solve the problem and pain points and jointly optimize the usability.

If the code of the corresponding issue needs to be modified, you can enter /assign or /assign @yourself in the Issue comment box to assign the issue to you and track and solve the problem.

## Appendixes ##

The project standard operator deliverables are as follows:

```
${op_class}                                          #Operator Category
├── ${op_name}                                       #Operator name
│   ├── op_host                                      #Operator definition and Tiling implementation
│   │   ├── ${op_name}_def.cpp                       #Operator definition file
│   │   ├── ${op_name}_tiling.cpp                    #Implementation file of the Tiling operator
│   │   └── CMakeLists.txt
│   ├── op_kernel                                    #Operator Kernel Directory
│   │   ├── ${op_name}.cpp                           #Kernel entry file, which contains the main function and scheduling logic.
│   │   ├── ${op_name}.h                             #Kernel implementation file, which defines the kernel header file, including the function description, structure definition, and logic implementation.
│   │   ├── ${op_name}_tiling_data.h                 #TilingData file, which stores configuration information about the Tiling policy.
│   │   └── ${op_name}_tiling_key.h                  #TilingKey file, which defines the key of the Tiling policy and identifies different division modes.
│   ├── CMakeLists.txt                               #Operator compilation configuration file. Retain the original file.
│   └── README.md                                    #Operator Description Document
│   └── tests                                        #Operator test file
│   │   ├── ut                                       #Operator UT test file
```

Requirements for PR inventory:

 *  Code deliverables: Tiling implementation of the op_host operator, Kernel implementation of the op_kernel operator, and UT test file of the operator must be provided. For details about the development process, see.[Operator Development Guide](docs/zh/develop/aicore_develop_guide.md)	.
 *  Document deliverables: The README document is mandatory. Other documents can be provided as required. For details about the document writing template and specifications, see.[Document Contribution Guide](docs/CONTRIBUTING_DOCS.md)	.
 *  Compliance Check:
    
     *  Whether the code complies with the[C++ Programming Specifications](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)	Check whether the standard operator basic programming specifications are complied with.
     *  Check whether the code is compiled successfully.
     *  Check whether the syntax of the Markdown document complies with the specifications.
 *  Contribution Directory: Submit the contribution to the specified directory according to the comments of the SIG members.`experimental/${op_class}`. For details, see the rules for storing existing operator files.
 *  PR submission: approved`git`Submit the PR of the target branch. Check whether the PR title is clear and whether the PR description is standard. (Specify the change content and reason, and whether the change is associated with the corresponding issue.) 2. Whether the CLA is signed.

