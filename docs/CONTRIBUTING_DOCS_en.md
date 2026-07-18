# Document Contribution Guide #

You are welcome to contribute to this project documentation. High-quality documentation is critical to the success of the project. This guide will help you efficiently submit compliant documents.

## Contribution scope ##

We welcome any contribution that can improve the quality of the document, including but not limited to:

 *  Error corrections and improvements: Fix typos, syntax errors, bad code examples, outdated information, or dead links.
 *  Clarification and optimization: Make the description clearer and easier to understand, optimize the sentence structure, and supplement the background knowledge.
 *  Content Supplement: Add usage examples, API documentation, FAQs, best practices, caution notes, and more to existing features.
 *  New content creation: Write new chapters or tutorials for new functions, such as operator README and API introduction documents. If you have any questions, you are advised to create an issue first.
 *  Localized translation: Help us translate or proofread documents in other languages.
 *  Style and navigation: Improve the layout, readability, and navigation structure of the document site.

## Contribution process ##

1.  **Preparations**
    
     *  Determine the task: If there are any document issues, create an issue. The recommended label type is as follows:`[Documentation|文档反馈]`and provide a detailed description. Determine the document issues to be resolved based on the existing issue list.
     *  Claiming a task: Comment on the corresponding issue.`/assign @yourself`to indicate that you will process it and avoid duplication of effort.
2.  **Document modification**
    
     *  Select a branch: Download the source code from the master branch or another tag branch to the local host.
     *  Follow the format:
        
         *  The Markdown format is recommended for this project.
         *  Follow the existing writing style of the project.
         *  Static resources such as images are stored in the corresponding directory. For example, images are stored in the docs directory.`figures`Folder, which can be adjusted in special cases.
     *  Exercise caution when adding or deleting content: Keep the original line width and line feed conventions when modifying content.
3.  **Commit Changes**
    
     *  Atomized commit: Each commit should focus on a single change. For example, "Fix spelling mistakes in the xx guide" and "Update sample code for API reference" should be submitted in two separate submissions.
     *  Write a clear submission:
        
        ```
        简短说明（不超过50字符）
        
        如有必要，在此处进行更详细描述。说明修改的原因和内容，而不是具体改了什么（代码本身会展示）。
        关联的 Issue: #123
        ```
4.  **Initiate a Pull Request.**
    
     *  Target Branch: Merge the PR into the target branch of the project.
     *  Title and Description:
        
         *  PR Title: The changes should be clearly outlined, for example:`[Docs] 修复快速入门中的配置示例`.
         *  PR Description: Specify your changes, motivation, and associated with the corresponding issue (use Closes \#123 or Fixes \#456).
     *  Preview Check: Check the effect of the document viewed locally or online in advance to ensure that the rendering is as expected.
     *  Awaiting review: Maintainers will review and may suggest changes. Please follow up the discussion in time.

## Writing Specifications ##

Before writing project documents, please read the following specifications. If you have any questions, please feel free to provide suggestions.

 *  Prerequisites: Learn the unified writing specifications provided by the CANN organization. For details, see.[CANN Document Writing Specifications](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)	.
    
     *  Document content requirements: This section describes the mandatory and optional document deliverables of the project.
     *  Directory structure specifications: describes the principles for dividing directories, such as Chinese and English management.
     *  Content element specifications: describes the rules for different writing elements, such as file names, titles, fonts, images, code blocks, and links.
 *  Precautions:
    
    In addition to the above rules, note the following:
    
     *  Tone: Use a friendly, professional, and neutral tone. For novices, avoid unnecessary jargon.
     *  Terminology: Maintain consistency of terms (e.g., "click" instead of "click"). Please refer to the Project Glossary, if any.
     *  Code example:
        
         *  Ensure that all code examples are operational and tested.
         *  Provides sufficient context and explanation.
         *  Specify the environment or prerequisites required for the code to run.
     *  Punctuation and formatting:
        
         *  When Chinese and English are mixed, use full-width punctuation, and the punctuation must conform to the Chinese and English context.
         *  Headers use the appropriate hierarchy (\#, \#\#, \#\#\#).
         *  Use lists and tables to organize complex information.
     *  Link: Use descriptive link text to avoid "click here" and ensure that the link resources are authentic and reliable.
     *  Picture:
        
         *  Common format: The .png format is recommended. The style should be the same as that of the existing image.
         *  Resolution and clarity: Clear and moderate size, avoiding blurring or overcompression.
         *  File size: It is recommended that the size of a single image be less than or equal to 10 MB.
     *  Copyright: All referenced images, documents, etc., please ensure compliance.

## Get Help ##

If you have any questions during the contribution process:

1.  Check existing documents: If the template or specification is incorrect, check the existing guide, API document, or README of the project first.
2.  Initiate a discussion: You can create an issue or leave a message in the related issue or PR.

## Operator README template ##

For the`experimental`The README operator is a mandatory document deliverable. You can refer to the simple template provided in this section and expand content based on this template.

 *  Document format: The Markdown file format is recommended. Native or HTML syntax is supported. Ensure that all syntax complies with official specifications.
 *  Function: This document describes the operator functions, implementation principles, parameter specifications, and operator invoking methods.
 *  Section title: The template section name (such as function description and parameter description) is preferred. The title level is \#\#. If there are special cases, add the title level in sequence. Chapters can be customized and expanded. Optional chapters are displayed as required.
 *  Content requirements: For the writing objectives and specifications of each chapter, see the following description. For ease of understanding,[AddExample](../examples/add_example/README.md)	The README operator is used as an example.

### Product Support ###

> **Writing specifications: It is recommended that the supported product models be listed in a table and marked with √. For details about the product models, see.**[Ascend Product Form Description](https://www.hiascend.com/document/redirect/CannCommunityProductForm)	.

| Product                                            |    Supported   |
|:-------------------------------------------------- |:--------------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term> |       √        |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |       √        |

### Function Description ###

> \[!NOTE\]
> 
> **Writing objective: Explain the operator functions, calculation principles, parameter specifications, invoking methods, and application scenarios.**
> 
> **Writing specifications: The unordered list is recommended. Generally, the following dimensions are included:**
> 
>  *  Operator function (mandatory): Please describe the function in one sentence.
>  *  Calculation formula (optional): For complex functions, you can use formulas to describe the operator implementation principle or the calculation process in different scenarios.
>  *  Other dimensions (optional): supports unordered list expansion. Customize the dimensions based on the site requirements, such as calculation examples and flowcharts.

 *  Operator function: performs tensor addition calculation.
 *  Calculation formula: $$y = x1 + x2 $$

### Parameter Description ###

> \[!NOTE\]
> 
> **Objective: To clarify the meanings, functions, and specifications of the parameters defined by the operator.**
> 
> **Writing rules: Use a table, including the following dimensions:**
> 
>  *  Parameter name: specifies the parameters in the operator definition file. The sequence must be the same, for example,`op_host/add_example_def.cpp`or the`op_graph/add_example_proto.h`.
>  *  Input/Output/Attribute: Specify the parameter location. By default, the parameter is mandatory. If the parameter is optional, the parameter is optional.
>  *  Description: describes the parameter meaning, function, and application scenario, including the mapping between the parameter and the preceding formula variables.
>  *  Data type: data type supported by the parameter. Generally, the tensor data type is`DT_XXX`Form. For the convenience of writing, do not bring`DT_`Prefix.
>  *  Data format: data layout mode supported by the parameter. The tensor format is as follows:`FORMAT_xxx`Form. For the convenience of writing, do not bring`FORMAT_`Prefix.
>  *  Other dimensions (optional): Table fields can be extended. Customize the table fields based on the site requirements, for example, shape specifications.

| Parameter name | Input/Output/Attribute | Description                                                                              | Data type                     | Data Format |
| -------------- | ---------------------- | ---------------------------------------------------------------------------------------- | ----------------------------- | ----------- |
| x1             | Entered                | Indicates the first tensor calculated by add_example, that is, in the formula.`x1`.      | FLOAT, FLOAT16, INT32         | ND          |
| x2             | Entered                | Indicates the second tensor calculated by add_example, that is, in the formula`x2`.      | Data type: consistent with x1 | ND          |
| y              | Output                 | Indicates the tensor of the add_example calculation result, that is, in the formula.`y`. | FLOAT, FLOAT16, INT32         | ND          |

### Constraint Description (Optional) ###

> \[!NOTE\]
> 
> **Objective: To describe the precautions for using operators, such as parameter combination constraints, application scenarios, impact on services, and operator performance or precision.**
> 
> **Writing Specifications: This chapter is optional and can be omitted unless otherwise specified. If yes, use an unordered list.**

None

### Invoking Description ###

> \[!NOTE\]
> 
> **Writing objective: Provide operator invoking methods. Try to copy and run sample code to facilitate quick verification.**
> 
> **Writing rules: It is recommended that the table form be used. If the content is complex, other forms can be used.**
> 
>  *  Invoking mode: This mode can be invoked in aclnn or graph mode, or can be customized. Provide at least one mode.
>  *  Sample code: In the`examples`The catalog provides sample code for calling, for example,`examples/test_aclnn_add_example.cpp`The file name format is test_$\{invoke_mode\}_$\{op_name\}, where $\{invoke_mode\} indicates the invoking mode and $\{op_name\} indicates the operator name.
>  *  Description: supplementary description of different invoking modes, such as the invoking scenario, invoking principle, and compilation and running guide. Customize the description based on the site requirements.

| Invoking mode  | Invoking Example                                                                       | Description                                                                                          |
| -------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| aclnn invoking | [test_aclnn_add_example](../examples/add_example/examples/test_aclnn_add_example.cpp)	 | Refer to[Operator Invoking](./zh/invocation/quick_op_invocation.md)	Compile and verify the operator. |

### Reference Resources (Optional) ###

> \[!NOTE\]
> 
> **Objective: Provide supplementary introductions to operator functions, specifications, and invoking, such as operator design documents (tiling/kernel design) and references.**
> 
> **Writing Specifications: This chapter is optional and can be omitted unless otherwise specified. If yes, use an unordered list.**

None

