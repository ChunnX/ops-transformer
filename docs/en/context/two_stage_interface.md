# Two-Stage Interface

When calling operator APIs based on the single-operator API execution method, it is typically divided into "two stages", with the style as follows:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

The first-stage interface aclxxXxxGetWorkspaceSize must be called first to calculate how much workspace memory is needed during this API call process. After obtaining the required workspaceSize, apply for NPU memory according to workspaceSize, and then call the second-stage interface aclxxXxx to execute computation.

Here "aclxx" represents the operator interface prefix, such as aclnn; and "Xxx" represents the corresponding operator type, such as the Add operator.

> Note:
>
>- Workspace refers to the temporary memory required by the API on the AI processor to complete computation, in addition to input/output.
>- The second-stage interface aclxxXxx(...) cannot be called repeatedly. The following calling pattern will cause an exception:

  ```cpp
  aclxxXxxGetWorkspaceSize(...)  
  aclxxXxx(...)   
  aclxxXxx(...)
  ```
