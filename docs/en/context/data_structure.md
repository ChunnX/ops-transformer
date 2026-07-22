# Data Structure

This chapter provides the basic data structures required for calling CANN operator APIs. **Developers do not need to focus on their internal implementation; use them directly.**

Note that these basic data structures can be created through [Operator Acceleration Library](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/aolapi/operatorlist_00001.html) in "Common Interface", such as aclCreateTensor.

- **aclTensor**

  A structure defined by the framework to manage and store tensor data (such as multi-dimensional data like vectors and matrices). Create this object through the **aclCreateTensor** interface.

  ```cpp
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A structure defined by the framework to manage and store scalar data (a single numerical value). Create this object through the **aclCreateScalar** interface.

  ```cpp
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  An array structure defined by the framework to manage and store integer data. Create this object through the **aclCreateIntArray** interface.

  ```cpp
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by the framework to manage and store float32 data. Create this object through the **aclCreateFloatArray** interface.

  ```cpp
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by the framework to manage and store boolean data. Create this object through the **aclCreateBoolArray** interface.
    
  ```cpp
  typedef struct aclBoolArray aclBoolArray
  ```
    
- **aclTensorList**

  An array structure defined by the framework to manage and store multiple tensor data. Create this object through the **aclCreateTensorList** interface.
    
  ```cpp
  typedef struct aclTensorList aclTensorList
  ```
    
- **aclScalarList**

  An array structure defined by the framework to manage and store scalar data. Create this object through the **aclCreateScalarList** interface.

  ```cpp
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by the framework, serving as a container for executing operator computation.

  When calling the first-stage interface aclxxXxxGetWorkspaceSize, the framework automatically creates aclOpExecutor; when calling the second-stage interface aclxxXxx, the framework automatically releases this object.

  ```cpp
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by the framework, used to manage and maintain the execution order of asynchronous operations.
    
  ```cpp
  typedef void *aclrtStream
  ```
