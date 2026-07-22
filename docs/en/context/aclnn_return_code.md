# aclnn Return Code

When calling aclnn APIs, common interface return codes are shown in [Table 1](#return-status-codes).
For abnormal status code values, you can obtain exception information through the aclGetRecentErrMsg interface (refer to [acl API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0005.html)). You can troubleshoot issues based on error prompts or contact technical support.

**Table 1** Return Status Codes

<a name="return-status-codes"></a>
| Status Code Name | Status Code Value | Status Code Description |
| :--- | :--- | :--- |
| ACLNN_SUCCESS | 0 | Success. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter validation error: illegal nullptr exists in parameters. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter validation error, such as two input data types not satisfying the input type derivation relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API internal call to NPU runtime interface exception. |
| ACLNN_ERR_INNER_XXX | 561xxx | API internal exception occurred. |

For more descriptions of ACLNN_ERR_INNER_XXX type status codes, see [Table 2](#exception-status-codes).

**Table 2** Exception Status Codes

<a name="exception-status-codes"></a>
| Status Code Name | Status Code Value | Status Code Description |
| :--- | :--- | :--- |
| ACLNN_ERR_INNER | 561000 | Internal exception: API internal exception occurred. |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 561001 | Internal exception: API internal output shape derivation error occurred. |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Internal exception: API internal NPU kernel tiling exception occurred. |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR | 561003 | Internal exception: API internal NPU kernel lookup exception (possibly because operator binary package is not installed). |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | Internal exception: API internal aclOpExecutor creation failed (possibly because of operating system exception). |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR | 561102 | Internal exception: API internal uniqueExecutor ReleaseTo was not called. |
| ACLNN_ERR_INNER_NULLPTR | 561103 | Internal exception: aclnn API internal nullptr exception occurred. |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE | 561104 | Internal exception: aclnn API internal operator attribute count exception. |
| ACLNN_ERR_INNER_KEY_CONFILICT | 561105 | Internal exception: aclnn API internal operator kernel matching hash key conflict. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE | 561106 | Internal exception: aclnn API internal operator implementation mode parameter error. |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | 561107 | Internal exception: aclnn API internal environment variable ASCEND_OPP_PATH not detected. |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED | 561108 | Internal exception: aclnn API internal operator kernel library operator information json file loading failed. |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND | 561109 | Internal exception: aclnn API internal operator kernel library operator information json file field loading failed. |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID | 561110 | Internal exception: aclnn API internal operator kernel library operator information json file format field has illegal value. |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID | 561111 | Internal exception: aclnn API internal operator kernel library operator information json file dtype field has illegal value. |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | 561112 | Internal exception: aclnn API internal operator binary kernel library not loaded. |
| ACLNN_ERR_INNER_OP_FILE_INVALID | 561113 | Internal exception: aclnn API internal operator json file field loading exception. |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND | 561114 | Internal exception: aclnn API internal operator attribute count inconsistent with operator information json, exceeding the attr count specified in json. |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH | 561115 | Internal exception: aclnn API internal operator attribute count inconsistent with operator information json, less than the attr count specified in json. |
