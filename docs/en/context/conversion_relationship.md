# Mutual Conversion Relationship

When an API (such as aclnnAdd, aclnnMul) has an **output aclTensor data type** that is inconsistent with the **computed type derived from input data types**, the API internally converts the computation result to the data type corresponding to the output type.

Data type conversion must satisfy the following rules. Conversions that do not satisfy the rules cannot be performed, and calling the API will result in parameter validation failure.

- Floating-point types: ACL\_FLOAT16, ACL\_FLOAT, ACL\_DOUBLE, ACL\_BF16.
- Integer types: ACL\_INT8, ACL\_UINT8, ACL\_INT16, ACL\_UINT16, ACL\_INT32, ACL\_UINT32, ACL\_INT64, ACL\_UINT64.
- Complex types: ACL\_COMPLEX64, ACL\_COMPLEX128.
- Integer types can be converted among themselves, and also support conversion to floating-point and complex types.
- Floating-point types can be converted among themselves, and also support conversion to complex types.
- Complex types can be converted among themselves.
- BOOL supports conversion to integer, floating-point, and complex types.

Besides the above scenarios, all other conversion scenarios are not supported.
