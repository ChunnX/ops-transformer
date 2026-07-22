# Broadcast Relationship

## Broadcast Concept

Broadcast describes how operators handle tensors (or arrays) of different shapes during computation. In most cases, it allows tensors (or arrays) of different shapes to automatically expand their shapes during element operations, making their dimensions compatible. Typically, the smaller tensor (or array) is "broadcast" to match the larger tensor (or array).

Currently, many CANN operator API parameter shapes support broadcasting, which can appropriately improve computation efficiency and reduce memory usage (especially in large-scale data scenarios). For more detailed broadcast technology introduction, refer to the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) official website.

## Broadcast Rules

When performing broadcast computation, the following rules generally need to be understood:

- Rule 1: If the number of dimensions between arrays is inconsistent, all arrays align to the longest-shaped array, and the insufficient shape parts are padded with 1 on the **left side** until the number of dimensions is the same.
  
  > Note:
  > - Sample 1: Number of Dimensions refers to the dimension count of the tensor (or array) corresponding to shape. For example, x.shape=(1,1,2,4), the number of dimensions is 4.  
  > - Sample 2: For example, computing a+b, where a.shape=(2, 2, 3) and b.shape=(2, 3), then array b will be broadcast to b.shape=(1, 2, 3).
  
- Rule 2: If the number of dimensions between arrays is consistent, and a certain dimension of an array is 1, then the array with dimension 1 will be stretched to match the corresponding dimension shape of the other array.

  > Note:
  > In this scenario, broadcasting only needs to be performed on a certain dimension. For example, computing a+b, where a.shape=(1, 3) and b.shape=(3, 1), then both arrays will be broadcast to a.shape=(3, 3) and b.shape=(3, 3).

- Rule 3: If the number of dimensions between arrays is inconsistent, and neither has a dimension equal to 1, an error will occur.

Based on the above rules, the broadcast process generally first expands dimensions according to **Rule 1**, then stretches shapes according to **Rule 2**. A specific example is as follows:

```text
Assuming a.shape=(2,2,3), values like:
[[[1 2 3],[4 5 6]],
[[1 2 3],[4 5 6]]]
Assuming b.shape=(2,3), values like:
[[1 2 3],
[-1 -2 -3]]
According to Rule 1, expand dimensions, b.shape=(1,2,3), values as follows:
[[[1 2 3],
[-1 -2 -3]]]
According to Rule 2, stretch shapes, b.shape=(2,2,3), values as follows:
[[[1 2 3],[-1 -2 -3]],
[[1 2 3],[-1 -2 -3]]]
Computing a+b, actual result as follows:
[[[2 4 6],[3 3 3]],
[[2 4 6],[3 3 3]]]
```

## Limitations

When the data types of two inputs a and b that satisfy the broadcast relationship, or the derived data types, are among COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, UINT64, in addition to satisfying the above broadcast rules, the following condition must also be met, otherwise broadcasting will fail and cause an operator execution error.
Condition: The dimension after merging consecutive axes that need broadcasting and consecutive axes that do not need broadcasting must be less than 6.
Examples:

- When a.shape=(5, 1, 5, 1, 5, 1) and b.shape=(5, 5, 5, 5, 5, 5), there are no axes to merge, the final dimension is 6, and broadcasting fails.
- When a.shape=(5, 1, 5, 5, 1, 1) and b.shape=(5, 5, 5, 5, 5, 5), dimensions 2 and 3 do not need broadcasting, dimensions 4 and 5 both need broadcasting, and they are merged separately as consecutive groups. After merging, the dimension is 4, and broadcasting succeeds.
