# Data Format

Data format (format) is used to describe the business semantics of axes in a multi-dimensional Tensor, representing the physical layout format of data, such as 1D, 2D, 3D, 4D, 5D, and so on. It is generally required in CNN (Convolutional Neural Networks) type APIs to describe specific formats.

For the **full range of data formats** supported by aclTensor, refer to [acl API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0005.html) in "Data Types and Their Operation Interfaces > aclFormat".

For **data format layout principles**, refer to [Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html) in "Concept Principles and Terminology > Neural Networks and Operators > Data Layout Format".

## Usage Instructions

Currently, most operator APIs support the ND data format. For example, the aclnnAdd interface specifies that the supported data format is ND (the rule for low-dimension-first contiguous layout of multi-dimensional Tensors). For aclnnConvolution, which is a CNN-type API, the input aclTensor must be set with a format that has business semantics, rather than the ND format. Such operators need to know the business semantics in the Tensor during computation. For example, in 2D convolution, the correspondence between the Batch dimension, Channel dimension, Height dimension, Width dimension, and Tensor dimensions needs to be known.

>**Note:**
>
>- In two-stage interface parameter descriptions, for simplified description, **the original data format "ACL\_FORMAT\_XXXX_" is abbreviated as "_XXXX_"**.
>- Dimension meanings in data formats: N (Batch) represents batch size, H (Height) represents feature map height, W (Width) represents feature map width, C (Channels) represents feature map channels, D (Depth) represents feature map depth, L (Length) represents feature map length.

## Common Data Formats

When creating aclTensor through the **aclCreateTensor** interface, set the data format according to API business requirements. The currently **supported data formats** are:

ACL\_FORMAT\_ND, ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, ACL\_FORMAT\_NDHWC, ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NC, ACL\_FORMAT\_NCL.

For non-ND Tensors, the Tensor dimension requirements must be consistent with the format description. For example:

- 5D Tensor: Requires ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NDHWC, or ACL\_FORMAT\_ND (if the API parameter description does not indicate ND support, setting the ND format will cause an API validation error).
- 4D Tensor: Requires ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, or ACL\_FORMAT\_ND.
- 3D Tensor: Requires ACL\_FORMAT\_NCL or ACL\_FORMAT\_ND.
- 2D Tensor: Requires ACL\_FORMAT\_NC or ACL\_FORMAT\_ND.
- Other dimension Tensors: Require ACL\_FORMAT\_ND.

## Private Data Formats

Besides the common data formats above, there are other data formats, such as ACL\_FORMAT\_NC1HWC0, ACL\_FORMAT\_FRACTAL\_Z, ACL\_FORMAT\_NC1HWC0\_C04, ACL\_FORMAT\_FRACTAL\_NZ, ACL\_FORMAT\_NDC1HWC0, ACL\_FORMAT\_FRACTAL\_Z\_3D, and so on.

These formats are private formats of the NPU. Currently, most aclnn APIs do not support these formats. If an individual API declares supported data formats, refer to the actual description of that API.
