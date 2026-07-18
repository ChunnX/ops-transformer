# Diagram Pattern Adaptation Guide #

## Overview ##

custom operator If the running diagram mode is required, the overall process and operator development guide ([AI Core Operator Development Guide]()	2) Consistent. Note that ACLN adaptation is not required. Only the following deliverables need to be adapted.

```
${op_name}                              #Replace it with the lowercase underscore (_) of the actual operator name.
├── op_host                             #Implemented on the host side
│   └── ${op_name}_infershape.cpp       #Implements the InferShape implementation, deduces the operator shape, and deduces the output shape during running.
├── op_graph                            #Figure Fusion-related implementation
│   ├── CMakeLists.txt                  #cmakelist file on the op_graph side
│   ├── ${op_name}_graph_infer.cpp      #InferDataType file, which is used to deduce operator types and output dataType during running.
└── └── ${op_name}_proto.h              #Operator prototype definition, which is used to identify operators in the graph optimization and fusion phases.
```

This article will be based on`AddExample`This section uses an operator (assuming the AI Core operator) as an example to describe the implementation of the image input deliverable. For details about the complete code, see.`examples`Directory`add_example`.

## Shape and DataType derivation ##

Two deliverables need to be completed for the diagram mode.`${op_name}_graph_infer.cpp` `${op_name}_infershape.cpp`

**Deliverable 1: $\{op_name\}_infershape.cpp**

The InferShape function is used to deduce the output shape based on the input shape.

Here's an example,`AddExample`For details about the complete code of the operator, see.`examples/add_example/op_host`Lower[add_example_infershape.cpp](../../../examples/add_example/op_host/add_example_infershape.cpp)	.

```
//The logic of the AddExample operator is the addition of two numbers. Therefore, the output shape is the same as the input shape.
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    //Obtains the input shape.
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    //Obtains the output shape.
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    //Obtains the input DimNum.
    auto xShapeSize = xShape->GetDimNum();
    //Sets the output DimNum.
    yShape->SetDimNum(xShapeSize);
    //Set the input dim value to the output value in sequence.
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
//InferShape Registration
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**Deliverable 2: $\{op_name\}_graph_infer.cpp**

The InferDataType function deduces the output DataType based on the input DataType.

Here's an example,`AddExample`For details about the complete code of the operator, see.`examples/add_example/op_graph`Lower[add_example_graph_infer.cpp](../../../examples/add_example/op_graph/add_example_graph_infer.cpp)	.

```
//The logic of the AddExample operator is the addition of two numbers. Therefore, the output dataType is the same as the input dataType.
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    //Obtains the input dataType.
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    //Set the input dataType to the output.
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

//Register InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## Operator prototype configuration ##

The operator prototype needs to be registered with the[Graph Engine](https://www.hiascend.com/cann/graph-engine)	(GE for short) so that the GE can identify the input, output, and attribute information of the operator. The registration is successful.`REG_OP`The interface is complete. The developer needs to define basic information such as the input and output tensor types and quantity of operators.

Examples of common tensor/attribute data types are as follows:

| Tensor type. | Attribute Type | Example       |
| ------------ | -------------- | ------------- |
| int64        | /              | DT_INT64      |
| int32        | /              | DT_INT32      |
| int16        | /              | DT_INT16      |
| int8         | /              | DT_INT8       |
| double       | /              | DT_DOUBLE     |
| float32      | /              | DT_FLOAT      |
| float16      | /              | DT_FLOAT16    |
| bfloat16     | /              | DT_BF16       |
| complex128   | /              | DT_COMPLEX128 |
| complex64    | /              | DT_COMPLEX64  |
| complex32    | /              | DT_COMPLEX32  |
| /            | int            | Int           |
| /            | bool           | Bool          |
| /            | string         | String        |
| /            | float          | Float         |
| /            | list           | ListInt       |

The basic information is as follows:

| Input/Output        | Keyword        | Example                                                          |
| ------------------- | -------------- | ---------------------------------------------------------------- |
| Mandatory Input     | INPUT          | .INPUT($\{name\}, TensorType(\{input_dtype\}))                   |
| Optional Input      | OPTIONAL_INPUT | .OPTIONAL_INPUT($\{name\}, TensorType(\{optional_input_dtype\})) |
| Mandatory Attribute | REQUIRED_ATTR  | .REQUIRED_ATTR($\{name\}, $\{dtype\})                            |
| Optional attributes | ATTR           | .ATTR($\{name\}, $\{dtype\}, $\{default_value\})                 |
| Output              | OUTPUT         | .OUTPUT($\{name\}, TensorType(\{output_dtype\}))                 |

The sample code is as follows, showing how to register`AddExample`Operator:

```
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

For details about the complete code, see.`examples/add_example/op_graph`Directory[add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h)	.

