# ConvertToTensor

## 功能说明

将不同数据类型的host侧数据，转为一个host侧aclTensor对象。

## 函数原型

```cpp
aclTensor *ConvertToTensor(const aclIntArray *value, DataType dataType)
```

```cpp
aclTensor *ConvertToTensor(const aclBoolArray *value, DataType dataType)
```

```cpp
aclTensor *ConvertToTensor(const aclFloatArray *value, DataType dataType)
```

```cpp
aclTensor *ConvertToTensor(const aclFp16Array *value, DataType dataType)
```

```cpp
aclTensor *ConvertToTensor(const aclBf16Array *value, DataType dataType)
```

```cpp
aclTensor *ConvertToTensor(const aclScalar *value, DataType dataType)
```

```cpp
template<typename T> 
aclTensor *ConvertToTensor(const T *value, uint64_t size, DataType dataType)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | host侧源数据。 |
| size | 输入 | 源数据的元素个数。 |
| dataType | 输入 | 将源数据转换为指定数据类型后，写入aclTensor。 |

## 返回值说明

返回转换后的host侧tensor。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 分别将一个aclScalar和int64_t转为host侧tensor
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    aclScalar *scalar = executor->AllocScalar(val);
    aclTensor *tensor = executor->ConvertToTensor(scalar, DT_INT64);
    tensor = executor->ConvertToTensor(&val, 1, DT_INT64);
}
```
