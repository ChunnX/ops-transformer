# 主要任务
验证算子的batch/seq轴数据一致性，通过给算子的aclnnfuzz测试用例增加输入输出的seed值方案，来固定算子的某个轴的输入信息，最后来验证输出信息是否满足要求。针对新生成的用例需要做一下覆盖度评估。

# 算子batch/seq 一致性说明
算子的batch/seq轴数据一致性说明，只要轴的相关输入数据一样，输出的对应数据也保持不变，输出不会因为batch/seq轴的大小而变化，包含下面两点内容：
1. 同一条用例的shape，batch/seq间数据位置不一样，如果对应的batch/seq轴的数据输入一样，则输出对应轴的数据也一样，比如[100,128]，batch/seq是第一个轴，如果使轴id是10,20,30的值数据保持一致，那输出的tensor对应轴id是10,20,30的数据也是一样的。
2. 不同的用例输出shape，batch/seq长度不一样，如果对应的batch/seq轴的数据输入一样，则输出对应的轴的数据也一样，比如两个case shape分别是[100,128]，[30,128]，batch/seq是第一个轴，如果保持第一个shape的id是80位置的输入和第二个shape的位置是10的输入数据一样，则这两个case输出tensor对应的值也会一样。

# 一致性验证方法

## tensor输入数据保持一致的方案
用例增加attr_name是`tensor_input_seed_info`属性，attr_type设置为`list_array`，attr_dtype设置为`list`，数据是一个list，里面值格式如下：

```
[[tensor_index, batch_axis, [batch_index_list], [range_min, range_max], seed, pre_shape_index], ...]
```

各字段说明：
- **tensor_index**：aclnn接口有多个tensor输入，当前参数是修改第几个输入tensor（从0开始）
- **batch_axis**：当前输入tensor的shape中，哪个轴是batch/seq轴（从0开始）
- **[batch_index_list]**：batch/seq轴里面的值索引列表，设置一个或多个batch/seq轴切片的数据，如[0]、[10,20,30]
- **[range_min, range_max]**：对应数据的range区间，**必须使用当前用例`input_tensor_range`中对应tensor的实际range值**。比如用例input_tensor_range是`[[2, 10], [0.01, 1]]`，则第1个tensor的seed range取`[2, 10]`，第2个tensor取`[0.01, 1]`
- **seed**：numpy固定的随机种子值，为了保持在数据range区间内数据一致使用。只有range输入一样且切片后的batch/seq shape一样，输入的tensor dtype也一样的时候才能设置相同的seed
- **pre_shape_index**：shape中batch/seq轴前面的轴索引列表，如果batch/seq轴前面有2个前轴则必须写2个值，也可以用`":"`代替全选。batch/seq轴为0时前面无前轴，写`[]`

### input seed_info 示例

**示例1**：`[0, 1, [0], [2, 10], 100, [0]]`
- 含义：aclnn接口第1个输入tensor(shape假设为[2,3,4,5])的第2根轴(axis=1)是batch/seq轴
- 设置batch/seq索引为0的切片，数据range区间是[2,10]，随机seed是100
- batch/seq轴前面的轴索引是[0]，即第1根轴取索引0
- 最终设置slice `input_i_data[0,0,:,:]` 的随机种子是100

**示例2**：`[1, 2, [1,2,3,4,5], [-10, 10], 10, [3,4]]`
- 含义：aclnn接口第2个输入tensor(shape假设为[100,200,300,4,5])的第3根轴(axis=2)是batch/seq轴
- 设置batch/seq索引为[1,2,3,4,5]的切片，数据range区间是[-10,10]，随机seed是10
- batch/seq轴前2个轴的索引是[3,4]
- 最终分别设置slice `[3,4,1,:,:]`、`[3,4,2,:,:]`、...、`[3,4,5,:,:]` 的数据区间是[-10,10]，随机种子是10

## tensor输出数据一致性标识方案
输出标识用于后期验证，用例增加attr_name是`tensor_output_seed_info`属性，attr_type设置为`list_array`，attr_dtype设置为`list`，数据是一个list，里面值格式如下：

```
[[tensor_index, batch_axis, [batch_index_list], output_seed, pre_shape_index], ...]
```

各字段说明：
- **tensor_index**：aclnn接口第几个输出tensor（从0开始）
- **batch_axis**：当前输出tensor的shape中，哪个轴是batch/seq轴
- **[batch_index_list]**：batch/seq轴里面的值索引列表
- **output_seed**：输出标识的seed值，一般是输入场景对应batch/seq的输入seed之和（或唯一定义的标识值）。同一个case中相同的输入seed对应的batch/seq需要设置相同的输出标识seed
- **pre_shape_index**：shape中batch/seq轴前面的轴索引，列表格式，有N个前轴必须写N个值

### output seed_info 示例

**示例**：`[[0, 1, [0], 102, [0]]]`
- 含义：aclnn接口第1个输出tensor(shape假设为[2,3,4,5])的第2根轴(axis=1)是batch/seq轴
- 设置batch/seq索引为0的输出标识seed是102
- batch/seq轴前面的轴索引是[0]
- 最终标识slice `[0,0,:,:]` 的输出数据标识为102

## 输入/输出标识一致说明
**1. 单个tensor输入可以是多个batch/seq使用一个seed或者不同的seed，多个tensor输入需要设置多个tensor输入对应的batch/seq节点的数据指定输入seed，tensor之间的seed可以不一样**
**2. 单个输出的tensor，对应的batch/seq和输入的batch/seq是对应的，输出指定的标识一般是输入tensor对应的batch/seq的seed和（或唯一定义值）**
**3. 如果输入的原始数据经过指定batch/seq切片后的shape不一样，输出和输入不能使用同样的seed**
**4. 不同case间比较时，对输出的batch/seq切片后的shape如果一样，则可以设置对应输入的切片seed一样，然后设置输出的seed也一样，用来验证不同用例间的输出batch/seq一致**

## seed和range绑定规则
- 相同的seed**必须**设置相同的range区间
- range值**必须**来自当前用例`input_tensor_range`中对应tensor的实际range，不能自定义随意值
- 举例：用例`input_tensor_range`=`[[2, 10], [0.01, 1]]`，tensor_index=0时range取`[2, 10]`，tensor_index=1时range取`[0.01, 1]`

# aclnnfuzz 测试用例表格格式说明

## sheet说明
excel表格有多个sheet页签，每个页签对应一种测试场景：
- **level0**：不同用例间的batch/seq比较（**必须包含**），验证跨case输出一致性
- **level1**：同一用例内的batch/seq比较，验证同一case内不同batch/seq位置输出标识正确性
- **level2**：边界场景，高维度、非标准axis等

## sheet字段说明

### 基础字段
| 字段名 | 说明 |
|--------|------|
| aclnn_name | 测试的aclnn接口名称 |
| case_name | 测试的case名称，实际测试框架输出的结果casename是aclnn_name和case_name字段的拼接：`{aclnn_name}_{case_name}` |
| bin_dir | 测试用例输入bin文件地址 |
| genetic | 用例网络标签，标识该用例属于哪些网络 |
| precision_mode | 精度模式 |
| precision_tolerance | 精度详细设置 |
| red_range | range缩减设置 |

### tensor输入输出参数
| 字段名 | 说明 |
|--------|------|
| input_tensor_shape | 算子多个输入tensor的shape信息，内容是list转的字符串，如`[[67, 145, 51], [9, 42, 3]]`，多个shape按顺序排列 |
| input_tensor_range | 算子多个输入tensor的值域区间，如`[[0, 0.001], [-51, 50]]`，多个range按顺序排列，**tensor_input_seed_info中的range字段值必须从此处对应tensor的range中取** |
| input_tensor_dtype | 算子多个输入tensor的dtype信息，如`['fp16', 'int64']` |
| input_tensor_format | 算子多个输入tensor的format信息，如`['ND', 'ND']` |
| input_tensor_type | 算子多个输入tensor的类型，一般为tensor，tensor_list时写tensor_list |
| input_tensor_index | 输入索引，表示shape/range/dtype/format对应顺序的参数在aclnn接口中的实际位置，如`[0, 1]` |
| output_tensor_shape | aclnn接口输出tensor的shape |
| output_tensor_range | aclnn接口输出tensor的range，一般为`[None]` |
| output_tensor_dtype | aclnn接口输出tensor的dtype |
| output_tensor_format | aclnn接口输出tensor的format |
| output_tensor_type | aclnn接口输出tensor的类型 |

### 属性参数（每组4列，可有多组）
| 字段名 | 说明 |
|--------|------|
| attr_name | 属性名称 |
| attr_type | 属性类型 |
| attr_dtype | 属性的数据类型 |
| attr_value | 属性值 |

每对tensor_input_seed_info / tensor_output_seed_info各占一组attr属性列（4列），依次排列在表格末尾。

### 特殊属性参数
attr_name是`tensor_data_{idx}`的是指实际接口输入的第idx的值是attr_value，用来给第idx输入的tensor填充值。

## batch/seq验证用例属性列结构

表格中batch/seq验证相关的属性列固定为2组（共8列），结构如下：

| 列序号 | attr_name | attr_type | attr_dtype | attr_value |
|--------|-----------|-----------|------------|------------|
| 第1组 | tensor_input_seed_info | list_array | list | `[[tensor_index, batch_axis, [batch_indices], [range], seed, pre_idx], ...]` |
| 第2组 | tensor_output_seed_info | list_array | list | `[[tensor_index, batch_axis, [batch_indices], output_seed, pre_idx], ...]` |

### input seed_info格式（6元素）
```
[tensor_index, batch_axis, [batch_index_list], [range_min, range_max], seed, pre_shape_index]
```

### output seed_info格式（5元素）
```
[tensor_index, batch_axis, [batch_index_list], output_seed, pre_shape_index]
```

### 数据类型映射关系
```python
DTYPE_MAPPING = {
    'float32': 'fp32',
    'float16': 'fp16',
    'float64': 'fp64',
    'bfloat16': 'bf16',
    'int32': 'int32',
    'int64': 'int64',
    'int16': 'int16',
    'int8': 'int8',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'uint32': 'uint32',
    'uint64': 'uint64',
    'bool': 'bool',
    'complex64': 'complex64',
    'complex128': 'complex128',
}
```

### 属性类型映射关系
```python
ATTR_TYPE_MAPPING = {
    'float': 'scalar',
    'float16': 'scalar',
    'bfloat16': 'scalar',
    'float32': 'scalar',
    'float64': 'scalar',
    'double': 'scalar',
    'int': 'buildins',
    'int8': 'buildins',
    'int16': 'buildins',
    'int32': 'buildins',
    'int64': 'buildins',
    'int8_t': 'buildins',
    'int16_t': 'buildins',
    'int32_t': 'buildins',
    'int64_t': 'buildins',
    'uint8': 'buildins',
    'uint16': 'buildins',
    'uint32': 'buildins',
    'uint64': 'buildins',
    'bool': 'buildins',
    'string': 'buildins',
    'list_array': 'list',
}
```

# 参考资料

## 实际参考用例
sample/aclnnAdd.xlsx（含tensor_input_seed_info和tensor_output_seed_info的完整参考）

## 参考用例数据示例

### aclnnAdd Row2 完整数据
```
aclnn_name: aclnnAdd
input_tensor_shape: [[1,1,3584],[1,1,3584]]
input_tensor_range: [[2, 10], [0.01, 1]]
input_tensor_dtype: ['bf16','bf16']
input_tensor_index: [0, 1]
output_tensor_shape: [[1,1,3584]]

attr_name: alpha                    | tensor_input_seed_info           | tensor_output_seed_info
attr_type: scalar                   | list_array                       | list_array
attr_dtype: bf16                    | list                             | list
attr_value: 1                       | [[0,1,[0],[2, 10],100,[0]],[1,1,[0],[0.01, 1],101,[0]]] | [[0,1,[0],102,[0]]]
```

说明：
- 第1个输入tensor(0)的range是[2,10]，input_seed_info中对应range取[2,10]，seed=100
- 第2个输入tensor(1)的range是[0.01,1]，input_seed_info中对应range取[0.01,1]，seed=101
- 输出output_seed_info=102，是输入seed 100+101的和(再可+1偏移)

# 用例场景优先级
- **level0**：不同用例间的比较（必须包含）—— 不同shape的case对，相同seed设置，验证跨case输出batch/seq一致性
- **level1**：同一用例内的比较 —— 同一case内多组batch/seq位置各自不同seed，验证组间输出标识正确
- **level2**：边界场景 —— 高维度shape、非标准batch/seq轴(axis>2)等

# batch/seq选取的位置要求
- 单挑case内/多case间，至少需要覆盖最小/中值/最大 batch/seq位置的值

# 用例数量要求
- **level0**：500条
- **level1**：200条
- **level2**：200条


# 用例输入tensor_input_seed_info 格式预检查
python3 compare_batch_tensor_by_result.py --case_file=aclnnAdd.xlsx --sheet_name=test --check_type=input
case_file 输入用例文件
sheet_name 用例表格的sheet页签

# 用例出tensor_output_seed_info 格式预检查
python3 compare_batch_tensor_by_result.py --case_file=aclnnAdd.xlsx --sheet_name=test --check_type=output
case_file 输入用例文件
sheet_name 用例表格的sheet页签

# 特殊算子batch/seq轴验证说明

## aclnnReduceSum batch/seq验证说明
1. 用例参数dims表示要reduce的轴，如果dims是空则表示所有轴都reduce，keepdims=True表示reduce的轴reduce之后保持1，False则reduce轴reduce之后不存在
2. 只有非reduce轴才能设置为batch/seq轴，输入设置batch/seq轴的时候要去掉reduce轴
3. 如果batch/seq的轴在reduce轴后面，那batch/seq轴前面的reduce轴要全选，比如输入是[1, 19080, 3072]，reduce轴是第2根轴，batch/seq轴是第3根轴，设置batch/seq 1024位置的随机种子时，reduce轴要所有都选，tensor_input_seed_info里面的属性batch/seq轴前轴索引reduce轴要设置全选：`[0, 2, [1024], [实际range], 7751, [0,:]]`
4. 输出如果keepdims是True时输出shape和输入一样，对应的batch/seq轴一样；如果keepdims是False，则实际输出对应的batch/seq要考虑缩轴的影响
5. 如果batch/seq轴的前轴里面有reduce轴，只有输入的reduce轴shape值一样的时候才能设置不同case的输入输出的seed一样
6. dims不能为空。
7. 用例覆盖度要求
- 不同用例间的比较要覆盖实际用例的所有非reduce轴
- 同一用例内的比较要覆盖实际用例的所有非reduce轴




## 逐元素算子（如aclnnRelu）batch/seq验证说明
1. 逐元素算子输入输出shape/dtype完全一致，单输入单输出
2. tensor_index固定为0（只有1个输入和1个输出）
3. output_seed_info与input_seed_info的batch_axis和batch_index_list保持一致
4. 用例覆盖度要求
- 不同用例间的比较要覆盖实际用例的所有轴，默认0~7轴
- 同一用例内的比较要覆盖实际用例的所有轴，默认0~7轴


## 多输入算子（如aclnnAdd）batch/seq验证说明
1. broadcast场景如果batch/seq轴值是1，则可以和另一个输入的对用的batch/seq轴的任何一个索引值都可以组合
2. scaler 输入的tensor 默认就是batch/seq轴的值,不能使用seed设置值，需要使用 `tensor_data_{idx}` 参数设置对应的tensor输入的值
3. output_seed_info与input_seed_info的batch_axis和batch_index_list保持一致
4. 用例覆盖度要求
- 不同用例间的比较要覆盖实际用例的所有轴，默认0~7轴
- 同一用例内的比较要覆盖实际用例的所有轴，默认0~7轴



