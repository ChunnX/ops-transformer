# 一、不切g模板

```c++
constexpr uint32_t SAS_META_SIZE = 1024;                            // SasMetaData大小
constexpr uint32_t FA_METADATA_SIZE = 8;
constexpr uint32_t FD_METADATA_SIZE = 8;
// FA Metadata Index Definitions                                    ------------FA阶段------------
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;                        // 当前AIC核是否有负载需要计算
constexpr uint32_t FA_BN2_START_INDEX = 1;                          // 当前AIC核的起始bn2Idx
constexpr uint32_t FA_M_START_INDEX = 2;                            // 当前AIC核的起始s1gIdx
constexpr uint32_t FA_S2_START_INDEX = 3;                           // 当前AIC核的起始s2Idx
constexpr uint32_t FA_BN2_END_INDEX = 4;                            // 当前AIC核的终止bn2Idx
constexpr uint32_t FA_M_END_INDEX = 5;                              // 当前AIC核的终止s1gIdx
constexpr uint32_t FA_S2_END_INDEX = 6;                             // 当前AIC核的终止s2Idx
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;        // 当前AIC核第一份规约任务在FD_DATA_WORKSPACE的idx

// FD Metadata Index Definitions                                    ------------FD阶段------------
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;                        // 当前AIV核是否有规约任务需要计算
constexpr uint32_t FD_BN2_IDX_INDEX = 1;                            // 当前AIV核上规约任务的bn2Idx
constexpr uint32_t FD_M_IDX_INDEX = 2;                              // 当前AIV核上规约任务的s1gIdx
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;                      // 当前AIV核上规约任务在FD_DATA_WORKSPACE的idx
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;                      // 当前AIV核上规约任务的s2切分数量
constexpr uint32_t FD_M_START_INDEX = 5;                            // 当前AIV核上规约任务的更小切分粒度的m轴offset
constexpr uint32_t FD_M_NUM_INDEX = 6;                              // 当前AIV核上规约任务的M轴切分大小
```



# 二、切g模板(n1=128)

```
constexpr uint32_t SAS_META_SIZE = 1024;                            // SasMetaData大小
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;
// FA Metadata Index Definitions                                    ------------FA阶段------------
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;                        // 当前AIC核是否有负载需要计算
constexpr uint32_t FA_BN2_START_INDEX = 1;                          // 当前AIC核的起始bn2Idx
constexpr uint32_t FA_M_START_INDEX = 2;                            // 当前AIC核的起始s1gIdx
constexpr uint32_t FA_S2_START_INDEX = 3;                           // 当前AIC核的起始s2Idx
constexpr uint32_t FA_BN2_END_INDEX = 4;                            // 当前AIC核的终止bn2Idx
constexpr uint32_t FA_M_END_INDEX = 5;                              // 当前AIC核的终止s1gIdx
constexpr uint32_t FA_S2_END_INDEX = 6;                             // 当前AIC核的终止s2Idx
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;        // 当前AIC核第一份规约任务在FD_DATA_WORKSPACE的idx
constexpr uint32_t FA_S2_MAX_NUM = 8;                               // 单个核最大s2基本块数量

// FD Metadata Index Definitions                                    ------------FD阶段------------
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;                        // 当前AIV核是否有规约任务需要计算
constexpr uint32_t FD_BN2_IDX_INDEX = 1;                            // 当前AIV核上规约任务的bn2Idx
constexpr uint32_t FD_M_IDX_INDEX = 2;                              // 当前AIV核上规约任务的s1gIdx
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;                      // 当前AIV核上规约任务在FD_DATA_WORKSPACE的idx
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;                      // 当前AIV核上规约任务的s2切分数量
constexpr uint32_t FD_M_START_INDEX = 5;                            // 当前AIV核上规约任务的更小切分粒度的m轴offset
constexpr uint32_t FD_M_NUM_INDEX = 6;                              // 当前AIV核上规约任务的M轴切分大小

struct SmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};
```



# 三、metadata结果打印方式（推荐）

```
FA_METADATA_SIZE = 9 # 切g模板置为9，其他模板置为8
FD_METADATA_SIZE = 8
AIC_CORE_MAX_NUM = 36
first_part_count = AIC_CORE_MAX_NUM * FA_METADATA_SIZE
# ===================== 打印aic分核结果 =====================
print("===== aic分核结果 =====")
for i in range(0, first_part_count, FA_METADATA_SIZE):
    # 每9个取一组并打印
    row = metadata[i:i+FA_METADATA_SIZE]
    print(row)

# ===================== 打印aiv分核结果 =====================
print("\n===== aiv分核结果 =====")
for i in range(first_part_count, len(metadata), FD_METADATA_SIZE):
    row = metadata[i:i+FD_METADATA_SIZE]
    print(row)
```

# 四、FD相关分核结果理解

![](images/FD%E7%9B%B8%E5%85%B3%E5%88%86%E6%A0%B8%E7%BB%93%E6%9E%9C%E7%90%86%E8%A7%A3.png)

