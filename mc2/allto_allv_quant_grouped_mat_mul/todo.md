# AlltoAllvQuantGroupedMatMul 修复记录

## 目标
实现多轮次通信调度策略，核心是修正通信逻辑语义问题。

## 已完成修复

### 2026-04-20: 修正LaunchCommMultiRound通信逻辑

**问题根因**：
1. 接收端错误：遍历e_个本地专家，但应遍历recvCounts_所有条目（0到e*ep-1）
2. 发送端错误：使用recvCounts_[currExpertIdx]（本卡的recvCount），但应使用preCommRecvGlobalBuffer_访问dstRank的recvCount

**修复方案**：

**接收端正确逻辑**：
```cpp
while (recvAccumulated < localBaseA_ && recvExpertIdx < e_ * epWorldSize_) {
    uint64_t recvCntValue = preCommRecvGlobalBuffer_.GetValue(rankId_ * e_ * epWorldSize_ + recvExpertIdx);
    
    int32_t remaining = recvCntValue - recvExpertOffset;
    int32_t toAdd = Mc2Min(remaining, localBaseA_ - recvAccumulated);
    
    uint32_t srcRank = recvExpertIdx / e_;
    alltoAllvRecvCnt[srcRank] += toAdd * axis;
    
    recvAccumulated += toAdd;
    recvExpertOffset += toAdd;
    
    if (recvExpertOffset >= recvCntValue) {
        recvExpertIdx++;
        recvExpertOffset = 0;
    }
}
```

**发送端正确逻辑**：
```cpp
while (dstAccumulated < baseAPerRank_[dstRank] && dstExpertIdx < e_ * epWorldSize_) {
    uint64_t dstRecvCntValue = preCommRecvGlobalBuffer_.GetValue(dstRank * e_ * epWorldSize_ + dstExpertIdx);
    
    int32_t remaining = dstRecvCntValue - dstOffset;
    int32_t toAdd = Mc2Min(remaining, baseAPerRank_[dstRank] - dstAccumulated);
    
    uint32_t srcRank = dstExpertIdx / e_;
    if (srcRank == rankId_) {
        sendAmount += toAdd;
    }
    
    dstAccumulated += toAdd;
    dstOffset += toAdd;
    
    if (dstOffset >= dstRecvCntValue) {
        dstExpertIdx++;
        dstOffset = 0;
    }
}
```

**关键理解**：
- recvCounts_[globalExpertIdx] = 本卡从rank(globalExpertIdx/e)接收的数据量
- preCommRecvGlobalBuffer_[dstRank * e * ep + globalExpertIdx] = dstRank的recvCounts_[globalExpertIdx]
- srcRank = globalExpertIdx / e，当srcRank == rankId_时，本卡需要发送该数据

## 编译验证
- 编译无错误（只有warnings）
- 构建失败是autogen文件缺失问题，非代码问题

## 下一步
- 需要实际运行验证通信配对是否正确
- 根据打印日志分析recvCnt/sendCnt配对情况