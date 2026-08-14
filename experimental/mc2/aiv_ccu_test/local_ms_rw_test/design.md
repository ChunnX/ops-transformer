# 目标
1. URMA/UBMEM能否从MS区获取数据，发送远端
2. 本地写远端MS软件层面是否通，需要验证。通过Hcomm注册远端MS，需要映射多个非连续地址
3. 客户诉求是提供连续的VA地址

# MS地址交织
MS全称是Memory Slice，位于CCUA上的地址空间。一个CCUA上的MS地址空间大小是1.5MB。950芯片上有四个CCUA，工CCUM管理。

CCUM对外呈现MS空间打平的地址，用户使用的是连续的6MB地址空间。但是6MB地址空间会映射到不同的四个CCUA上。映射规则如下：

按照4KB为粒度，将6MB的地址空间分为MS0～MS1535。

CCUA0中的MS编号为：MS0～MS7；MS32～MS39；...；MS1504 ～ MS1511；

CCUA1中的MS编号为：MS8～MS15；MS40～MS47；...；MS1512 ～ MS1519；

CCUA2中的MS编号为：MS16～MS23；MS48～MS55；...；MS1520 ～ MS1527；

CCUA3中的MS编号为：MS24～MS31；MS56～MS63；...；MS1528 ～ MS1511；

# 场景
每个场景需要测读，写，混合读写。数据量从4KB->6MB。打点获取读写时间。但是现在可以先实现写。

## 本地读写MS
现在有个需求，需要写一个简单的host kernel侧代码，然后通过<<<>>>调用kernel侧函数，kernel侧函数是本地写memoryslice的逻辑。

### 讨论笔记
数据量从4KB -> 8KB -> 16K -> ... ->6MB

HOST侧：

1. 从HrtGetCcuMemInfo接口中获取VA地址dstaddr
2. 传递给Device

Device侧：

1. 生成数据var
2. 打点starttime
3. for循环datacopy(dstaddr, var)
4. barrier确保搬运完毕
5. 打点endtime
6. 计算时间

### Host实现
main函数前面配置和一般算子一样，比如申请通信域，申请流等等。

1、通过Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId_)接口，获取tlvHandle

2、通过HcclResult HrtGetCcuMemInfo(void* tlv_handle, uint32_t udieIdx, uint64_t memTypeBitmap, struct CcuMemInfo *memInfoList, uint32_t count) 接口，获取MS地址；  
备注：  
1）tlv_handle 来自GetTlvHandle获取；   
2）udieIdx可填0或1；   
3）memTypeBitmap，填写要读取的对象地址；

以上两个接口都可以在/Users/coke/Desktop/Ascend/hcomm这个仓找到实现。

enum CcuMemTypeBitmap : uint64_t;

// 当前需要申请权限的内存为18块  
inline const std::array<CcuMemTypeBitmap, 18>& GetMemTypeVector() {  
    static const std::array<CcuMemTypeBitmap, 18> kMemTypeArray = {{  
        CCU_MEMTYPE_INS,  
        CCU_MEMTYPE_GSA,  
        CCU_MEMTYPE_XN,  
        CCU_MEMTYPE_CKE,  
        CCU_MEMTYPE_PFE,  
        CCU_MEMTYPE_CHN,  
        CCU_MEMTYPE_JETTY_CTX,  
        CCU_MEMTYPE_MISSION_CTX,  
        CCU_MEMTYPE_LOOP_CTX,  
        CCU_MEMTYPE_MISSION_SQE,  
        CCU_MEMTYPE_CQE_BLOCK0,  
        CCU_MEMTYPE_CQE_BLOCK1,  
        CCU_MEMTYPE_CQE_BLOCK2,  
        CCU_MEMTYPE_WQEBB,  
        CCU_MEMTYPE_MS_BLOCK0,  
        CCU_MEMTYPE_MS_BLOCK1,  
        CCU_MEMTYPE_MS_BLOCK2,  
        CCU_MEMTYPE_MS_BLOCK3  
    }};

将申请的CcuMemInfo取出，按照count应该有四块，对应MSAddr0，MSAddr1，MSAddr2，MSAddr3

### Kernel实现
入口：Aivccu_COMM<<<blockDim, nullptr, stream>>>(mode0, mode1,  datasize, MSAddr0, MSAddr1, MSAddr2, MSAddr3, timeaddr)

mode0：读/写/读写混合场景

mode1:  交织/不交织。因为输入的是四个MS地址，交织需要手动交织。



核函数内部：

1. 不交织场景：搬运指定数据，例如写，CCUA中的MS顺序为CCUA0，CCUA1，CCUA2，CCUA3
2. 交织场景：搬运指定数据，例如写，按照之前的MS地址交织算法。先CCUA0(MS0~MS7)，用完了就CCUA1(MS8～MS15).......



## UBMEM读写远端MS
数据量从4KB -> 8KB -> 16K -> ... ->6MB

HOST侧：

1. 从HrtGetCcuMemInfo接口中获取VA地址dstaddr
2. dstAddrList = registerCCLBuffer(dstaddr)
3. 传递给Device

Device侧：

1. 生成数据var
2. 打点starttime
3. for循环datacopy(dstAddrList[], var)
4. barrier确保搬运完毕
5. 打点endtime
6. 计算时间

## URMA读写远端MS
数据量从4KB -> 8KB -> 16K -> ... ->6MB

HOST侧：

1. 从HrtGetCcuMemInfo接口中获取VA地址dstaddr
2. peerAddr = registerCHannelBuffer(dstAddr)
3. channel = {peerAddr, }
4. 传递给Device

Device侧：

1. 生成数据GMVar
2. 打点starttime
3. for循环 writeNbi(channel->peerAddr, GMVar,)
4. barrier确保搬运完毕
5. 打点endtime
6. 计算时间
