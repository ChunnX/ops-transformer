现在有一个任务
背景：allGatherMatmul算子需要支持GatherOut输出，也就是说需要把从各张卡上Gather回来的数据直接输出，
当前的实现：allgatherMatmul算子会把其他卡的数据gather回来，但是存放的位置是win区，需要把Gather回来的数据从win区拷贝到UB，再从UB拷贝到GM上，可以使用开始多buffer加速。

初步方案：allgatherMatmul会把Gather的数据再M方向上切块，我现在希望，后rankSize个核通信，前blockNum - rankSize个核进行win->UB->gatherOutGM的拷贝。假设loopCnt是通信的切分轮次
伪代码
     for i in loopCnt：
        allGather(i);
        if (i > 0):
            CopyDataFormWin(i - 1):
        syncAll<true>();
    CopyDataFormWin(loopCnt - 1);
    syncAll<true>();

V核上采用以上伪代码的流水，请帮我先设计好这个方案。
当前代码路径：/home/huzhijian/ops-transformer/mc2/common/op_kernel/apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_urma_impl.h
CopyDataFormWin实现/home/huzhijian/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode.h中的MoveResultFromPeerMemToOut函数

方案输出到/home/huzhijian/ops-transformer/mc2/all_gather_matmul_v3/GatherOutDesign.md