令  
- $I = \{1,2,\dots,N\}$ 为 token 索引集合，$N = \text{num\_tokens}$；  
- $d = \text{hidden}$ 为特征维度；  
- 每个 token $i \in I$ 的特征向量 $\mathbf{x}_i \in \mathbb{R}^{1 \times d}$（行向量），且 $\mathbf{x}_i = \mathbf{x}[i,:]$；  
- $K = \text{num\_topk}$ 为每个 token 选择的专家数；  
- $E$ 为专家总数。  

对于每个 $i \in I$，定义  
$$e_{i,k} = \text{topk\_idx}[i,k],\quad k = 1,\dots,num\_topk$$  
其中 $e_{i,k} \in \{1,\dots,num\_experts\}$。
  
在Dispatch的过程中，每个 token $i$ 将其特征向量 $\mathbf{x}_i$ 同时发送给专家 $e_{i,1}, e_{i,2}, \dots, e_{i,num\_topk}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

令 $S_e = \{\, i \mid \exists k,\ \text{topk\_idx}[i,k]=e \,\}$，$N_e = |S_e|$。  
$$\mathbf{X}_e \in \mathbb{R}^{N_e \times d}$$  
是由所有满足 $i \in S_e$ 的特征向量 $\mathbf{x}_i$ **按任意固定顺序行堆叠** 而成的矩阵。

**行索引映射**：  
对于每个 token $i$ 及其被分配到的某个专家 $e_{i,k}$，存在唯一的行索引  
$$\text{idx}_{e_{i,k}}(i) \in \{1,\dots,N_{e_{i,k}}\}$$  
使得  
$$\mathbf{X}_{e_{i,k}}[\text{idx}_{e_{i,k}}(i), :] = \mathbf{x}_i.$$  
该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。