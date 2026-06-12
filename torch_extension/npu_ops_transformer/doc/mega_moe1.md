## MoE 完整流程：Token‑i 的 Dispatch/Combine + 专家批量计算

### 输入与维度约定

- $\mathbf{x} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden}}$，所有 token 的输入特征
- $\text{topk\_idx} \in \mathbb{Z}^{\text{num\_tokens} \times \text{num\_topk}}$，每个 token 选中的专家编号
- $\text{topk\_weights} \in \mathbb{R}^{\text{num\_tokens} \times \text{num\_topk}}$，对应的门控权重（已归一化）
- $\text{l1\_weights} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden} \times (2\cdot\text{intermediate\_hidden})}$
- $\text{l2\_weights} \in \mathbb{R}^{\text{num\_experts} \times \text{intermediate\_hidden} \times \text{hidden}}$
- 当前 rank 的实际 token 数 $\text{num\_tokens} \le \text{num\_max\_tokens\_per\_rank}$

---

### Step 1 – Dispatch

对于某个 token $i\;(0\le i < \text{num\_tokens})$，其特征向量为 $\mathbf{x}_i = \mathbf{x}[i,:]$。  
路由结果为：

$$
e_k = \text{topk\_idx}[i,k],\qquad
w_k = \text{topk\_weights}[i,k],\qquad k = 1,\dots,\text{num\_topk}
$$

**Dispatch 的含义**：token $i$ 的输入 $\mathbf{x}_i$ 被发送给专家 $e_1,\dots,e_{\text{num\_topk}}$，每个专家都会收到相同的 $\mathbf{x}_i$。

> **过渡到批量实现**：实际执行时，所有 token 的 dispatch 会**以专家为中心**并行完成。对于每个专家 $e$，系统收集所有满足 $\text{topk\_idx}[i,k]=e$ 的 token $i$ 的特征向量，堆叠为矩阵 $\mathbf{X}_e$：
> $$
> \mathbf{X}_e = \bigl\{\,\mathbf{x}_i \;\big|\; \exists\,k \text{ 使得 } \text{topk\_idx}[i,k] = e\,\bigr\} \;\in\; \mathbb{R}^{N_e \times \text{hidden}}
> $$
> 其中 $N_e$ 是分配给专家 $e$ 的 token 总数。  
> token $i$ 在专家 $e_k$ 的批次 $\mathbf{X}_{e_k}$ 中占据某一行，该行的位置记为 $\text{idx}_{e_k}(i)$。  
> 门控权重 $w_k$ 未被收集到专家端，而是仍然保存在 token 侧的 $\text{topk\_weights}[i,k]$ 中。

---

### Step 2 – 专家 $e$ 的内部批量计算（专家视角）

此步骤以专家 $e$ 的批次 $\mathbf{X}_e$ 为单位，对每个专家独立进行。

**Linear1 投影（gate + up）**

$$
\mathbf{H}_e = \mathbf{X}_e \cdot \text{l1\_weights}[e]
\quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}\bigr)
$$

**拆分 gate 与 up**

$$
\mathbf{H}_{\text{gate}} = \mathbf{H}_e\bigl[:,\;\text{:intermediate\_hidden}\bigr], \qquad
\mathbf{H}_{\text{up}} = \mathbf{H}_e\bigl[:,\;\text{intermediate\_hidden:}\bigr]
$$

**SwiGLU 激活**

$$
\mathbf{A}_e = \text{SiLU}\bigl(\mathbf{H}_{\text{gate}}\bigr) \;\odot\; \mathbf{H}_{\text{up}}
\quad\bigl(\in \mathbb{R}^{N_e \times \text{intermediate\_hidden}}\bigr)
$$

其中 $\text{SiLU}(z)=z\cdot\sigma(z)$，$\odot$ 为逐元素乘法。

**Linear2 投影（down projection）**

$$
\mathbf{Y}_e = \mathbf{A}_e \cdot \text{l2\_weights}[e]
\quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
$$

经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 token $i$，它在专家 $e_k$ 中的输出行即为 $\mathbf{Y}_{e_k}\bigl[\text{idx}_{e_k}(i),\,:\bigr]$。

---

### Step 3 – Combine（Token‑i 视角）

回到 token $i$ 的视角：它需要将分布在各专家中的输出聚合起来。

利用之前记录的位置索引 $\text{idx}_{e_k}(i)$，从专家 $e_k$ 的输出矩阵中取出属于 token $i$ 的行，并与 token 侧保存的门控权重 $w_k$ 相乘后求和：

$$
\mathbf{y}_i = \sum_{k=1}^{\text{num\_topk}} w_k \;\cdot\;
\mathbf{Y}_{e_k}\!\bigl[\,\text{idx}_{e_k}(i),\,:\,\bigr]
\qquad\bigl(\in \mathbb{R}^{\text{hidden}}\bigr)
$$

其中 $w_k = \text{topk\_weights}[i,k]$ 就是 token $i$ 对专家 $e_k$ 的门控权重。

所有 token 的输出堆叠为最终序列输出 $\mathbf{Y} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden}}$。

---

### 流程总结

- **Step 1 – Dispatch**：以 token $i$ 的视角描述分发，只将特征 $\mathbf{x}_i$ 发送给选中专家，门控权重留在 token 侧。实现时转为以专家为中心的批量收集 $\mathbf{X}_e$。
- **Step 2 – 专家批量计算**：各专家独立对 $\mathbf{X}_e$ 执行 Linear1（gate+up）→ SwiGLU → Linear2，产生输出 $\mathbf{Y}_e$。
- **Step 3 – Combine**：以 token $i$ 的视角，利用位置索引取出其输出行，直接使用 $\text{topk\_weights}[i,k]$ 加权求和，恢复 token 最终表示。