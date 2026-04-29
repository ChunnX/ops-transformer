# SparseFlashAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。
    本次公布的`sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>attention结构的Query输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
     <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>attention结构的Key输入</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为(block_num, block_size, KV_N, D)。</li>
                <li>layout_kv为BSND时，shape为(B, S2, KV_N, D)。</li>
                <li>layout_kv为TND时，shape为(T2, KV_N, D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>attention结构的Value输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>shape与key的shape一致。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输入</td>
      <td>离散取kvCache的索引。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)。</li>
                <li>layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTable</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>第二维长度不小于所有batch中最大的S2对应的block数量，即S2_max / block_size向上取整。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2/block_size)。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery</td>
      <td>输入</td>
      <td>表示不同Batch中query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKv</td>
      <td>输入</td>
      <td>表示不同Batch中key和value的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li>该参数中每个Batch的有效token数不超过key/value中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>当layout_kv为TND或PA_BSND时，该入参必须传入。</li>
                <li>layout_kv为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryRope</td>
      <td>输入</td>
      <td>表示MLA结构中的query的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_query为BSND时，shape为(T1,N1,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRope</td>
      <td>输入</td>
      <td>表示MLA结构中的key的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_kv为BSND时，shape为(T1,N1,Dr)。</li>
                <li>layout_kv为PA_BSND时，shape为(block_num,block_size,N2,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>代表缩放系数。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize</td>
      <td>可选属性</td>
      <td>代表sparse阶段的block大小。</td>
      <td>
          <ul>
                <li>sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。</li>
                <li>sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQuery</td>
      <td>可选属性</td>
      <td>标识输入query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入BSND和TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>可选属性</td>
      <td>标识输入key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入TND、BSND和PA_BSND，其中PA_BSND在使能PageAttention时使用。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>可选属性</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表全部计算。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>可选属性</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>可选属性</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode</td>
      <td>可选属性</td>
      <td>-</td>
      <td>仅支持传入2，表示MLA-absorb模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse</td>
      <td>可选属性</td>
      <td>用于表示是否返回softmax_max和softmax_sum。</td>
      <td>
          <ul>
                <li>True表示返回，False表示不返回；默认值为False。</li>
                <li>该参数仅在训练且layout_kv不为PA_BSND场景支持。</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果，取max得到softmax_max。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>Attention算法query乘key的结果减去softmax_max, 再取exp，接着求sum，得到softmax_sum。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- N1支持1/2/4/8/16/32/64/128。
- block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
- 参数query中的D和key、value的D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 支持sparse_block_size整除block_size。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_sparse_flash_attention.cpp">test_aclnn_sparse_flash_attention
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashAttention.md">aclnnSparseFlashAttention
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
