import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
torch.manual_seed(42)


def aligned_up(x: int, granularity: int) -> int:
    """向上对齐函数"""
    return ((x + granularity - 1) // granularity) * granularity


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力机制 (GQA) 的 PyTorch 实现，适用于 Qwen2.5-32B 模型。
    将 num_attention_heads 个 query 头划分为 num_kv_heads 组，每组共享一个 Key/Value 头。
    维度约束：
      - num_attention_heads = 40 // tp_size
      - num_kv_heads = 8 // tp_size
      - head_dim = 128
    """

    def __init__(
        self,
        tp_size: int,
        qk_head_dim: int = 128,
        vo_head_dim: int = 128,
        softmax_scale: float = 1.0,
        # softmax_scale: float = 128**-0.5,
    ):
        super().__init__()
        self.num_attention_heads = 40 // tp_size
        self.num_kv_heads = 8 // tp_size
        self.qk_head_dim = qk_head_dim
        self.vo_head_dim = vo_head_dim
        self.softmax_scale = softmax_scale
        self.is_naive = True  # 是否使用 naive 方法

    def forward(
        self,
        query: torch.Tensor,  # (num_attention_heads, total_tokens, head_dim)
        key_cache: torch.Tensor,  # (total_pages, num_kv_heads, page_len, head_dim)
        value_cache: torch.Tensor,  # (total_pages, num_kv_heads, page_len, head_dim)
        cu_seqlens_query: torch.Tensor,  # (batch_size + 1)
        seqlens_kv: torch.Tensor,  # (batch_size)
        page_ids: torch.Tensor,  # (batch_size, max_num_blocks)
    ) -> torch.Tensor:



        key_cache = key_cache.to(torch.float16)
        value_cache = value_cache.to(torch.float16)

        batch_size = seqlens_kv.size(0)
        outputs = []

        # 对每个sequence分别计算attention
        for i in range(batch_size):
            # 获取当前sequence的query
            q_start = cu_seqlens_query[i]
            q_end = cu_seqlens_query[i + 1]
            q = query[
                q_start:q_end
            ]  # (seqlen_query, num_attention_heads, head_dim)
            seqlen_query = q.shape[0]
            # 获取当前sequence的page_ids和有效长度
            seqlen_kv = seqlens_kv[i]
            curr_blocks = (seqlen_kv + page_len - 1) // page_len
            curr_pages = page_ids[i, :curr_blocks]

            # 收集当前sequence的key和value
            k = key_cache[
                curr_pages
            ]  # (num_blocks, num_kv_heads, page_len, head_dim)
            v = value_cache[curr_pages]

            if self.is_naive:
                # 方式一：计算attention
                k = torch.permute(k, (0, 2, 1, 3)).reshape(
                    -1, self.num_kv_heads, self.qk_head_dim
                )[
                    :seqlen_kv
                ]  # (seqlen_kv, num_kv_heads, head_dim)
                v = torch.permute(v, (0, 2, 1, 3)).reshape(
                    -1, self.num_kv_heads, self.vo_head_dim
                )[:seqlen_kv]
                # 重复KV以匹配query heads
                repeat = self.num_attention_heads // self.num_kv_heads
                k = k.repeat_interleave(
                    repeat, dim=1
                )  # (seqlen_kv, num_attention_heads, head_dim)
                v = v.repeat_interleave(repeat, dim=1)
                cur_q = q.transpose(0, 1).to(torch.float32)
                k = k.transpose(0, 1).to(
                    torch.float32
                )  # （num_attention_heads, seqlen_kv, head_dim)
                v = v.transpose(0, 1).to(torch.float32)
                scores = (
                    torch.matmul(cur_q, k.transpose(-2, -1))
                    * self.softmax_scale
                )

                mask = torch.ones(seqlen_query, seqlen_kv, dtype=torch.bool, device=scores.device)
                if seqlen_query > 1:  # 如果有生成 tokens
                    # 历史 tokens 可以互相关注（双向）
                    mask[:seqlen_kv - seqlen_query, :seqlen_kv - seqlen_query] = True
                    # 生成 tokens 只能关注前面的 tokens（单向）
                    for t in range(seqlen_kv - seqlen_query, seqlen_kv):
                        mask[t - (seqlen_kv - seqlen_query), :t+1] = True
                        mask[t - (seqlen_kv - seqlen_query), t+1:] = False

                # 应用mask
                scores = scores.masked_fill(~mask, float('-inf'))
                max = scores.max(dim=-1,keepdim=True).values
                exp_scores = (scores-max).exp()
                sum_exp_scores = (scores-max).exp().sum(dim=-1,keepdim=True)
                temp_atten = exp_scores @ v

                attn = F.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(0, 1).to(
                    torch.float16
                )  # (num_attention_heads, seqlen_query, head_dim)
            else:
                # 方式二：
                q = q.view(
                    -1,
                    self.num_kv_heads,
                    self.num_attention_heads // self.num_kv_heads,
                    self.qk_head_dim,
                )  # (seqlen_q, nhead_kv, group, head_dim)
                out = torch.zeros_like(
                    q
                )  # (seqlen_query, nhead_kv, group, head_dim)

                k = k.reshape(
                    -1, self.num_kv_heads, page_len, self.qk_head_dim
                )  # (num_blocks, num_kv_heads, page_len, head_dim)
                print(f"kcache.shape={key_cache.shape}, k.shape={k.shape}")
                v = v.reshape(
                    -1, self.num_kv_heads, page_len, self.vo_head_dim
                )



                for q_idx in range(seqlen_query):
                    cur_q = q[
                        q_idx
                    ]  # (N, G, D) note: prefill阶段可能需要对q做tile，这里只考虑了decode阶段每个sequence q_len=1的情况
                    # 初始化
                    m = torch.full(
                        (
                            self.num_kv_heads,
                            self.num_attention_heads // self.num_kv_heads,
                            1,
                        ),
                        float("-inf"),
                    ).half()
                    l = torch.zeros(
                        (
                            self.num_kv_heads,
                            self.num_attention_heads // self.num_kv_heads,
                            1,
                        )
                    )
                    O = torch.zeros_like(cur_q)
                    for curr_page_id in range(len((curr_pages))):
                        # page_id = page_ids[0][curr_page_id]
                        start_idx = curr_page_id * page_len
                        end_idx = min(start_idx + page_len, seqlen_kv.item())
                        cur_page_valid_len = end_idx - start_idx
                        # 计算当前page的key和value
                        k_page = k[
                            curr_page_id, :,:cur_page_valid_len
                        ]  # (num_kv_heads, page_len(valid_len), head_dim)
                        v_page = v[
                            curr_page_id, :,:cur_page_valid_len
                        ]  # (num_kv_heads, page_len(valid_len), head_dim)
                        S = (
                            torch.einsum("hgd,htd->hgt", cur_q, k_page)
                            * self.softmax_scale
                        )  # (num_kv_heads, group_size, seqlen_kv)

                        # 本块最大值 & 全局更新
                        m_loc, _ = S.max(
                            dim=-1, keepdim=True
                        )  # (num_kv_heads, group_size, 1)
                        m_new = torch.maximum(m, m_loc)

                        # 本块归一化常数 && 全局更新
                        exp_diff = torch.exp(m - m_new)
                        sum_exp = torch.sum(
                            torch.exp(S - m_new), dim=-1, keepdim=True
                        )  # (num_kv_heads, group_size, 1)
                        l_new = l * exp_diff + sum_exp

                        # 本块加权输出
                        P = torch.exp(
                            S - m_new
                        )  # (num_kv_heads, group_size, seqlen_kv) 分子
                        O_block = torch.einsum(
                            "hgt,htd->hgd", P, v_page
                        )  # (num_kv_heads, group_size, head_dim)

                        # 在线累积输出
                        # O = O * exp_diff * (1 / l_new) + O_block * (1 / l_new)
                        O = O * exp_diff + O_block 

                        # 写回全局状态
                        m, l = m_new, l_new
                    out[q_idx] = O / l
                out = out.view(
                    seqlen_query, self.num_attention_heads, self.vo_head_dim
                )  # (seqlen_query, num_attention_heads, head_dim)
            outputs.append(out)

        # 拼接所有sequence的结果
        return torch.cat([o for o in outputs], dim=0)


# ======================= 测试与数据构造 =======================
if __name__ == "__main__":
    import argparse
    import numpy as np
    import os

    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    tp_size = 1
    qk_head_dim = 192
    vo_head_dim = 128
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--seqlen-kv", type=str, default="63,128")
    parser.add_argument("--cu-seqlen-q", type=str, default="0,1,2")
    args = parser.parse_args()

    cur_len = args.seqlen_kv.split(",")
    cu_seq_len = args.cu_seqlen_q.split(",")
    batch_size = len(cu_seq_len) - 1

    seqlens_kv = torch.from_numpy(np.array(cur_len).astype(np.int32))
    max_kv = int(seqlens_kv.max())

    cu_seqlens_query = torch.from_numpy(np.array(cu_seq_len).astype(np.int32))
    total_tokens = int(cu_seqlens_query[-1])  # 总token数
    total_kv_tokens = seqlens_kv.sum()
    page_len = 256
    max_num_blocks = aligned_up(max_kv, page_len) // page_len

    # 构造 page_ids: (batch_size, max_num_blocks)
    total_pages = 1024  # 固定页数
    page_ids = torch.zeros(batch_size, max_num_blocks, dtype=torch.int32)

    # 预计算每个sequence需要的blocks数
    blocks_per_seq = [
        aligned_up(length, page_len) // page_len
        for length in seqlens_kv.tolist()
    ]
    valid_seqs = 0
    total_blocks = 0

    # 找出能够容纳的sequence数量
    for blocks in blocks_per_seq:
        if total_blocks + blocks <= total_pages:
            total_blocks += blocks
            valid_seqs += 1
        else:
            break

    # 调整batch_size
    if valid_seqs < batch_size:
        batch_size = valid_seqs
        seqlens_kv = seqlens_kv[:batch_size]
        page_ids = page_ids[:batch_size]

    # 分配page_ids
    offset = 0
    for i, blocks in enumerate(blocks_per_seq[:batch_size]):
        candidate_ids = torch.randperm(blocks)[:blocks]  # 不再翻倍range
        page_ids[i, :blocks] = candidate_ids + offset
        offset += blocks

    # 创建KV cache
    num_attention_heads = 40 // tp_size
    num_kv_heads = 8 // tp_size  # KV头数

    # num_attention_heads = 40/8
    # num_kv_heads = 8/8

    kcache = torch.ones(total_pages, num_kv_heads, page_len, qk_head_dim).fill_(float('-inf')).to(
        torch.half
    )
    vcache = torch.ones(total_pages, num_kv_heads, page_len, vo_head_dim).fill_(float('-inf')).to(
        torch.half
    )

    # seqlens_kv[-1]

    all_k = torch.randn(num_kv_heads, total_kv_tokens, qk_head_dim).to(torch.half)
    all_v = torch.randn(num_kv_heads, total_kv_tokens, vo_head_dim).to(torch.half)
    # 为每个sequence生成并填充KV cache
    all_page_ids = []
    # print(type(total_kv_tokens))
    # print(total_kv_tokens)
    # print(total_kv_tokens[i])

    num_page_ids = len(seqlens_kv.tolist())
    kvlens_list = seqlens_kv.tolist()
    for i, length in enumerate(kvlens_list):
        print(f"i{i}")
        print(f"length{length}")
        # 直接生成正确shape的key和value
        # k = torch.randn(num_kv_heads, length, qk_head_dim).to(torch.half)
        # v = torch.randn(num_kv_heads, length, vo_head_dim).to(torch.half)
        if (i==0):
            k = all_k[:, 0:kvlens_list[i], :]
            v = all_v[:, 0:kvlens_list[i], :]
        else:
            k = all_k[:, kvlens_list[i-1]:kvlens_list[i-1]+kvlens_list[i], :]
            v = all_v[:, kvlens_list[i-1]:kvlens_list[i-1]+kvlens_list[i], :]
        print(f"k{k}")
        print(f"v{v}")


        # 按page_len分块并填充到cache
        num_valid_blocks = aligned_up(length, page_len) // page_len
        for j in range(num_valid_blocks):
            start_idx = j * page_len
            end_idx = min(start_idx + page_len, length)
            page_id = page_ids[i, j]

            # 填充该page的实际长度
            kcache[page_id, :, : (end_idx - start_idx), :] = k[
                :, start_idx:end_idx, :
            ]
            vcache[page_id, :, : (end_idx - start_idx), :] = v[
                :, start_idx:end_idx, :
            ]
            all_page_ids.append(page_id)
            print(f"all_page_ids{all_page_ids}")

    # 随机输入向量 - 改为总token数维度
    query_temp = torch.randn(total_tokens//batch_size, num_attention_heads, qk_head_dim).half()
    query = query_temp.repeat(batch_size, 1, 1)



    # 初始化并运行 GQA
    gqa = GroupedQueryAttention(tp_size=tp_size,qk_head_dim=qk_head_dim,vo_head_dim=vo_head_dim,softmax_scale=qk_head_dim **-0.5).half()
    output = gqa(
        query,
        kcache,
        vcache,
        cu_seqlens_query,
        seqlens_kv,
        page_ids,
    )

    kcache = kcache.reshape(total_pages,num_kv_heads,-1)
    vcache = vcache.reshape(total_pages,num_kv_heads,-1)
    for ids in  all_page_ids:
        for h in range(num_kv_heads) :
            kcache[ids,h,:] = kcache[ids,h,:].reshape(page_len,-1,16).permute(1,0,2).reshape(-1)
            vcache[ids,h,:] = vcache[ids,h,:].reshape(page_len,-1,16).permute(1,0,2).reshape(-1)
    kcache.numpy().tofile("./input/input_k.bin")
    vcache.numpy().tofile("./input/input_v.bin")

    query.numpy().tofile("./input/input_q.bin")
    cu_seqlens_query.numpy().tofile("./input/input_cu_seqlens_query.bin")
    seqlens_kv.numpy().tofile("./input/input_seqlens_kv.bin")
    page_ids.numpy().tofile("./input/input_page_ids.bin")
    output.numpy().tofile("./output/golden_output.bin")