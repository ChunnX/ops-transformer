import importlib
import math
import os
import struct

import torch
import torch_npu  # noqa: F401: registers NPU and npugraph_ex

# ScheduleContext ABI: common/include/op_kernel/attention_ffn_schedule.h.
# The first 6 uint32 words describe dimensions and the byte stride of token data.
CONTEXT_BYTES = 1024
FFN_AREA_OFFSET = 384
POLLING_INDEX_OFFSET = FFN_AREA_OFFSET + 4 * 8
MASK_ID = torch.iinfo(torch.int32).max


class AsyncLayerExample:
    """Keep all device allocations alive while context contains their pointers."""

    def __init__(self, device):
        self.sessions, self.batches, self.bs, self.k, self.h = 3, 2, 2, 2, 256
        self.layers, self.experts = 2, 8
        self.selected_batch = 1
        self.selected_sessions = (0, 1)
        # Both sessions use the SAME layer-local expert IDs, but different layers.
        self.local_ids = torch.tensor([[1, 0], [3, 1]], dtype=torch.int32)
        self.layer_ids = (1, 0, 1)
        shape = (self.sessions, self.batches, self.bs, self.k, self.h)
        self.tokens_cpu = (torch.arange(math.prod(shape)) % 97).half().reshape(shape)
        self.tokens = self.tokens_cpu.to(device)
        self.descriptors_cpu = torch.zeros(
            (self.sessions, self.batches, 2 + self.bs * self.k), dtype=torch.int32
        )
        for session, layer in enumerate(self.layer_ids):
            # FfnDataDesc = [flag, layer_id, expert_ids[BS*K]].
            self.descriptors_cpu[session, :, 1] = layer
            self.descriptors_cpu[session, :, 2:] = self.local_ids.flatten()
        # Batch 0 is empty. Only two of three sessions are ready in batch 1.
        # The async receiver must skip batch 0 and must not wait for session 2.
        self.descriptors_cpu[list(self.selected_sessions), self.selected_batch, 0] = 1
        self.descriptors = self.descriptors_cpu.to(device)
        context = bytearray(CONTEXT_BYTES)
        struct.pack_into(
            "<6I",
            context,
            0,
            self.sessions,
            self.batches,
            self.bs,
            self.k,
            self.experts // self.layers,
            self.h * self.tokens.element_size(),
        )
        struct.pack_into(
            "<4Q",
            context,
            FFN_AREA_OFFSET,
            self.descriptors.data_ptr(),
            self.descriptors.numel() * self.descriptors.element_size(),
            self.tokens.data_ptr(),
            self.tokens.numel() * self.tokens.element_size(),
        )
        # common.expert_num is per-layer; the operator attribute expert_num is
        # the total across layers. RECV does not use the NORM layer_ids_buf.
        self.context_cpu = torch.frombuffer(context, dtype=torch.int8).clone()
        self.context = self.context_cpu.to(device)

    def publish(self):
        # RECV consumes flag/IDs and advances polling_index. Restore IN PLACE
        # before each replay, preserving every address captured by the graph.
        # Copies and the consumer run on the same stream, so publication is ordered.
        self.descriptors.copy_(self.descriptors_cpu)
        self.context.copy_(self.context_cpu)

    def check(self, outputs):
        y, groups, sessions, batches, tokens, offsets, _, count = [
            t.cpu() for t in outputs
        ]
        count = int(count.item())
        expected = {}
        for session in self.selected_sessions:
            for token in range(self.bs):
                for choice in range(self.k):
                    expert = int(self.local_ids[token, choice])
                    expected[(session, self.selected_batch, token, choice)] = (
                        self.layer_ids[session] * (self.experts // self.layers) + expert
                    )
        indices = list(
            zip(
                sessions[:count].tolist(),
                batches[:count].tolist(),
                tokens[:count].tolist(),
                offsets[:count].tolist(),
            )
        )
        assert count == len(expected) and len(set(indices)) == count
        assert set(indices) == set(expected), indices
        keys = [expected[index] for index in indices]
        assert keys == sorted(keys), keys
        # Do not assume a particular order among tokens with equal expert IDs.
        torch.testing.assert_close(
            y[:count],
            torch.stack([self.tokens_cpu[i] for i in indices]),
            rtol=0,
            atol=0,
        )
        expected_groups = torch.zeros((self.experts, 2), dtype=torch.int64)
        for row, expert in enumerate(sorted(set(expected.values()))):
            expected_groups[row] = torch.tensor(
                [expert, list(expected.values()).count(expert)]
            )
        torch.testing.assert_close(groups, expected_groups, rtol=0, atol=0)
        after = self.descriptors_cpu.clone()
        for session in self.selected_sessions:
            after[session, self.selected_batch, 0] = 0
            after[session, self.selected_batch, 2:] = MASK_ID
        torch.testing.assert_close(self.descriptors.cpu(), after, rtol=0, atol=0)
        context_bytes = self.context.cpu().numpy().tobytes()
        assert struct.unpack_from("<Q", context_bytes, POLLING_INDEX_OFFSET)[0] == 0
        print("actual_token_num:", count)
        print("sorted cross-layer expert IDs:", keys)
        print("group_list [expert_id, token_count]:", groups[groups[:, 1] > 0].tolist())
        print(
            "PASS: token reorder, group_list, descriptor consumption and unselected sessions"
        )


def main():
    torch.npu.set_device(0)
    # Explicit loading also works for selectively packaged vendor wheels.
    op = importlib.import_module(
        "cann_ops_transformer.ops.ffn.ffn_worker_batching"
    ).ffn_worker_batching
    case = AsyncLayerExample("npu:0")

    def call(context):
        return op(
            context,
            case.experts,
            [case.sessions, case.bs, case.k, case.h],
            token_dtype=0,
            need_schedule=1,
            layer_num=case.layers,
            sync_flag=1,
        )

    with torch.inference_mode():
        # Publish fresh descriptors before the consuming call.
        case.publish()
        outputs = call(case.context)
        torch.npu.synchronize()
        case.check(outputs)


if __name__ == "__main__":
    main()
