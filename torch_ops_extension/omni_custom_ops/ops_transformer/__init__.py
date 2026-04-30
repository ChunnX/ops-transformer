"""
本文件通过mokey patch的方式用部分自定义接口替换主线接口
1、inherit_globals:获取被替换接口在原模块中的globals环境
2、patch_graphs:获取原始接口后替换为自定义接口
"""

def inherit_globals(original_func):
    def decorator(new_func):
        original_globals = original_func.__globals__
        new_globals = new_func.__globals__
        new_globals.update(original_globals)
        return new_func
    return decorator

def patch_graphs():
    from torch_npu.npu.graphs import _GraphDispatchMode
    original_update_capture_record = _GraphDispatchMode.update_capture_record
    original_torch_dispatch = _GraphDispatchMode.__torch_dispatch__
    original_append_dispatch_record = _GraphDispatchMode._append_dispatch_record
    
    @inherit_globals(original_update_capture_record)
    def update_capture_record(self, cpu_update_input):
        if len(cpu_update_input) == 1:
            new_list = [cpu_update_input[0].copy() for _ in range(len(self.graph_dispatch_records))]
            cpu_update_input = new_list 
        if len(self.graph_dispatch_records) != len(self.graph_dispatch_records):
            raise RuntimeError(f"Currently, there are {len(self.graph_dispatch_records)} operators that need to be updated by capture, "
                f"and there are only {len(self.graph_dispatch_records)} elements in the incoming cpu_update_input list", pta_error(ErrCode.PARAM))
        with torch.npu.stream(self.update_stream):
            for graph_dispatch_record, update_input in zip(self.graph_dispatch_records, cpu_update_input):
                graph_task_update_begin(self.update_stream, graph_dispatch_record.handle)
                for key in update_input:
                    if key in graph_dispatch_record.kwargs:
                        graph_dispatch_record.kwargs[key] = update_input[key]
                # When parameters are passed through args, position is position of the updated parameter.
                if graph_dispatch_record.op_cache_entry.__name__ in ["npu_fused_infer_attention_score", "npu_fused_infer_attention_score.out"]:
                    position, key = 6, "actual_seq_lengths_kv"
                elif graph_dispatch_record.op_cache_entry.__name__ in ["npu_fused_infer_attention_score_v2", "npu_fused_infer_attention_score_v2.out"]:
                    position, key = 8, "actual_seq_kvlen"
                elif graph_dispatch_record.op_cache_entry.__name__ in ["npu_sparse_flash_attention_gqa", "npu_sparse_flash_attention_gqa.out"]:
                    position, key = 7, "actual_seq_lengths_kv"
                elif graph_dispatch_record.op_cache_entry.__name__ == "_npu_paged_attention.default":
                    position, key = 7, "context_lens"
                elif graph_dispatch_record.op_cache_entry.__name__ == "npu_multi_head_latent_attention.out":
                    position, key = 5, "context_lens"
                elif graph_dispatch_record.op_cache_entry.__name__ in ["npu_esa_select_topk", "npu_esa_select_topk.out", "npu_esa_select_topk.default"]:
                    position, key = 8, "actual_seq_k_len_optional"
                if len(graph_dispatch_record.args) >= (position + 1):
                    graph_dispatch_record.args[position] = update_input[key]
                graph_dispatch_record.op_cache_entry(*graph_dispatch_record.args, **graph_dispatch_record.kwargs)
                graph_task_update_end(self.update_stream)
                graph_dispatch_record.event.record(self.update_stream)

    @inherit_globals(original_append_dispatch_record)
    def _append_dispatch_record(self, event, handle, args, kwargs, func):
        args_ref = []
        for element in args:
            if torch.is_tensor(element) and "npu" in str(element.device):
                args_ref.append(TensorWeakRef(element))
            else:
                args_ref.append(deepcopy(element))
        kwargs_ref = {}

        for key, vaule in kwargs.items():
            if key == "out":
                if isinstance(vaule, torch.Tensor):
                    kwargs_ref[key] = TensorWeakRef(vaule)
                else:
                    kwargs_ref[key] = [TensorWeakRef(vaule[0]), TensorWeakRef(vaule[1])]
            elif key in self.tensor_schema_name[str(func.__name__)]:
                kwargs_ref[key] = TensorWeakRef(vaule)
            else:
                kwargs_ref[key] = deepcopy(vaule)
        return _GraphDispatchRecord(event=event, handle=handle, kwargs=kwargs_ref, args=list(args_ref), op_cache_entry=func)

    @inherit_globals(original_torch_dispatch)
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.__name__ in ["npu_esa_select_topk.out"]:
            # 记录schema
            self.update_schema(str(func.__name__), str(func._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # begin graph task
            graph_task_group_begin(stream)
            result = func(*args, **kwargs)
            handle = graph_task_group_end(stream)

            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func))
            return kwargs["out"]
        elif func.__name__ in ["npu_esa_select_topk.default", "npu_esa_select_topk"]:
            func_out = torch.ops.custom.npu_esa_select_topk.out
            self.update_schema(str(func_out.__name__), str(func_out._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            workspace = torch.ops.custom._npu_esa_select_topk_get_max_workspace(*args, **kwargs)
            out_args = [args[0], args[1], args[3], args[4], args[5], args[6]]
            kwargs["out"] = torch.ops.custom._npu_esa_select_topk_infer_output(*out_args)
            # begin graph task
            graph_task_group_begin(stream)
            func_out(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func_out))
            return kwargs["out"]
        elif func.__name__ in ["npu_fused_infer_attention_score", "npu_fused_infer_attention_score.default"]:
            func_out = torch_npu.npu_fused_infer_attention_score.out
            self.update_schema(str(func_out.__name__), str(func_out._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # apply tensor
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(*args, **kwargs)
            out_args = [args[0], args[2]]
            out_kwargs_keys = [
                'input_layout', 
                'quant_scale2',
                'block_table',
                'num_heads',
                'num_key_value_heads',
                'softmax_lse_flag',
                'query_rope']
            out_kwargs = {key: kwargs[key] for key in out_kwargs_keys if key in kwargs}
            output, softmax_lse = torch_npu._npu_fused_infer_attention_score_infer_output(*out_args, **out_kwargs)
            kwargs["workspace"] = workspace
            kwargs["out"] = [output, softmax_lse]
            # begin graph task
            graph_task_group_begin(stream)
            func_out(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func_out))
            return kwargs["out"]
        elif func.__name__ in ["npu_fused_infer_attention_score_v2", "npu_fused_infer_attention_score_v2.default"]:
            func_out = torch_npu.npu_fused_infer_attention_score_v2.out
            self.update_schema(str(func_out.__name__), str(func_out._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # apply tensor
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(*args, **kwargs)
            out_args = [args[0], args[2]]
            out_kwargs_keys = [
                'query_dtype',
                'value_dtype',
                'input_layout',
                'quant_scale_out',
                'block_table',
                'num_query_heads',
                'num_key_value_heads',
                'return_softmax_lse',
                'query_rope',
                'out_dtype']
            out_kwargs = {key: kwargs[key] for key in out_kwargs_keys if key in kwargs}
            output, softmax_lse = torch_npu._npu_fused_infer_attention_score_v2_infer_output(*out_args, **out_kwargs)
            kwargs["workspace"] = workspace
            kwargs["out"] = [output, softmax_lse]
            # begin graph task
            graph_task_group_begin(stream)
            func_out(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func_out))
            return kwargs["out"]
        elif func.__name__ in ["npu_sparse_flash_attention_gqa", "npu_sparse_flash_attention_gqa.default"]:
            func_out = torch.ops.custom.npu_sparse_flash_attention_gqa.out
            self.update_schema(str(func_out.__name__), str(func_out._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # apply tensor
            workspace = torch.ops.custom.npu_sparse_flash_attention_gqa_get_max_workspace(*args, **kwargs)
            out_args = [args[0], args[2]]
            out_kwargs_keys = [
                'block_table',
                'num_query_heads',
                'num_key_value_heads',
                'input_layout',
                'return_softmax_lse']
            out_kwargs = {key: kwargs[key] for key in out_kwargs_keys if key in kwargs}
            output, softmax_lse = torch.ops.custom.npu_sparse_flash_attention_gqa_infer_output(*out_args, **out_kwargs)
            kwargs["workspace"] = workspace
            kwargs["out"] = [output, softmax_lse]
            # begin graph task
            graph_task_group_begin(stream)
            func_out(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func_out))
            return kwargs["out"]
        elif func.__name__ in ["npu_fused_infer_attention_score.out", "npu_fused_infer_attention_score_v2.out", "npu_sparse_flash_attention_gqa.out"]:
            self.update_schema(str(func.__name__), str(func._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # begin graph task
            graph_task_group_begin(stream)
            func(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func))
            return kwargs["out"]
        elif func.__name__ in ["_npu_paged_attention.default", "npu_multi_head_latent_attention.out"]:
            self.update_schema(str(func.__name__), str(func._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # begin graph task
            graph_task_group_begin(stream)
            result = func(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func))
            return result
        else:
            return func(*args, **kwargs)
    
    _GraphDispatchMode.update_capture_record = update_capture_record
    _GraphDispatchMode.__torch_dispatch__ = __torch_dispatch__
    _GraphDispatchMode._append_dispatch_record = _append_dispatch_record

patch_graphs()