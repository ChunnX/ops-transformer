#以AclnnLayerNorm算子为例，该算子期望的tensor为非连续tensor，希望tensor里的值每隔一个数取一个值，这种情况下需要手动指定tensor的storage shape，示例代码实现如下，get_storage_shape的取值规则与get_format一致

import copy
@register("layer_norm_backward")
class AclnnLayerNormBackward(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.Input_Data = copy.deepcopy(input_data)  # 备份一个做切分之前的原始参数
        input_data.kwargs["gradOut"] = input_data.kwargs["gradOut"][..., ::2]
        input_data.kwargs["input"] = input_data.kwargs["input"][..., ::2]
        input_data.kwargs["normalizedShape"][-1] = input_data.kwargs["input"].shape[-1]
        input_data.kwargs["weightOptional"] = input_data.kwargs["weightOptional"][..., ::2]
        input_data.kwargs["biasOptional"] = input_data.kwargs["biasOptional"][..., ::2]

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name is not None:
            return self.Input_Data.kwargs[name].shape  # 把原始参数里的shape作为storage shape给aclnn用
        else:
            return None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOut = input_data.kwargs["gradOut"]
        input = input_data.kwargs["input"]
        normalizedShape = input_data.kwargs["normalizedShape"]
        mean = input_data.kwargs["mean"]
        rstd = input_data.kwargs["rstd"]
        weight = input_data.kwargs["weightOptional"]
        bias = input_data.kwargs["biasOptional"]
        outMask = input_data.kwargs["outputMask"]
        if self.device == 'cpu' and (input.dtype == torch.float16 or input.dtype == torch.bfloat16):
            res = torch.ops.aten.native_layer_norm_backward(gradOut.to(torch.float32), input.to(torch.float32),
                                                            normalizedShape, mean.to(torch.float32),
                                                            rstd.to(torch.float32), weight.to(torch.float32),
                                                            bias.to(torch.float32), outMask)
            output = tuple(tensor.to(input_data.kwargs["input"].dtype) for tensor in res)
        else:
            output = torch.ops.aten.native_layer_norm_backward(gradOut, input, normalizedShape, mean, rstd, weight,
                                                               bias, outMask)
        return output