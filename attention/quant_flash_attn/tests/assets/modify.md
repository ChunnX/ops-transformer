quant_flash_attn tests assets整改

1、删除多余的spec
spec.py中"qfa_metadata_wrapper.run_metadata"，"qfa_main_wrapper.run_main"都用不到，删除相关的文件和代码

2、拆分metadata
npu_qfa中的调用会调用quant_flash_attn_golden.py中的golden，比如npu_gqa_fp8_fa，npu_hif8_fa，npu_mxfp8_fa，这些函数都是两段式的，需要先调用metadata接口，然后再调用算子接口，现在要把metadata单独剥离出去，放到attention\quant_flash_attn\tests\assets\impl路径下，新建一个npu_preprocess.py文件，参考attention\lightning_indexer_v2\tests\assets\impl\npu_preprocess.py

3、spec修改
由于metadata剥离出去了，所以不需要wrapper把两个调用封装起来，只需要保留一个，同时把QfaMxfp8Spec修改名称为QuantFlashAttnSpec：
 __spec__ = {	 
     "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnSpec",
 }
QuantFlashAttnSpec类中要添加npu_preprocess函数，供ttk接口调用