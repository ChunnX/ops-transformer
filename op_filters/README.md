输入文件位置（op_filters目录）：
- changed_files.txt
- whitelist.txt
- result.json
执行示例：
# 逐步执行
python ops-transformer/op_filters/op_whitelist_step1.py
python ops-transformer/op_filters/op_ignored_step2.py
python ops-transformer/op_filters/op_non_mc2_step3.py
python ops-transformer/op_filters/op_common_3rd_step4.py
python ops-transformer/op_filters/op_extractor_step5.py
python ops-transformer/op_filters/op_dependency_step6.py
python ops-transformer/op_filters/op_output_step7.py
# 或统一调用
python ops-transformer/op_filters/select_op_tests_main.py --pr ops-transformer/op_filters/changed_files.txt