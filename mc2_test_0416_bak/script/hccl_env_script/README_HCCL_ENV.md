## 脚本执行方式


```
#用法：./run_hccl_env.sh env_file.csv "command"
bash run_hccl_env.sh hccl_env.csv "python3 moe_operator_test_v1.py"
```
## 配置须知
- 对应环境变量需要配置在如上命令入参的csv里，如下图所示，每一行为每一次执行指定命令时export的环境变量，如果单次执行需要同时export两个环境变量，请参考16行将两个环境变量以空格分离
![image](pic/hccl_env_csv.png 'hccl_env_csv.png')

- 需注意，脚本内对于环境变量的分隔是按照空格分隔处理，如下环境变量，需要把空格去掉 --> xxx=x,x,x 
![image](pic/hccl_env_csv1.png 'hccl_env_csv1.png')

## 结果汇总
最终执行结果如下图所示，
![image](pic/hccl_env_res.png 'hccl_env_res.png')
如执行失败，则会有如下打印，可查看对应日止定位
![image](pic/hccl_env_res1.png 'hccl_env_res1.png')