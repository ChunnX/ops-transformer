# 环境配置
## 1. 环境信息
Host 121.37.53.41
  HostName 121.37.53.41
  Port 10001
  User root
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
  ServerAliveCountMax 3
  PassWord A#E(8eKqj4

## 2. 环境变量配置
```bash
conda activate rrh
source /opt/rrh/zhengbao/0609/ascend-toolkit/set_env.sh 
export ASCEND_CUSTOM_OPP_PATH=/opt/rrh/zhengbao/0609/cann/opp
cd /opt/rrh/code/ops-transformer/mhc/mhc_pre_backward
```

## 3. 编译命令
```bash
bash build.sh --pkg --soc=ascend910b --ops=mhc_pre_backward
```

## 3. 说明
1. 本地修改之后，会立刻同步到环境上，不需要手动同步。
2. 同一个问题，修改超过三次未解决，需要把问题暴露出来，和我一起讨论。

## 4. ATK
1. 在远程环境/opt/rrh/test/mhc/mhcPreBackward/目录下，运行ATK。
2. 运行命令：atk task -c result/MhcPreBackward/json/all_MhcPreBackward.json -n nodes.yaml -p gdd_executor_aclnnMhcPreBackward.py -wl [8]