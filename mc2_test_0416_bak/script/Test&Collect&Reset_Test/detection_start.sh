IPs=('100.102.180.164'
'100.102.180.168'
  )

for ip in ${IPs[*]};
do
echo "11111111"
echo $ip
# nohup  ./exp_cmd.ex $ip "mkdir -p /home/0619" &
cd /home/w00836691/collectop/
nohup  ./exp_cmd.ex $ip  "cd /home/w00836691/2server;bash  rundetection.sh 2 ResetV2_fanhua_caselist_50.xlsx detection_server3.py $1 $1 > train.log 2>&1" &
done
